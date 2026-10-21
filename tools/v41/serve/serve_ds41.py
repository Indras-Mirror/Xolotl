#!/usr/bin/env python3
"""Strata's API server (~/AI/Strata/serve/server.py: OpenAI + Anthropic endpoints, streaming, sessions, llama.cpp-style
metrics) in front of the DeepSeek-V4.1-Flash engine (`ds41_generate --serve`).  The server is Qwen3.8's; this shim
swaps in V4.1's pieces (same shape as ~/AI/Strata-GLM/tools/glm/serve/serve_glm.py):

- tokenizer: the GGUF's BPE (tokenizer dir from export_tokenizer.py) with llama.cpp's `joyai-llm` pre-split
  (Strata-DS4's strata_tokenizer.py; == HF tokenizers on 465,645 tokens)
- chat template: DeepSeek's own encoder (dsv41_encoding.encode_messages), not Jinja
- reasoning effort: V4.1's numeric 1-100 ("Reasoning Effort: N" system prefix).  Names as DwarfStar's server
  (antirez/ds4 ds4_server.c parse_reasoning_effort_name + ds4_deepseek41_reasoning_effort_text): high / xhigh = 75,
  max = 100, a number 1-100 as given, none / 0 = thinking off; low = 50 (DeepSeek's encoder), medium = 62
- end of turn: <｜end▁of▁sentence｜>
- tool calls: DSML (`<｜DSML｜ invoke name="X">` + `<｜DSML｜ parameter name="K" string="true|false">V</｜DSML｜ parameter>`,
  note the LEADING SPACE in the tag names) parsed as DeepSeek's parse_tool_calls does; the `calls` wrapper is
  stripped from the stream.  A value may carry DwarfStar's escape of its own closing tag (ds4_tool_text.h): undone.

  serve_ds41.py --engine strata --config strata-ds41.json --port 8150
"""
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V41 = HERE.parent
# xolotl: the server is THIS tree's serve/ (it used to import ~/AI/Strata's, a separate upstream checkout at 0.1.39)
STRATA = V41.parent.parent
sys.path.insert(0, str(STRATA))
sys.path.insert(0, str(STRATA / "tools"))
sys.path.insert(0, str(V41.parent))          # tools/: strata_tokenizer.py with the joyai-llm pre-split
sys.path.insert(0, str(V41))

import strata_tokenizer as ST  # noqa: E402
assert "joyai-llm" in getattr(ST, "PRE_PATTERNS", {}), ST.__file__
import dsv41_encoding as E  # noqa: E402
from serve import frontend as F  # noqa: E402
from serve import server as S  # noqa: E402

# ---- tokenizer: the server builds ST.Tokenizer(tokens, merges, types); give it V4.1's pre-split
_Tok = ST.Tokenizer


class Ds41Tokenizer(_Tok):
    def __init__(self, tokens, merges, token_types=None, pre="joyai-llm", **k):
        super().__init__(tokens, merges, token_types, pre="joyai-llm", **k)
        self._literal = {i: t.encode("utf-8") for t, i in self.special_tokens.items()}

    def token_bytes(self, i: int) -> bytes:
        # V4.1's special tokens (<｜User｜>, <｜end▁of▁sentence｜>, ...) are not in the byte alphabet: they decode to
        # their own text, as llama.cpp detokenizes control / user-defined tokens (Qwen's are ASCII, so never hit this)
        lit = self._literal.get(i)
        return lit if lit is not None else super().token_bytes(i)


ST.Tokenizer = Ds41Tokenizer

# ---- reasoning effort: client spellings -> template kwargs (reasoning_effort = an int 1-100, or thinking off)
EFFORT = {"none": 0, "off": 0, "disabled": 0, "false": 0, "minimal": 50, "low": 50, "medium": 62, "high": 75,
          "xhigh": 75, "max": 100, "maximum": 100}


def effort_kwargs(value) -> dict:
    if value is None or value == "":
        return {}
    if value is False:
        return {"enable_thinking": False}
    key = str(value).strip().lower()
    if key.isdigit() and 0 <= int(key) <= 100:
        level = int(key)
    elif key in EFFORT:
        level = EFFORT[key]
    else:
        raise ValueError(f"unknown reasoning effort {value!r}: use none, low, medium, high, max or 1-100")
    return {"enable_thinking": False} if level == 0 else {"reasoning_effort": level}


def budget_effort(tokens) -> dict:
    """Anthropic's thinking budget: under 2K low, under 8K medium, else high (as the server's own)."""
    try:
        n = int(tokens)
    except (TypeError, ValueError):
        return {}
    return {"reasoning_effort": 50 if n < 2048 else 62 if n < 8192 else 75}


for mod in (F, S):
    if hasattr(mod, "effort_kwargs"):
        mod.effort_kwargs = effort_kwargs
    if hasattr(mod, "budget_effort"):
        mod.budget_effort = budget_effort


def _level(effort):
    """Whatever reaches the template (ours, or the server's own names) -> 1..100, or None for the default (75)."""
    if effort is None or effort == "":
        return None
    if isinstance(effort, int):
        return max(1, min(100, effort))
    key = str(effort).strip().lower()
    if key.isdigit():
        return max(1, min(100, int(key)))
    return EFFORT.get(key) or None


# ---- chat template: DeepSeek's encoder
GEN_SUFFIXES = (E.ASSISTANT_SP_TOKEN + E.thinking_start_token, E.ASSISTANT_SP_TOKEN + E.thinking_end_token)


def _text(content):
    if content is None or isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in ("text", "input_text", None))


def to_encoder_messages(messages, tools):
    out = []
    for m in messages:
        m = dict(m)
        content = m.get("content")
        if isinstance(content, list) and any(p.get("type") in ("image", "image_url", "input_image")
                                              for p in content if isinstance(p, dict)):
            m["content"] = [{**p, "type": "image_url" if p.get("type") == "input_image" else
                            "text" if p.get("type") == "input_text" else p.get("type")}
                            for p in content if isinstance(p, dict)]
        else:
            m["content"] = _text(content)
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = dict(c.get("function") or {})
                fn.setdefault("arguments", "{}")
                calls.append({"type": "function", "id": c.get("id"), "function": fn})
            m["tool_calls"] = calls
        out.append(m)
    if tools:   # the server passes flat {"name", "description", "parameters"}; the encoder wants OpenAI's wrapper
        tools = [t if "function" in t else {"type": "function", "function": t} for t in tools]
        if out and out[0].get("role") == "system":
            out[0] = dict(out[0], tools=tools)
        else:
            out.insert(0, {"role": "system", "content": "", "tools": tools})
    return out


_ChatTemplate = F.ChatTemplate   # the server's own, kept for its capability probe


class Ds41Template:
    """Stands in for F.ChatTemplate: render(messages, tools, add_generation_prompt, **kwargs) -> prompt text."""
    source = "{# DeepSeek-V4.1: rendered by serve_ds41.py with DeepSeek's encoder (dsv41_encoding.py) #}"

    def __init__(self, path=None):
        self._caps = None

    @property
    def caps(self):
        """xolotl: the 0.1.44 server reports the template's capabilities (props: chat_template_caps), detected by
        F.ChatTemplate._detect_caps through render() and the tool-call parser - both DeepSeek's here."""
        if self._caps is None:
            try:
                self._caps = _ChatTemplate._detect_caps(self)
            except Exception:   # noqa: BLE001 - a hint only (as in the server): never fail a request over it
                self._caps = {"supports_tools": True, "supports_tool_calls": True, "supports_system_role": True,
                              "supports_parallel_tool_calls": True, "supports_preserve_reasoning": False}
        return self._caps

    def render(self, messages, tools=None, add_generation_prompt=True, **kw):
        mode = "chat" if kw.get("enable_thinking") is False else "thinking"
        prompt = E.encode_messages(to_encoder_messages(messages, tools), thinking_mode=mode,
                                   reasoning_effort=_level(kw.get("reasoning_effort")))
        if not add_generation_prompt:
            for s in GEN_SUFFIXES:
                if prompt.endswith(s):
                    return prompt[:-len(s)]
        return prompt


F.ChatTemplate = Ds41Template
S.ChatTemplate = Ds41Template

# ---- end of turn
S.IM_END = E.eos_token
_svc_init = S.Service.__init__


def _svc_init_ds41(self, engine, tokenizer, template, *a, **k):
    _svc_init(self, engine, tokenizer, template, *a, **k)
    # REPLACE, not add to: the server also encodes Qwen's "<|endoftext|>", which V4.1's vocabulary splits into plain
    # pieces ("<", "|", ...) - every DSML tag would then end the reply
    eos = tokenizer.encode(E.eos_token, parse_special=True)
    assert len(eos) == 1, eos
    self.stop_ids = set(eos)
    self.effort_end = False      # #458's tail-turn trick is Qwen text; V4.1's effort lives in the first system turn
    self.literals = {**self.literals, E.IMAGE_PLACEHOLDER: "\U000f0e20"}


S.Service.__init__ = _svc_init_ds41


# ---- the dashboard's routing note (Service.mode_note): the config's decode biases until a request names its own
def _config_routing():
    """(route_bias, route_bias_file, protect, pool) from the --config the server was started with; the engine's
    defaults (1.0 / 0.6, no smart router) for what it doesn't set."""
    try:
        cfg = json.load(open(sys.argv[sys.argv.index("--config") + 1], encoding="utf-8"))
    except (ValueError, IndexError, OSError):
        return 1.0, 0.6, 0, 0
    args, env = [str(x) for x in cfg.get("args", [])], cfg.get("env") or {}
    def flag(name, dflt):
        try:
            return float(args[args.index(name) + 1])
        except (ValueError, IndexError):
            return dflt
    def rank(name):
        try:
            return max(0, int(env.get(name, 0)))
        except (TypeError, ValueError):
            return 0
    return flag("--route-bias", 1.0), flag("--route-bias-file", 0.6), rank("STRATA_RB_PROTECT"), rank("STRATA_RB_POOL")


_ROUTING = _config_routing()


def _routing_note(rb, rbf, protect, pool):
    if protect or pool:
        return f"smart routing (top {protect}, pool {pool}, {rb:g} / {rbf:g})"
    if rb == 0 and rbf == 0:
        return "exact routing"
    for name, (a, b) in (("light", (0.25, 0.15)), ("fast", (0.5, 0.3))):
        if abs(rb - a) < 1e-9 and abs(rbf - b) < 1e-9:
            return f"{name} routing ({rb:g} / {rbf:g})"
    return f"route bias {rb:g} / {rbf:g}"


_svc_init_routing = S.Service.__init__


def _svc_init_note(self, *a, **k):
    _svc_init_routing(self, *a, **k)
    self.mode_note = _routing_note(*_ROUTING)


S.Service.__init__ = _svc_init_note


def _expand_images_ds41(self, ids, encoded):
    marker = self.tok.encode(E.IMAGE_PLACEHOLDER, parse_special=True)
    if len(marker) != 1:
        raise ValueError("the tokenizer must encode the image marker as one token")
    out, k = [], 0
    for token in ids:
        if token == marker[0]:
            if k == len(encoded):
                raise ValueError("the prompt and its images do not match")
            out.extend([token] * encoded[k][1])
            k += 1
        else:
            out.append(token)
    if k != len(encoded):
        raise ValueError("the prompt and its images do not match")
    return out


S.Service.expand_image_tokens = _expand_images_ds41
_Vision = S.Vision


class Ds41Vision(_Vision):
    """Run the reference-based encoder with the server's existing cache and ENC protocol."""

    def __init__(self, cfg, log=None, env=None, lazy=False):
        super().__init__(cfg, log=log, env=env, lazy=True)
        args, log, env = self.spawn
        args = [cfg.get("python") or sys.executable, str(V41 / "vision" / "ds41_vision.py"), *args[1:]]
        if cfg.get("reference_dir"):
            args += ["--reference-dir", os.path.abspath(cfg["reference_dir"])]
        # "weights": the full-precision tower in place of the sidecar's Q8_0 copy (a safetensors file with the
        # checkpoint's vision.* / aligner.* / image_*, the checkpoint folder, or an mmproj GGUF); "mmproj" stays the config
        if cfg.get("weights"):
            args += ["--vision-weights", os.path.abspath(os.path.expanduser(str(cfg["weights"])))]
        # the encoder's numerics / placement (ds41_vision.py): "dtype" f32 (default) | bf16 | f16, "device" cpu | cuda
        # (the server's own "gpu": true already passes --gpu = cuda); on the card "offload": true keeps the weights in
        # pinned RAM between images, "tf32": true, "warmup_tokens": N (the warm-up grid; default max_tokens / 1024)
        if cfg.get("dtype"):
            args += ["--dtype", str(cfg["dtype"])]
        if cfg.get("device"):
            args += ["--device", str(cfg["device"])]
        if cfg.get("offload") is True:
            args.append("--offload")
        if cfg.get("tf32") is True:
            args.append("--tf32")
        if cfg.get("warmup_tokens"):
            args += ["--warmup-tokens", str(int(cfg["warmup_tokens"]))]
        self.spawn = (args, log, env)
        if not lazy and not self.lazy_vision and not self.elastic:
            self._start()


S.Vision = Ds41Vision

# ---- tool calls: DSML
CALLS_OPEN = f"<{E.dsml_token}{E.tool_calls_block_name}>"           # <｜DSML｜ calls>
CALLS_CLOSE = f"</{E.dsml_token}{E.tool_calls_block_name}>"
INVOKE_OPEN = f"<{E.dsml_token}{E.tool_call_tag_name}"               # <｜DSML｜ invoke   (name="X">)
INVOKE_CLOSE = f"</{E.dsml_token}{E.tool_call_tag_name}>"
P_OPEN = f"<{E.dsml_token}{E.tool_parameter_tag_name}"
P_CLOSE = f"</{E.dsml_token}{E.tool_parameter_tag_name}>"
_P_RE = re.compile(r' name="(.*?)" string="(true|false)">(.*?)' + re.escape(P_CLOSE), re.DOTALL)


def unescape(value: str) -> str:
    """DwarfStar escapes a value's own closing tag as &lt;... (and &amp; chains before it); DeepSeek's encoder never
    escapes, so only that exact spelling is undone."""
    tail = P_CLOSE[1:]
    return re.sub(r"&((?:amp;)*)lt;" + re.escape(tail),
                  lambda m: ("&" + m.group(1)[:-4] + "lt;" if m.group(1) else "<") + tail, value)


def ds41_call_end(text: str) -> int:
    return text.find(INVOKE_CLOSE)


def ds41_parse_tool_call(body: str, schema=None):
    m = re.match(r'\s*name="(.*?)">', body, re.DOTALL)
    if not m:
        raise ValueError("malformed tool call: " + body[:80])
    args = {}
    for key, is_str, value in _P_RE.findall(body[m.end():]):
        value = unescape(value)
        if is_str == "true":
            args[key] = value
        else:
            try:
                args[key] = json.loads(value)
            except ValueError:
                args[key] = value
    return F.ToolCall(name=m.group(1), arguments=args)


F.CALL_START, F.CALL_END = INVOKE_OPEN, INVOKE_CLOSE
# xolotl: the 0.1.44 parser treats a call opener as a call only when Qwen's `<function=` follows it (else it is
# prose naming the format); this model's follower is DSML (` name="...">`), so with FUNC_START = "" that guard always passes -
# the pre-0.1.44 behaviour.  Only the opt-in tool_call_recovery reads FUNC_START otherwise (off here).
F.FUNC_START = ""
F.call_end = ds41_call_end
F.parse_tool_call = ds41_parse_tool_call
# frontend.forced_call writes the forced-call (tool_choice) prefix in Qwen's `<function=NAME>` shape; V4.1's
# DSML puts the name in the call OPENER itself, so a forced call has to be written in that shape here.
def ds41_forced_call(tool_choice, tools):
    """tool_choice -> the DSML text that opens the call the reply must make (None: the model decides)."""
    kind, name = F.tool_choice_of(tool_choice)
    names = {t.get("name") for t in tools or [] if isinstance(t, dict)}
    if kind in ("auto", "none"):
        return None
    if kind == "required" and len(names) == 1:
        kind, name = "named", next(iter(names))
    if kind == "required" and names:
        return INVOKE_OPEN + " "
    if kind == "named" and name in names:
        return INVOKE_OPEN + f' name="{name}">'
    return None


F.forced_call = ds41_forced_call
if hasattr(S, "forced_call"):
    S.forced_call = ds41_forced_call
_op_init, _op_feed, _op_finish = F.OutputParser.__init__, F.OutputParser.feed, F.OutputParser.finish
WRAPPERS = (CALLS_OPEN, CALLS_CLOSE)


def _op_init_ds41(self, *a, **k):
    _op_init(self, *a, **k)
    self.stream_tools = False          # the streaming scanner reads Qwen's <function=...> body
    self.dsml_hold = ""
    self.echo_check, self.echo_hold = True, ""


# 2026-10-19: after the server closes a thinking budget with REASONING_WRAP_UP, the 1017 sometimes repeats that sentence
# as the first line of its answer (seen in the probe runs: the sentence correctly in the thinking AND again in the
# answer).  Drop an exact repeat at the very start of the answer; anything else passes through unchanged.
WRAP_SENTENCE = S.REASONING_WRAP_UP.split("</think>")[0].strip()


def _drop_wrap_echo(self, evs):
    out = []
    for ev in evs:
        if not self.echo_check or ev.kind != "content":
            if self.echo_check and self.echo_hold and ev.kind != "reasoning":   # flush a held prefix before anything else
                out.append(F.Event("content", self.echo_hold)); self.echo_hold = ""; self.echo_check = False
            out.append(ev)
            continue
        self.echo_hold += ev.text
        head = self.echo_hold.lstrip()
        if not head or WRAP_SENTENCE.startswith(head):
            continue                                   # still could be the echo: hold it
        self.echo_check = False
        if head.startswith(WRAP_SENTENCE):
            rest = head[len(WRAP_SENTENCE):].lstrip("\n")
            if rest:
                out.append(F.Event("content", rest))
        else:
            out.append(F.Event("content", self.echo_hold))
        self.echo_hold = ""
    return out


def _strip_wrappers(self, delta: str, final: bool) -> str:
    """Drop <｜DSML｜ calls> / </｜DSML｜ calls> from the stream, holding a partial one until it is decided."""
    buf = self.dsml_hold + delta
    for w in WRAPPERS:
        buf = buf.replace(w, "")
    keep = 0 if final else self._hold(buf, WRAPPERS)
    self.dsml_hold = buf[len(buf) - keep:] if keep else ""
    return buf[:len(buf) - keep] if keep else buf


def _op_feed_ds41(self, delta: str):
    return _drop_wrap_echo(self, _op_feed(self, _strip_wrappers(self, delta, False)))


def _op_finish_ds41(self, *a, **k):   # *a/**k: the 0.1.44 server's finish(reason=...)
    out = _op_feed(self, _strip_wrappers(self, "", True)) if self.dsml_hold else []
    out = _drop_wrap_echo(self, out + _op_finish(self, *a, **k))
    if self.echo_hold:                 # ended while matching: the whole sentence = the echo, a part of it = real text
        if self.echo_hold.strip() != WRAP_SENTENCE:
            out.append(F.Event("content", self.echo_hold))
        self.echo_hold = ""
    return out


F.OutputParser.__init__ = _op_init_ds41
F.OutputParser.feed = _op_feed_ds41
F.OutputParser.finish = _op_finish_ds41
S.OutputParser = F.OutputParser

# ---- "parallel": N (#465's batch slots) on the DeepSeek-V4.1 engine.  ds41_generate --serve runs Strata's slot protocol
# itself (BGEN / BADM / BT / BDONE / BSTOP / BYIELD: the server's generate_batched, unchanged) when started with
# --batch N (= its --batch-slots N): INFO batch_slots=N.  The server's own parallel_args is Strata's: it skips the flag
# whenever the args hold "--slots" - for ds41 that is the VRAM EXPERT cache, not batch slots - and caps at 8, where a
# ds41 window takes 16 rows (one per slot).  When the engine cannot run the slots (no conversation pool, a build
# without them) it says why in its log and the server says "N asked, the engine runs one at a time".
DS41_PARALLEL_MAX = 16


def ds41_parallel_args(cfg: dict, args: list[str]) -> list[str]:
    if "--batch" in args or "--batch-slots" in args:
        return []
    n = cfg.get("parallel")
    if n is None or n is False:
        return []
    if isinstance(n, bool) or not isinstance(n, int):
        print(f'[strata] "parallel" must be a whole number of requests (2..{DS41_PARALLEL_MAX}), not {n!r}: ignored',
              flush=True)
        return []
    if n <= 1:
        return []
    if os.environ.get("STRATA_DS41_PARALLEL", "1") == "0":
        print(f'[strata] "parallel": {n} ignored: the DeepSeek-V4.1 engine\'s batch slots are turned off '
              "(STRATA_DS41_PARALLEL=0); one request at a time", flush=True)
        return []
    if "--conversation-cache-mib" in args and args[args.index("--conversation-cache-mib") + 1:][:1] == ["0"]:
        print(f'[strata] "parallel": {n} ignored: the DeepSeek-V4.1 engine\'s batch slots need its conversation pool '
              "(--conversation-cache-mib 0 turns it off); one request at a time", flush=True)
        return []
    if n > DS41_PARALLEL_MAX:
        print(f'[strata] "parallel": {n} - the DeepSeek-V4.1 engine runs at most {DS41_PARALLEL_MAX} requests together '
              f"(a batch window is {DS41_PARALLEL_MAX} rows); it will use {DS41_PARALLEL_MAX}", flush=True)
        n = DS41_PARALLEL_MAX
    return ["--batch", str(n)]


S.parallel_args = ds41_parallel_args

# ---- full text log (llama.cpp-style): every request's thinking / answer / tool calls as they are generated, so a run
# can be watched (`tail -f`) and checked afterwards (loops, what it was thinking).  STRATA_TEXT_LOG=<path> (default
# /tmp/strata-ds41-text.log), STRATA_TEXT_LOG=0 = off.
import threading as _th
import time as _time

_TEXT_LOG = os.environ.get("STRATA_TEXT_LOG", os.path.join(__import__("tempfile").gettempdir(), "strata-ds41-text.log"))
_text_lock = _th.Lock()
_text_state = {"kind": None, "who": None, "n": 0}
_text_req = _th.local()   # "parallel": requests write at once - each chunk says whose it is when the writer changes
_KIND_LABEL = {"reasoning": "thinking", "content": "answer", "tool_start": "tool call", "tool_call": "tool call"}


def _text_write(s: str):
    if not _TEXT_LOG or _TEXT_LOG == "0":
        return
    try:
        with _text_lock, open(_TEXT_LOG, "a", encoding="utf-8") as f:
            f.write(s)
    except OSError:
        pass


_svc_note, _svc_run = S.Service._note, S.Service.run


def _note_ds41(self, n, evs, st=None, rate=None):
    out = []
    for ev in evs:
        if not ev.text:
            continue
        label = _KIND_LABEL.get(ev.kind, ev.kind)
        who = getattr(_text_req, "n", None)
        if label != _text_state["kind"] or who != _text_state["who"]:
            _text_state["kind"], _text_state["who"] = label, who
            out.append(f"\n--- {label} ---\n" if who is None or not getattr(self.engine, "batch", 0) else f"\n--- [#{who}] {label} ---\n")
        out.append(ev.text)
    if out:
        _text_write("".join(out))
    return _svc_note(self, n, evs, st, rate)


# a model name ending in one of these (a config alias) = that routing for the request, for the strata_tune keys the
# request doesn't name: -exact = no decode biases (quality mode), -fast = the stronger 0.5 / 0.3 nudge (speed mode); the
# plain name gets the config's --route-bias / --route-bias-file
ROUTE_SUFFIXES = {"-exact": {"route_bias": 0.0, "route_bias_file": 0.0, "route_bias_remote": 0.0},
                  "-fast": {"route_bias": 0.5, "route_bias_file": 0.3}}


def _run_ds41(self, ids, thinking, tools, max_new, sampling, cancel, **kw):   # **kw: the server's newer keywords (0.1.44: force=)
    model = sampling.get("model") if isinstance(sampling, dict) else None
    suffix = next((s for s in ROUTE_SUFFIXES if isinstance(model, str) and model.endswith(s)), None)
    if suffix:
        tune = dict(sampling.get("strata_tune") or {}) if isinstance(sampling.get("strata_tune"), dict) else {}
        for k, v in ROUTE_SUFFIXES[suffix].items():
            tune.setdefault(k, v)
        sampling = {**sampling, "strata_tune": tune}
    tune = sampling.get("strata_tune") if isinstance(sampling, dict) and isinstance(sampling.get("strata_tune"), dict) else {}
    def num(k, dflt):
        v = tune.get(k)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else dflt
    self.mode_note = _routing_note(num("route_bias", _ROUTING[0]), num("route_bias_file", _ROUTING[1]),
                                   int(num("route_protect", _ROUTING[2])), int(num("route_pool", _ROUTING[3])))
    t0 = _time.time()
    _text_state["kind"] = None
    with _text_lock:
        _text_state["n"] += 1
        _text_req.n = _text_state["n"]
    _text_write(f"\n\n===== {_time.strftime('%Y-%m-%d %H:%M:%S')}  request #{_text_req.n}: prompt {len(ids):,} tokens, max {max_new}, "
                f"thinking {'on' if thinking else 'off'}, tools {len(tools or [])} =====\n")
    end = "(stream closed)"
    try:
        for item in _svc_run(self, ids, thinking, tools, max_new, sampling, cancel, **kw):
            if item[0] == "done" and isinstance(item[1], dict):
                d = item[1]
                end = f"finish {d.get('finish')}, {d.get('completion_tokens')} tokens"
            yield item
    finally:
        _text_write(f"\n===== end #{_text_req.n}: {end}, {_time.time() - t0:.1f} s =====\n")


S.Service._note = _note_ds41
S.Service.run = _run_ds41


def _cache_slots_value(slots):
    """Engine flag value for --conversation-cache-slots; None = omit (engine default)."""
    if slots is None:
        return None
    if isinstance(slots, str) and slots.strip().lower() in ("auto", "default", ""):
        return None
    return str(int(slots))


def expand_ds41_config_cache(cfg: dict) -> None:
    """Flatten nested conversation_cache / prompt_cache; accept slots: \"auto\" (do not int() it)."""
    args = list(cfg.get("args") or [])
    cc = cfg.get("conversation_cache")
    if isinstance(cc, dict):
        mib = cc.get("mib", cc.get("budget_mib"))
        if mib is not None and "--conversation-cache-mib" not in args:
            args += ["--conversation-cache-mib", str(int(mib))]
        slots = _cache_slots_value(cc.get("slots"))
        if slots is not None and "--conversation-cache-slots" not in args:
            args += ["--conversation-cache-slots", slots]
        every = cc.get("prompt_cache_every", cc.get("every"))
        if every is not None and "--prompt-cache-every" not in args:
            args += ["--prompt-cache-every", str(int(every))]
        pc = cc.get("prompt_cache")
        if pc is not None and "--prompt-cache" not in args:
            if isinstance(pc, str) and pc.strip().lower() in ("auto", "default", ""):
                pass
            else:
                args += ["--prompt-cache", str(int(pc))]
        cfg["args"] = args


_engine_args = S.engine_args


def engine_args_ds41(cfg: dict) -> list[str]:
    expand_ds41_config_cache(cfg)
    return _engine_args(cfg)


S.engine_args = engine_args_ds41

if __name__ == "__main__":
    for i, v in enumerate(sys.argv[:-1]):   # paths are the caller's; the server runs from Strata's directory
        if v in ("--config", "--tokenizer", "--mcp-config"):
            sys.argv[i + 1] = os.path.abspath(sys.argv[i + 1])
    os.chdir(str(STRATA))
    sys.exit(S.main())
