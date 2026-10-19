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
_op_init, _op_feed, _op_finish = F.OutputParser.__init__, F.OutputParser.feed, F.OutputParser.finish
WRAPPERS = (CALLS_OPEN, CALLS_CLOSE)


def _op_init_ds41(self, *a, **k):
    _op_init(self, *a, **k)
    self.stream_tools = False          # the streaming scanner reads Qwen's <function=...> body
    self.dsml_hold = ""


def _strip_wrappers(self, delta: str, final: bool) -> str:
    """Drop <｜DSML｜ calls> / </｜DSML｜ calls> from the stream, holding a partial one until it is decided."""
    buf = self.dsml_hold + delta
    for w in WRAPPERS:
        buf = buf.replace(w, "")
    keep = 0 if final else self._hold(buf, WRAPPERS)
    self.dsml_hold = buf[len(buf) - keep:] if keep else ""
    return buf[:len(buf) - keep] if keep else buf


def _op_feed_ds41(self, delta: str):
    return _op_feed(self, _strip_wrappers(self, delta, False))


def _op_finish_ds41(self, *a, **k):   # *a/**k: the 0.1.44 server's finish(reason=...)
    out = _op_feed(self, _strip_wrappers(self, "", True)) if self.dsml_hold else []
    return out + _op_finish(self, *a, **k)


F.OutputParser.__init__ = _op_init_ds41
F.OutputParser.feed = _op_feed_ds41
F.OutputParser.finish = _op_finish_ds41
S.OutputParser = F.OutputParser

# ---- full text log (llama.cpp-style): every request's thinking / answer / tool calls as they are generated, so a run
# can be watched (`tail -f`) and checked afterwards (loops, what it was thinking).  STRATA_TEXT_LOG=<path> (default
# /tmp/strata-ds41-text.log), STRATA_TEXT_LOG=0 = off.
import threading as _th
import time as _time

_TEXT_LOG = os.environ.get("STRATA_TEXT_LOG", os.path.join(__import__("tempfile").gettempdir(), "strata-ds41-text.log"))
_text_lock = _th.Lock()
_text_state = {"kind": None}
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
        if label != _text_state["kind"]:
            _text_state["kind"] = label
            out.append(f"\n--- {label} ---\n")
        out.append(ev.text)
    if out:
        _text_write("".join(out))
    return _svc_note(self, n, evs, st, rate)


# a model name ending in one of these (a config alias) = that routing for the request, for the strata_tune keys the
# request doesn't name: -exact = no decode biases (quality mode), -fast = the stronger 0.5 / 0.3 nudge (speed mode); the
# plain name gets the config's --route-bias / --route-bias-file
ROUTE_SUFFIXES = {"-exact": {"route_bias": 0.0, "route_bias_file": 0.0},
                  "-fast": {"route_bias": 0.5, "route_bias_file": 0.3}}


def _run_ds41(self, ids, thinking, tools, max_new, sampling, cancel, **kw):   # **kw: the server's newer keywords (0.1.44: force=)
    model = sampling.get("model") if isinstance(sampling, dict) else None
    suffix = next((s for s in ROUTE_SUFFIXES if isinstance(model, str) and model.endswith(s)), None)
    if suffix:
        tune = dict(sampling.get("strata_tune") or {}) if isinstance(sampling.get("strata_tune"), dict) else {}
        for k, v in ROUTE_SUFFIXES[suffix].items():
            tune.setdefault(k, v)
        sampling = {**sampling, "strata_tune": tune}
    t0 = _time.time()
    _text_state["kind"] = None
    _text_write(f"\n\n===== {_time.strftime('%Y-%m-%d %H:%M:%S')}  request: prompt {len(ids):,} tokens, max {max_new}, "
                f"thinking {'on' if thinking else 'off'}, tools {len(tools or [])} =====\n")
    end = "(stream closed)"
    try:
        for item in _svc_run(self, ids, thinking, tools, max_new, sampling, cancel, **kw):
            if item[0] == "done" and isinstance(item[1], dict):
                d = item[1]
                end = f"finish {d.get('finish')}, {d.get('completion_tokens')} tokens"
            yield item
    finally:
        _text_write(f"\n===== end: {end}, {_time.time() - t0:.1f} s =====\n")


S.Service._note = _note_ds41
S.Service.run = _run_ds41

if __name__ == "__main__":
    for i, v in enumerate(sys.argv[:-1]):   # paths are the caller's; the server runs from Strata's directory
        if v in ("--config", "--tokenizer", "--mcp-config"):
            sys.argv[i + 1] = os.path.abspath(sys.argv[i + 1])
    os.chdir(str(STRATA))
    sys.exit(S.main())
