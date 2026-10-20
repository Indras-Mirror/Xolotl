#!/usr/bin/env python3
"""build_corpus.py - render calibration sources into DeepSeek-V4.1 prompt text for v41_calib.py prep --corpus.

Every chat/agent sample goes through DeepSeek's own V4.1 encoder (dsv41_encoding.py, vendored from
deepseek-ai/DeepSeek-V4.1-Flash @ 2cba9e42), so tool calls use the V4.1 DSML tags and thinking is kept the way an
agent client sends it.  Output: $V41_WORK/corpus/<pool>.jsonl, one {"text", "src", "project"} per line.

  sessions   Claude Code / Quetza / GLM session transcripts (~/.claude/projects, ~/.quetza-data, ~/.claude-data):
             each session -> messages (user / assistant reasoning+text+tool calls / tool results) -> the longest
             prefix that fits --max-tokens.  Secrets are scrubbed; tool results are clipped; sidechains skipped.
  antirez_*  antirez/ds4 imatrix prompts (docs/ds4/calib-refs/antirez-ds4-imatrix), V4 DSML parsed back into
             structured tool calls and re-rendered for V4.1.  One pool per category.

  build_corpus.py [--max-tokens 8191] [--sessions 40] [--exclude-session ID]
"""
import argparse, glob, hashlib, json, os, re, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dsv41_encoding as E  # noqa: E402

WORK = os.environ.get("V41_WORK", "/media/Crucial1TB/v41-calib")
MODEL = os.environ.get("V41_MODEL", "/media/Crucial1TB/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8")
ANTIREZ = os.path.join(HERE, "../../docs/ds4/calib-refs/antirez-ds4-imatrix/dataset/prompts.jsonl")
TOOL_RESULT_CHARS = 6000

SECRETS = [
    (re.compile(r"hf_[A-Za-z0-9]{30,}"), "hf_REDACTED"),
    (re.compile(r"sk-(?:ant-|or-|proj-)?[A-Za-z0-9_\-]{20,}"), "sk-REDACTED"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"), "ghp_REDACTED"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AKIAREDACTED"),
    (re.compile(r"xox[abpr]-[A-Za-z0-9\-]{10,}"), "xoxb-REDACTED"),
    (re.compile(r"AIza[0-9A-Za-z_\-]{30,}"), "AIzaREDACTED"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{16,}"), r"\1REDACTED"),
    (re.compile(r"(?i)((?:api[_-]?key|token|secret|password|passwd)[\"']?\s*[:=]\s*[\"']?)[^\s\"',}]{6,}"), r"\1REDACTED"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "PRIVATE_KEY_REDACTED"),
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), "user@example.com"),
    (re.compile(r"(?<!\d)(?:\+?61\s?|0)4\d{2}[\s\-]?\d{3}[\s\-]?\d{3}(?!\d)"), "0400 000 000"),
    (re.compile(r"\b(?:5[1-5]\d{2}|4\d{3})(?:[\s\-]?\d{4}){3}\b"), "CARD_REDACTED"),
]


def scrub(t):
    for rx, rep in SECRETS:
        t = rx.sub(rep, t)
    return t


def scrub_obj(o):
    """scrub every string inside a JSON value (scrubbing the serialized JSON could break its quoting)."""
    if isinstance(o, str):
        return scrub(o)
    if isinstance(o, list):
        return [scrub_obj(x) for x in o]
    if isinstance(o, dict):
        return {k: scrub_obj(v) for k, v in o.items()}
    return o


def synth_tools(calls):
    """OpenAI-format tool schemas from the calls a sample actually makes (name -> argument keys and JSON types)."""
    props = defaultdict(dict)
    for name, args in calls:
        for k, v in args.items():
            t = ("boolean" if isinstance(v, bool) else "integer" if isinstance(v, int) else "number"
                 if isinstance(v, float) else "array" if isinstance(v, list) else "object" if isinstance(v, dict)
                 else "string")
            props[name].setdefault(k, {"type": t})
    return [{"type": "function", "function": {"name": n, "description": f"The {n} tool.",
                                              "parameters": {"type": "object", "properties": p}}}
            for n, p in props.items()]


def tool_call(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


def clip(t, n=TOOL_RESULT_CHARS):
    return t if len(t) <= n else t[: n // 2] + f"\n... [{len(t) - n} chars clipped] ...\n" + t[-n // 2:]


# ------------------------------------------------------------------------------------------- session transcripts
def _text_of(content):
    if isinstance(content, str):
        return content
    out = []
    for b in content or []:
        if isinstance(b, dict) and b.get("type") == "text":
            out.append(b.get("text", ""))
    return "\n".join(out)


def session_messages(path):
    """Claude-Code-format jsonl -> (system, messages, calls).  None if it is not a usable agent session."""
    recs = []
    for line in open(path, errors="ignore"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("type") in ("user", "assistant") and isinstance(r.get("message"), dict) and not r.get("isSidechain") \
                and not r.get("isMeta"):
            recs.append(r)
    msgs, calls, cur, cur_id = [], [], None, None
    for r in recs:
        m = r["message"]
        if r["type"] == "assistant":
            if cur is None or m.get("id") != cur_id:
                cur = {"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": []}
                cur_id = m.get("id"); msgs.append(cur)
            for b in m.get("content") or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "thinking" and b.get("thinking"):
                    cur["reasoning_content"] += scrub(b["thinking"])
                elif b.get("type") == "text":
                    cur["content"] += scrub(b.get("text", ""))
                elif b.get("type") == "tool_use":
                    args = b.get("input") or {}
                    args = scrub_obj(args)
                    cur["tool_calls"].append(tool_call(b.get("id", ""), b.get("name", "tool"), args))
                    calls.append((b.get("name", "tool"), args))
        else:
            cur = None
            c = m.get("content")
            if isinstance(c, list) and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c):
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        msgs.append({"role": "tool", "tool_call_id": b.get("tool_use_id", ""),
                                     "content": clip(scrub(_text_of(b.get("content"))))})
            else:
                t = scrub(_text_of(c)).strip()
                if t:
                    msgs.append({"role": "user", "content": t})
    for a in msgs:
        if a["role"] == "assistant" and not a["tool_calls"]:
            del a["tool_calls"]
    # tool results whose call is missing (truncated logs) would break the encoder's ordering: drop them
    ids = {tc["id"] for a in msgs if a["role"] == "assistant" for tc in a.get("tool_calls", [])}
    msgs = [x for x in msgs if x["role"] != "tool" or x["tool_call_id"] in ids]
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    if len(calls) < 8 or sum(x["role"] == "user" for x in msgs) < 1:
        return None
    return msgs, calls


SYSTEM = ("You are an autonomous coding agent working in the user's Linux workstation. Use the tools to inspect, "
          "edit and run code; keep going until the task is complete, then report the outcome.")


def pi_session_messages(path):
    """pi-agent JSONL session (type=message events) -> (msgs, calls), same shape as session_messages()."""
    recs = []
    for line in open(path, errors="ignore"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("type") == "message" and isinstance(r.get("message"), dict):
            recs.append(r["message"])
    msgs, calls, cur, cur_id = [], [], None, None
    for j, m in enumerate(recs):
        role = m.get("role")
        if role == "assistant":
            mid = m.get("id") or f"__pi{j}"   # pi events carry no per-message id: one turn per message
            if cur is None or mid != cur_id:
                cur = {"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": []}
                cur_id = mid
                msgs.append(cur)
            for b in m.get("content") or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "thinking" and b.get("thinking"):
                    cur["reasoning_content"] += scrub(b["thinking"])
                elif b.get("type") == "text":
                    cur["content"] += scrub(b.get("text", ""))
                elif b.get("type") == "toolCall":
                    args = scrub_obj(b.get("arguments") or {})
                    cur["tool_calls"].append(tool_call(b.get("id", ""), b.get("name", "tool"), args))
                    calls.append((b.get("name", "tool"), args))
        elif role == "user":
            cur = None
            t = scrub(_text_of(m.get("content"))).strip()
            if t:
                msgs.append({"role": "user", "content": t})
        elif role == "toolResult":
            msgs.append({"role": "tool", "tool_call_id": m.get("toolCallId", ""),
                         "content": clip(scrub(_text_of(m.get("content"))))})
        # system / other roles: skipped - render_session injects its own SYSTEM prompt
    for a in msgs:
        if a["role"] == "assistant" and not a["tool_calls"]:
            del a["tool_calls"]
    ids = {tc["id"] for a in msgs if a["role"] == "assistant" for tc in a.get("tool_calls", [])}
    msgs = [x for x in msgs if x["role"] != "tool" or x["tool_call_id"] in ids]
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    if len(calls) < 8 or sum(x["role"] == "user" for x in msgs) < 1:
        return None
    return msgs, calls


# Canonical schema block for OUR serving toolset — spliced into SWE preambles so the
# advertised tools match the (translated) calls. Style mirrors the pi-agent schema block.
OUR_TOOLS_SCHEMA = (
    '### Available Tool Schemas\n\n'
    '{"name": "bash", "description": "The bash tool.", "parameters": {"type": "object", '
    '"properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}}}\n'
    '{"name": "read", "description": "The read tool.", "parameters": {"type": "object", '
    '"properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}}}\n'
    '{"name": "edit", "description": "The edit tool.", "parameters": {"type": "object", '
    '"properties": {"path": {"type": "string"}, "edits": {"type": "array"}}}}\n'
    '{"name": "write", "description": "The write tool.", "parameters": {"type": "object", '
    '"properties": {"path": {"type": "string"}, "content": {"type": "string"}}}}\n'
)

def _swap_schema_block(t):
    """Replace the originating framework's tool-schema block in a SWE preamble with ours."""
    i = t.find("### Available Tool Schemas")
    if i < 0:
        return t
    j = t.find("You MUST strictly follow", i)
    end = j if j > 0 else len(t)
    return t[:i] + OUR_TOOLS_SCHEMA + t[end:]


def _translate_swe_tool(name, args):
    """OpenHands/SWE tool -> our serving toolset (read/bash/edit/write), semantically.
    Returns (new_name, new_args, kind) where kind in {"tool","reason","final","drop"}:
      reason -> absorb the call into the assistant turn's reasoning (think)
      final  -> absorb into the assistant turn's content (finish/submit)
      drop   -> bookkeeping only (task_tracker); drop call + its result
    Syntax is part of the calibration distribution: the model must see OUR tool names/args,
    not the originating agent framework's (OpenMoE token-driven routing; DSML rendering)."""
    a = args if isinstance(args, dict) else {}
    if name in ("execute_bash", "bash"):
        return "bash", {"command": a.get("command", "")}, "tool"
    if name == "str_replace_editor":
        cmd = a.get("command", "")
        if cmd == "view":
            vr = a.get("view_range") or []
            out = {"path": a.get("path", "")}
            if len(vr) >= 1: out["offset"] = int(vr[0])
            if len(vr) >= 2: out["limit"] = int(vr[1]) - int(vr[0])
            return "read", out, "tool"
        if cmd == "create":
            return "write", {"path": a.get("path", ""), "content": a.get("file_text", "")}, "tool"
        if cmd == "str_replace":
            return "edit", {"path": a.get("path", ""),
                            "edits": [{"oldText": a.get("old_str", ""), "newText": a.get("new_str", "")}]}, "tool"
        if cmd == "insert":  # ~0.03% of calls; line-anchored insert has no exact edit() form
            return "edit", {"path": a.get("path", ""),
                            "edits": [{"oldText": "", "newText": a.get("text_to_insert", "")}]}, "tool"
        return name, a, "tool"
    if name == "think":
        return None, None, "reason" if a.get("thought") else "drop"
    if name in ("finish", "submit"):
        return None, None, "final" if a.get("response") else "drop"
    if name == "task_tracker":
        return None, None, "drop"
    return name, a, "tool"  # unknown tools pass through untranslated


def swe_row_messages(row):
    """One OpenAI-style row ({"messages": [...], "tools": [...]}) -> (msgs, calls) in the internal shape.
    Used for public SWE trajectory datasets (nebius SWE-rebench, nvidia SWE-Hero, SWE-smith tool).
    Tool calls are translated to our serving toolset via _translate_swe_tool."""
    msgs, calls, cur, cur_id = [], [], None, None
    for j, m in enumerate(row.get("messages") or []):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "assistant":
            mid = m.get("id") or f"__swe{j}"   # no per-message id in these datasets: one turn per message
            if cur is None or mid != cur_id:
                cur = {"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": []}
                cur_id = mid
                msgs.append(cur)
            c = m.get("content")
            if c:
                cur["content"] += scrub(c if isinstance(c, str) else json.dumps(c, ensure_ascii=False))
            if m.get("reasoning_content"):
                cur["reasoning_content"] += scrub(m["reasoning_content"])
            for tc in m.get("tool_calls") or []:
                f = tc.get("function") or {}
                args = f.get("arguments", "")
                if isinstance(args, str):   # these datasets JSON-encode the arguments
                    try: args = json.loads(args) if args else {}
                    except Exception: pass
                args = scrub_obj(args if isinstance(args, dict) else {"raw": args})
                nm, na, kind = _translate_swe_tool(f.get("name", "tool"), args)
                if kind == "reason":
                    cur["reasoning_content"] += scrub(str(args.get("thought", "")) if isinstance(args, dict) else "")
                    continue  # absorbed: its tool-result is dropped by the id filter below
                if kind == "final":
                    cur["content"] += scrub(str(args.get("response", "")) if isinstance(args, dict) else "")
                    continue
                if kind == "drop":
                    continue
                cur["tool_calls"].append(tool_call(tc.get("id", ""), nm, na))
                calls.append((nm, na))
        elif role == "user":
            cur = None
            t = scrub(_text_of(m.get("content"))).strip()
            if t:
                t = _swap_schema_block(t)  # advertise OUR tools, matching the translated calls
                msgs.append({"role": "user", "content": t})
        elif role == "tool":
            msgs.append({"role": "tool", "tool_call_id": m.get("tool_call_id", ""),
                         "content": clip(scrub(_text_of(m.get("content"))))})
        # system: skipped - render_session injects its own SYSTEM prompt
    for a in msgs:
        if a["role"] == "assistant" and not a["tool_calls"]:
            del a["tool_calls"]
    ids = {tc["id"] for a in msgs if a["role"] == "assistant" for tc in a.get("tool_calls", [])}
    msgs = [x for x in msgs if x["role"] != "tool" or x["tool_call_id"] in ids]
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    if len(calls) < 8 or sum(x["role"] == "user" for x in msgs) < 1:
        return None
    return msgs, calls


def render_session(msgs, calls, tok, max_tokens):
    """Longest message prefix (ending on an assistant turn) whose V4.1 rendering fits max_tokens."""
    tools = synth_tools(calls)
    ends = [i + 1 for i, m in enumerate(msgs) if m["role"] == "assistant"]

    def render(k):
        ms = [{"role": "system", "content": SYSTEM, "tools": tools}] + msgs[:k]
        return E.encode_messages(ms, thinking_mode="thinking")
    lo, hi, best = 0, len(ends) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        t = render(ends[mid])
        n = len(tok.encode(t, add_special_tokens=False))
        if n <= max_tokens:
            best, lo = (t, n), mid + 1
        else:
            hi = mid - 1
    if best is None and msgs:
        # whole prefix too long (giant task preamble): take the longest TAIL that fits, starting on an
        # assistant boundary (orphan tool results before the window are dropped by the caller's filter)
        def render_tail(start):
            ms = [{"role": "system", "content": SYSTEM, "tools": tools}] + msgs[start:]
            return E.encode_messages(ms, thinking_mode="thinking")
        def fits(start):
            return len(tok.encode(render_tail(start), add_special_tokens=False)) <= max_tokens
        bounds = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]   # valid window starts
        lo, hi, ans = 0, len(bounds) - 1, None
        while lo <= hi:                 # binary search the LONGEST tail that fits (smallest start)
            mid = (lo + hi) // 2
            if fits(bounds[mid]):
                ans, hi = bounds[mid], mid - 1
            else:
                lo = mid + 1
        if ans is not None:
            best = (render_tail(ans), len(tok.encode(render_tail(ans), add_special_tokens=False)))
    return best


def build_sessions(a, tok):
    roots = [os.path.expanduser(p) for p in ("~/.claude/projects", "~/.quetza-data", "~/.claude-data")]
    # private agent logs (e.g. ~/.pi/agent/sessions) enter a corpus only when named with --session-root
    roots += [os.path.expanduser(p) for p in a.session_root]
    files = []
    for r in roots:
        for f in glob.glob(os.path.join(r, "**", "*.jsonl"), recursive=True):
            if os.path.getsize(f) > 50_000 and not any(x and x in f for x in a.exclude_session):
                files.append(f)
    cands = []
    for f in files:
        parse = pi_session_messages if "/.pi/agent/" in f else session_messages
        try:
            got = parse(f)
        except Exception as e:
            print(f"  skip {os.path.basename(f)}: {type(e).__name__}", file=sys.stderr)
            continue
        if not got:
            continue
        msgs, calls = got
        src = "pi" if "/.pi/agent/" in f else ("quetza" if "/.quetza-data/" in f else ("claude-data" if "/.claude-data/" in f else "claude-code"))
        cwd = ""
        for line in open(f, errors="ignore"):
            if '"cwd"' in line:
                try:
                    cwd = json.loads(line).get("cwd", ""); break
                except Exception:
                    pass
        project = "/".join(cwd.rstrip("/").split("/")[-2:]) if cwd else "?"
        cands.append((f, src, project, msgs, calls))
    print(f"sessions: {len(files)} files, {len(cands)} usable agent sessions", file=sys.stderr)
    # diversity: round-robin over (source, project), longest sessions first within each group
    groups = defaultdict(list)
    for c in sorted(cands, key=lambda c: -len(c[4])):
        groups[(c[1], c[2])].append(c)
    order, keys = [], sorted(groups, key=lambda k: (k[0] != "quetza", -len(groups[k])))
    while any(groups[k] for k in keys):
        for k in keys:
            if groups[k]:
                order.append(groups[k].pop(0))
    out, stats = [], Counter()
    for f, src, project, msgs, calls in order:
        r = render_session(msgs, calls, tok, a.max_tokens)
        if not r or r[1] < a.max_tokens // 2:     # too short to be a long-context sample
            continue
        out.append({"text": r[0], "src": src, "project": project, "tokens": r[1],
                    "file": hashlib.sha1(f.encode()).hexdigest()[:12]})
        stats[src] += 1
        if len(out) >= a.sessions:
            break
    # public SWE trajectory rows (OpenAI-style JSONL, one session per line): nebius/SWE-Hero/SWE-smith
    for f in (a.extra_sessions or []):
        for line in open(f, errors="ignore"):
            try:
                row = json.loads(line)
            except Exception:
                continue
            got = swe_row_messages(row)
            if not got:
                continue
            msgs, calls = got
            r = render_session(msgs, calls, tok, a.max_tokens)
            if not r or r[1] < a.max_tokens // 2:
                continue
            out.append({"text": r[0], "src": "swe-" + str(row.get("src", "?")),
                        "project": str(row.get("instance_id", "?")), "tokens": r[1],
                        "file": hashlib.sha1(f.encode()).hexdigest()[:12]})
            stats["swe"] += 1
    print(f"sessions: kept {len(out)} {dict(stats)}; projects {Counter(o['project'] for o in out).most_common(12)}",
          file=sys.stderr)
    return out


# ------------------------------------------------------------------------------------------- antirez prompts
DSML4_INVOKE = re.compile(r'<｜DSML｜invoke name="([^"]+)">(.*?)</｜DSML｜invoke>', re.S)
DSML4_PARAM = re.compile(r'<｜DSML｜parameter name="([^"]+)" string="(true|false)">(.*?)</｜DSML｜parameter>', re.S)
V4_TOOLS_HDR = "## Tools"


def _v4_calls(dsml):
    calls = []
    for name, body in DSML4_INVOKE.findall(dsml or ""):
        args = {}
        for k, s, v in DSML4_PARAM.findall(body):
            if s == "true":
                args[k] = v
            else:
                try:
                    args[k] = json.loads(v)
                except Exception:
                    args[k] = v
        calls.append((name, args))
    return calls


def _v4_system_tools(system):
    """Split a V4-rendered system prompt into (plain text, OpenAI tool schemas found as JSON lines)."""
    i = system.find(V4_TOOLS_HDR)
    if i < 0:
        return system, []
    tools = []
    for line in system[i:].splitlines():
        line = line.strip()
        if line.startswith('{"name"'):
            try:
                t = json.loads(line)
                tools.append({"type": "function", "function": t})
            except Exception:
                pass
    return system[:i].rstrip(), tools


def build_antirez():
    pools = defaultdict(list)
    bad = 0
    for line in open(ANTIREZ):
        r = json.loads(line)
        msgs, all_calls, n = [], [], 0
        tools = []
        for m in r["messages"]:
            role = m["role"]
            if role == "system":
                text, tools = _v4_system_tools(m["content"])
                msgs.append({"role": "system", "content": text})
            elif role == "assistant":
                am = {"role": "assistant", "content": m.get("content", "")}
                if m.get("reasoning"):
                    am["reasoning_content"] = m["reasoning"]
                calls = _v4_calls(m.get("dsml"))
                if calls:
                    am["tool_calls"] = []
                    for name, args in calls:
                        cid = f"call_{n}"; n += 1
                        am["tool_calls"].append(tool_call(cid, name, args)); all_calls.append((name, args))
                msgs.append(am)
            elif role == "tool":
                prev = [x for x in msgs if x["role"] == "assistant" and x.get("tool_calls")]
                used = {x["tool_call_id"] for x in msgs if x["role"] == "tool"}
                free = [tc["id"] for x in prev for tc in x["tool_calls"] if tc["id"] not in used]
                msgs.append({"role": "tool", "tool_call_id": free[0] if free else "", "content": m["content"]})
            else:
                msgs.append({"role": role, "content": m["content"]})
        if all_calls and msgs and msgs[0]["role"] == "system":
            msgs[0]["tools"] = tools or synth_tools(all_calls)
        try:
            text = E.encode_messages(msgs, thinking_mode="thinking" if r["mode"] == "think" else "chat")
        except Exception:
            bad += 1
            continue
        pools[r["category"]].append({"text": text, "src": "antirez", "project": r["source"].split(":")[0],
                                     "id": r["id"]})
    print(f"antirez: {sum(map(len, pools.values()))} rendered, {bad} failed; "
          f"{ {k: len(v) for k, v in pools.items()} }", file=sys.stderr)
    return pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=8191)
    ap.add_argument("--sessions", type=int, default=40)
    ap.add_argument("--exclude-session", action="append", default=[])
    ap.add_argument("--session-root", action="append", default=[],
                    help="extra session dir to scan (repeatable; e.g. ~/.pi/agent/sessions). Off by default: "
                         "private agent logs enter a corpus only when named here")
    ap.add_argument("--extra-sessions", action="append", default=[],
                    help="OpenAI-style SWE trajectory JSONL (one {messages,tools,src,instance_id,resolved} per line)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    out = os.path.join(WORK, "corpus")
    os.makedirs(out, exist_ok=True)
    pools = {"sessions": build_sessions(a, tok)}
    pools.update({f"antirez_{k}": v for k, v in build_antirez().items()})
    for k, v in pools.items():
        with open(os.path.join(out, f"{k}.jsonl"), "w") as fh:
            for x in v:
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("wrote", out, {k: len(v) for k, v in pools.items()}, file=sys.stderr)


if __name__ == "__main__":
    main()
