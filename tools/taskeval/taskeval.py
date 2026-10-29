#!/usr/bin/env python3
"""Real-task screen for the DS4 decode route-bias arms (Mal 2026-10-14: does the bias hurt actual coding / tool calling
/ agentic work, not just token agreement?).  Talks to the OpenAI endpoint of serve_ds41.py.  Greedy (temperature 0) so
arms are paired; thinking off (effort none) to keep it fast - a screen, not the deployed setting.

  taskeval.py run   <url> <outdir> [--only he,tool,agent]
  taskeval.py score <outdir> [<outdir2> ...]

Sets: he     = HumanEval every 8th problem (21), graded by its own check()
      tool   = 10 single-step tool-choice cases (right tool + args, or no call), graded structurally
      agent  = 3 multi-step bug-fix loops over a mock filesystem (read_file / write_file / run_tests), graded by the
               hidden tests on the file the model wrote
"""
import json, os, re, subprocess, sys, tempfile, time, urllib.request
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def chat(url, messages, tools=None, max_tokens=1024):
    body = {"model": os.environ.get("TASKEVAL_MODEL", "x"), "messages": messages, "temperature": 0,
            "max_tokens": max_tokens, "stream": False}
    # local engines: reasoning_effort none = no thinking (a screen, not the deployed setting).
    # TASKEVAL_NO_EFFORT=1 drops it - a remote API (the DeepSeek endpoint) rejects it.
    if not os.environ.get("TASKEVAL_NO_EFFORT"):
        body["reasoning_effort"] = "none"
    # TASKEVAL_EFFORT=low|high|max: an explicit effort (the DeepSeek API's knob; "max" = the model at full potential)
    if os.environ.get("TASKEVAL_EFFORT"):
        body["reasoning_effort"] = os.environ["TASKEVAL_EFFORT"]
    # TASKEVAL_THINK_BUDGET=N: the models think (their serve default effort) with a budget of N thinking tokens, and the
    # answer keeps its own max_tokens on top - the way they are served (every Xolotl config: reasoning_budget_tokens 32768)
    tb = int(os.environ.get("TASKEVAL_THINK_BUDGET", "0") or 0)
    if tb > 0:
        body.pop("reasoning_effort", None)
        body["reasoning_budget_tokens"] = tb
        body["max_tokens"] = max_tokens + tb
    # TASKEVAL_API_THINK=N: a remote thinking API (DeepSeek) has no thinking budget and its max_tokens covers the chain
    # of thought too, so the answer cap would cut it off: max_tokens = answer cap + N (the local engines' 32K + answer)
    at = int(os.environ.get("TASKEVAL_API_THINK", "0") or 0)
    if at > 0:
        body["max_tokens"] = max_tokens + at
    if tools:
        body["tools"] = tools
    # TASKEVAL_KEY: Bearer token for a remote OpenAI-compatible endpoint (the local engines need none)
    headers = {"Content-Type": "application/json"}
    if os.environ.get("TASKEVAL_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["TASKEVAL_KEY"]
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        j = json.load(r)
    j["_secs"] = time.time() - t0
    # REASONING FIELD NORMALISATION. This harness grew up on DeepSeek, which returns the chain of
    # thought as `message.reasoning_content` and REQUIRES it back with the assistant turn for tool
    # calling to behave. OpenRouter (and other OpenAI-compatible gateways) return the same data as
    # `message.reasoning` / `reasoning_details`. Without this, an OpenRouter run silently loses the
    # reasoning on every turn: the model still thinks and is billed for it, but the caller reads
    # only `reasoning_content`, finds nothing, and drops it from the history - so the run degrades
    # at multi-turn tool calling and reports "think 0 ch" throughout. That reads as "this model
    # does not reason" when in fact the harness did not carry it. Measured 2026-10-29 on the same
    # model: 4/7 with the fields dropped, 5/7 when carried.
    #
    # Only fills a field that is absent, so DeepSeek and local-engine runs are unaffected.
    try:
        _m = (j.get("choices") or [{}])[0].get("message") or {}
        if not _m.get("reasoning_content"):
            _r = _m.get("reasoning")
            if not _r:
                _r = "".join(d.get("text", "") for d in (_m.get("reasoning_details") or [])
                             if isinstance(d, dict))
            if _r:
                _m["reasoning_content"] = _r
    except (IndexError, AttributeError, TypeError):
        pass
    # every response's server timings (prompt / decode tok/s), saved with the results: the speed DURING the real tasks
    TIMINGS.append({"secs": j["_secs"], "usage": j.get("usage", {}), "timings": j.get("timings", {})})
    return j


TIMINGS = []


def fn(name, desc, props, req=None):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": req if req is not None else list(props)}}}


S = {"type": "string"}
I = {"type": "integer"}
TOOLS = [
    fn("get_weather", "Current weather for a city.", {"city": S, "unit": {"type": "string", "enum": ["c", "f"]}}, ["city"]),
    fn("read_file", "Read a text file from the workspace.", {"path": S}),
    fn("write_file", "Write (overwrite) a text file in the workspace.", {"path": S, "content": S}),
    fn("run_shell", "Run a shell command in the workspace and return stdout/stderr.", {"command": S}),
    fn("search_code", "Search the codebase for a regex; returns matching file:line entries.", {"pattern": S, "glob": S}, ["pattern"]),
    fn("create_event", "Create a calendar event.", {"title": S, "date": {"type": "string", "description": "YYYY-MM-DD"},
                                                     "start": {"type": "string", "description": "HH:MM 24h"},
                                                     "duration_min": I}, ["title", "date", "start"]),
    fn("convert_currency", "Convert an amount between currencies (ISO codes).", {"amount": {"type": "number"}, "from": S, "to": S}),
    fn("send_email", "Send an email.", {"to": S, "subject": S, "body": S}),
]

SYS = "You are a helpful assistant with tools. Call a tool when it is needed; otherwise answer directly."

# (id, user message, check(tool_calls, content) -> bool)
def _c(calls, name):
    return [c for c in calls if c["name"] == name]


TOOL_CASES = [
    ("weather_f", "What's the weather like in Osaka right now? Give it to me in Fahrenheit.",
     lambda tc, t: len(tc) >= 1 and tc[0]["name"] == "get_weather" and "osaka" in tc[0]["args"].get("city", "").lower()
     and tc[0]["args"].get("unit") == "f"),
    ("no_tool_math", "What is 17 * 23? Just the number.",
     lambda tc, t: len(tc) == 0 and "391" in t),
    ("read_cfg", "Show me what's in config/settings.yaml.",
     lambda tc, t: len(tc) >= 1 and tc[0]["name"] == "read_file" and tc[0]["args"].get("path", "").endswith("config/settings.yaml")),
    ("grep_todo", "Find every TODO comment in the Python files of this repo.",
     lambda tc, t: len(tc) >= 1 and tc[0]["name"] in ("search_code", "run_shell")
     and "todo" in json.dumps(tc[0]["args"]).lower()
     and (tc[0]["name"] != "search_code" or "py" in tc[0]["args"].get("glob", "py"))),
    ("event", "Book 'Dentist' on 2026-11-03 at 2:30pm for 45 minutes.",
     lambda tc, t: len(tc) >= 1 and tc[0]["name"] == "create_event" and tc[0]["args"].get("date") == "2026-11-03"
     and tc[0]["args"].get("start") in ("14:30",) and int(tc[0]["args"].get("duration_min", 0) or 0) == 45
     and "dentist" in tc[0]["args"].get("title", "").lower()),
    ("currency", "How much is 250 Australian dollars in Japanese yen?",
     lambda tc, t: len(tc) >= 1 and tc[0]["name"] == "convert_currency" and float(tc[0]["args"].get("amount", 0)) == 250
     and tc[0]["args"].get("from", "").upper() == "AUD" and tc[0]["args"].get("to", "").upper() == "JPY"),
    ("shell_count", "How many lines are in src/main.rs? Use the shell.",
     lambda tc, t: len(tc) >= 1 and tc[0]["name"] == "run_shell" and "wc" in tc[0]["args"].get("command", "")
     and "src/main.rs" in tc[0]["args"].get("command", "")),
    ("no_tool_fact", "In one word: what is the capital of France?",
     lambda tc, t: len(tc) == 0 and "paris" in t.lower()),
    ("email", "Email bob@example.com with subject 'Build fixed' and tell him the CI is green again.",
     lambda tc, t: len(tc) >= 1 and tc[0]["name"] == "send_email" and tc[0]["args"].get("to") == "bob@example.com"
     and tc[0]["args"].get("subject") == "Build fixed" and "green" in tc[0]["args"].get("body", "").lower()),
    ("write_hello", "Create hello.py that prints exactly: Hello, world!",
     lambda tc, t: len(tc) >= 1 and tc[0]["name"] == "write_file" and tc[0]["args"].get("path", "").endswith("hello.py")
     and _runs_hello(tc[0]["args"].get("content", ""))),
]


def _runs_hello(code):
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() == "Hello, world!"
    except Exception:
        return False


# ---- agentic: mock workspace, hidden tests ----
AGENT_CASES = [
    {"id": "stats_bug",
     "task": "The tests in test_stats.py fail. Find and fix the bug in stats.py. Do not edit the tests. Use run_tests to check.",
     "files": {"stats.py": "def mean(xs):\n    return sum(xs) / len(xs)\n\n\ndef median(xs):\n    s = sorted(xs)\n    n = len(s)\n    if n % 2:\n        return s[n // 2]\n    return (s[n // 2] + s[n // 2 + 1]) / 2\n\n\ndef variance(xs):\n    m = mean(xs)\n    return sum((x - m) ** 2 for x in xs) / len(xs)\n",
               "test_stats.py": "from stats import mean, median, variance\n\ndef test_all():\n    assert mean([1, 2, 3]) == 2\n    assert median([3, 1, 2]) == 2\n    assert median([4, 1, 3, 2]) == 2.5\n    assert variance([1, 1, 1]) == 0\n"},
     "target": "stats.py",
     "hidden": "from stats import mean, median, variance\nassert median([4, 1, 3, 2]) == 2.5\nassert median([10, 20]) == 15\nassert median([5]) == 5\nassert median([7, 1, 3, 9, 5, 11]) == 6\nassert abs(variance([1, 2, 3, 4]) - 1.25) < 1e-9\nprint('OK')\n"},
    {"id": "slug_bug",
     "task": "slugify() in text_utils.py is buggy: test_text.py fails. Fix text_utils.py so the tests pass (do not change the tests).",
     "files": {"text_utils.py": "import re\n\n\ndef slugify(s):\n    s = s.lower()\n    s = re.sub(r'[^a-z0-9]', '-', s)\n    return s\n",
               "test_text.py": "from text_utils import slugify\n\ndef test_slug():\n    assert slugify('Hello World') == 'hello-world'\n    assert slugify('  Rock & Roll!! ') == 'rock-roll'\n    assert slugify('a--b') == 'a-b'\n"},
     "target": "text_utils.py",
     "hidden": "from text_utils import slugify\nassert slugify('Hello World') == 'hello-world'\nassert slugify('  Rock & Roll!! ') == 'rock-roll'\nassert slugify('a--b') == 'a-b'\nassert slugify('Already-slug') == 'already-slug'\nassert slugify('x  y   z') == 'x-y-z'\nassert slugify('-lead-and-trail-') == 'lead-and-trail'\nprint('OK')\n"},
    {"id": "lru_feature",
     "task": "Implement the missing LRUCache class in cache.py (get returns -1 when absent; put evicts the least recently used key when over capacity; get counts as a use). test_cache.py shows the expected behaviour.",
     "files": {"cache.py": "class LRUCache:\n    def __init__(self, capacity):\n        raise NotImplementedError\n\n    def get(self, key):\n        raise NotImplementedError\n\n    def put(self, key, value):\n        raise NotImplementedError\n",
               "test_cache.py": "from cache import LRUCache\n\ndef test_lru():\n    c = LRUCache(2)\n    c.put(1, 1); c.put(2, 2)\n    assert c.get(1) == 1\n    c.put(3, 3)\n    assert c.get(2) == -1\n"},
     "target": "cache.py",
     "hidden": "from cache import LRUCache\nc = LRUCache(2)\nc.put(1, 1); c.put(2, 2)\nassert c.get(1) == 1\nc.put(3, 3)\nassert c.get(2) == -1\nc.put(4, 4)\nassert c.get(1) == -1 and c.get(3) == 3 and c.get(4) == 4\nc.put(3, 30)\nc.put(5, 5)\nassert c.get(4) == -1 and c.get(3) == 30\nd = LRUCache(1)\nd.put('a', 1); d.put('b', 2)\nassert d.get('a') == -1 and d.get('b') == 2\nprint('OK')\n"},
]
AGENT_TOOLS = [TOOLS[1], TOOLS[2], fn("run_tests", "Run the workspace's pytest suite; returns the output.", {})]


def run_pytest(files):
    with tempfile.TemporaryDirectory() as d:
        for k, v in files.items():
            open(os.path.join(d, k), "w").write(v)
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "--no-header", "-p", "no:cacheprovider"],
                           cwd=d, capture_output=True, text=True, timeout=60)
        return (r.stdout + r.stderr)[-1500:]


def run_hidden(code, hidden):
    with tempfile.TemporaryDirectory() as d:
        return code, hidden, d


def hidden_ok(files, target, hidden):
    with tempfile.TemporaryDirectory() as d:
        for k, v in files.items():
            open(os.path.join(d, k), "w").write(v)
        try:
            r = subprocess.run([sys.executable, "-c", hidden], cwd=d, capture_output=True, text=True, timeout=20)
            return r.stdout.strip().endswith("OK")
        except Exception:
            return False


def parse_calls(msg):
    out = []
    for c in msg.get("tool_calls") or []:
        f = c.get("function", {})
        try:
            a = json.loads(f.get("arguments") or "{}")
        except Exception:
            a = {"_unparsed": f.get("arguments")}
        out.append({"id": c.get("id"), "name": f.get("name"), "args": a if isinstance(a, dict) else {"_v": a}})
    return out


def agent_loop(url, case, max_steps=8):
    files = dict(case["files"])
    msgs = [{"role": "system", "content": "You are a coding agent working in a small Python workspace. Files: "
             + ", ".join(sorted(files)) + ". Use the tools; when the tests pass, reply DONE."},
            {"role": "user", "content": case["task"]}]
    log, toks, secs = [], 0, 0.0
    for step in range(max_steps):
        j = chat(url, msgs, AGENT_TOOLS, max_tokens=1536)
        toks += j.get("usage", {}).get("completion_tokens", 0); secs += j["_secs"]
        m = j["choices"][0]["message"]
        calls = parse_calls(m)
        log.append({"content": m.get("content"), "calls": calls})
        if not calls:
            break
        # reasoning_content goes back with the turn (DeepSeek's thinking-mode tool-call protocol requires it; the
        # Gauntlet does the same) - TASKEVAL_DROP_REASONING=1 = the old behaviour
        keep = {"reasoning_content": m["reasoning_content"]} if m.get("reasoning_content") and \
            os.environ.get("TASKEVAL_DROP_REASONING") != "1" else {}
        msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": m["tool_calls"], **keep})
        for c in calls:
            a = c["args"]
            if c["name"] == "read_file":
                res = files.get(a.get("path", "").lstrip("./"), f"error: no such file {a.get('path')}")
            elif c["name"] == "write_file":
                p = a.get("path", "").lstrip("./")
                if p.startswith("test_"):
                    res = "error: tests are read-only"
                else:
                    files[p] = a.get("content", ""); res = f"wrote {len(files[p])} bytes to {p}"
            elif c["name"] == "run_tests":
                res = run_pytest(files)
            else:
                res = f"error: unknown tool {c['name']}"
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": res})
    ok = hidden_ok(files, case["target"], case["hidden"])
    return {"id": case["id"], "ok": ok, "steps": len(log), "tokens": toks, "secs": secs, "log": log,
            "final": files.get(case["target"])}


def he_cases():
    df = pd.read_parquet(os.path.join(HERE, "he/openai_humaneval/test-00000-of-00001.parquet"))
    # TASKEVAL_HE_ALL=1: all 164 problems (the default every-8th subset of 21 is saturated by every Xolotl model)
    return [df.iloc[i] for i in range(0, len(df), 1 if os.environ.get("TASKEVAL_HE_ALL") == "1" else 8)]


def extract_code(text):
    m = re.findall(r"```(?:python|py)?\n(.*?)```", text or "", re.S)
    return max(m, key=len) if m else (text or "")


def he_ok(row, text):
    code = extract_code(text)
    if f"def {row['entry_point']}" not in code:
        code = row["prompt"] + code   # body-only answer
    prog = code + "\n\n" + row["test"] + f"\ncheck({row['entry_point']})\nprint('OK')\n"
    try:
        r = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, timeout=15)
        return r.stdout.strip().endswith("OK")
    except Exception:
        return False


def run(url, out, only):
    os.makedirs(out, exist_ok=True)
    res = {}
    if "he" in only:
        rs = []
        for row in he_cases():
            j = chat(url, [{"role": "user", "content": "Complete this Python function. Reply with the complete function "
                                                        "(imports included) in one ```python block and nothing else.\n\n"
                                                        "```python\n" + row["prompt"] + "```"}], max_tokens=1024)
            t = j["choices"][0]["message"].get("content") or ""
            rs.append({"id": row["task_id"], "ok": he_ok(row, t), "tokens": j.get("usage", {}).get("completion_tokens", 0),
                       "secs": j["_secs"], "text": t})
            print(f"he {row['task_id']}: {'PASS' if rs[-1]['ok'] else 'fail'} ({rs[-1]['tokens']} tok, {j['_secs']:.0f}s)", flush=True)
        res["he"] = rs
    if "tool" in only:
        rs = []
        for cid, q, chk in TOOL_CASES:
            j = chat(url, [{"role": "system", "content": SYS}, {"role": "user", "content": q}], TOOLS, max_tokens=512)
            m = j["choices"][0]["message"]
            calls = parse_calls(m)
            try:
                ok = bool(chk(calls, m.get("content") or ""))
            except Exception:
                ok = False
            rs.append({"id": cid, "ok": ok, "calls": calls, "content": m.get("content"),
                       "tokens": j.get("usage", {}).get("completion_tokens", 0), "secs": j["_secs"]})
            print(f"tool {cid}: {'PASS' if ok else 'fail'} {calls}", flush=True)
        res["tool"] = rs
    if "agent" in only:
        rs = []
        for c in AGENT_CASES:
            r = agent_loop(url, c)
            rs.append(r)
            print(f"agent {c['id']}: {'PASS' if r['ok'] else 'fail'} ({r['steps']} steps, {r['tokens']} tok, {r['secs']:.0f}s)", flush=True)
        res["agent"] = rs
    res["timings"] = TIMINGS
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1)


def score(outs):
    print(f"{'run':24s} {'he':>7s} {'tool':>7s} {'agent':>7s}  {'tok':>6s} {'secs':>6s} {'tok/s':>6s}")
    for out in outs:
        r = json.load(open(os.path.join(out, "results.json")))
        cells, tok, secs = [], 0, 0.0
        for k in ("he", "tool", "agent"):
            rs = r.get(k, [])
            cells.append(f"{sum(x['ok'] for x in rs)}/{len(rs)}" if rs else "-")
            tok += sum(x["tokens"] for x in rs); secs += sum(x["secs"] for x in rs)
        print(f"{os.path.basename(out.rstrip('/')):24s} {cells[0]:>7s} {cells[1]:>7s} {cells[2]:>7s}  {tok:6d} {secs:6.0f} {tok / max(secs, 1e-9):6.1f}")


if __name__ == "__main__":
    if sys.argv[1] == "speed":   # taskeval.py speed <outdir>... : prefill / decode tok/s measured during the tasks
        for d in sys.argv[2:]:
            ts = [t["timings"] for t in json.load(open(os.path.join(d, "results.json"))).get("timings", []) if t.get("timings")]
            pn = sum(t.get("prompt_n", 0) for t in ts); pms = sum(t.get("prompt_ms", 0) for t in ts)
            dn = sum(t.get("predicted_n", 0) for t in ts); dms = sum(t.get("predicted_ms", 0) for t in ts)
            print(f"{d}: {len(ts)} requests | prefill {pn} tok at {1000 * pn / max(pms, 1e-9):.0f} tok/s | "
                  f"decode {dn} tok at {1000 * dn / max(dms, 1e-9):.1f} tok/s")
        sys.exit(0)
    if sys.argv[1] == "run":
        only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else ["he", "tool", "agent"]
        run(sys.argv[2], sys.argv[3], only)
    else:
        score(sys.argv[2:])
