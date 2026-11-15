#!/usr/bin/env python3
"""gauntlet.py - the Gauntlet: small, hard, machine-graded agentic coding tasks (see DESIGN.md).

  gauntlet.py run   <url> <outdir> [--only 01,02]    one model through every task (OpenAI endpoint at <url>/v1)
  gauntlet.py score <outdir> ...                     pass/10, tokens, seconds, decode tok/s during the run

Uses tools/taskeval's chat() (same request shape, TASKEVAL_THINK_BUDGET, timings recording) and parse_calls().
Decoding is greedy (temperature 0), so a rerun repeats the same tokens; TASKEVAL_TEMPERATURE=served (the server's
configured sampling) or =<t>, with TASKEVAL_SEED=1,2,3..., gives independent sampled runs to average.
Each task's workspace is staged in a temp dir; the model gets list_files / read_file / write_file / run_tests /
submit; after it submits (or runs out of steps) the hidden tests are copied in and pytest decides pass / fail.
A task whose task.json carries "answer" (an exact-answer task: no workspace, no hidden tests) is graded instead on
whether that regex matches the model's final assistant message. A task with neither hidden tests nor an answer is
ungradable and fails loudly - it is never scored PASS by accident.

Step budget: every request asks for GAUNTLET_MAX_TOKENS completion tokens (default 36864 = 4096 answer + 32768
thinking, the same total a local engine gets from TASKEVAL_THINK_BUDGET=32768). On a thinking API the cap covers the
chain of thought too: the first calibration used 4096 and 47 of its 118 steps were cut off mid-thought (05 and 09
never wrote a file), so a step that ends on finish_reason "length" is counted per task and shown by `score`.
reasoning_content is passed back on every assistant turn, as DeepSeek's thinking-mode tool-call protocol requires
(serve_ds41's encoding renders it the same way).

The workspace may not decide the grade: write_file refuses the names below (see _write_refusal), anything planted
another way is stripped before the hidden run (_strip_grading_names), and every subprocess that runs model-supplied
code gets a minimal environment (_child_env) so the harness's credentials are never in reach of a snippet.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "taskeval"))
import taskeval as T  # noqa: E402

fn = T.fn
TOOLS = [fn("list_files", "List the workspace's files.", {}),
         fn("read_file", "Read a workspace file.", {"path": {"type": "string"}}, ["path"]),
         fn("write_file", "Create or overwrite a workspace file (tests are read-only).",
            {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
         fn("run_tests", "Run the workspace's visible pytest tests; returns their output (truncated).", {}),
         fn("run_python", "Run a Python snippet with the workspace as the current directory (20 s limit); returns "
            "stdout + stderr (truncated).", {"code": {"type": "string"}}, ["code"]),
         fn("submit", "Finish: your work is graded now.", {"note": {"type": "string"}})]
SYSTEM = ("You are a careful senior engineer working as a coding agent in a small Python workspace. You can only act "
          "through the tools. The visible tests are not the whole grade: hidden tests check the documented behaviour. "
          "Fix causes, not symptoms. Call submit when you are done.")

# --- grading integrity: the one place these two lists live ------------------------------------------------
# The staged workspace IS the rootdir of the pytest that grades the task and is on sys.path for it (pytest
# inserts every test file's basedir; the hidden tests land in the workspace root). A file planted there can
# therefore rewrite the grade instead of earning it - a conftest.py with a pytest_collection_modifyitems hook
# turns 01-ledger-rootcause's "13 failed, 20 passed" into a green "33 passed" without a line of real work.
GRADING_CONFIG_NAMES = frozenset({
    "conftest.py",        # loaded before collection; hooks can skip, rewrite or fake every test
    "pytest.ini",         # pytest's own config for the grading run (testpaths, addopts, -p plugin autoload)
    "pyproject.toml",     # idem (and [tool.pytest.ini_options])
    "setup.cfg",          # idem
    "tox.ini",            # idem
    "setup.py",           # executed by packaging tooling; never a task deliverable
    "sitecustomize.py",   # imported at interpreter start of every subprocess, including pytest's
    "usercustomize.py",   # idem
})
# The stdlib top-level modules the tasks import (hidden tests included), so a workspace module of the same
# name would shadow the real one for them:
#   grep -rhoE '^\s*(import|from) [A-Za-z_]\w*' tasks --include=*.py | sed -E 's/^\s*(import|from) +//' | sort -u
STDLIB_SHADOW_NAMES = frozenset({
    "ast", "collections", "csv", "datetime", "decimal", "functools", "io", "itertools", "json", "math", "os",
    "pathlib", "pytest", "random", "re", "shutil", "signal", "sys", "threading", "time", "unittest", "zoneinfo",
})
INTERFERE = "that name would let the workspace interfere with grading"


def _shadow_name(parts: tuple[str, ...]) -> str | None:
    """The stdlib module this workspace-relative path would shadow, or None. Covers `json.py`, a `json/`
    package directory, and a sourceless `json.pyc` (measured: a bare .pyc at the top of the search path
    does resolve as `import json`); a nested `sub/json.py` is refused too - harmless in today's tasks, but
    the rule stays one line instead of a path-depth argument."""
    for i, part in enumerate(parts):
        name = part
        if i == len(parts) - 1:
            for suffix in (".py", ".pyc"):
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
        if name in STDLIB_SHADOW_NAMES:
            return name
    return None


def _write_refusal(d: Path, p: Path | None) -> str | None:
    """Why write_file must refuse the resolved workspace path `p`, or None if the write is allowed.

    Refusals are plain error strings, never exceptions: a model that trips one should read the reason and
    work around it, not have the run die. None of these names is a deliverable of any of the ten tasks, so
    a legitimate model loses nothing (validate.py proves every task still stages and grades unchanged).
    """
    if p is None or p.name.startswith("test_"):
        return "error: tests are read-only (and paths must name a file inside the workspace)"
    rel = p.relative_to(d)
    if p.name in GRADING_CONFIG_NAMES:
        return (f"error: refusing to write {rel}: {INTERFERE} (the harness loads {p.name} from the workspace "
                "when it grades you)")
    shadow = _shadow_name(rel.parts)
    if shadow:
        return (f"error: refusing to write {rel}: {INTERFERE} (it would shadow the stdlib module '{shadow}', "
                "which the tests import)")
    return None


def _strip_grading_names(d: Path) -> list[str]:
    """Delete any workspace entry that could hijack grading, just before the hidden tests are copied in.

    write_file refuses these names, but a snippet run through run_python can still create them (it is code,
    not a path check), and the interpreter-start hook sitecustomize.py is loaded by the grading pytest itself.
    The model's own modules are never named one of these, so stripping them cannot cost a real solution."""
    gone = []
    for p in sorted(d.iterdir()):
        if p.name in GRADING_CONFIG_NAMES or _shadow_name(p.relative_to(d).parts):
            gone.append(p.name)
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
    return gone


def _child_env(home: Path) -> dict[str, str]:
    """The environment for every subprocess that runs model-supplied code (run_python, and both pytests,
    which import the model's modules).

    This shell carries DEEPSEEK_API_KEY and a dozen other credentials; inheriting os.environ would hand them
    to a snippet the model wrote. Pass exactly these six variables and nothing else."""
    return {"PATH": os.environ.get("PATH", ""), "HOME": str(home), "PYTHONPATH": "",
            "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8", "TMPDIR": str(home)}


def _pytest(d: Path, hidden: bool, timeout: int = 300) -> tuple[bool, str]:
    files = sorted(str(p.relative_to(d)) for p in d.rglob("test_*.py") if p.name.startswith("test_hidden_") == hidden)
    if not files:
        return True, "no tests"
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *files], cwd=d,
                       capture_output=True, text=True, timeout=timeout, env=_child_env(d))
    return r.returncode == 0, (r.stdout + r.stderr)[-4000:]


def _safe(d: Path, p: str) -> Path | None:
    q = (d / p.lstrip("/")).resolve()
    s = str(d.resolve())
    return q if (str(q) == s or str(q).startswith(s + os.sep)) and q != d.resolve() else None


def _grade_answer(pattern, text: str) -> tuple[bool, str]:
    """An exact-answer task: PASS iff `pattern` (a regex from task.json's "answer") matches the final message."""
    text = text or ""
    try:
        m = re.search(pattern, text)
    except re.error as e:
        return False, f"error: task.json 'answer' is not a valid regex ({e}): {pattern!r}"
    if m:
        return True, f"answer regex {pattern!r} matched the final message: {m.group(0)!r}"
    return False, f"answer regex {pattern!r} did NOT match the final assistant message ({len(text)} chars): {text[-400:]!r}"


def _partial(ok: bool, out: str) -> float:
    """Partial credit = the fraction of hidden tests passed, from pytest's last summary line (1.0 / 0.0 for an
    answer task). A finer signal than pass/10 when comparing two engines on the same frozen set - but the headline
    stays pass/10, because a task is either done or not."""
    if ok:
        return 1.0
    last = out.strip().splitlines()[-1] if out.strip() else ""
    n = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|error|errors)\b", last)}
    total = n.get("passed", 0) + n.get("failed", 0) + n.get("error", 0) + n.get("errors", 0)
    return round(n.get("passed", 0) / total, 4) if total else 0.0


def _grade_snapshot(d: Path, task: Path, timeout: int) -> tuple[bool, str, list[str]]:
    """The hidden tests on a COPY of the workspace (the live one never sees them): strip anything that could grade
    itself, copy hidden/ in, run pytest. Returns (ok, output, stripped names)."""
    g = Path(tempfile.mkdtemp(prefix="gauntlet-grade-"))
    try:
        shutil.copytree(d, g, dirs_exist_ok=True)
        gone = _strip_grading_names(g)  # plant a harness file and it is gone before it can grade
        shutil.copytree(task / "hidden", g, dirs_exist_ok=True)
        try:
            ok, out = _pytest(g, True, timeout)
        except subprocess.TimeoutExpired:
            ok, out = False, f"hidden grading exceeded its {timeout} s budget"
        return ok, out, gone
    finally:
        shutil.rmtree(g, ignore_errors=True)


_START = {}  # task id -> the hidden fraction its UNTOUCHED workspace already scores (free credit, subtracted)


def start_fraction(task: Path) -> float:
    """What the starting workspace scores on the hidden tests before the model does anything (12's stub passed 1 of
    14 - credit nobody earned). Progress is measured from here."""
    meta = json.loads((task / "task.json").read_text())
    if meta["id"] not in _START:
        if meta.get("answer") is not None or not (task / "hidden").is_dir() or not (task / "workspace").is_dir():
            _START[meta["id"]] = 0.0
        else:
            ok, out, _ = _grade_snapshot(task / "workspace", task, meta.get("grade_timeout", 300))
            _START[meta["id"]] = _partial(ok, out)
    return _START[meta["id"]]


def progress(frac: float, start: float) -> float:
    """Normalised progress: 0 = where the task started, 1 = solved (never below 0)."""
    return 1.0 if frac >= 1.0 else max(0.0, round((frac - start) / (1.0 - start), 4)) if start < 1.0 else 0.0


def run_task(url: str, task: Path) -> dict:
    meta = json.loads((task / "task.json").read_text())
    d = Path(tempfile.mkdtemp(prefix="gauntlet-run-"))
    if (task / "workspace").is_dir():  # exact-answer tasks have no workspace
        shutil.copytree(task / "workspace", d, dirs_exist_ok=True)
    hidden_tests = sorted((task / "hidden").rglob("test_hidden_*.py")) if (task / "hidden").is_dir() else []
    answer, final_content = meta.get("answer"), ""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": meta["prompt"]}]
    log, toks, secs, test_runs, py_runs, submitted, truncated = [], 0, 0.0, 0, 0, False, 0
    max_tokens = int(os.environ.get("GAUNTLET_MAX_TOKENS", "36864"))
    # GAUNTLET_TASK_TOKENS=N: a hard budget of N completion tokens (thinking included) for the WHOLE task - the same for
    # every engine, so a run's length is bounded by budget / decode speed. Out of budget = graded as it stands.
    budget = int(os.environ.get("GAUNTLET_TASK_TOKENS", "0") or 0)
    tb_env = int(os.environ.get("TASKEVAL_THINK_BUDGET", "0") or 0)
    integrity = {"refused": [], "removed": []}  # what the workspace tried to do to its own grade
    # GAUNTLET_TURNS=N (default 10): every task gets the same N turns (one model request each), whatever its
    # task.json says; 0 = the task's own max_steps. The hidden tests grade a copy of the workspace after every turn
    # that could have changed it (the model never sees that result), so a run records WHERE each model got to and the
    # turn it first solved the task; the task ends at that turn (solved) or at the limit.
    turns = int(os.environ.get("GAUNTLET_TURNS", "10") or 0) or meta.get("max_steps", 12)
    per_turn = hidden_tests and answer is None and os.environ.get("GAUNTLET_PER_TURN", "1") != "0"
    turn_timeout = min(meta.get("grade_timeout", 300), int(os.environ.get("GAUNTLET_TURN_GRADE_SECS", "120")))
    start = start_fraction(task) if hidden_tests and answer is None else 0.0
    curve, solved_turn, grade_secs = [], None, 0.0
    chat_error = None
    for _ in range(turns):
        step_cap = max_tokens
        if budget:
            if budget - toks < 1024:
                break
            step_cap = min(max_tokens, budget - toks)
        # a failed request (s6: q27's conversation ran past its 200K context on turn 8) ends the task where it got
        # to - the workspace is still graded below - instead of discarding the whole task as a runner error
        try:
            if tb_env:  # chat() adds the thinking budget on top of the answer's: split this step's cap between them
                ans_cap = min(4096, step_cap // 2)   # NOT `answer`: that is the exact-answer task's pattern (graded below)
                os.environ["TASKEVAL_THINK_BUDGET"] = str(max(256, min(tb_env, step_cap - ans_cap)))
                try:
                    j = T.chat(url, msgs, TOOLS, max_tokens=ans_cap)
                finally:
                    os.environ["TASKEVAL_THINK_BUDGET"] = str(tb_env)
            else:
                j = T.chat(url, msgs, TOOLS, max_tokens=step_cap)
        except Exception as e:
            chat_error = f"turn {len(log) + 1}: {e!r}"[:300]
            print(f"  [{meta['id'][:2]} step {len(log) + 1:2d}] request failed: {chat_error}", flush=True)
            break
        toks += j.get("usage", {}).get("completion_tokens", 0)
        secs += j["_secs"]
        m = j["choices"][0]["message"]
        truncated += j["choices"][0].get("finish_reason") == "length"
        keep = {"reasoning_content": m["reasoning_content"]} if m.get("reasoning_content") else {}
        final_content = m.get("content") or final_content  # the last text the model produced (incl. the submit turn)
        calls = T.parse_calls(m)
        tm, us = j.get("timings") or {}, j.get("usage") or {}
        log.append({"content": m.get("content"), "calls": calls, "reasoning": m.get("reasoning_content"),
                    "finish": j["choices"][0].get("finish_reason"), "secs": round(j["_secs"], 2), "usage": us,
                    "timings": tm})
        if os.environ.get("GAUNTLET_QUIET") is None:  # one live line per step: prefill (new/reused), decode, thinking
            print(f"  [{meta['id'][:2]} step {len(log):2d}] {j['_secs']:6.1f}s  prompt {us.get('prompt_tokens', 0):6d} "
                  f"(reused {tm.get('cache_n', (us.get('prompt_tokens_details') or {}).get('cached_tokens', 0))}, "
                  f"{tm.get('prompt_per_second', 0):.0f} tok/s)  out {us.get('completion_tokens', 0):5d} "
                  f"(think {len(m.get('reasoning_content') or '')} ch, {tm.get('predicted_per_second', 0):.1f} tok/s)  "
                  f"{','.join(c['name'] for c in calls) or '-'}  [{toks}/{budget or '-'} tok]", flush=True)
        if not calls:
            msgs.append({"role": "assistant", "content": m.get("content") or "", **keep})
            msgs.append({"role": "user", "content": "Use the tools (call submit when you are done)."})
            continue
        msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": m["tool_calls"], **keep})
        for c in calls:
            a, name = c["args"], c["name"]
            if name == "list_files":
                res = "\n".join(sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()
                                       and "__pycache__" not in p.parts))
            elif name == "read_file":
                p = _safe(d, a.get("path", ""))
                try:
                    res = p.read_text() if p and p.is_file() else f"error: no such file {a.get('path')}"
                except OSError as e:
                    res = f"error: cannot read {a.get('path')}: {e}"
            elif name == "write_file":
                p = _safe(d, a.get("path", ""))
                res = _write_refusal(d, p)
                if res:
                    integrity["refused"].append({"path": a.get("path", ""), "reason": res})
                else:
                    try:
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_text(a.get("content", ""))
                        res = f"wrote {len(a.get('content', ''))} bytes to {p.relative_to(d)}"
                    except OSError as e:
                        res = f"error: cannot write {a.get('path')}: {e}"
            elif name == "run_tests":
                test_runs += 1
                res = "error: test-run limit reached" if test_runs > meta.get("max_test_runs", 6) else _pytest(d, False)[1]
            elif name == "run_python":
                py_runs += 1
                if py_runs > meta.get("max_python_runs", 12):
                    res = "error: run_python limit reached"
                else:
                    try:
                        r = subprocess.run([sys.executable, "-c", a.get("code", "")], cwd=d, capture_output=True,
                                           text=True, timeout=20, env=_child_env(d))
                        res = (r.stdout + r.stderr)[-4000:] or "(no output)"
                    except subprocess.TimeoutExpired:
                        res = "error: timed out after 20 s"
            elif name == "submit":
                submitted, res = True, "submitted"
            else:
                res = f"error: unknown tool {name}"
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": res})
        if per_turn and any(c["name"] in ("write_file", "run_python") for c in calls):
            g0 = time.time()
            ok_t, out_t, _ = _grade_snapshot(d, task, turn_timeout)
            grade_secs += time.time() - g0
            curve.append([len(log), _partial(ok_t, out_t)])
            log[-1]["hidden_fraction"] = curve[-1][1]
            if ok_t:
                solved_turn = len(log)
                break
        if submitted:
            break
    if answer is not None:  # exact-answer task: the final message is the deliverable, there is nothing to run
        ok, out = _grade_answer(answer, final_content)
    elif not hidden_tests:  # nothing grades this task - say so loudly instead of passing it by default
        ok, out = False, (f"error: task {meta['id']} has no hidden test_hidden_*.py files and no 'answer' key - "
                          "nothing can grade it; refusing to report a silent pass")
    else:
        ok, out, gone = _grade_snapshot(d, task, meta.get("grade_timeout", 300))
        integrity["removed"] += gone
    shutil.rmtree(d, ignore_errors=True)
    part = _partial(ok, out)
    if ok and solved_turn is None:
        solved_turn = len(log)
    return {"id": meta["id"], "ok": ok, "submitted": submitted, "steps": len(log), "test_runs": test_runs,
            "truncated": truncated, "partial": part, "start": start, "progress": progress(part, start),
            "solved_turn": solved_turn, "turns": turns, "curve": curve, "grade_secs": round(grade_secs, 1),
            "tokens": toks, "budget": budget, "secs": round(secs, 1), "integrity": integrity,
            "hidden_output": out[-1500:], "log": log, "chat_error": chat_error}


def run(url, out, only):
    os.makedirs(out, exist_ok=True)
    tasks = sorted(p for p in (HERE / "tasks").iterdir() if p.is_dir() and (not only or p.name[:2] in only))
    res = []
    for t in tasks:
        try:
            r = run_task(url, t)
        except Exception as e:  # a task must never abort the whole run (it still has to be scored)
            r = {"id": t.name, "ok": False, "submitted": False, "steps": 0, "test_runs": 0, "tokens": 0, "secs": 0.0,
                 "integrity": {"refused": [], "removed": []},
                 "hidden_output": f"runner error: {e!r}", "log": []}
        res.append(r)
        print(f"{r['id']}: {'PASS' if r['ok'] else 'fail'} ({r['steps']} steps, {r['tokens']} tok, {r['secs']:.0f}s, "
              f"partial {r.get('partial', 0):.2f}, progress {r.get('progress', 0):.2f}, solved at turn "
              f"{r.get('solved_turn') or '-'}, curve {[f for _, f in r.get('curve', [])]}, "
              f"truncated {r.get('truncated', 0)})", flush=True)
    json.dump({"tasks": res, "timings": T.TIMINGS}, open(os.path.join(out, "gauntlet.json"), "w"), indent=1)


def score(dirs):
    """One row per run. SCORE = the sum over tasks of normalised progress (0 = the starting workspace, 1 = solved; a
    solved task counts 1), so a run that gets most of the way on hard tasks outranks one that does nothing - the
    pass count alone ties too often. turns = mean turn of first solve over the solved tasks.
    Throughput (local engines; the API reports no timings): decode = generated tokens / decode time, prefill = NEW
    prompt tokens / prefill time (reused prefix excluded), e2e = completion tokens / model wall time."""
    print(f"{'run':24s} {'solved':>7s} {'SCORE':>6s} {'turns':>5s} {'trunc':>5s} {'tok':>7s} {'secs':>6s} "
          f"{'decode':>7s} {'prefill':>8s} {'new/reused prompt':>19s} {'e2e':>6s}  per task (progress, solved@turn)")
    for d in dirs:
        g = json.load(open(os.path.join(d, "gauntlet.json")))
        ts = [t["timings"] for t in g.get("timings", []) if t.get("timings")]
        dn, dms = sum(t.get("predicted_n", 0) for t in ts), sum(t.get("predicted_ms", 0) for t in ts)
        pn, pms = sum(t.get("prompt_n", 0) for t in ts), sum(t.get("prompt_ms", 0) for t in ts)
        cn = sum(t.get("cache_n", 0) for t in ts)
        rs = g["tasks"]
        prog = [r.get("progress", r.get("partial", _partial(r["ok"], r.get("hidden_output", "")))) for r in rs]
        st = [r["solved_turn"] for r in rs if r.get("solved_turn")]
        tok, secs = sum(r['tokens'] for r in rs), sum(r['secs'] for r in rs)
        rate = lambda n, ms: f"{1000 * n / ms:.1f}" if ms > 0 else "-"
        per = " ".join(f"{r['id'][:2]}:{p:.2f}{'@' + str(r['solved_turn']) if r.get('solved_turn') else ''}"
                       for r, p in zip(rs, prog))
        print(f"{os.path.basename(d.rstrip('/')):24s} {sum(r['ok'] for r in rs):>3d}/{len(rs):<3d} {sum(prog):>6.2f} "
              f"{(sum(st) / len(st)) if st else 0:>5.1f} "
              f"{sum(r.get('truncated', 0) for r in rs):>5d} {tok:>7d} {secs:>6.0f} {rate(dn, dms):>7s} "
              f"{rate(pn, pms):>8s} {f'{pn}/{cn}':>19s} {tok / max(secs, 1e-9):>6.1f}  {per}")

if __name__ == "__main__":
    if sys.argv[1] == "run":
        only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
        run(sys.argv[2], sys.argv[3], only)
    else:
        score(sys.argv[2:])
