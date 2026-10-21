#!/usr/bin/env python3
"""Batch-slot protocol check for ds41_generate --serve --batch-slots N (CPU backend, real model, no server).

The engine under test runs the slot protocol of serve/server.py's generate_batched (BGEN / BADM / BT / BDONE / BSTOP /
BYIELD); a second engine without slots answers the same prompts one at a time (GEN, greedy) as the reference.  With
--mode serial (the default: the core's per-slot states, every window slot by slot through the old single-sequence path)
or --mode stub (one state, host copies) a slot's tokens must equal the reference's exactly; with --mode core (batched
windows) the rows differ only by the row-batched matmuls, so tokens are compared for agreement and logits by KL.

  check_batch_slots.py --bin build/ds41_generate --model M.gguf --engram A:B --tokenizer tokenizer.json \
      [--scenario all|order|cancel|...]

Scenarios (each prints a JSON line; SUMMARY at the end):
  order      three BGEN while the others decode (reads alone: STRATA_BATCH_MIX_MAX=0): every slot's tokens = reference
  cancel     BGEN into a busy slot is refused (ERR BGEN); BSTOP ends a slot early (BDONE cancel), its tokens a prefix
  classes    -exact and -fast requests in the slots at once: no window mixes the two routing classes (engine trace)
  solo       a slot BSTOPped to go solo, then GEN with prompt + output: continues in the same slot (RESUME), = reference
  promote    a solo GEN STOPped, then BGEN with prompt + output into another slot: the conversation moves (swap), = ref
  yield      BYIELD during a slot's long read: YIELDED + DONE cancel + BADM 0, the read goes on from there, = reference
  yieldsolo  the same for a solo read (into a free slot the "server" names)
  edit       solo turns: a side request in another slot, the next turn, then an edit of it: resumes from the pool
  mix        a short BGEN prompt read in the windows beside a decoding slot (its rows mixed in): the decoder = reference
             exactly, the mixed read's first-token logits vs the reference's (KL, top-1)
"""
import argparse
import json
import math
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time

import numpy as np

NV = 129280


class Engine:
    def __init__(self, binary, args, env, workdir, name):
        self.name = name
        self.dump = Path(workdir) / (name + "-dump")
        self.dump.mkdir(parents=True, exist_ok=True)
        for f in self.dump.glob("req*.f32"):
            f.unlink()
        self.log_path = Path(workdir) / (name + ".log")
        self.log = self.log_path.open("w")
        e = {k: v for k, v in os.environ.items() if not k.startswith(("STRATA_", "DS41_"))}
        e.update(CUDA_VISIBLE_DEVICES="-1", HIP_VISIBLE_DEVICES="-1", STRATA_ENGRAM_CACHE_ROWS="65536")
        e.update(env)
        e["STRATA_SERVE_DUMP"] = str(self.dump)
        self.p = subprocess.Popen([str(binary), *args, "--serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.log, text=True, env=e, bufsize=1)
        self.ctl = queue.Queue()
        self.slot = {}
        self.lock = threading.Lock()
        self.info = {}
        self.nreq = 0          # requests that ran a prompt (each dumps req%03d.f32)
        threading.Thread(target=self._pump, daemon=True).start()
        t0 = time.time()
        while True:
            ln = self.ctl.get(timeout=1800)
            if ln is None:
                raise RuntimeError(f"{name}: engine exited at start; see {self.log_path}")
            if ln.startswith("INFO"):
                for kv in ln.split()[1:]:
                    k, _, v = kv.partition("=")
                    self.info[k] = v
            if ln.startswith("READY"):
                break
        print(f"[{name}] ready in {time.time() - t0:.0f} s, info {self.info}", flush=True)

    def _q(self, s):
        with self.lock:
            return self.slot.setdefault(s, queue.Queue())

    def _pump(self):
        for ln in self.p.stdout:
            ln = ln.rstrip("\n")
            if ln.startswith(("BT ", "BDONE ")):
                self._q(int(ln.split()[1])).put(ln)
            else:
                self.ctl.put(ln)
        self.ctl.put(None)
        with self.lock:
            for q in self.slot.values():
                q.put(None)

    def send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def control(self, until=("DONE",), stop_after=0, on_t=None):
        """Control lines up to (and including) the first starting with `until` (or ERR)."""
        out = {"resume": 0, "tokens": [], "lines": []}
        while True:
            ln = self.ctl.get(timeout=14400)   # (a card chain may freeze this test for hours)
            if ln is None:
                raise RuntimeError(f"{self.name}: engine died; see {self.log_path}")
            out["lines"].append(ln)
            if ln.startswith("RESUME "):
                out["resume"] = int(ln.split()[1])
            elif ln.startswith("T "):
                out["tokens"].append(int(ln.split()[1]))
                if on_t:
                    on_t(len(out["tokens"]))
            elif ln.startswith("YIELDED "):
                out["yielded"] = [int(x) for x in ln.split()[1:3]]
            elif ln.startswith("DONE"):
                out["done"] = ln.split()
            elif ln.startswith("BADM "):
                out["badm"] = int(ln.split()[2])
            elif ln.startswith("ERR"):
                out["err"] = ln
                return out
            if ln.startswith(until):
                return out

    def _dumped(self, out):   # every request whose prompt phase ended (DONE, not a yield) dumped its last-row logits
        f = self.dump / f"req{self.nreq:03d}.f32"
        out["logits"] = np.fromfile(f, dtype=np.float32) if f.exists() else None
        self.nreq += 1

    def gen(self, ids, max_new, keys="", stop_after=0, then=None):
        """A solo request (GEN): its tokens; `stop_after`: STOP after that many tokens; `then`: a line sent right after."""
        self.send(f"GEN {max_new} temperature=0{(' ' + keys) if keys else ''} {','.join(map(str, ids))}")
        if then:
            self.send(then)
        out = self.control(("DONE",), on_t=(lambda n: self.send("STOP") if stop_after and n == stop_after else None))
        if "done" in out and "yielded" not in out:
            self._dumped(out)
        return out

    def bgen(self, slot, ids, max_new, keys="", then=None):
        """An admission (BGEN): its first token and BADM (1 = the slot decodes on)."""
        self.send(f"BGEN {slot} {max_new} temperature=0{(' ' + keys) if keys else ''} {','.join(map(str, ids))}")
        if then:
            self.send(then)
        out = self.control(("BADM",))
        if "done" in out and "yielded" not in out:
            self._dumped(out)
        return out

    def slot_run(self, slot, stop_after=0, timeout=14400):
        """A decoding slot's BT tokens up to its BDONE."""
        q, toks, end = self._q(slot), [], None
        while True:
            ln = q.get(timeout=timeout)
            if ln is None:
                raise RuntimeError(f"{self.name}: engine died; see {self.log_path}")
            f = ln.split()
            if f[0] == "BT":
                toks.append(int(f[2]))
                if stop_after and len(toks) == stop_after:
                    self.send(f"BSTOP {slot}")
            else:
                end = f
                break
        return toks, end

    def close(self):
        try:
            self.send("QUIT")
            self.p.wait(timeout=60)
        except Exception:
            self.p.kill()
        self.log.close()


def kl(a, b):
    pa = np.exp(a - a.max()); pa /= pa.sum()
    lb = b - b.max(); lb = lb - math.log(np.exp(lb).sum())
    return float(np.sum(pa * (np.log(pa + 1e-30) - lb)))


def cmp_logits(a, b):
    if a is None or b is None:
        return {"logits": "missing"}
    return {"bitexact": bool(np.array_equal(a, b)), "top1": bool(np.argmax(a) == np.argmax(b)),
            "kl": float(f"{kl(a, b):.3e}"), "maxdiff": round(float(np.max(np.abs(a - b))), 5)}


def corpus(tokenizer_json):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer_json))
    root = Path(__file__).resolve().parents[3]
    return [tok.encode((root / f).read_text(errors="replace")).ids for f in ("README.md", "HANDOFF.md", "howto_eng.md")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--ref-bin", help="the reference engine (default: --bin)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--engram", required=True)
    ap.add_argument("--tokenizer", default=os.environ.get("DS41_TOKENIZER_JSON"),
                    help="the model's tokenizer.json (default: $DS41_TOKENIZER_JSON)")
    ap.add_argument("--scenario", default="all")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out", default="/tmp/check_batch_slots")
    ap.add_argument("--mode", default="serial", choices=("serial", "stub", "core"),
                    help="serial: the core's slots, windows slot by slot through the old path (STRATA_BATCH_SERIAL=1; "
                         "tokens must equal the reference); stub: one state, host copies (STRATA_BATCH_STUB=1; equal too); "
                         "core: batched windows (Ds41RowsStep; agreement and KL reported)")
    ap.add_argument("--extra", default="", help="extra engine args (both engines)")
    a = ap.parse_args()
    if not a.tokenizer:
        ap.error("--tokenizer (or DS41_TOKENIZER_JSON) is required")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    common = ["-m", a.model, "--engram", a.engram, "--backend", "cpu", "--experts", "cpu", "--arena-gib", "0",
              "--ctx", "4096", "--threads", str(a.threads), "--prompt-batch", "32", "--prefill-chunk", "16",
              "--stop", "1", "--no-enc-skip"] + (a.extra.split() if a.extra else [])
    env_b = {"STRATA_SERVE_PARK_MIN": "16", "STRATA_BATCH_TRACE": "1"}
    if a.mode == "stub":
        env_b["STRATA_BATCH_STUB"] = "1"
    if a.mode == "serial":
        env_b["STRATA_BATCH_SERIAL"] = "1"
    exact = a.mode != "core"
    T = corpus(a.tokenizer)
    BOS = [0]
    A, B, C = BOS + T[0][:39], BOS + T[1][:47], BOS + T[2][:35]
    Y, Y2 = BOS + T[0][400:499], BOS + T[1][600:699]     # 100 tokens: four layer-major batches of 32
    D = BOS + T[2][200:239]
    want = a.scenario.split(",") if a.scenario != "all" else ["order", "cancel", "classes", "solo", "promote", "yield",
                                                              "yieldsolo", "edit", "mix"]
    res = []

    def report(d):
        print(json.dumps(d), flush=True)
        res.append(d)

    def same(toks, ref, label):
        n = min(len(toks), len(ref))
        ok = toks[:n] == ref[:n] if exact else sum(x == y for x, y in zip(toks, ref)) >= 0  # core: agreement reported
        agree = next((i for i in range(n) if toks[i] != ref[i]), n)
        return {"label": label, "tokens": toks, "ref": ref[:len(toks)], "equal": toks[:n] == ref[:n], "agree": agree,
                "ok": bool(ok)}

    # ---- the reference: one request at a time, no slots
    ref = Engine(a.ref_bin or a.bin, common, {"STRATA_SERVE_PARK_MIN": "16"}, out, "ref")
    R = {}
    for k, ids, n in (("A", A, 14), ("B", B, 6), ("C", C, 6), ("D", D, 6), ("Y", Y, 4), ("Y2", Y2, 4)):
        r = ref.gen(ids, n)
        R[k] = r["tokens"]
        R[k + "_logits"] = r.get("logits")
        print(f"[ref] {k}: prompt {len(ids)} -> {R[k]} ({r['done'][3]} ms prompt)", flush=True)
    ref.close()

    def engine(name, extra_env=None, slots=3):
        env = dict(env_b, **(extra_env or {}))
        return Engine(a.bin, common + ["--batch-slots", str(slots)], env, out, name)

    if any(s in want for s in ("order", "cancel", "classes", "solo", "promote", "yield", "yieldsolo", "edit")):
        E = engine("slots", {"STRATA_BATCH_MIX_MAX": "0"})
        if E.info.get("batch_slots") != "3":
            report({"scenario": "start", "ok": False, "info": E.info})
            E.close()
            raise SystemExit(1)
        if "order" in want:
            t0 = time.time()
            r0 = E.bgen(0, A, 6)
            r1 = E.bgen(1, B, 6)
            r2 = E.bgen(2, C, 6)
            got = {}
            for s, r in ((0, r0), (1, r1), (2, r2)):
                toks, end = E.slot_run(s) if r.get("badm") == 1 else ([], None)
                got[s] = (r["tokens"] + toks, end)
            d = {"scenario": "order", "sec": round(time.time() - t0, 1), "badm": [r0.get("badm"), r1.get("badm"), r2.get("badm")],
                 "slots": [same(got[0][0], R["A"], "A"), same(got[1][0], R["B"], "B"), same(got[2][0], R["C"], "C")],
                 "bdone": [got[s][1] for s in range(3)],
                 "admission_logits": [cmp_logits(r0.get("logits"), R["A_logits"]), cmp_logits(r1.get("logits"), R["B_logits"]),
                                      cmp_logits(r2.get("logits"), R["C_logits"])]}
            d["ok"] = all(x["ok"] and x["equal"] for x in d["slots"]) and all(e and e[3] == "length" for e in d["bdone"])
            report(d)
        if "cancel" in want:
            r0 = E.bgen(0, A, 40)
            E.send(f"BGEN 0 4 temperature=0 {','.join(map(str, B))}")    # slot 0 is decoding: refused
            refused = E.control(("BADM", "DONE"))
            r1 = E.bgen(1, B, 6)
            toks0, end0 = E.slot_run(0, stop_after=3)
            toks1, end1 = E.slot_run(1)
            d = {"scenario": "cancel", "refused": refused.get("err"), "bdone0": end0, "bdone1": end1,
                 "slot0": same(r0["tokens"] + toks0, R["A"], "A cancelled"), "slot1": same(r1["tokens"] + toks1, R["B"], "B")}
            d["ok"] = bool(refused.get("err", "").startswith("ERR BGEN")) and end0[3] == "cancel" and int(end0[2]) < 40 \
                and d["slot0"]["equal"] and d["slot1"]["equal"] and end1[3] == "length"
            report(d)
        if "classes" in want:
            mark = E.log_path.stat().st_size
            ex, fa = "route_bias=0.0 route_bias_file=0.0", "route_bias=0.5 route_bias_file=0.3"
            r0 = E.bgen(0, A, 6, ex)
            r1 = E.bgen(1, B, 6, fa)
            r2 = E.bgen(2, C, 6, ex)
            got = [r["tokens"] + E.slot_run(s)[0] for s, r in ((0, r0), (1, r1), (2, r2))]
            time.sleep(1)
            with open(E.log_path) as f:
                f.seek(mark)
                logt = f.read()
            wins = re.findall(r"strata batch: window \d+ class (\d+) \(bias ([\d.]+)/([\d.]+)\) decode \[([\d,]*)\]", logt)
            cls_of = {0: "0.00", 1: "0.50", 2: "0.00"}
            mixed = [w for w in wins if any(cls_of[int(s)] != w[1] for s in w[3].split(",") if s)]
            d = {"scenario": "classes", "windows": len(wins), "mixed_windows": len(mixed),
                 "classes": sorted(set((w[1], w[2]) for w in wins)),
                 "slots": [same(got[0], R["A"], "A exact"), same(got[1], R["B"], "B fast"), same(got[2], R["C"], "C exact")]}
            d["ok"] = len(wins) > 0 and not mixed and len(d["classes"]) == 2 and all(x["equal"] for x in d["slots"])
            report(d)
        if "solo" in want:
            r0 = E.bgen(0, A, 40)
            toks0, end0 = E.slot_run(0, stop_after=1)      # alone: the "server" sends it back to the solo path
            sofar = r0["tokens"] + toks0
            g = E.gen(A + sofar, 14 - len(sofar))
            d = {"scenario": "solo", "bdone0": end0, "resume": g["resume"], "expect_resume": len(A) + len(sofar) - 1,
                 "tokens": same(sofar + g["tokens"], R["A"], "A slot then solo")}
            d["ok"] = end0[3] == "cancel" and g["resume"] == len(A) + len(sofar) - 1 and d["tokens"]["equal"] \
                and len(sofar + g["tokens"]) == 14
            report(d)
        if "promote" in want:
            mark = E.log_path.stat().st_size
            g = E.gen(D, 6, stop_after=2)                  # solo, STOPped after 2 (another request came)
            sofar = g["tokens"]
            time.sleep(0.5)
            with open(E.log_path) as f:
                f.seek(mark)
                m = re.search(r"strata batch: request in slot (\d+) \(GEN\)", f.read())
            solo_slot = int(m.group(1)) if m else 0
            tgt = (solo_slot + 1) % 3                      # the "server" picks another slot: the conversation must move
            r = E.bgen(tgt, D + sofar, 6 - len(sofar))
            toks, end = E.slot_run(tgt) if r.get("badm") == 1 else ([], None)
            time.sleep(0.5)
            with open(E.log_path) as f:
                f.seek(mark)
                moved = f"slot {tgt} takes slot {solo_slot}'s conversation" in f.read()
            again = E.gen(D, 2)                            # the same prompt again: the turn's snapshot is still at its end
            d = {"scenario": "promote", "solo_done": g["done"][5], "solo_slot": solo_slot, "slot": tgt, "moved": moved,
                 "resume": r["resume"], "expect_resume": len(D) + len(sofar) - 1,
                 "tokens": same(sofar + r["tokens"] + toks, R["D"], "D solo then another slot"), "bdone": end,
                 "again_resume": again["resume"], "again": same(again["tokens"], R["D"], "D again")}
            d["ok"] = g["done"][5] == "cancel" and moved and r["resume"] == len(D) + len(sofar) - 1 and d["tokens"]["equal"] \
                and again["resume"] == len(D) and d["again"]["equal"]
            report(d)
        if "yield" in want:
            r = E.bgen(1, Y, 4, then="BYIELD 1")
            r2 = E.bgen(1, Y, 4)
            toks, end = E.slot_run(1) if r2.get("badm") == 1 else ([], None)
            d = {"scenario": "yield", "yielded": r.get("yielded"), "first": {"done": r.get("done"), "badm": r.get("badm")},
                 "resume": r2["resume"], "tokens": same(r2["tokens"] + toks, R["Y"], "Y yielded then read on"),
                 "logits": cmp_logits(r2.get("logits"), R["Y_logits"])}
            yq = (r.get("yielded") or [None, None])[1]
            d["ok"] = (r.get("yielded") or [None])[0] == 1 and yq in (32, 64) and r.get("badm") == 0 and r["done"][5] == "cancel" \
                and r2["resume"] == yq and d["tokens"]["equal"]
            report(d)
        if "yieldsolo" in want:
            g = E.gen(Y2, 4, then="BYIELD 2")
            r2 = E.bgen(2, Y2, 4)
            toks, end = E.slot_run(2) if r2.get("badm") == 1 else ([], None)
            d = {"scenario": "yieldsolo", "yielded": g.get("yielded"), "done": g.get("done"), "resume": r2["resume"],
                 "tokens": same(r2["tokens"] + toks, R["Y2"], "Y2 solo yielded, then slot 2"),
                 "logits": cmp_logits(r2.get("logits"), R["Y2_logits"])}
            yq = (g.get("yielded") or [None, None])[1]
            d["ok"] = (g.get("yielded") or [None])[0] == 2 and yq in (32, 64) and g["done"][5] == "cancel" and r2["resume"] == yq \
                and d["tokens"]["equal"]
            report(d)
        if "edit" in want:
            # solo turns only: a conversation, a side request (another slot), its next turn (continues the live state
            # in its slot), then an EDIT of that turn (shares only the first turn's prompt): it must resume from the
            # first turn's boundary in the pool, not read everything again (20.10 card: c3edit read 8185 tokens)
            P1 = BOS + T[2][400:439]
            g1 = E.gen(P1, 4)
            E.gen(B, 2)
            P2 = P1 + g1["tokens"] + T[2][500:510]
            E.gen(P2, 2)
            P3 = P1 + T[2][600:612]
            g3 = E.gen(P3, 2)
            d = {"scenario": "edit", "resume": g3["resume"], "expect_resume": len(P1)}
            d["ok"] = g3["resume"] == len(P1)
            report(d)
        E.close()
    if "mix" in want:
        E = engine("slots-mix", {}, slots=3)
        mark = 0
        r0 = E.bgen(0, A, 14)
        r1 = E.bgen(1, B, 6)              # read in the windows beside slot 0
        r2 = E.bgen(2, C, 6)              # ... beside slots 0 and 1
        got = [r["tokens"] + E.slot_run(s)[0] for s, r in ((0, r0), (1, r1), (2, r2))]
        time.sleep(1)
        logt = E.log_path.read_text()
        wins = re.findall(r"decode \[([\d,]*)\] read (\d+) of slot (\d+)", logt)
        d = {"scenario": "mix", "mixed_windows": len(wins), "read_rows": sum(int(w[1]) for w in wins),
             "slot0": same(got[0], R["A"], "A decoding while B, C are read"),
             "slot1": same(got[1], R["B"], "B read in the windows"), "slot2": same(got[2], R["C"], "C read in the windows"),
             "logits1": cmp_logits(r1.get("logits"), R["B_logits"]), "logits2": cmp_logits(r2.get("logits"), R["C_logits"])}
        d["ok"] = len(wins) > 0 and d["slot0"]["equal"] and d["logits1"].get("top1") and d["logits2"].get("top1")
        report(d)
        E.close()
    ok = all(d.get("ok") for d in res)
    print("SUMMARY " + json.dumps({"ok": ok, "scenarios": {d["scenario"]: d.get("ok") for d in res}}), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
