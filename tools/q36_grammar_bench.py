#!/usr/bin/env python3
"""Native q36 grammar acceptance, distribution and decode-cost evidence.

Reserve the GPU through your scheduler, without concurrent builds/benchmarks. All generated
requests and responses are retained in --out; validation uses independent Python
parsers of synthetic application formats and jsonschema, never the decoder's grammar state.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from strata_tokenizer import Tokenizer
from serve.structured import format_grammar

TEXT = r"[^;\n]*"
SAID = r"[^;\n]+"
KEYS = "presay|power|emote|gesture|gesture_amount|delegate|arm_task|approach|base|base_amount|needs_view|recall|look|halt|hush"
ACT = "none|gesture|arm_task|base|approach|power|view|delegate"
TURN = rf"act=({ACT})(;({KEYS})={TEXT})*;say={SAID};tone={TEXT}"
SCENE = rf"speak=false|speak=true;what={TEXT};kind=(appeared|gone|moved|still);tone={TEXT};presay={TEXT};change={TEXT}(;note={TEXT})?(;screen=(1|0))?;say={SAID}"
DIARY = {"type": "object", "properties": {
    "event": {"type": "string", "maxLength": 60},
    "mood": {"type": "string", "enum": ["calm", "happy", "curious"]},
    "importance": {"type": "integer", "minimum": 1, "maximum": 5},
    "people": {"type": "array", "maxItems": 2, "items": {"type": "string", "maxLength": 20}},
    "actions": {"type": "array", "maxItems": 2, "items": {"type": "string", "enum": ["wave", "look", "wait"]}}},
    "required": ["event", "mood", "importance", "people", "actions"], "additionalProperties": False}


def write_example_grammars(directory):
    """Create synthetic application fixtures; no external conversation data is needed."""
    directory.mkdir(parents=True, exist_ok=True)
    text_rules = 'text ::= [^;\\n]*\nsaid ::= [^;\\n]+\n'
    scene = ('root ::= "speak=false" | "speak=true;what=" text ";kind=" kind '
             '";tone=" text ";presay=" text ";change=" text '
             '(";note=" text)? (";screen=" ("1" | "0"))? ";say=" said\n'
             'kind ::= "appeared" | "gone" | "moved" | "still"\n' + text_rules)
    keys = ' | '.join(json.dumps(key) for key in KEYS.split('|'))
    acts = ' | '.join(json.dumps(key) for key in ACT.split('|'))
    turn_rule = 'turn ::= "act=" action (";" key "=" text)* ";say=" said ";tone=" text\n'
    common = turn_rule + f'action ::= {acts}\nkey ::= {keys}\n' + text_rules
    for name, grammar in (("scene", scene), ("turn", 'root ::= turn\n' + common),
                          ("turn_quiet", 'root ::= "quiet" | turn\n' + common)):
        (directory / (name + '.gbnf')).write_text(grammar)
    return directory


class Engine:
    def __init__(self, args, mtp, baseline=False):
        self.args, self.mtp = args, mtp
        self.log = (args.out / f"engine-{'baseline' if baseline else 'grammar'}-{mtp}.log").open("a")
        exe = args.baseline_engine if baseline else args.engine
        self.proc = subprocess.Popen([str(exe), "--model", str(args.model), "--max-context", "8192",
                                      "--mtp", str(mtp), "--serve"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1,
                                     env={**os.environ, "CUDA_VISIBLE_DEVICES": args.device})
        self.info = {}
        while True:
            line = self.read()
            if line.startswith("INFO "):
                self.info.update(item.split("=", 1) for item in line.split()[1:])
            if line.startswith("READY "):
                break
        if not baseline and self.info.get("grammar") != "1":
            self.close()
            raise RuntimeError("engine did not advertise grammar=1")
        if baseline and self.info.get("grammar") == "1":
            self.close()
            raise RuntimeError("baseline unexpectedly advertises grammar support")

    def read(self):
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"engine ended ({self.proc.poll()}); see {self.log.name}")
        return line.strip()

    def gen(self, prompt, grammar=None, temperature=.9, limit=200, seed=1000, penalty=0., after=None):
        keys = f" temperature={temperature} top_p=0.9 top_k=20 seed={seed}"
        if penalty:
            keys += f" penalty_present={penalty} penalty_last_n=64"
        if grammar:
            keys += f" grammar={grammar}"
        if after is not None:
            keys += f" grammar_after={after}"
        request = f"GEN {limit}{keys} " + ",".join(map(str, prompt))
        self.proc.stdin.write(request + "\n")
        self.proc.stdin.flush()
        tokens = []
        start = time.monotonic()
        while True:
            line = self.read()
            if line.startswith("T "):
                tokens.append(int(line[2:]))
            elif line.startswith("ERR "):
                return {"error": line, "tokens": tokens}
            elif line.startswith("DONE "):
                fields = line.split()
                return {"tokens": tokens, "text": self.args.tok.decode([t for t in tokens if t != self.args.eos]),
                        "decode_ms": float(fields[4]), "prompt_ms": float(fields[3]), "finish": fields[5],
                        "accepted": int(fields[6]), "offered": int(fields[7]), "wall_s": time.monotonic() - start}

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=60)
        self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def prompt(args, user, system="Follow the final instruction. Respond briefly in the requested format, without explanations.", thinking=False):
    suffix = "<think>\n" if thinking else "<think>\n\n</think>\n\n"
    return args.tok.encode(f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{suffix}", parse_special=True)


def example_prompt(args, kind, index=0):
    if kind == "scene":
        if index % 2 == 0:
            user = "The room is unchanged. There is nothing to announce. Return speak=false."
        else:
            user = ("A visitor entered the room and waved. Return one short scene report: "
                    "speak=true;what=visitor;kind=appeared;tone=happy;presay=;change=entered;say=Hello!")
    elif kind in ("turn", "turn_quiet"):
        if kind == "turn_quiet" and index % 2 == 0:
            user = "There is no request to handle. Return quiet."
        else:
            user = (f"A visitor says hello. Example {index}. Return one short action report: "
                    "act=gesture;gesture=wave;emote=happy;say=Hello, visitor!;tone=happy")
    elif kind == "yesno":
        user = f"Is {index} even? Answer only yes or no."
    else:
        user = (f"Event {index}: a visitor arrived and the assistant waved. Return a short JSON object with "
                'event, mood (calm/happy/curious), importance (1..5), people (an array), actions (wave/look/wait).')
    return prompt(args, user)


def valid(kind, row):
    if "error" in row or row["finish"] != "stop":
        return False
    if kind == "diary":
        import jsonschema
        try:
            jsonschema.validate(json.loads(row["text"]), DIARY)
            return True
        except (ValueError, jsonschema.ValidationError):
            return False
    pattern = {"scene": SCENE, "turn": TURN, "turn_quiet": "quiet|" + TURN, "yesno": "yes|no"}[kind]
    return re.fullmatch(pattern, row["text"]) is not None


def summarize(rows):
    tokens = sum(len(row["tokens"]) for row in rows)
    decode_ms = sum(row["decode_ms"] for row in rows)
    accepted, offered = sum(row["accepted"] for row in rows), sum(row["offered"] for row in rows)
    return {"n": len(rows), "valid": sum(row.get("valid", True) for row in rows), "tokens": tokens,
            "decode_ms": decode_ms, "tok_s": tokens * 1000 / decode_ms,
            "mtp_acceptance": accepted / offered if offered else None,
            "accepted": accepted, "offered": offered}


def write(args, name, data):
    (args.out / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def validity(args):
    summaries = []
    with (args.out / "validity.jsonl").open("a") as evidence:
        for mtp in (3, 0):
            with Engine(args, mtp) as engine:
                # Invalid compilation must not execute or corrupt the following request.
                bad = args.out / "invalid.gbnf"
                bad.write_text('root ::= [')
                check = engine.gen(example_prompt(args, "scene"), bad)
                write(args, f"invalid-grammar-{mtp}.json", check)
                assert check.get("error", "").startswith("ERR grammar:") and not check["tokens"], check
                for kind in ("scene", "turn", "turn_quiet", "yesno", "diary"):
                    modes = ("greedy", "sampling", "application") if kind in ("scene", "turn", "turn_quiet") else ("greedy", "sampling")
                    for mode in modes:
                        rows = []
                        temperature = 0. if mode == "greedy" else (.4 if kind == "scene" and mode == "application" else .9)
                        penalty = 1.5 if mode == "sampling" else 0.
                        for i in range(args.n):
                            row = engine.gen(example_prompt(args, kind, i), args.paths[kind], temperature,
                                             700 if kind in ("turn", "turn_quiet") else 200, 3000 + i,
                                             penalty)
                            row.update(kind=kind, mode=mode, mtp=mtp, index=i,
                                       temperature=temperature, presence_penalty=penalty)
                            row["valid"] = valid(kind, row)
                            evidence.write(json.dumps(row, ensure_ascii=False) + "\n")
                            evidence.flush()
                            rows.append(row)
                        summary = {"kind": kind, "mode": mode, "mtp": mtp, "temperature": temperature,
                                   "presence_penalty": penalty, **summarize(rows)}
                        summaries.append(summary)
                        write(args, "validity-summary.json", summaries)
                        print(json.dumps(summary), flush=True)
                # Lazy activation after actual model reasoning (both MTP settings).
                end_think = args.tok.encode("</think>", parse_special=True)[0]
                row = engine.gen(prompt(args, "Think briefly: is two plus two four? After thinking answer yes or no.", thinking=True),
                                 args.paths["yesno"], .9, 2048, after=end_think)
                write(args, f"thinking-{mtp}.json", row)
                assert end_think in row["tokens"], row
                tail = row["tokens"][row["tokens"].index(end_think) + 1:]
                row["answer"] = args.tok.decode([t for t in tail if t != args.eos])
                assert row["finish"] == "stop" and row["answer"] in ("yes", "no"), row
                write(args, f"thinking-{mtp}.json", row)


def tv(a, b):
    ca, cb = collections.Counter(a), collections.Counter(b)
    return .5 * sum(abs(ca[k] / len(a) - cb[k] / len(b)) for k in ca.keys() | cb.keys())


def chi2(a, b):
    from scipy.stats import chi2 as distribution
    ca, cb = collections.Counter(a), collections.Counter(b)
    keys = [k for k in ca.keys() | cb.keys() if ca[k] + cb[k] >= 10]
    cells = [(ca[k], cb[k]) for k in keys]
    rest = (len(a) - sum(ca[k] for k in keys), len(b) - sum(cb[k] for k in keys))
    if sum(rest):
        if sum(rest) < 10 and cells:
            # Keep expected counts >= 5 in the balanced two-sample table.
            index = min(range(len(cells)), key=lambda i: sum(cells[i]))
            old = cells.pop(index)
            rest = (rest[0] + old[0], rest[1] + old[1])
        cells.append(rest)
    if len(cells) <= 1:
        return 0., 0, 1.
    na, nb = len(a), len(b)
    stat = sum((oa - (oa + ob) * na / (na + nb)) ** 2 / ((oa + ob) * na / (na + nb)) +
               (ob - (oa + ob) * nb / (na + nb)) ** 2 / ((oa + ob) * nb / (na + nb)) for oa, ob in cells)
    return stat, len(cells) - 1, float(distribution.sf(stat, len(cells) - 1))


def stats(args):
    results = {}
    cases = [(kind, temperature) for kind, temperature in (("scene", .4), ("turn", .9))
             if args.stats_kind in ("all", kind)]
    for mtp in (3, 0):
        with Engine(args, mtp) as engine:
            for kind, temperature in cases:
                user = ("A visitor looks at the assistant and stays silent. Decide whether to announce anything. "
                        "Return speak=false or speak=true;what=visitor;kind=still;tone=calm;presay=;change=;say=Hello."
                        if kind == "scene" else
                        "A visitor waves and smiles. Return a short report: act=...;say=...;tone=...")
                ids = prompt(args, user)
                write(args, f"stats-prompt-{kind}.json", {"ids": ids, "text": args.tok.decode(ids), "temperature": temperature})
                rows = []
                with (args.out / f"stats-{kind}-{mtp}.jsonl").open("w") as evidence:
                    for i in range(args.samples):
                        row = engine.gen(ids, args.paths[kind], temperature, 6,
                                         seed=args.seed_offset + (10000 if mtp else 100000) + i)
                        if "error" in row:
                            raise RuntimeError(row)
                        evidence.write(json.dumps(row, ensure_ascii=False) + "\n")
                        evidence.flush()
                        rows.append(row)
                results[kind, mtp] = rows
                print(f"stats {kind} MTP={mtp}: {len(rows)} samples", flush=True)
    report = {}
    for kind, _ in cases:
        spec, plain = results[kind, 3], results[kind, 0]
        # EOS is absorbing, so early completion does not condition later positions
        # on surviving responses (scene speak=false can end before position six).
        pad = lambda row: (row["tokens"] + [args.eos] * 6)[:6]
        sa, sb = [pad(r) for r in spec], [pad(r) for r in plain]
        positions = []
        for pos in range(6):
            a, b = [r[pos] for r in sa], [r[pos] for r in sb]
            stat, df, p = chi2(a, b)
            half = len(b) // 2
            positions.append({"position": pos + 1, "tv": tv(a, b),
                              "noise_plain": tv(b[:half], b[half:]),
                              "noise_spec": tv(a[:half], a[half:]), "chi2": stat, "df": df, "p": p,
                              "distinct": len(set(a) | set(b))})
        ordered = sorted(range(6), key=lambda i: positions[i]["p"])
        adjusted = 0.
        for rank, index in enumerate(ordered):
            adjusted = max(adjusted, min(1., positions[index]["p"] * (6 - rank)))
            positions[index]["p_holm"] = adjusted
        report[kind] = {"samples_per_mode": args.samples, "seed_offset": args.seed_offset, "positions": positions,
                        "mtp_acceptance": summarize(spec)["mtp_acceptance"]}
    write(args, "stats-summary.json", report)
    print(json.dumps(report), flush=True)


def perf(args):
    report = []
    for mtp in (3, 0):
        with Engine(args, mtp) as engine:
            for kind, answer in (("scene", "speak=true;what=visitor;kind=appeared;tone=happy;presay=;change=entered;say=Hello, visitor!"),
                                 ("turn", "act=gesture;gesture=wave;emote=happy;presay=;say=Hello, visitor! Good to see you.;tone=happy")):
                for mode in (("greedy", "sampling") if args.perf_sampling else ("greedy",)):
                    groups = {False: [], True: []}
                    warmup = []
                    temperature = 0. if mode == "greedy" else (.4 if kind == "scene" else .9)
                    for i in range(args.perf_repeats):  # ABBA; first four warm the caches.
                        constrained = i % 4 in (1, 2)
                        # Both paths receive each case twice. Vary free text so
                        # cost reflects reusable grammar states, not identical replies.
                        variant = answer.replace("visitor", f"visitor {i // 4}")
                        ids = prompt(args, "Repeat this line exactly, without quotes or explanation:\n" + variant)
                        row = engine.gen(ids, args.paths[kind] if constrained else None, temperature, 200, seed=7000 + i)
                        row["case"] = i // 4
                        if constrained:
                            row["valid"] = valid(kind, row)
                            assert row["valid"], row
                        if i >= 4:
                            groups[constrained].append(row)
                        else:
                            warmup.append({"grammar": constrained, **row})
                    plain, gram = summarize(groups[False]), summarize(groups[True])
                    cases = collections.defaultdict(list)
                    for rs in groups.values():
                        for row in rs:
                            cases[row["case"]].append(row["tokens"])
                    same = all(all(tokens == rows[0] for tokens in rows) for rows in cases.values())
                    item = {"kind": kind, "mtp": mtp, "mode": mode, "plain": plain, "grammar": gram,
                            "same_tokens": same, "overhead_percent": 100 * (plain["tok_s"] / gram["tok_s"] - 1)}
                    report.append(item)
                    write(args, "perf-summary.json", report)
                    write(args, f"perf-{kind}-{mtp}-{mode}.json", {"warmup": warmup, **{str(k): v for k, v in groups.items()}})
                    print(json.dumps(item), flush=True)


def baseline(args):
    outputs = {}
    for original in (True, False):
        for mtp in (0, 3):
            with Engine(args, mtp, baseline=original) as engine:
                rows = []
                for kind, index in (("scene", 0), ("scene", 1), ("turn", 0), ("turn_quiet", 0), ("diary", 0)):
                    for penalty in (0., 1.5):
                        row = engine.gen(example_prompt(args, kind, index), temperature=0., limit=96, penalty=penalty)
                        row.update(kind=kind, index=index, presence_penalty=penalty)
                        rows.append(row)
                outputs[original, mtp] = rows
    report = {str(mtp): {"all_identical": all(a["tokens"] == b["tokens"] for a, b in zip(outputs[True, mtp], outputs[False, mtp])),
                         "before": outputs[True, mtp], "after": outputs[False, mtp]} for mtp in (0, 3)}
    write(args, "baseline-summary.json", report)
    print("baseline", {k: v["all_identical"] for k, v in report.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("engine", "model", "tokenizer", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--grammars", type=Path, help="optional custom scene/turn/turn_quiet GBNF directory")
    parser.add_argument("--device", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"), help="CUDA visible-device selection")
    parser.add_argument("--baseline-engine", type=Path)
    parser.add_argument("--phase", choices=["validity", "stats", "perf", "baseline", "all"], default="all")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--stats-kind", choices=["all", "scene", "turn"], default="all")
    parser.add_argument("--seed-offset", type=int, default=0, help="independent, reproducible replication seeds")
    parser.add_argument("--perf-repeats", type=int, default=24, help="ABBA requests including four warmups")
    parser.add_argument("--perf-sampling", action="store_true", help="also measure application temperatures")
    args = parser.parse_args()
    if args.perf_repeats < 8 or args.perf_repeats % 4:
        parser.error("--perf-repeats must be a multiple of four, at least eight")
    args.out.mkdir(parents=True, exist_ok=True)
    vocab = json.loads((args.tokenizer / "vocab.json").read_text())
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    args.tok = Tokenizer(tokens, (args.tokenizer / "merges.txt").read_text().split("\n"),
                         json.loads((args.tokenizer / "token_type.json").read_text()))
    args.eos = args.tok.encode("<|im_end|>", parse_special=True)[0]
    if args.grammars is None:
        args.grammars = write_example_grammars(args.out / "grammars")
    args.paths = {name: args.grammars / (name + ".gbnf") for name in ("scene", "turn", "turn_quiet")}
    args.paths["yesno"] = args.out / "yesno.gbnf"
    args.paths["yesno"].write_text('root ::= "yes" | "no"\n')
    args.paths["diary"] = args.out / "diary.gbnf"
    args.paths["diary"].write_text(format_grammar({"type": "json_schema", "json_schema": {"schema": DIARY}}, True))
    write(args, "diary-schema.json", DIARY)
    write(args, "run-metadata.json", {
        "engine": str(args.engine), "engine_sha256": hashlib.sha256(args.engine.read_bytes()).hexdigest(),
        "model": str(args.model), "n": args.n, "samples": args.samples, "phase": args.phase,
        "stats_kind": args.stats_kind, "seed_offset": args.seed_offset,
        "grammar_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in args.paths.items()}})
    for name, fn in (("baseline", baseline), ("validity", validity), ("stats", stats), ("perf", perf)):
        if args.phase in (name, "all"):
            if name == "baseline" and not args.baseline_engine:
                continue
            fn(args)


if __name__ == "__main__":
    main()
