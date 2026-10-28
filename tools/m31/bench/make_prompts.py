#!/usr/bin/env python3
"""Write deterministic short or long numerical fixtures for the vocabulary-only oracle.

No network or inference. Tokenization truncates each text to its .length count.
Every passage starts with a real task and has varying content across blocks.
"""
import argparse
import json
from pathlib import Path

LENGTHS = [64, 127, 128, 129, 256, 384, 512, 768, 1024, 1536,
           1792, 2048, 2049, 2050, 2176, 2304, 2560, 2816, 3000, 3000]
TASKS = {
    "ru": ("Объясни, как устроен поиск по разреженному индексу. Сопоставь примеры и найди ошибки.\n",
           "Запись {i}: в каталоге {j} было {a} документов; после обновления осталось {b}. "
           "Автор отметил строку «и\u0306, е\u0308, кафе\u0301», временную метку {t}:17 и ключ shard_{i}.\n"),
    "en": ("Read the incident log, track each change and explain which facts support the conclusion.\n",
           "Event {i}: partition {j} accepted {a} messages and retried {b}; "
           "the worker wrote café and cafe\u0301 at {t}:17. The operation was idempotent only for key item_{i}.\n"),
    "code": ("Review this Python code. Track state across operations and propose tests for boundary cases.\n",
             "# transaction {i}, shard {j}\n"
             "def step_{i}(state, value={a}):\n"
             "    old = state.get('counter', {b})\n"
             "    state['counter'] = old + value\n"
             "    return (state['counter'], {i} % 7)\n\n"),
    "math": ("Solve the following problems step by step; preserve exact fractions until the final answer.\n",
             "Problem {i}: a train travels {a} km in {j} hours, then {b} km in {t} hours. "
             "Compute the average speed. Also simplify ({a}*x+{b})-({j}*x-{i}) and verify at x=3.\n"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--suite", choices=("gate20", "long", "long-public", "cache-observatory"), default="gate20")
    p.add_argument("--corpus", type=Path, help="long suite: use this corpus, matching speed.py")
    a = p.parse_args()
    a.out.mkdir(exist_ok=False)
    manifest = []
    if a.suite == "long":
        if not a.corpus:
            p.error("long suite requires --corpus")
        text = a.corpus.read_text(encoding="utf-8")[:1200000]
        for length, offset in ((16384, 0), (32768, 7919)):
            name = f"corpus_{length}"
            (a.out / f"{name}.txt").write_text(text, encoding="utf-8")
            (a.out / f"{name}.length").write_text(f"{length}\n")
            (a.out / f"{name}.offset").write_text(f"{offset}\n")
        return
    if a.suite == "cache-observatory":
        if a.corpus:
            p.error("cache-observatory is self-contained; --corpus belongs to --suite long")
        text = "\n".join(f"Record {i}: The observatory measured clear skies over the northern river. "
                         f"Observer number {i % 71} entered the temperature and checked the previous report."
                         for i in range(7000))
        for length, reuse in ((257, 256), (1324, 1024)):
            name = f"cache_observatory_{length}"
            (a.out / f"{name}.txt").write_text(text, encoding="utf-8")
            (a.out / f"{name}.length").write_text(f"{length}\n")
            manifest.append(dict(case=name, group="cache", tokens=length, reuse=reuse,
                                 whole_chunks=[length], split_chunks=[reuse, length - reuse]))
        (a.out / "suite.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return
    if a.suite == "long-public":
        if a.corpus:
            p.error("long-public is self-contained; --corpus belongs to --suite long")
        for group in ("ru", "code"):
            prefix, passage = TASKS[group]
            # More than 32K tokens in the pinned vocabulary; every record varies.
            # The vocabulary-only helper enforces the requested exact token count.
            text = prefix + "".join(passage.format(i=k, j=k % 19 + 1, a=17 * k + 23,
                                                  b=11 * k + 7, t=k % 23 + 1) for k in range(2048))
            for length in (16384, 32768):
                name = f"{group}_long_{length}"
                (a.out / f"{name}.txt").write_text(text, encoding="utf-8")
                (a.out / f"{name}.length").write_text(f"{length}\n")
                manifest.append(dict(case=name, group=group, tokens=length, passages=2048))
        (a.out / "suite.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        return
    for i, length in enumerate(LENGTHS):
        group = list(TASKS)[i % 4]
        prefix, passage = TASKS[group]
        # >6000 tokens even for efficient English/code tokenization. No generation or model loading.
        text = prefix + "".join(passage.format(i=k, j=k % 19 + 1, a=17 * k + 23,
                                              b=11 * k + 7, t=k % 23 + 1) for k in range(400))
        name = f"{group}_{i:02d}_{length}"
        (a.out / f"{name}.txt").write_text(text, encoding="utf-8")
        (a.out / f"{name}.length").write_text(f"{length}\n")
        manifest.append(dict(case=name, group=group, tokens=length))
    (a.out / "suite.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
