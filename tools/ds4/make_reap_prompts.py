#!/usr/bin/env python3
"""Calibration + held-out eval prompts for REAP pruning on DS4 (ds4_generate --saliency / --prune).

Writes <out>/cal.i32 and <out>/eval.i32 (concatenated domain segments, DS4 tokenizer) and <out>/eval.segments
("name start end" in prompt positions) so one eval run gives per-domain perplexity from DS4_PPL_DUMP.
Held-out design (strata-glm's lesson: a static prune list holds only for calibrated domains): calibration takes the
FIRST slice of de/zh/es and different code/prose files; eval takes LATER slices of de/zh (same language, unseen text)
and fr/ja/pt, which calibration never sees.
  python3 tools/ds4/make_reap_prompts.py bench/ds4-2026-10-09/reap
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from strata_tokenizer import Tokenizer  # noqa: E402
import numpy as np  # noqa: E402

MODEL = "/media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf"
N = 1200
CAL = [("cpp", "src/core/expert_cache.cpp", 0), ("py", "setup.py", 0), ("prose", "docs/ds4/ENGINE_MOE.md", 0),
       ("json", "tests/ds4/tokenizer_golden.json", 0), ("de", "data/multilingual/README.de.md", 0), ("zh", "data/multilingual/README.zh-CN.md", 0),
       ("es", "data/multilingual/README.es.md", 0)]
# English + Chinese + code only (Mal's question: do non-en/zh languages own prunable experts?)
CAL_ENZH = [("cpp", "src/core/expert_cache.cpp", 0), ("py", "setup.py", 0), ("prose", "docs/ds4/ENGINE_MOE.md", 0),
            ("json", "tests/ds4/tokenizer_golden.json", 0), ("prose2", "docs/ds4/ENGINE_DENSE.md", 0),
            ("zh", "data/multilingual/README.zh-CN.md", 0), ("prose3", "docs/ds4/PORT_PLAN.md", 0)]
EVAL = [("cpp", "tools/ds4/ds4_moe.cpp", 0), ("py", "serve/responses.py", 0), ("prose", "docs/ds4/DSV4_ARCH_SPEC.md", 0),
        ("de", "data/multilingual/README.de.md", N), ("zh", "data/multilingual/README.zh-CN.md", N), ("fr*", "data/multilingual/README.fr.md", N), ("ja*", "data/multilingual/README.ja.md", N),
        ("pt*", "data/multilingual/README.pt-BR.md", N)]


def build(tok, spec, bos):
    ids, segs = ([bos] if bos is not None else []), []
    for name, path, off in spec:
        t = tok.encode((REPO / path).read_text(encoding="utf-8"))[off:off + N]
        if len(t) < N // 2:
            sys.exit(f"{path}: only {len(t)} tokens at offset {off}")
        segs.append((name, len(ids), len(ids) + len(t)))
        ids += t
    return np.array(ids, dtype=np.int32), segs


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_gguf(MODEL)
    bos = tok.special_ids.get("tokenizer.ggml.bos_token_id")
    for name, spec in (("cal", CAL), ("cal_enzh", CAL_ENZH), ("eval", EVAL)):
        ids, segs = build(tok, spec, bos)
        ids.tofile(out / f"{name}.i32")
        (out / f"{name}.segments").write_text("".join(f"{n} {a} {b}\n" for n, a, b in segs))
        print(name, len(ids), "tokens:", " ".join(f"{n}={b - a}" for n, a, b in segs))


if __name__ == "__main__":
    main()
