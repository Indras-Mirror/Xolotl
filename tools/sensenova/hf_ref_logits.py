#!/usr/bin/env python3
"""Reference log-probabilities from the HF SenseNova-Loop model (its own remote code), for the xolotl-dense parity check.

    python3 tools/sensenova/hf_ref_logits.py ids  <hf_dir> <out.ids> [--tokens 256]
        tokenizes a fixed slice of data/quant_quality/v1 (code + english + agent, chat-free, --tokens each) -> ids
    python3 tools/sensenova/hf_ref_logits.py run  <hf_dir> <in.ids> <out.tsv> [--layers-loop N]
        one teacher-forced BF16 forward over the ids (use_cache=False), rows in xolotl-dense --logits format:
        pos, argmax, then the 64 most likely `id:logp`.  --layers-loop N overrides eval_count (1 = no loop) to get
        a "what if the loop were dropped" baseline.  --dtype fp32 --device cpu: the full-precision ground truth
        (36 GB of host RAM for the 9B).  --fake-q8: round every 2-D weight to Q8_0 first (what the GGUF holds),
        to tell the model's own quant sensitivity apart from engine error.  --fake-act: also round every Linear's
        input to q8_1 (what the engine's GEMVs read).  Plain rope by default; --yarn keeps the config's YaRN.
    python3 tools/sensenova/hf_ref_logits.py compare REF.tsv RUN.tsv
        top-1 agreement and KL(REF || RUN) over REF's 64 most likely tokens (both renormalized to that set).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SET = ROOT / "data" / "quant_quality" / "v1"


def cmd_ids(hf_dir: str, out: str, n: int) -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=True)
    ids: list[int] = []
    for name in ("code.txt", "english.txt", "agent.txt"):
        ids += tok((SET / name).read_text(), add_special_tokens=False)["input_ids"][:n]
    Path(out).write_text(" ".join(map(str, ids)) + "\n")
    print(f"{len(ids)} ids -> {out}")


def cmd_run(hf_dir: str, ids_path: str, out: str, loops: int | None, dtype: str = "bf16", device: str = "cuda:0",
            fake_q8: bool = False, fake_act: bool = False, yarn: bool = False) -> None:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    ids = [int(x) for x in Path(ids_path).read_text().split()]
    cfg = AutoConfig.from_pretrained(hf_dir, trust_remote_code=True)
    if loops is not None:
        cfg.loop_settings = {**cfg.loop_settings, "counts": [loops], "eval_count": loops}
    if not yarn:
        # the shipped config applies YaRN (factor 2 over 262144) at every position; tests run the plain rope
        rp = {k: v for k, v in cfg.text_config.rope_parameters.items()
              if k not in ("factor", "original_max_position_embeddings", "beta_fast", "beta_slow", "attention_factor")}
        rp["rope_type"] = "default"
        cfg.text_config.rope_parameters = rp
    dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(hf_dir, config=cfg, trust_remote_code=True, dtype=dt,
                                                 device_map={"": device}).eval()
    if fake_q8:
        # every matrix the GGUF stores as Q8_0 (all 2-D weights of the text model; the mHC routers stay F32):
        # 32-value blocks along the input dim, scale = absmax / 127 rounded to f16, values rounded to int8
        n = 0
        with torch.no_grad():
            for name, prm in model.named_parameters():
                if prm.ndim != 2 or "visual" in name or "mhc_" in name or prm.shape[-1] % 32:
                    continue
                w = prm.data.float().reshape(prm.shape[0], -1, 32)
                d = (w.abs().amax(-1, keepdim=True) / 127.0).half().float()
                q = torch.where(d > 0, torch.round(w / d.clamp_min(1e-30)), torch.zeros_like(w)).clamp(-127, 127)
                prm.data.copy_((q * d).reshape(prm.shape).to(prm.dtype))
                n += 1
        print(f"fake Q8_0 on {n} matrices")
    if fake_act:
        # the engine's GEMVs read their input as q8_1: 32-value blocks, scale = absmax / 127 (f16), int8 values.
        # Every Linear of the text model (head included); the mHC routers are F32 matmuls in the engine, so not them.
        def q8_1(_mod, args):
            (h,) = args
            sh = h.shape
            w = h.float().reshape(*sh[:-1], -1, 32)
            d = (w.abs().amax(-1, keepdim=True) / 127.0).half().float()
            q = torch.where(d > 0, torch.round(w / d.clamp_min(1e-30)), torch.zeros_like(w)).clamp(-127, 127)
            return ((q * d).reshape(sh).to(h.dtype),)
        n = 0
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and "visual" not in name and "mhc_" not in name and mod.in_features % 32 == 0:
                mod.register_forward_pre_hook(q8_1)
                n += 1
        print(f"fake q8_1 activations on {n} linears")
    x = torch.tensor([ids], device=device)
    with torch.inference_mode():
        logits = model(input_ids=x, use_cache=False).logits[0].float()
    lp = torch.log_softmax(logits, dim=-1)
    tgt = torch.tensor(ids[1:], device=lp.device)
    nll = -lp[:-1].gather(1, tgt[:, None]).squeeze(1)
    print(f"text NLL {nll.mean().item():.4f} (ppl {math.exp(nll.mean().item()):.3f}) over {len(ids) - 1} positions")
    top = torch.topk(lp, 64, dim=-1)
    with open(out, "w") as f:
        for p in range(len(ids)):
            v, i = top.values[p].tolist(), top.indices[p].tolist()
            f.write(f"{p}\t{i[0]}\t" + "\t".join(f"{a}:{b:.6f}" for a, b in zip(i, v)) + "\n")
    print(f"{len(ids)} positions -> {out}")


def read(path: str) -> list[tuple[int, dict[int, float]]]:
    rows = []
    for line in Path(path).read_text().splitlines():
        f = line.split("\t")
        rows.append((int(f[1]), {int(a): float(b) for a, b in (x.split(":") for x in f[2:])}))
    return rows


def read_pos(path: str) -> dict[int, tuple[int, dict[int, float]]]:
    out = {}
    for line in Path(path).read_text().splitlines():
        f = line.split("\t")
        out[int(f[0])] = (int(f[1]), {int(a): float(b) for a, b in (x.split(":") for x in f[2:])})
    return out


def cmd_compare(ref: str, run: str) -> None:
    # matched by position, so a sparse run (xolotl-dense XOLOTL_LOGITS_PREFILL=N) compares against a full reference
    RA, RB = read_pos(ref), read_pos(run)
    common = sorted(set(RA) & set(RB))
    A, B = [RA[k] for k in common], [RB[k] for k in common]
    n = len(common)
    agree, kls = 0, []
    for (am, ad), (bm, bd) in zip(A[:n], B[:n]):
        agree += am == bm
        floor = min(bd.values())
        keys = list(ad)
        pa = [math.exp(ad[k]) for k in keys]
        pb = [math.exp(bd.get(k, floor)) for k in keys]
        za, zb = sum(pa), sum(pb)
        kls.append(sum((a / za) * math.log((a / za) / (b / zb)) for a, b in zip(pa, pb) if a > 0))
    kls.sort()
    print(f"positions {n}  top-1 agreement {100.0 * agree / n:.2f} %  KL mean {sum(kls) / n:.5f}  "
          f"p50 {kls[n // 2]:.5f}  p99 {kls[min(n - 1, int(n * 0.99))]:.5f}  max {kls[-1]:.4f}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "ids":
        cmd_ids(a[1], a[2], int(a[a.index("--tokens") + 1]) if "--tokens" in a else 256)
    elif a[0] == "run":
        cmd_run(a[1], a[2], a[3], int(a[a.index("--layers-loop") + 1]) if "--layers-loop" in a else None,
                a[a.index("--dtype") + 1] if "--dtype" in a else "bf16",
                a[a.index("--device") + 1] if "--device" in a else "cuda:0", "--fake-q8" in a, "--fake-act" in a, "--yarn" in a)
    elif a[0] == "compare":
        cmd_compare(a[1], a[2])
    else:
        sys.exit(__doc__)
