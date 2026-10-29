#!/usr/bin/env python3
"""Alpha pre-flight: how much of the abliteration edit survives each ggml quantisation tier?

The question
------------
The abliteration edit is small (~3% of a tensor's Frobenius norm, measured on the existing IQ3_XXS
bake).  A quantiser can only store it where the perturbation exceeds half a quantisation step, so
a low-bit type can be mathematically uncapable of holding the edit: on the current 3-bit Flash-Next
bake, Q2_0 stores **0.0000** of it (alpha = 0, signal 1.0%) while IQ4_NL stores 0.3273 with only
17.5% of what it stores being signal rather than requantisation noise.

That matters more for Qwen than for DeepSeek because two published results agree that the routed
experts carry most of the refusal behaviour on this architecture:

  * Ae55667/halogen-qwen3.8-flash-next-abliterated (this exact model): "The overlay alone loads but
    still refuses; the MoE expert down_proj tensors carry most of the refusal behaviour."  They
    raise the edited tensors to Q8G64 where the base is 4-bit.
  * arXiv 2609.09793 (GLM-5.3-Flash, same hyper-connection residual): attention 0.039, dense 0.016,
    routed experts 0.148; all three jointly 0.776.

And the format fights back: the down projection's row length is n_ff = 640, which is not a multiple
of 256, so every IQ2_*/IQ3_*/Q4_K..Q6_K type is IMPOSSIBLE for it.  The only sub-8-bit choices for
down are Q2_0 (64) and IQ4_NL (32) - exactly the two types that barely hold the edit.  Q8_0 is the
only faithful option and it is expensive.

So the decision this tool settles: is it worth paying for Q8_0 on the expert down projections, or
does IQ4_NL already hold enough of the edit to be worth shipping?  It answers with numbers from our
own weights instead of borrowed ones.

The metric
----------
    d_intended = -lam * u (u^T W)                the edit the adapter specifies
    W_edit     = W + d_intended
    d_real     = Q(W_edit) - W                   what actually ends up on disk
    alpha      = <d_real, d_intended> / <d_intended, d_intended>
                 1.0 = the edit is stored exactly, 0.0 = the type cannot express it at all
    signal%    = ||alpha*d_intended||^2 / ||d_real||^2
                 the rest of d_real is requantisation noise injected across the whole tensor
    noise      = ||d_real - alpha*d_intended|| / ||W||
                 how much noise we add to the tensor relative to its own magnitude

alpha is the number that decides.  A tier with alpha near 0 ships an un-abliterated model at that
tensor; a tier with low signal% buys the edit at the cost of injecting noise.

  python3 tools/qwen38/alpha_preflight.py \
      --model-dir "/media/mal/Team CX2/models/Qwen3.8-Flash-Next-FP8" \
      --adapter /media/mal/NVME1TB/Models/Qwen3.8-Flash-Next-Abliterated-Adapter/Q5_K_M/adapter.gguf \
      --layers 1,20 --experts 4 --out /tmp/alpha.json

CPU only, no GPU, no engine build.  Reads a handful of experts per layer (6.5 MB each at f32), so a
dozen layers costs seconds.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import screen as S  # noqa: E402  (FP8Model, GGML)

GGUF_PY = pathlib.Path("/home/mal/AI/llama.cpp-master-rebase/gguf-py")

# adapter family -> FP8 checkpoint template.  {L} layer, {E} expert.
# The dense families matter as much as the experts: the published MoE result (arXiv 2609.09793)
# is that the joint intervention does the work (0.776) where any one family alone manages
# 0.039-0.148.  The whole dense edited set is only 0.84B params, so Q8_0 on all of it costs
# ~0.89 GB total, +0.42 GB over IQ4_NL.
FAMILIES = {
    "exp_down": "model.language_model.layers.{L}.mlp.experts.{E}.down_proj.weight",
    "sh_down": "model.language_model.layers.{L}.mlp.shared_expert.down_proj.weight",
    "attn_out": "model.language_model.layers.{L}.self_attn.o_proj.weight",
    "ssm_out": "model.language_model.layers.{L}.linear_attn.out_proj.weight",
    "ple_value": "model.language_model.layers.{L}.ple.value_proj.weight",
}
# only the routed experts are per-expert tensors; everything else is one tensor per layer
EXPERT_FAMILIES = {"exp_down"}
# family -> how the adapter names it (blk.<L>.<this>.weight)
ADAPTER_NAMES = {
    "exp_down": "ffn_down_exps",
    "sh_down": "ffn_down_shexp",
    "attn_out": "attn_output",
    "ssm_out": "ssm_out",
    "ple_value": "ple_value",
}

# the ladder down can actually hold at n_ff = 640 (see the module docstring)
DOWN_TIERS = ["q2_0", "iq4_nl", "q8_0"]


def load_adapter(path):
    """-> {base: {'u': [out,E] or [out], 'lam': float, 'rank': int}} for the down families."""
    if str(GGUF_PY) not in sys.path:
        sys.path.insert(0, str(GGUF_PY))
    import gguf

    r = gguf.GGUFReader(str(path))
    pairs = {}
    for t in r.tensors:
        base, kind = t.name.rsplit(".lora_", 1)
        pairs.setdefault(base, {})[kind] = t

    lam = None
    for f in r.fields.values():
        if f.name == "adapter.reexport.lambda":
            lam = float(np.asarray(f.contents()).ravel()[0])
    if lam is None:                     # apetersson's original: measured 1.483-1.507 on 145 targets
        lam = 1.5

    out = {}
    for base, ab in pairs.items():
        # adapter names look like "blk.<L>.<ADAPTER_NAMES[f]>.weight"
        fam = next((f for f, nm in ADAPTER_NAMES.items() if f".{nm}.weight" in base), None)
        if fam is None:
            continue
        B = np.asarray(ab["b"].data, dtype=np.float32)   # [1,out] or [1,out,E]
        rank = int(np.asarray(ab["b"].data).shape[0])
        if B.ndim == 3:                                   # expert tensor: [1, out, E]
            B = B[0]                                      # [out, E]
        nb = np.linalg.norm(B, axis=0)
        nb = np.atleast_1d(nb) if nb.ndim else nb.reshape(1)
        u = B / nb[None, :] if B.ndim == 2 else B / nb[0]
        if u.ndim == 1:
            u = u[:, None]
        same = float(np.min(np.abs(u.T @ u[:, 0]))) if u.shape[1] > 1 else 1.0
        out[base] = {"u": u, "lam": lam, "rank": rank, "same_dir": same, "family": fam}
    return out


def alpha_for(W, u, lam, ggml, tier):
    """Measure how much of the rank-1 edit `tier` can store.  W [out, in], u [out].

    Two alphas, because they answer different questions:

      alpha       <Q(W')-W, d_int> / <d_int,d_int>
                  how much of the edit appears in the stored tensor.  Mixes the edit with the
                  quantiser's own error on W', so a coarse type can score ~1 by luck of alignment
                  while storing none of the edit (see signal_pct).
      alpha_eff   <Q(W')-Q(W), d_int> / <d_int,d_int>
                  THE number that matters: compare against the same tensor quantised WITHOUT the
                  edit.  That subtracts the base quantisation noise, so alpha_eff is purely "does
                  the edit survive this type".  1.0 = fully survives, 0.0 = the type cannot hold it.
    """
    d_int = -lam * np.outer(u, u @ W)                     # [out, in], the intended delta
    W_edit = (W + d_int).astype(np.float32)
    Wq_edit, nbytes = ggml.quantize_dequantize(W_edit, tier, None)
    Wq_base, _ = ggml.quantize_dequantize(W.astype(np.float32), tier, None)

    denom = float(np.sum(d_int * d_int))
    if denom <= 0:
        return None
    wn = float(np.linalg.norm(W)) or 1e-30
    d_real = Wq_edit - W                                   # edit + the type's error on W'
    alpha = float(np.sum(d_real * d_int) / denom)
    d_edit = Wq_edit - Wq_base                             # edit alone, base noise removed
    alpha_eff = float(np.sum(d_edit * d_int) / denom)
    resid = d_edit - alpha_eff * d_int
    return {
        "alpha": alpha,
        "alpha_eff": alpha_eff,
        "signal_pct": 100.0 * (alpha_eff * alpha_eff * denom)
                      / (float(np.sum(d_edit * d_edit)) or 1e-30),
        "noise_rel": float(np.linalg.norm(resid)) / wn,
        "base_qerr_rel": float(np.linalg.norm(Wq_base - W)) / wn,
        "bytes_per_row": int(nbytes // max(W.shape[0], 1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="FP8 (or F16) HF checkpoint directory")
    ap.add_argument("--adapter", required=True, help="the abliteration LoRA GGUF")
    ap.add_argument("--layers", default="1,20")
    ap.add_argument("--experts", type=int, default=4, help="experts sampled per layer")
    ap.add_argument("--families", default="exp_down,sh_down")
    ap.add_argument("--tiers", default=",".join(DOWN_TIERS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    layers = [int(x) for x in a.layers.split(",") if x.strip()]
    families = [f.strip() for f in a.families.split(",") if f.strip()]
    tiers = [t.strip() for t in a.tiers.split(",") if t.strip()]

    G = S.GGML()
    model = S.FP8Model(a.model_dir)
    dirs = load_adapter(a.adapter)
    print(f"adapter: {len(dirs)} down tensors, lam={next(iter(dirs.values()))['lam']}")

    # drop tiers this tensor geometry cannot hold, per family (n_ff=640 for both down families)
    results, notes = {}, []
    for fam in families:
        for L in layers:
            key = f"blk.{L}.{ADAPTER_NAMES[fam]}.weight"
            if key not in dirs:
                notes.append(f"{key}: not in the adapter (layer has no {fam})")
                continue
            d = dirs[key]

            n_exp = d["u"].shape[1]
            if fam in EXPERT_FAMILIES:
                ids = np.linspace(0, n_exp - 1, min(a.experts, n_exp)).astype(int)
            else:
                ids = np.array([0])

            acc = {t: [] for t in tiers}
            usable_seen = set()
            for e in ids:
                name = FAMILIES[fam].format(L=L, E=int(e))
                try:
                    W = model.tensor(name)
                except KeyError:
                    notes.append(f"{name}: not in this checkpoint")
                    continue
                except FileNotFoundError:
                    notes.append(f"{name}: shard not downloaded yet")
                    continue
                # a tier is only possible if its block size divides this tensor's row length
                usable = [t for t in tiers if G.row_bytes(t, W.shape[1]) > 0]
                dropped = [t for t in tiers if t not in usable]
                if dropped:
                    notes.append(f"{key}: cannot hold {dropped} at in={W.shape[1]}")
                usable_seen |= set(usable)
                u = d["u"][:, 0] if d["u"].shape[1] == 1 else d["u"][:, int(e)]
                for t in usable:
                    r = alpha_for(W.astype(np.float32), u, d["lam"], G, t)
                    if r:
                        acc[t].append(r)
            usable = sorted(usable_seen)

            row = {}
            for t in usable:
                if not acc[t]:
                    continue
                row[t] = {
                    "alpha": float(np.median([x["alpha"] for x in acc[t]])),
                    "alpha_eff": float(np.median([x["alpha_eff"] for x in acc[t]])),
                    "signal_pct": float(np.median([x["signal_pct"] for x in acc[t]])),
                    "noise_rel": float(np.median([x["noise_rel"] for x in acc[t]])),
                    "base_qerr_rel": float(np.median([x["base_qerr_rel"] for x in acc[t]])),
                    "n": len(acc[t]),
                }
            if row:
                results[key] = {"family": fam, "layer": L, "rank": d["rank"],
                                "same_dir": d["same_dir"], "tiers": row}

    # ---- report
    print()
    print(f"{'tensor':32s} {'tier':7s} {'alpha':>7s} {'alpha_eff':>9s} {'signal%':>8s} "
          f"{'base_err':>8s}  n")
    for key, v in results.items():
        for t, m in v["tiers"].items():
            print(f"{key:32s} {t:7s} {m['alpha']:7.4f} {m['alpha_eff']:9.4f} "
                  f"{m['signal_pct']:8.1f} {m['base_qerr_rel']:8.4f}  {m['n']}")
        print()

    # ---- the recommendation that decides the 31 GB
    agg = {}
    for v in results.values():
        for t, m in v["tiers"].items():
            agg.setdefault(t, []).append(m)
    if agg:
        print("median across the sampled tensors:")
        print(f"{'tier':8s} {'alpha':>7s} {'alpha_eff':>9s} {'signal%':>8s} {'base_err':>8s}  verdict")
        for t, ms in agg.items():
            al = float(np.median([m["alpha"] for m in ms]))
            ae = float(np.median([m["alpha_eff"] for m in ms]))
            sg = float(np.median([m["signal_pct"] for m in ms]))
            bq = float(np.median([m["base_qerr_rel"] for m in ms]))
            # alpha_eff alone is not enough: q2_0 scores ~1.0 while destroying the tensor, because
            # the projection of a large random requantisation noise onto d_intended can still land
            # near 1.  A tier is only usable if it stores the edit (signal%) AND does not wreck the
            # tensor (base_err).
            if bq > 0.25:
                verdict = f"UNUSABLE - format damages the tensor ({bq:.2f} rel err)"
            elif sg < 30.0:
                verdict = f"edit buried in requant noise ({sg:.0f}% signal)"
            elif sg > 80.0:
                verdict = "edit stored cleanly"
            else:
                verdict = f"edit stored, {sg:.0f}% signal"
            print(f"{t:8s} {al:7.4f} {ae:9.4f} {sg:8.1f} {bq:8.4f}  {verdict}")

    if notes:
        print("\nnotes:")
        for n in sorted(set(notes)):
            print("  -", n)

    if a.out:
        json.dump({"model_dir": a.model_dir, "adapter": a.adapter,
                   "layers": layers, "families": families, "tiers": tiers,
                   "results": results, "notes": sorted(set(notes))},
                  open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
