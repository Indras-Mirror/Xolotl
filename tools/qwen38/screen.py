#!/usr/bin/env python3
"""Per-expert, per-tier error screening for Qwen3.8-Flash-Next routed experts.

This is the qwen4exp counterpart of `tools/v41/v41_calib.py`'s `screen_expert`.  It answers, for
one layer and one expert at a time: if I store this expert's gate/up/down at candidate tier T, how
far does the expert's OUTPUT move on the tokens that actually route to it?

Why per-expert matters: the XQ allocator in `tools/v41/v41_alloc.py` picks a tier per expert from
measured held-out error.  On DeepSeek-V4.1 that per-expert choice re-tiers 41.5% of experts versus
collapsing to one tier per layer (measured from V41-XQ-1017-v4.gguf.experts.json), so the
measurement has to be per expert even if a given output format later cannot express it.

Inputs
------
  --dump DIR     per-layer MoE dumps written by the Xolotl engine (see dump_moe.patch):
                 moe_L<NN>.bin = {int64 T, int64 N, int64 K} then T*N float32 X then T*K int32 ids
  --model-dir    the FP8 safetensors checkpoint (per-expert tensors, convenient here)
  --layer L
  --tiers        comma list of ggml type names (default iq2_xxs,q2_0,iq2_xs,iq2_s,iq3_xxs,iq3_s,iq4_nl)

Output
------
  stats/L<NN>.json in exactly the schema `v41_alloc.py` reads:
    {summary: {...}, experts: {"<e>": {n_fit, n_eval, y_energy, egu_<t>, edn_<t>, e_<t>, rtn_<t>}}}

Quantisation goes through ggml itself (ggml_qshim.cpp -> ggml_quantize_chunk), never a numpy
re-implementation: the IQ types are lattice formats whose error depends on the imatrix weights, and
a hand-written version would measure a different format than the one we ship.

  python3 tools/qwen38/screen.py --model-dir "$FP8" --dump "$DUMPS" --layer 5 \
      --out /tmp/stats/L05.json
"""
import argparse
import ctypes
import json
import os
import pathlib
import re
import struct
import sys

import numpy as np

SHIM = pathlib.Path("/tmp/libgqshim.so")

FGATE, FUP, FDOWN = "gate_proj", "up_proj", "down_proj"


# ----------------------------------------------------------------------------- ggml shim
class GGML:
    def __init__(self, path=SHIM):
        if not pathlib.Path(path).exists():
            raise SystemExit(
                f"missing {path}\n"
                "build it: g++ -O2 -fPIC -shared tools/qwen38/ggml_qshim.cpp "
                "-I<llama.cpp>/ggml/include -L<llama.cpp>/build-cpu-screen/bin "
                "-lggml -lggml-base -Wl,-rpath,<...>/bin -o /tmp/libgqshim.so")
        self.L = ctypes.CDLL(str(path))
        L = self.L
        L.gq_id.argtypes, L.gq_id.restype = [ctypes.c_char_p], ctypes.c_int
        L.gq_row_bytes.argtypes, L.gq_row_bytes.restype = [ctypes.c_int, ctypes.c_int64], ctypes.c_int64
        L.gq_quantize.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
                                 ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_float)]
        L.gq_quantize.restype = ctypes.c_int64
        L.gq_dequantize.argtypes = [ctypes.c_int, ctypes.c_void_p,
                                    ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
        L.gq_dequantize.restype = ctypes.c_int

    def id(self, name):
        t = self.L.gq_id(name.encode())
        if t < 0:
            raise SystemExit(f"this ggml build has no type {name!r}")
        return t

    def row_bytes(self, type_name: str, cols: int) -> int:
        """Bytes per row, or <= 0 when this type's block size does not divide `cols`."""
        if type_name == "ptq1":
            return cols // 128 * 28 if cols % 128 == 0 else 0
        return int(self.L.gq_row_bytes(self.id(type_name), cols))

    def quantize_dequantize(self, W: np.ndarray, type_name: str, imatrix: np.ndarray | None):
        """W [rows, cols] f32 -> (Wq_dequantized [rows, cols] f32, bytes used).

        PTQ1_0 is outside stock ggml. Use v41_formats' writer encoder, then
        its bit-tested decoder: measurement and emitted bytes must agree.
        """
        rows, cols = W.shape
        if type_name == "ptq1":
            sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "v41"))
            import v41_formats as F
            raw = F.encode_rtn(W, "ptq1", imatrix)
            out = np.stack([F.decode(F.PTQ1_0, row, cols) for row in raw])
            return out, int(raw.nbytes)
        t = self.id(type_name)
        rb = self.L.gq_row_bytes(t, cols)
        if rb <= 0:
            raise SystemExit(f"{type_name}: bad row bytes for cols={cols} (not a multiple of blck)")
        src = np.ascontiguousarray(W, dtype=np.float32)
        buf = ctypes.create_string_buffer(rb * rows)
        # ggml takes ONE importance weight per input column (length n_per_row), not one per
        # element: llama-quant passes `imatrix_data->data()` after checking its size >= n_per_row.
        #
        # A NULL pointer is NOT "no importance" for the IQ2_*/IQ3_* types: ggml asserts
        # `imatrix != NULL` in ggml_quantize_chunk and CALLS abort(), which kills the whole
        # process - not an exception. One such type reached with None would take down a
        # multi-hour screen with no output. Uniform importance is the correct RTN meaning,
        # and it is a no-op for the types that ignore the matrix (Q2_0, IQ4_NL, Q8_0).
        if imatrix is None:
            im = np.ones(cols, dtype=np.float32)
        else:
            im = np.ascontiguousarray(np.asarray(imatrix, dtype=np.float32).ravel())
            if im.size != cols:
                raise SystemExit(f"imatrix has {im.size} entries, expected {cols} (one per column)")
        got = self.L.gq_quantize(
            t, src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), buf,
            rows, cols, im.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if got <= 0:
            raise SystemExit(f"{type_name}: ggml_quantize_chunk returned {got} for {rows}x{cols}")
        out = np.zeros(rows * cols, dtype=np.float32)
        rc = self.L.gq_dequantize(t, buf, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                  rows * cols)
        if rc != 0:
            raise SystemExit(f"{type_name}: dequantize failed rc={rc}")
        return out.reshape(rows, cols), got


# ----------------------------------------------------------------------------- checkpoint
class GGUFModel:
    """Read Qwen4Exp expert matrices from the exact calibration GGUF.

    Expert stacks are dequantized one expert at a time. The reader never
    materializes an entire 40 GB expert tensor in host memory.
    """
    def __init__(self, path):
        sys.path.insert(0, "/home/mal/AI/llama.cpp-master-rebase/gguf-py")
        import gguf
        from gguf import quants
        self._quants = quants
        self._gguf = gguf
        self._reader = gguf.GGUFReader(str(path))
        self.path = str(path)
        self.ts = {t.name: t for t in self._reader.tensors}
        arch = self._reader.fields.get("general.architecture")
        if arch is None or arch.contents() != "qwen4exp":
            raise SystemExit(f"{path}: expected general.architecture=qwen4exp")

    def tensor(self, name):
        t = self.ts[name]
        return self._quants.dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32)

    def expert_dims(self, layer, e, proj):
        t = self.ts[f"blk.{layer}.ffn_{proj.removesuffix('_proj')}_exps.weight"]
        shape = tuple(int(x) for x in t.shape)
        return (shape[1], shape[0])

    def expert(self, layer, e, proj):
        role = proj.removesuffix("_proj")
        name = f"blk.{layer}.ffn_{role}_exps.weight"
        t = self.ts[name]
        if len(t.shape) != 3:
            raise ValueError(f"{name}: expected stacked experts")
        cols, rows, experts = map(int, t.shape)
        if not 0 <= e < experts:
            raise KeyError(name)
        nb = t.n_bytes // experts
        data = np.asarray(t.data).reshape(experts, -1)[e]
        return self._quants.dequantize(data, t.tensor_type).astype(np.float32).reshape(rows, cols)

    def router(self, layer):
        return self.tensor(f"blk.{layer}.ffn_gate_inp.weight")


class FP8Model:
    """Reader for the Qwen FP8 checkpoint: float8_e4m3fn weights x bfloat16 block scales.

    Uses torch's lazy `safe_open` rather than loading shards whole: a shard is up to 2.1 GB and a
    run touches a dozen of them.  torch is required because the tensors are float8_e4m3fn and
    bfloat16, neither of which numpy's safetensors reader understands.
    """

    def __init__(self, model_dir):
        from safetensors import safe_open
        self._open = safe_open
        self.dir = pathlib.Path(model_dir)
        idx = self.dir / "model.safetensors.index.json"
        if not idx.is_file():
            raise SystemExit(f"no {idx} - this reads the HF checkpoint, not a GGUF")
        self.map = json.load(open(idx))["weight_map"]
        self._handles = {}
        self._order = []

    def _shard(self, name):
        f = self.map.get(name)
        if f is None:
            raise KeyError(name)
        if f in self._handles:
            self._order.remove(f)
            self._order.append(f)
            return self._handles[f]
        h = self._open(str(self.dir / f), framework="pt")
        self._handles[f] = h
        self._order.append(f)
        while len(self._order) > 3:            # keep a few resident, close the rest
            old = self._order.pop(0)
            dead = self._handles.pop(old, None)
            if dead is not None:
                try:
                    dead.__exit__(None, None, None)
                except Exception:
                    pass
        return self._handles[f]

    def tensor(self, name):
        """Any tensor as float32 numpy, applying its block scale when the checkpoint has one."""
        import torch
        w = self._shard(name).get_tensor(name)
        scale_name = name.replace(".weight", ".weight_scale_inv")
        if scale_name in self.map:
            s = self._shard(scale_name).get_tensor(scale_name)
            return self._deblock(w, s)
        return w.to(torch.float32).numpy()

    def names(self, layer, suffix):
        base = f"model.language_model.layers.{layer}.mlp.experts."
        return [n for n in self.map if n.startswith(base) and n.endswith(suffix)]

    def expert_dims(self, layer, e, proj):
        """(out, in) of an expert projection, from the tensor's own shape (no weight read)."""
        name = f"model.language_model.layers.{layer}.mlp.experts.{e}.{proj}.weight"
        t = self._shard(name).get_tensor(name)
        return tuple(int(x) for x in t.shape)

    def expert(self, layer, e, proj):
        """-> dequantised f32 weight [out, in]."""
        return self.tensor(
            f"model.language_model.layers.{layer}.mlp.experts.{e}.{proj}.weight")

    @staticmethod
    def _deblock(w, s):
        """E4M3 weights x per-block scales -> f32.  Block size comes from the scale shape, so any
        [128,128]-style checkpoint works."""
        import torch
        out, inn = int(w.shape[0]), int(w.shape[1])
        so, si = int(s.shape[0]), int(s.shape[1])
        bo, bi = -(-out // so), -(-inn // si)
        wf = w.to(torch.float32)
        sf = s.to(torch.float32)
        scale = sf.repeat_interleave(bo, dim=0).repeat_interleave(bi, dim=1)[:out, :inn]
        return (wf * scale).to(torch.float32).numpy()

    def router(self, layer):
        return self.tensor(f"model.language_model.layers.{layer}.mlp.gate.weight")


# ----------------------------------------------------------------------------- dumps
def dump_files(dump_dir, layer):
    """Every dump file for one layer, in chunk order.

    The engine's STRATA_DUMP_MOE_ALL writes one file per (layer, chunk) -
    `moe_L05_c0003.bin` - because the layer loop runs inside the per-chunk call and a single
    append-only file would interleave under concurrent sessions.  A single concatenated
    `moe_L05.bin` is also accepted, for dumps produced before that change.
    """
    d = pathlib.Path(dump_dir)
    pat = re.compile(rf"^moe_L{layer:02d}_c(\d+)\.bin$")
    kept, ignored = [], []
    for p in d.glob(f"moe_L{layer:02d}_c*.bin"):
        m = pat.match(p.name)
        # A file matching the glob but NOT the numeric pattern must be dropped, not sorted to
        # the front.  A well-formed foreign file (say moe_L05_cX.bin) is not truncated, so the
        # truncation check cannot catch it, and a -1 sort key would silently prepend it as
        # chunk 0 - a plausible-looking wrong quality number with no exception anywhere.
        (kept if m else ignored).append((int(m.group(1)) if m else -1, p))
    if ignored:
        print(f"  WARNING: ignoring {len(ignored)} file(s) whose chunk index is not numeric: "
              f"{[p.name for _, p in ignored]}", file=sys.stderr)
    if kept:
        # sort by the NUMERIC chunk index, not the path string: the engine zero-pads to 4 digits
        # (%04lld) so lexicographic order is right up to 9999 chunks, but past that "c10000"
        # sorts before "c9999" and the concatenation would silently be out of order.
        kept.sort(key=lambda t: t[0])
        return [p for _, p in kept]
    one = d / f"moe_L{layer:02d}.bin"
    return [one] if one.is_file() else []


def read_dump(paths):
    """{T, N, K} int64, then T*N f32 X, then T*K i32 ids - per file, concatenated over chunks."""
    if isinstance(paths, (str, pathlib.Path)):
        paths = [paths]
    Xs, idss = [], []
    for p in paths:
        b = pathlib.Path(p).read_bytes()
        T, N, K = struct.unpack_from("<3q", b, 0)
        off = 24
        want = off + T * N * 4 + T * K * 4
        if len(b) < want:
            raise SystemExit(f"{p}: truncated - header says T={T} N={N} K={K}, "
                             f"needs {want} bytes, has {len(b)}")
        Xs.append(np.frombuffer(b, dtype=np.float32, count=T * N, offset=off).reshape(T, N))
        off += T * N * 4
        idss.append(np.frombuffer(b, dtype=np.int32, count=T * K, offset=off).reshape(T, K))
    if len(Xs) > 1:
        return np.concatenate(Xs, axis=0), np.concatenate(idss, axis=0)
    return Xs[0], idss[0]


# ----------------------------------------------------------------------------- expert math
def swiglu(gate, up):
    """silu(gate) * up, no clamp (qwen4exp's experts have none, unlike DeepSeek's `lim`).

    Stable form: the naive 1/(1+exp(-x)) overflows for very negative x, and a single overflow
    poisons the whole error measurement with inf/NaN.
    """
    sig = np.where(gate >= 0, 1.0 / (1.0 + np.exp(-np.abs(gate))),
                   np.exp(-np.abs(gate)) / (1.0 + np.exp(-np.abs(gate))))
    return (gate * sig) * up


def expert_out(X, Wg, Wu, Wd):
    return swiglu(X @ Wg.T, X @ Wu.T) @ Wd.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--tiers", default="ptq1,iq2_xxs,q2_0,iq2_xs,iq2_s,iq3_xxs,iq3_s,iq4_nl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--edit-report", help="required for an ablated run: q8_ablate's completed report")
    ap.add_argument("--dn-tiers", default="iq4_nl,q8_0", help="protected edited-down ladder")
    ap.add_argument("--importance-out", help="optional .npz of exact fitted per-expert gu/dn importances for the writer")
    ap.add_argument("--min-eval", type=int, default=8)
    ap.add_argument("--min-fit", type=int, default=32,
                    help="fit tokens needed before the imatrix is built from data; below it "
                         "V4 uses an identity Hessian (v41_calib.py --min-fit, default 32)")
    ap.add_argument("--eval-frac", type=float, default=0.25)
    ap.add_argument("--limit-expert", type=int, default=-1, help="debug: only this expert id")
    a = ap.parse_args()

    tiers = [t.strip() for t in a.tiers.split(",") if t.strip()]
    G = GGML()
    model = GGUFModel(a.model_dir) if str(a.model_dir).endswith(".gguf") else FP8Model(a.model_dir)
    if a.edit_report:
        er = json.load(open(a.edit_report))
        if er.get("state") != "edited" or er.get("layers") != "all" or len(er.get("targets", [])) != 145:
            raise SystemExit("--edit-report is not a completed all-target edit")
        if pathlib.Path(er["model"]).resolve() != pathlib.Path(a.model_dir).resolve():
            raise SystemExit("edit report is for a different calibration model")
    elif str(a.model_dir).endswith(".gguf"):
        raise SystemExit("GGUF calibration requires --edit-report; refusing to mix pristine and edited runs")

    dump = dump_files(a.dump, a.layer)
    if not dump:
        raise SystemExit(f"no dumps for layer {a.layer} in {a.dump}\n"
                         f"  expected moe_L{a.layer:02d}_cNNNN.bin - run the engine with "
                         f"STRATA_DUMP_MOE_ALL=<dir>")
    print(f"  {len(dump)} chunk file(s) for layer {a.layer}", file=sys.stderr)
    X, ids = read_dump(dump)
    T, N = X.shape
    K = ids.shape[1]

    router = model.router(a.layer)
    n_router = router.shape[0]
    if not 0 < K <= n_router:
        raise SystemExit(f"dump says top_k={K} but layer {a.layer}'s router has {n_router} experts")
    logits = X @ router.T
    top = np.argpartition(-logits, K - 1, axis=1)[:, :K]
    z = np.take_along_axis(logits, top, axis=1)
    p = np.exp(z - z.max(axis=1, keepdims=True))
    w = p / p.sum(axis=1, keepdims=True)

    # Routing weights, keyed by EXPERT ID, not by slot.  This matters: the engine's kernel
    # (src/kernels/cuda/native_router.cu:route_topk) selects the max at every rank, so its ids
    # come out sorted by DESCENDING weight, while argpartition's within-k order is arbitrary --
    # measured only ~45% slot agreement.  Pairing `w[kcol]` against a dumped id would then
    # attach the wrong expert's weight on most rows.  A full scatter makes the pairing exact
    # regardless of either side's ordering, at T x n_router floats (~100 MB at 49k tokens).
    w_full = np.zeros((T, n_router), dtype=np.float32)
    w_full[np.arange(T)[:, None], top] = w

    # A ggml type is only usable on a tensor whose row length is a multiple of its block size.
    # This is not a preference: qwen4exp's n_ff is 640, which is not a multiple of 256, so every
    # IQ2_*/IQ3_* type is IMPOSSIBLE for the down projection.  down is limited to Q2_0 (64),
    # IQ4_NL (32) and Q8_0 (32) - which is exactly why the shipped third-party quants alternate
    # Q2_0 and IQ4_NL for down and nothing else.  gate/up see n_embd 2560 = 10 x 256, so they get
    # the full ladder.  Screening a type the format cannot hold would report an error for a file
    # that cannot be built, so drop those tiers per projection and say so.
    n_embd, n_ff = N, None
    for e in range(int(ids.max()) + 1):
        try:
            n_ff = model.expert_dims(a.layer, e, FDOWN)[1]
            break
        except KeyError:
            continue
    if n_ff is None:
        raise SystemExit("could not read any expert's geometry")
    tiers_gu = [t for t in tiers if G.row_bytes(t, n_embd) > 0]
    tiers_dn = [t for t in tiers if G.row_bytes(t, n_ff) > 0]
    if a.edit_report:
        tiers_dn = [t.strip() for t in a.dn_tiers.split(",") if t.strip() and G.row_bytes(t.strip(), n_ff) > 0]
    if any(t not in ("iq4_nl", "q8_0") for t in tiers_dn) and a.edit_report:
        raise SystemExit("edited down tensors allow only IQ4_NL or Q8_0")
    dropped = [t for t in tiers if t not in tiers_dn]
    if dropped:
        print(f"L{a.layer}: down (n_ff={n_ff}) cannot hold {dropped}; "
              f"down ladder = {tiers_dn}")
    if not tiers_gu or not tiers_dn:
        raise SystemExit(f"no usable tiers: gu={tiers_gu} dn={tiers_dn}")

    rng = np.random.RandomState(1234 + a.layer)
    held = rng.rand(T) < a.eval_frac

    n_expert = int(ids.max()) + 1
    unmatched_total = 0
    dropped_experts = []
    experts = {}
    importance = {}
    for e in range(n_expert):
        if a.limit_expert >= 0 and e != a.limit_expert:
            continue
        # kcol is deliberately discarded: weights are looked up by expert id (w_full[rows, e]),
        # not by the slot the id happened to occupy in the dump.
        rows, _ = np.where(ids == e)
        if rows.size == 0:
            continue
        if e >= n_router:
            raise SystemExit(f"dump routes to expert {e} but layer {a.layer}'s router has "
                             f"only {n_router} experts")
        wgt = w_full[rows, e]
        # A zero here means the engine routed a token to this expert but screen.py's own top-k
        # did not include it -- the set of routed experts disagreed.  Those rows would silently
        # contribute nothing, biasing the measurement, so count them rather than let it pass.
        n_zero = int((wgt <= 0).sum())
        if n_zero:
            unmatched_total += n_zero
        fit = ~held[rows]
        ev = held[rows]
        n_fit = int(fit.sum())
        n_eval = int(ev.sum())           # PRE-expansion, as v41_calib.py records it
        st_extra = {}
        if n_eval < a.min_eval:
            ev = np.ones_like(ev)          # nothing held out: measure on all, flagged below
            st_extra["eval_on_fit"] = True
        Xe = X[rows]
        try:
            Wg = model.expert(a.layer, e, FGATE)
            Wu = model.expert(a.layer, e, FUP)
            Wd = model.expert(a.layer, e, FDOWN)
        except KeyError:
            continue

        Y = expert_out(Xe[ev], Wg, Wu, Wd) * wgt[ev, None]
        y_energy = float(np.square(Y).sum())
        st = {"n_fit": n_fit, "n_eval": n_eval, "y_energy": y_energy}
        st.update(st_extra)

        # V4 methodology (tools/v41/v41_calib.py:418-428): the imatrix the quantiser is fitted
        # with comes from the FIT tokens, the output error from the HELD-OUT tokens.  Too few fit
        # tokens and V4 uses an identity Hessian - no token data at all - rather than falling back
        # to the rows it is about to be scored on.  Here the analogue is a uniform imatrix: this
        # path is RTN through ggml (no GPTQ), so the identity lands on the per-column importance
        # rather than on a Hessian.  Same intent: uniform, data-free, and never the eval rows.
        if n_fit >= a.min_fit:
            # per input column, weighted by routing weight and |x|
            im_gu = (np.abs(Xe[fit]) * wgt[fit, None]).mean(axis=0)
            # down sees the expert's own hidden activation, not X
            H = swiglu(Xe[fit] @ Wg.T, Xe[fit] @ Wu.T)
            im_dn = (np.abs(H) * wgt[fit, None]).mean(axis=0)
            st["n_im"] = n_fit
        else:
            im_gu = np.ones(Wg.shape[1], dtype=np.float32)
            im_dn = np.ones(Wd.shape[1], dtype=np.float32)
            st["n_im"] = 0
            st["lowdata"] = True

        # Export the EXACT importances used below. The final writer re-encodes the same
        # weights and must reproduce the same PTQ1_0 scale choices; a global imatrix or a
        # uniform fallback would silently pick different scales than were measured. Stored
        # for every expert including ones later dropped, so a layer whose allocation is
        # ptq1 can still encode all 512 experts.
        importance[f"L{a.layer}_gu_{e}"] = np.ascontiguousarray(im_gu, dtype=np.float32)
        importance[f"L{a.layer}_dn_{e}"] = np.ascontiguousarray(im_dn, dtype=np.float32)

        # gate+up tiers and down tiers are different ladders (see the block-size note above)
        Yd_fp = dict()
        for t in tiers_dn:
            Wdq, _ = G.quantize_dequantize(Wd, t, im_dn)
            Yd_fp[t] = expert_out(Xe[ev], Wg, Wu, Wdq) * wgt[ev, None]
            st[f"edn_{t}"] = float(np.square(Yd_fp[t] - Y).sum())
        for t in tiers_gu:
            Wgq, _ = G.quantize_dequantize(Wg, t, im_gu)
            Wuq, _ = G.quantize_dequantize(Wu, t, im_gu)
            Yg = expert_out(Xe[ev], Wgq, Wuq, Wd) * wgt[ev, None]
            st[f"egu_{t}"] = float(np.square(Yg - Y).sum())
            # both halves at the same tier, for the case a single-type file wants one number
            if t in tiers_dn:
                Wdq, _ = G.quantize_dequantize(Wd, t, im_dn)
                Yq = expert_out(Xe[ev], Wgq, Wuq, Wdq) * wgt[ev, None]
                full = float(np.square(Yq - Y).sum())
                st[f"e_{t}"] = full
                st[f"rtn_{t}"] = full      # the ggml path has no GPTQ variant to compare against

        # A non-finite error would reach v41_alloc.py as an ordinary float.  min() over NaN picks
        # unpredictably, so one bad expert could silently drive a whole layer's tiers with no
        # exception anywhere.  DROP the expert rather than the layer: the allocator treats an
        # absent expert as unseen and floors it at --unseen-floor q2_0, which is the conservative
        # outcome, whereas aborting would throw away the other 511 experts in this layer -- and,
        # with an unattended chain, stall the whole run over one poisoned row.
        # The isinstance target must include np.floating: `isinstance(np.float32(nan), float)` is
        # False, so a numpy-typed value would slip past a plain `float` test and silently disable
        # this guard -- with no test failing.  Every st value is float()-wrapped at assignment
        # today, but a later "optimisation" that drops the wrapper must not be able to reintroduce
        # the hole, so test the numeric ABC instead of one concrete type.
        numeric = (int, float, np.integer, np.floating)
        bad = [k for k, v in st.items()
               if isinstance(v, numeric) and not isinstance(v, bool) and not np.isfinite(v)]
        if bad or not np.isfinite(st["y_energy"]) or st["y_energy"] <= 0.0:
            dropped_experts.append((e, sorted(bad) or ["y_energy<=0"], n_eval, n_fit))
            continue
        experts[str(e)] = st

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": {"layer": a.layer, "tokens": int(T), "top_k": int(K),
                           "n_expert": n_expert, "tiers": tiers,
                           "gu_tiers": tiers_gu, "dn_tiers": tiers_dn,
                           "model": str(pathlib.Path(a.model_dir).resolve()),
                           "edit_report": str(pathlib.Path(a.edit_report).resolve()) if a.edit_report else None,
                           "encoders": {"ptq1": "v41_formats.encode_rtn", "other": "ggml_quantize_chunk"},
                           "eval_frac": a.eval_frac,
                           "unmatched_route_rows": unmatched_total,
                           "n_dropped_experts": len(dropped_experts),
                           "dropped_experts": [d[0] for d in dropped_experts]},
               "experts": experts}, open(out, "w"), indent=1)
    if a.importance_out:
        imp = pathlib.Path(a.importance_out)
        imp.parent.mkdir(parents=True, exist_ok=True)
        tmp = imp.with_suffix(imp.suffix + ".partial")
        # np.savez appends .npz to a PATH argument, so pass a handle for the partial file.
        with open(tmp, "wb") as f:
            np.savez(f, **importance)
        os.replace(tmp, imp)
        print(f"L{a.layer}: {len(importance) // 2} expert importances -> {imp}")
    print(f"L{a.layer}: {len(experts)} experts, {T} tokens -> {out}")
    if dropped_experts:
        print(f"L{a.layer}: WARNING dropped {len(dropped_experts)} expert(s) with a non-finite "
              f"error or y_energy<=0: {dropped_experts[:8]}"
              f"{' ...' if len(dropped_experts) > 8 else ''}. They are ABSENT from the stats, so "
              f"v41_alloc treats them as unseen and floors them at --unseen-floor.",
              file=sys.stderr)
    if unmatched_total:
        print(f"L{a.layer}: WARNING {unmatched_total} routed rows were not in screen.py's own "
              f"top-{K} (engine/screen set disagreement); they carry weight 0 and were dropped "
              f"from the measurement", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
