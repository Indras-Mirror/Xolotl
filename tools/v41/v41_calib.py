#!/usr/bin/env python3
"""v41_calib.py - layer-streaming calibration forward + per-expert GPTQ tier screening, DeepSeek-V4.1-Flash.

Runs DeepSeek's own model.py (on tools/v41/torch_kernels) ONE LAYER AT A TIME straight from the HF safetensors, so a
layer can be processed as soon as its shard has downloaded.  State between layers lives in WORK:
  tokens.npy [N,S] int32, hashes.npy [N,S,2,24] int64 (engram n-gram ids), meta.json (domain + fit/eval per sequence)
  h/LNN.bin  [N,S,4,5120] bf16  hyper-connection stream entering layer NN (engram NOT yet added)
  pre/LNN.npy [N,S,4] f32       pre_mix for layer NN's attention (= previous FFN's hc_mixes)
  shared/LNN.pt                 SharedAttentionRuntime per batch (compressed KV / index keys / top-k / candidates)
  route/LNN.npz                 routing (indices int16, weights f16) for every token
  stats/LNN.json                per-expert screening: REAP saliency, token counts, held-out output error per tier
Per layer (`layer L`): attention half for every batch -> MoE expert-major over ALL tokens (each FP4 expert dequantized
once; exactly the reference Expert.forward) -> per-expert GPTQ at ptq1/q2_0/q3_k + RTN control -> hc_post -> h/L+1.
Screening error = ||f(x; W) - f(x; Q)||^2 summed over HELD-OUT sequences (GPTQ fits on the others), gate weight
included, same fp32 math for teacher and student (so it is weight damage only, not act-quant noise).

  v41_calib.py prep  [--n 128 --seq 2048]       tokenize the calibration mix + engram hashes (CPU)
  v41_calib.py init                             h/L00 from the embedding (shard 2)
  v41_calib.py layer L [--no-gptq] [--batch 2]  one layer (GPU; run under tools/ds4/memguard.sh)
Env: V41_WORK (default /media/Crucial1TB/v41-calib).
"""
import argparse, glob, hashlib, json, mmap, os, struct, sys, time
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.expanduser("~/AI/gsq-rco/dsv41-inference")
sys.path[:0] = [os.path.join(HERE, "torch_kernels"), REF, HERE]
import model as M  # noqa: E402
import kernel as K  # noqa: E402  (our torch stand-in)
import v41_gptq as G  # noqa: E402

# V41_SRC="dir1:dir2:..." = another checkpoint (e.g. the 1017): dir1 = MAIN (config, index, tokenizer), every dir is
# searched for <shard>.ok.  Unset = the dealignai FP8 below.  Read by v41_calib, calib_layers.sh, v41_gguf, v41_sidecar.
_SRC = [d for d in os.environ.get("V41_SRC", "").split(os.pathsep) if d]
MAIN = _SRC[0] if _SRC else "/media/Crucial1TB/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8"
DIRS = _SRC or [MAIN, "/media/mal/NVME1TB/Models/DeepSeek-V4.1-Flash-UNCENSORED-FP8",   # shard 48 (Engram L14) copied here 2026-10-11: NVMe, not Team CX2
                "/media/mal/Crucible/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8",
                "/media/mal/Team CX2/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8"]
WORK = os.environ.get("V41_WORK", "/media/Crucial1TB/v41-calib")
D, HC, NE, TOPK = 5120, 4, 384, 6
TIERS = list(G.TIERS)
BF16 = torch.bfloat16


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def wpath(*p):
    return os.path.join(WORK, *p)


def shard_file(name):
    """Verified local path of a shard (by its .ok marker), else None."""
    for d in DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p + ".ok"):
            return p
    return None


def weight_map():
    return json.load(open(os.path.join(MAIN, "model.safetensors.index.json")))["weight_map"]


def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def model_args(batch, seq):
    """DeepSeek's inference-format config (field names = ModelArgs).  The HF config.json in the model dir uses the
    transformers schema - feeding it to ModelArgs silently leaves the TINY defaults - so it is only cross-checked."""
    cfg = json.load(open(os.path.join(REF, "config.json")))
    unknown = set(cfg) - set(M.ModelArgs.__dataclass_fields__)
    assert not unknown, f"config keys ModelArgs does not know: {unknown}"
    args = M.ModelArgs(**cfg)
    hf = json.load(open(os.path.join(MAIN, "config.json")))
    hf = hf.get("text_config", hf)
    for mine, theirs in [("dim", "hidden_size"), ("n_layers", "num_hidden_layers"), ("n_routed_experts", None),
                         ("moe_inter_dim", "moe_intermediate_size"), ("n_activated_experts", "num_experts_per_tok"),
                         ("compress_ratios", None), ("kv_source_layers", "kv_source_layer_ids"),
                         ("index_source_layers", "index_source_layer_ids"), ("swiglu_limit", None),
                         ("route_scale", "routed_scaling_factor"), ("engram_num_embeddings", None),
                         ("engram_compressed_vocab_size", None), ("candidate_source_layer", "candidate_source_layer_id"),
                         ("window_size", "sliding_window"), ("norm_eps", "rms_norm_eps"), ("head_dim", None)]:
        a_, b_ = getattr(args, mine), hf[theirs or mine]
        assert (list(a_) if isinstance(a_, (list, tuple)) else a_) == b_, (mine, a_, b_)
    args.max_batch_size, args.max_seq_len = batch, seq
    return args


# ---------------------------------------------------------------------------------------------------------- prep
def _cjk_ratio(t):
    s = [c for c in t if not c.isspace()]
    if not s:
        return 0.0, 0.0
    cjk = sum("\u4e00" <= c <= "\u9fff" for c in s)
    kana = sum("\u3040" <= c <= "\u30ff" for c in s)
    return cjk / len(s), kana / len(s)


def _local_texts():
    """Repo code + zh/de/es prose from local files, deduplicated.  fr/ja/pt are deliberately excluded (held out)."""
    code, zh, eu, seen = [], [], [], set()

    def add(lst, p):
        try:
            t = open(p, encoding="utf-8").read()
        except Exception:
            return
        h = hashlib.sha1(t.encode()).hexdigest()
        if h not in seen and len(t) > 2000:
            seen.add(h); lst.append(t)
    for pat in ["~/AI/Xolotl/src/**/*.cpp", "~/AI/Xolotl/src/**/*.cu", "~/AI/Xolotl/tools/**/*.py",
                "~/AI/QuetzaCodetl/src/**/*.ts", "~/AI/Odin/mcp-server/*.js"]:
        for p in sorted(glob.glob(os.path.expanduser(pat), recursive=True)):
            add(code, p)
    for p in sorted(glob.glob(os.path.expanduser("~/AI/*/*.md")) + glob.glob(os.path.expanduser("~/AI/*/docs/*.md"))
                    + glob.glob(os.path.expanduser("~/AI/*/docs/**/*.md"), recursive=True)):
        if os.path.getsize(p) < 3000 or "node_modules" in p:
            continue
        try:
            t = open(p, encoding="utf-8").read()
        except Exception:
            continue
        c, k = _cjk_ratio(t[:20000])
        if c > 0.25 and k < 0.02:
            add(zh, p)
        elif os.path.basename(p) in ("README.de.md", "README.es.md"):
            add(eu, p)
    return code, zh, eu


def cmd_prep(a):
    import datasets
    from transformers import AutoTokenizer
    os.makedirs(WORK, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MAIN)
    BOS, EOS = tok.bos_token_id, tok.eos_token_id
    S, rng = a.seq, np.random.default_rng(0)
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    pile = datasets.load_dataset("NeelNanda/pile-10k", split="train")
    by = defaultdict(list)
    for i in rng.permutation(len(pile)):
        r = pile[int(i)]
        by[r["meta"]["pile_set_name"]].append(r["text"])

    def chat(t):  # StackExchange "Q:\n\n..\n\nA:\n\n.." -> DeepSeek non-thinking chat turn
        if "\nA:\n" not in t:
            return None
        q, ans = t.split("\nA:\n", 1)
        q = q.replace("Q:\n", "", 1).strip()
        return f"<｜User｜>{q}<｜Assistant｜></think>{ans.strip()}<｜end▁of▁sentence｜>"

    code_local, zh, eu = _local_texts()
    web_sets = ["Pile-CC", "OpenWebText2", "Wikipedia (en)", "HackerNews", "USPTO Backgrounds", "FreeLaw",
                "Enron Emails", "OpenSubtitles", "Books3", "BookCorpus2", "Ubuntu IRC"]
    sci_sets = ["PubMed Abstracts", "PubMed Central", "ArXiv", "NIH ExPorter", "DM Mathematics"]
    pools = {
        "web": [t for s in web_sets for t in by[s]],
        "sci": [t for s in sci_sets for t in by[s]],
        "chat": [c for c in map(chat, by["StackExchange"]) if c],
        "code": by["Github"][:400] + code_local,
        "zh": zh, "de_es": eu,
    }
    for k in ("web", "sci", "code", "zh", "de_es"):
        rng.shuffle(pools[k])
    want = {"web": 56, "sci": 10, "chat": 16, "code": 24, "zh": 14, "de_es": 8}
    if a.corpus:  # V4.1-rendered pools from build_corpus.py: agentic-first mix (docs/ds4/calib-refs/README.md)
        def cj(*names):
            out = []
            for n in names:
                f = os.path.join(a.corpus, f"{n}.jsonl")
                out += [json.loads(l)["text"] for l in open(f)] if os.path.exists(f) else []
            rng.shuffle(out)
            return out

        def mix(*lists):  # interleave so every source is represented in the first sequences
            out = []
            for i in range(max(map(len, lists))):
                out += [l[i] for l in lists if i < len(l)]
            return out
        sess = [json.loads(l)["text"] for l in open(os.path.join(a.corpus, "sessions.jsonl"))]
        rng.shuffle(sess)
        pools = {
            "agentic_long": sess,
            "agent": cj("antirez_agent"),
            "code": mix(cj("antirez_source", "antirez_programming", "antirez_algorithms"), pools["code"]),
            "reason": mix(cj("antirez_eval_reasoning", "antirez_long_context"), cj("antirez_language"), pools["sci"]),
            "web": pools["web"],
            "zh": zh,
            "chat": mix(cj("antirez_general"), pools["chat"]),
            "multi": mix(cj("antirez_translation"), eu),
        }
        want = {"agentic_long": 10, "agent": 5, "code": 6, "reason": 4, "web": 2, "zh": 3, "chat": 1, "multi": 1}
    for kv in filter(None, (a.mix or "").split(",")):   # --mix chat=4,agentic_long=7: per-domain sequence weights
        k, v = kv.split("=")
        assert k in want, f"--mix: unknown domain {k} (have {sorted(want)})"
        want[k] = int(v)
    scale = a.n / sum(want.values())
    want = {k: max(1, round(v * scale)) for k, v in want.items()}

    def pack(texts, n):
        seqs, cur = [], [BOS]
        for t in texts:
            special = t.startswith("<｜User｜>") or t.startswith("<｜begin▁of▁sentence｜>")
            if t.startswith("<｜begin▁of▁sentence｜>"):
                t = t[len("<｜begin▁of▁sentence｜>"):]   # pack() already opens every sequence with BOS
            ids = tok.encode(t, add_special_tokens=False)[: S - 1]
            if not special:
                ids = ids + [EOS]
            while ids:
                room = S - len(cur)
                cur += ids[:room]; ids = ids[room:]
                if len(cur) == S:
                    seqs.append(cur); cur = [BOS]
                    if len(seqs) == n:
                        return seqs
                    ids = ids[: S // 2]  # a long document continues once into the next sequence, then moves on
        return seqs
    seqs, doms = [], []
    for k, n in want.items():
        got = pack(pools[k], n)
        log(f"prep: {k}: {len(got)}/{n} sequences ({len(pools[k])} texts)")
        seqs += got; doms += [k] * len(got)
    short = a.n - len(seqs)
    if short > 0:  # a domain ran dry: fill with more web text
        got = pack(pools["web"][len(pools["web"]) // 2:], short)
        seqs += got; doms += ["web"] * len(got)
    seqs, doms = seqs[: a.n], doms[: a.n]
    # held-out split: every 4th sequence of each domain
    tot, is_eval, cnt = Counter(doms), [], defaultdict(int)
    for d_ in doms:
        n_ev = tot[d_] // 4 + (1 if tot[d_] % 4 == 3 else 0)
        is_eval.append(cnt[d_] >= tot[d_] - n_ev); cnt[d_] += 1
    tokens = np.array(seqs, dtype=np.int32)
    np.save(wpath("tokens.npy"), tokens)
    json.dump({"n": len(seqs), "seq": S, "domains": doms, "is_eval": is_eval, "corpus": a.corpus}, open(wpath("meta.json"), "w"))
    # engram n-gram hash ids for every position (depend on token ids only)
    args = model_args(len(seqs), S)
    layout = M.EngramLayout.from_args(args)
    hs = M.NgramHashState(args, layout, tok)
    with torch.inference_mode():
        hashes = hs(torch.from_numpy(tokens).long(), 0)
    np.save(wpath("hashes.npy"), hashes.numpy())
    log(f"prep: {tokens.shape} tokens, {sum(is_eval)} eval sequences, hashes {tuple(hashes.shape)} "
        f"max id {int(hashes.max())} (tables {args.engram_num_embeddings})")


def cmd_prep_ids(a):
    """One sequence from int32 token files (concatenated, as given - e.g. an engine prompt + its continuation), for a
    teacher forward over exactly the engine's tokens (tools/v41/dspark_accept.py --teacher)."""
    from transformers import AutoTokenizer
    os.makedirs(WORK, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MAIN)
    ids = np.concatenate([np.fromfile(f, dtype=np.int32) for f in a.ids])
    tokens = ids.reshape(1, -1).astype(np.int32)
    np.save(wpath("tokens.npy"), tokens)
    json.dump({"n": 1, "seq": int(tokens.shape[1]), "domains": [a.domain], "is_eval": [True], "corpus": None},
              open(wpath("meta.json"), "w"))
    args = model_args(1, int(tokens.shape[1]))
    layout = M.EngramLayout.from_args(args)
    hs = M.NgramHashState(args, layout, tok)
    with torch.inference_mode():
        hashes = hs(torch.from_numpy(tokens).long(), 0)
    np.save(wpath("hashes.npy"), hashes.numpy())
    log(f"prep-ids: {tokens.shape} tokens from {a.ids}")


# ---------------------------------------------------------------------------------------------------------- init
def h_memmap(L, mode, N, S):
    os.makedirs(wpath("h"), exist_ok=True)
    return np.memmap(wpath("h", f"L{L:02d}.bin"), dtype=np.int16, mode=mode, shape=(N, S, HC, D))


def cmd_init(a):
    tokens = np.load(wpath("tokens.npy"))
    N, S = tokens.shape
    from safetensors import safe_open
    with safe_open(shard_file("model-00002-of-00048.safetensors"), "pt") as f:
        emb = f.get_tensor("embed.weight")
    assert emb.dtype == BF16 and emb.shape[1] == D, (emb.dtype, emb.shape)
    out = h_memmap(0, "w+", N, S)
    for i in range(N):
        e = emb[torch.from_numpy(tokens[i]).long()]                        # [S, D]
        out[i] = e.unsqueeze(1).expand(S, HC, D).contiguous().view(torch.int16).numpy()
    out.flush()
    pre = np.zeros((N, S, HC), np.float32); pre[..., 0] = 1.0
    os.makedirs(wpath("pre"), exist_ok=True)
    np.save(wpath("pre", "L00.npy"), pre)
    log(f"init: h/L00 {N}x{S} from the embedding")


# ---------------------------------------------------------------------------------------------------------- layer
class MemmapEngramEmbedding(nn.Module):
    """Drop-in for model.ParallelEngramEmbedding: rows read from the shard on disk (the 98 GB table never loads).
    Same math: fp8 row * its E8M0 scale per 32, to bf16."""
    SRC = None  # (path, weight name, scale name) - set before the Block is built

    def __init__(self, num_embeddings, dim):
        super().__init__()
        path, wn, sn = self.SRC
        h, base = st_header(path)
        (wa, wb), (sa, sb) = h[wn]["data_offsets"], h[sn]["data_offsets"]
        assert h[wn]["shape"] == [num_embeddings, dim] and h[wn]["dtype"] == "F8_E4M3", h[wn]
        assert h[sn]["shape"] == [num_embeddings, dim // 32] and h[sn]["dtype"] == "F8_E8M0", h[sn]
        self.w = np.memmap(path, np.uint8, "r", offset=base + wa, shape=(num_embeddings, dim))
        self.s = np.memmap(path, np.uint8, "r", offset=base + sa, shape=(num_embeddings, dim // 32))
        for m in (self.w, self.s):
            m._mmap.madvise(mmap.MADV_RANDOM)
        self.dim, self.bytes_read, self.rows = dim, 0, 0

    def forward(self, indices):
        dev = indices.device
        u, inv = np.unique(indices.reshape(-1).cpu().numpy(), return_inverse=True)
        w = torch.from_numpy(np.ascontiguousarray(self.w[u])).to(dev).view(torch.float8_e4m3fn).float()
        s = torch.from_numpy(np.ascontiguousarray(self.s[u])).to(dev).to(torch.int32)
        s = torch.ldexp(torch.ones_like(s, dtype=torch.float32), s - 127)
        v = (w.view(len(u), -1, 32) * s.unsqueeze(-1)).view(len(u), self.dim).to(BF16)
        self.rows += len(u)
        return v[torch.from_numpy(inv.reshape(-1)).to(dev)].view(*indices.shape, self.dim)


def load_block(blk, L):
    """Copy layer L's tensors from the shards into the Block (no TP convert; wo_a dequantized like convert.py)."""
    from safetensors import safe_open
    wm = weight_map()
    pre = f"layers.{L}."
    names = [k for k in wm if k.startswith(pre)]
    params = dict(blk.named_parameters())
    loaded, by_file = set(), defaultdict(list)
    for k in names:
        by_file[wm[k]].append(k)
    for fn, ks in by_file.items():
        path = shard_file(fn)
        assert path, f"{fn} not downloaded/verified yet (needed for layer {L})"
        with safe_open(path, "pt") as f:
            for k in ks:
                rel = k[len(pre):]
                if rel.startswith("engram.embed.") or rel == "attn.wo_a.scale":
                    continue
                t = f.get_tensor(k)
                if rel == "attn.wo_a.weight":
                    s = f.get_tensor(k.replace(".weight", ".scale")).float()
                    bo, bi = t.shape[0] // s.shape[0], t.shape[1] // s.shape[1]
                    t = (t.float().unflatten(0, (-1, bo)).unflatten(-1, (-1, bi)) * s[:, None, :, None]).flatten(2, 3).flatten(0, 1)
                p = params[rel]
                assert tuple(p.shape) == tuple(t.shape), (k, p.shape, t.shape)
                if p.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.float4_e2m1fn_x2):
                    assert t.element_size() == 1, (k, t.dtype)
                    p.data.view(torch.uint8).copy_(t.view(torch.uint8))
                else:
                    p.data.copy_(t.to(p.dtype))
                loaded.add(rel)
    missing = set(params) - loaded
    assert not missing, f"layer {L}: params not in checkpoint: {sorted(missing)[:10]}"
    return len(loaded)


SHARED = ("compress_kv", "index_k", "topk_idxs", "candidates")


ROWS = int(os.environ.get("V41_ROWS", "32768"))   # token rows per GPU call for a hot expert (OOM retries halve it)


def expert_f(x, W1, W3, W2, g, lim):
    """fp32 SwiGLU expert with the training clamps and the routing weight (as Expert.forward, no act quant).
    Rows are independent: done ROWS at a time so a hot expert never needs a whole-batch intermediate."""
    if len(x) > ROWS:
        return torch.cat([expert_f(x[c:c + ROWS], W1, W3, W2, g[c:c + ROWS], lim) for c in range(0, len(x), ROWS)])
    x, g = x.float(), g.float()        # callers pass bf16 rows: fp32 only per ROWS chunk (a hot expert's fp32 copy OOMs)
    gate = x @ W1.T
    up = x @ W3.T
    if lim > 0:
        up = up.clamp(-lim, lim); gate = gate.clamp(max=lim)
    return (F.silu(gate) * up * g) @ W2.T


def sq_err(x, W1, W3, W2, g, lim, Y):
    """sum((expert_f(x) - Y)^2), ROWS at a time: no full-size temporaries for a hot expert."""
    return sum(float((expert_f(x[c:c + ROWS], W1, W3, W2, g[c:c + ROWS], lim) - Y[c:c + ROWS]).square().sum())
               for c in range(0, len(x), ROWS))


def act_hessians(xf, W1, W3, gf, lim):
    """H1 = x^T x, H2 = a^T a (a = gated SwiGLU activation), accumulated ROWS at a time."""
    H1 = torch.zeros((xf.shape[1], xf.shape[1]), dtype=torch.float32, device=xf.device)
    H2 = torch.zeros((W1.shape[0], W1.shape[0]), dtype=torch.float32, device=xf.device)
    for c in range(0, len(xf), ROWS):
        x, g = xf[c:c + ROWS].float(), gf[c:c + ROWS].float()
        H1 += x.T @ x
        a = F.silu((x @ W1.T).clamp(max=lim)) * (x @ W3.T).clamp(-lim, lim) * g
        H2 += a.T @ a
    return H1, H2


# V41_SCREEN_RTN=1: screen with RTN only (no per-tier GPTQ - that was ~88 % of a calibration layer).  Measured on the
# 1017 stats (2026-10-19, 15,096 experts): spearman(RTN err, GPTQ err) 0.992 / 0.995 / 0.996 per tier, and the map
# picked from RTN errors scaled by these median GPTQ/RTN ratios had -0.77 % TRUE GPTQ cost vs the GPTQ-picked map
# (88.6 % same tier, same bytes).  e_/egu_/edn_ then hold RTN x ratio (GPTQ-equivalent); rtn_ stays the raw RTN.
SCREEN_RTN = os.environ.get("V41_SCREEN_RTN") == "1"
RTN2GPTQ = {"ptq1": 0.657, "q2_0": 0.728, "q3_k": 0.742}


def screen_expert(exp, xe, ge, fit, lim, min_fit, damps=(0.01,), layer=None, expert=None):
    """GPTQ + RTN at every tier on the fit tokens, output error on the held-out tokens.  Returns a stats dict."""
    W1 = K.dequant_fp4_weight(exp.w1.weight, exp.w1.scale)
    W3 = K.dequant_fp4_weight(exp.w3.weight, exp.w3.scale)
    W2 = K.dequant_fp4_weight(exp.w2.weight, exp.w2.scale)
    xf, gf = xe[fit], ge[fit]          # bf16; expert_f / act_hessians convert per chunk
    xv, gv = xe[~fit], ge[~fit]
    st = {"n_fit": int(fit.sum()), "n_eval": int((~fit).sum())}
    if st["n_eval"] == 0:              # nothing held out: measure on the fit tokens (flagged)
        xv, gv, st["eval_on_fit"] = xf, gf, True
    do_gptq = st["n_fit"] >= min_fit
    if do_gptq:
        H1, H2 = act_hessians(xf, W1, W3, gf, lim)
    else:                               # too few tokens for a Hessian: weight-space importance only
        H1 = torch.eye(D, device=xe.device); H2 = torch.eye(W2.shape[1], device=xe.device)
        st["lowdata"] = True
    if not len(xv):
        return st
    Y = expert_f(xv, W1, W3, W2, gv, lim)
    st["y_energy"] = float(Y.square().sum())
    err = lambda q1, q3, q2: sq_err(xv, q1, q3, q2, gv, lim, Y)
    if SCREEN_RTN:
        _, Qr1 = G.quantize_stack([W1, W3], H1, TIERS, damp=damps[0], do_gptq=False)
        _, Qr2 = G.quantize_stack([W2], H2, TIERS, damp=damps[0], do_gptq=False)
        st["screen"] = "rtn"
        for t in TIERS:
            f = RTN2GPTQ[t]
            st[f"rtn_{t}"] = e = err(Qr1[t][0], Qr1[t][1], Qr2[t][0])
            st[f"e_{t}"] = f * e
            st[f"egu_{t}"] = f * err(Qr1[t][0], Qr1[t][1], W2)
            st[f"edn_{t}"] = f * err(W1, W3, Qr2[t][0])
        return st
    for di, damp in enumerate(damps):   # several damps (sweep): keys after the first get an "@damp" suffix
        sfx = "" if di == 0 else f"@{damp:g}"
        Qg1, Qr1 = G.quantize_stack([W1, W3], H1, TIERS, damp=damp, do_gptq=do_gptq)
        Qg2, Qr2 = G.quantize_stack([W2], H2, TIERS, damp=damp, do_gptq=do_gptq)
        Qg1, Qg2 = (Qg1, Qg2) if do_gptq else (Qr1, Qr2)
        for t in TIERS:
            st[f"e_{t}{sfx}"] = err(Qg1[t][0], Qg1[t][1], Qg2[t][0])            # whole expert at tier t (GPTQ)
            st[f"egu_{t}{sfx}"] = err(Qg1[t][0], Qg1[t][1], W2)                 # gate+up only
            st[f"edn_{t}{sfx}"] = err(W1, W3, Qg2[t][0])                        # down only
            if di == 0:
                st[f"rtn_{t}"] = err(Qr1[t][0], Qr1[t][1], Qr2[t][0])          # RTN control
    return st


def cmd_layer(a):
    L = a.layer
    meta = json.load(open(wpath("meta.json")))
    tokens = np.load(wpath("tokens.npy"))
    N, S = tokens.shape
    B = a.batch or next(b for b in range(max(1, 4096 // S), 0, -1) if N % b == 0)   # ~4K tokens per attention batch (2 x 2048 or 1 x 8192 fit a 24 GB card), dividing N
    assert N % B == 0
    dev = a.device
    torch.set_default_dtype(BF16)
    args = model_args(B, S)
    layout = M.EngramLayout.from_args(args)
    t0 = time.time()
    if L in args.engram_layer_ids:
        e = args.engram_layer_ids.index(L)
        MemmapEngramEmbedding.SRC = (shard_file(weight_map()[f"layers.{L}.engram.embed.weight"]),
                                     f"layers.{L}.engram.embed.weight", f"layers.{L}.engram.embed.scale")
        M.ParallelEngramEmbedding = MemmapEngramEmbedding
        hashes = np.load(wpath("hashes.npy"), mmap_mode="r")
    with torch.device(dev):
        blk = M.Block(L, args, layout)
    n = load_block(blk, L)
    if getattr(a, "block_hook", None) is not None:   # v41_e2e: e.g. fake-quantize the dense weights
        a.block_hook(L, blk)
    torch.set_default_device(dev)
    log(f"layer {L}: {n} tensors loaded ({time.time() - t0:.0f} s), ratio {args.compress_ratios[L]}, "
        f"engram {blk.engram is not None}, kv_src {blk.attn.is_kv_source}, idx_src {blk.attn.is_index_source}")

    h_in = h_memmap(L, "r", N, S)
    pre_in = np.load(wpath("pre", f"L{L:02d}.npy"))
    sp = wpath("shared", f"L{L:02d}.pt")
    shared = torch.load(sp) if os.path.exists(sp) else [dict.fromkeys(SHARED) for _ in range(N // B)]
    resid = torch.empty((N, S, HC, D), dtype=BF16, device="cpu")
    X = torch.empty((N * S, D), dtype=BF16, device=dev)
    post = torch.empty((N, S, HC), dtype=torch.float32, device=dev)
    comb = torch.empty((N, S, HC, HC), dtype=torch.float32, device=dev)
    nxt = torch.empty((N, S, HC), dtype=torch.float32, device=dev)
    t1 = time.time()
    with torch.inference_mode():
        for bi, i0 in enumerate(range(0, N, B)):
            i1 = i0 + B
            for k in SHARED:
                v = shared[bi][k]
                setattr(M.shared_attn, k, None if v is None else v.to(dev))
            x = torch.from_numpy(np.asarray(h_in[i0:i1])).to(dev).view(BF16)
            pm = torch.from_numpy(pre_in[i0:i1]).to(dev)
            if blk.engram is not None:
                x = blk.engram(x, torch.from_numpy(np.asarray(hashes[i0:i1, :, e, :])).to(dev), None)
            residual = x
            attn_pre, attn_post, attn_comb = blk.hc_mixes(x, blk.hc_attn_fn, blk.hc_attn_scale, blk.hc_attn_base)
            y = blk.attn(blk.attn_norm(blk.hc_pre(x, pm)), 0)
            x = blk.hc_post(y, residual, attn_post, attn_comb)
            ffn_pre, post[i0:i1], comb[i0:i1] = blk.hc_mixes(x, blk.hc_ffn_fn, blk.hc_ffn_scale, blk.hc_ffn_base)
            nxt[i0:i1] = ffn_pre
            X[i0 * S:i1 * S] = blk.ffn_norm(blk.hc_pre(x, attn_pre)).view(-1, D)
            resid[i0:i1] = x.cpu()
            shared[bi] = {k: (None if getattr(M.shared_attn, k) is None else getattr(M.shared_attn, k).clone().cpu())
                          for k in SHARED}
        assert torch.isfinite(X).all(), "non-finite MoE input"
        log(f"layer {L}: attention half {N}x{S} in {time.time() - t1:.0f} s"
            + (f", engram rows read {blk.engram.embed.rows:,}" if blk.engram is not None else ""))

        # ---- MoE, expert-major over every token (reference Gate / Expert / shared expert)
        t2 = time.time()
        T = N * S
        wts = torch.empty((T, TOPK), dtype=torch.float32, device=dev)
        idx = torch.empty((T, TOPK), dtype=torch.int64, device=dev)
        for c in range(0, T, 65536):
            wts[c:c + 65536], idx[c:c + 65536] = blk.ffn.gate(X[c:c + 65536])
        is_eval_tok = torch.tensor(meta["is_eval"], device=dev).repeat_interleave(S)
        if getattr(a, "want_ctx", False):   # v41_repair: the hooks see the layer's MoE input and routing
            a.moe_ctx = (X, wts, idx)
        Y = torch.zeros((T, D), dtype=torch.float32, device=dev)
        stats, lim = {}, args.swiglu_limit
        hook = getattr(a, "expert_hook", None)
        for ex in range(NE):
            tok, slot = torch.where(idx == ex)
            if len(tok) == 0:
                stats[ex] = {"n": 0}
                continue
            xe, ge = X[tok], wts[tok, slot, None]
            # chunked: a hot expert (88K tokens at layer 6) OOMs the 24 GB card in one call; rows are independent
            out = torch.cat([blk.ffn.experts[ex](xe[c:c + ROWS], ge[c:c + ROWS]) for c in range(0, len(tok), ROWS)])
            st = {"n": len(tok), "gsum": float(ge.sum()), "reap": float(out.float().norm(dim=-1).mean())}
            if hook is not None:        # v41_quant: quantize this expert; its output replaces the teacher's
                out, extra = hook(L, ex, blk.ffn.experts[ex], xe, ge, ~is_eval_tok[tok], lim, out)
                st.update(extra)
            Y.index_add_(0, tok, out.float())
            if not a.no_gptq and ex < a.gptq_experts and ex % a.expert_stride == 0:
                st.update(screen_expert(blk.ffn.experts[ex], xe, ge, ~is_eval_tok[tok], lim, a.min_fit, a.damps))
            stats[ex] = st
            if ex % 64 == 63:
                log(f"layer {L}: experts {ex + 1}/{NE} ({time.time() - t2:.0f} s)")
        routed_energy = float(Y.square().sum())
        if getattr(a, "moe_done_hook", None) is not None:   # v41_quant: experts no token reached, write the layer
            a.moe_done_hook(L, blk, stats)
        for c in range(0, T, 65536):
            Y[c:c + 65536] += blk.ffn.shared_experts(X[c:c + 65536])
        log(f"layer {L}: MoE{'' if a.no_gptq else ' + GPTQ screening'} in {time.time() - t2:.0f} s")

        if a.screen_only:               # sweep run: stats only, the h chain is left alone
            json.dump({"summary": {"layer": L, "damps": a.damps, "expert_stride": a.expert_stride,
                                   "seconds": round(time.time() - t0, 1)}, "experts": stats},
                      open(wpath("stats", f"L{L:02d}_{a.screen_only}.json"), "w"))
            log(f"layer {L}: screen-only '{a.screen_only}' done in {time.time() - t0:.0f} s")
            return
        # ---- finish the block, write layer L+1's input
        h_out = h_memmap(L + 1, "w+", N, S)
        for i0 in range(0, N, B):
            i1 = i0 + B
            y = Y[i0 * S:i1 * S].view(B, S, D).to(BF16)
            xo = blk.hc_post(y, resid[i0:i1].to(dev), post[i0:i1], comb[i0:i1])
            assert torch.isfinite(xo).all(), f"non-finite output, batch {i0}"
            h_out[i0:i1] = xo.view(torch.int16).cpu().numpy()
        h_out.flush()
        np.save(wpath("pre", f"L{L + 1:02d}.npy"), nxt.cpu().numpy())
        os.makedirs(wpath("shared"), exist_ok=True)
        torch.save(shared, wpath("shared", f"L{L + 1:02d}.pt"))
        os.makedirs(wpath("route"), exist_ok=True)
        np.savez(wpath("route", f"L{L:02d}.npz"), idx=idx.to(torch.int16).cpu().numpy(), w=wts.half().cpu().numpy())
        counts = torch.bincount(idx.flatten(), minlength=NE).float()
        summ = {"layer": L, "ablate": ABL.manifest_id() or "none",
                "tokens": T, "n_eval_tok": int(is_eval_tok.sum()), "routed_energy": routed_energy,
                "h_rms_out": float(xo.float().square().mean().sqrt()),
                "load_cv": float(counts.std() / counts.mean()), "unused_experts": int((counts == 0).sum()),
                "seconds": round(time.time() - t0, 1), "tiers": TIERS, "min_fit": a.min_fit, "damps": a.damps}
        os.makedirs(wpath("stats"), exist_ok=True)
        json.dump({"summary": summ, "experts": stats}, open(wpath("stats", f"L{L:02d}.json"), "w"))
    if a.gc and L >= 1:
        for p in (wpath("h", f"L{L - 1:02d}.bin"), wpath("shared", f"L{L - 1:02d}.pt")):   # shared: 3 GB/layer from L21
            if os.path.exists(p):
                os.remove(p)
    log(f"layer {L}: done in {time.time() - t0:.0f} s; load CV {summ['load_cv']:.3f}, unused {summ['unused_experts']}, "
        f"out rms {summ['h_rms_out']:.3f}")


def cmd_head(a):
    """End-of-chain gate: last layer's hc_pre (pre/L.npy mix) -> final norm -> head on h/L (default 40 = after the
    last block), next-token NLL / top-1 accuracy per domain and fit/held-out.  A broken layer anywhere in the chain
    shows up here as a ppl far from the model's (Transformer.forward: h = hc_pre(h, pre_mix); head(norm(h)))."""
    from safetensors import safe_open
    meta = json.load(open(wpath("meta.json")))
    tokens = np.load(wpath("tokens.npy"))
    N, S = tokens.shape
    L, dev = a.layer, a.device
    torch.set_default_dtype(BF16)
    args = model_args(1, S)
    wm = weight_map()
    with torch.device(dev):
        norm = M.RMSNorm(D, args.norm_eps)
    with safe_open(shard_file(wm["norm.weight"]), "pt") as f:
        norm.weight.data.copy_(f.get_tensor("norm.weight"))
    with safe_open(shard_file(wm["head.weight"]), "pt") as f:
        W = f.get_tensor("head.weight").to(dev, torch.float32)       # ParallelHead keeps fp32
    h_in = h_memmap(L, "r", N, S)
    pre = np.load(wpath("pre", f"L{L:02d}.npy"))
    res = {}
    am = np.zeros((N, S - 1), np.int32) if a.save_argmax else None   # argmax at position p = the model's guess for p+1
    t0 = time.time()
    with torch.inference_mode():
        for i in range(N):
            x = torch.from_numpy(np.asarray(h_in[i:i + 1])).to(dev).view(BF16)
            pm = torch.from_numpy(pre[i:i + 1]).to(dev)
            y = norm(torch.sum(pm.unsqueeze(-1) * x.float(), dim=2).to(BF16))[0]     # [S, D]
            tgt = torch.from_numpy(tokens[i, 1:].astype(np.int64)).to(dev)
            nll, hit = 0.0, 0
            for c in range(0, S - 1, 1024):
                lg = y[c:min(c + 1024, S - 1)].float() @ W.T
                t = tgt[c:c + lg.shape[0]]
                nll += float(F.cross_entropy(lg, t, reduction="sum"))
                hit += int((lg.argmax(-1) == t).sum())
                if am is not None:
                    am[i, c:c + lg.shape[0]] = lg.argmax(-1).cpu().numpy()
            key = (meta["domains"][i], "eval" if meta["is_eval"][i] else "fit")
            r = res.setdefault(key, [0.0, 0, 0])
            r[0] += nll; r[1] += hit; r[2] += S - 1
    out = {f"{d}/{k}": {"ppl": float(np.exp(v[0] / v[2])), "top1": v[1] / v[2], "tokens": v[2]} for (d, k), v in sorted(res.items())}
    for k in ("fit", "eval"):
        tot = [sum(v[j] for (d, kk), v in res.items() if kk == k) for j in range(3)]
        if tot[2]:
            out[f"ALL/{k}"] = {"ppl": float(np.exp(tot[0] / tot[2])), "top1": tot[1] / tot[2], "tokens": tot[2]}
    os.makedirs(wpath("stats"), exist_ok=True)
    json.dump({"layer": L, "seconds": round(time.time() - t0, 1), "by_domain": out},
              open(wpath("stats", f"head_L{L:02d}.json"), "w"), indent=1)
    if am is not None:
        np.save(wpath("stats", f"head_argmax_L{L:02d}.npy"), am)
    for k, v in out.items():
        log(f"head L{L}: {k:22s} ppl {v['ppl']:9.3f}  top1 {v['top1']:.3f}  ({v['tokens']} tokens)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep"); p.add_argument("--n", type=int, default=128); p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--corpus", default=None, help="build_corpus.py output dir (agentic-first V4.1 mix)")
    p.add_argument("--mix", default=None, help="override domain weights, e.g. agentic_long=7,chat=4,multi=2 (default = the V4.1 mix)")
    sub.add_parser("init")
    p = sub.add_parser("prep-ids"); p.add_argument("ids", nargs="+"); p.add_argument("--domain", default="custom")
    p = sub.add_parser("head"); p.add_argument("--layer", type=int, default=40, help="h/L to read (40 = after the last block)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--save-argmax", action="store_true", help="also write stats/head_argmax_LNN.npy (int32 [N, S-1])")
    p = sub.add_parser("layer"); p.add_argument("layer", type=int); p.add_argument("--batch", type=int, default=0, help="sequences per attention batch (0: ~4K tokens)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-gptq", action="store_true"); p.add_argument("--gptq-experts", type=int, default=NE); p.add_argument("--min-fit", type=int, default=32)
    p.add_argument("--gc", action="store_true", help="delete h/L-1 after success")
    p.add_argument("--damp", dest="damps", type=lambda v: [float(x) for x in v.split(",")], default=[1.0],
                   help="GPTQ damping (x mean diag H); a comma list sweeps, extra ones stored as e_<tier>@<damp>")
    p.add_argument("--expert-stride", type=int, default=1, help="screen every K-th expert only")
    p.add_argument("--screen-only", default=None, metavar="TAG",
                   help="write stats/LNN_TAG.json only; no h/pre/shared/route output, no --gc")
    a = ap.parse_args()
    {"prep": cmd_prep, "prep-ids": cmd_prep_ids, "init": cmd_init, "layer": cmd_layer, "head": cmd_head}[a.cmd](a)


if __name__ == "__main__":
    main()
