"""v41_from_llamacpp.py - a llama.cpp `deepseek41` GGUF (e.g. Unsloth's DeepSeek-V4.1-Flash-1017 UD-* quants, split
or not) -> the single-file GGUF the ds41 engine reads: v41_gguf_meta.py's keys and ds41_check_tensors' tensor map.

What changes on the way (everything else is copied byte for byte, the routed experts included):
  - token_embd Q8_0, engram_kv Q6_K -> F16 (dequantized with ggml's arithmetic, then rounded to F16)
  - hc_{attn,ffn}_fn F32, the compressor / indexer BF16 tensors -> F16 (the types ds41_check_tensors wants)
  - engram_gate (llama.cpp folds q_weight * k_weight into it) -> engram_q_norm = engram_gate, engram_k_norm = 1:
    the engine only ever uses their product (ds41_dense.cpp wqk), so the gate is exact
  - per_layer_token_embd (the Engram tables, Q6_K) is NOT copied: deepseek41.engram.encoding = "q6_k_gguf" makes the
    engine read its rows from the llama.cpp file itself (give that file or its folder to --engram)
The Engram hash layout (primes, multipliers) is recomputed here as engram.py does and must equal the file's; the
config keys come from the checkpoint's config.json (--hf-config) and must equal the file's own keys.

  python -I v41_from_llamacpp.py SRC-00001-of-0000N.gguf --hf-config config.json --out OUT.gguf [--tokenizer-dir DIR]
         [--dry-run]
  --tokenizer-dir writes vocab.json / merges.txt / token_type.json / chat_template.jinja for serve_ds41.py, from the
  GGUF's own tokenizer tables (what export_tokenizer.py writes from the HF tokenizer.json).
"""
import argparse
import functools
import json
import os
import queue
import re
import struct
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v41_gguf_meta as MD  # noqa: E402

ALIGN = MD.GGUF_ALIGNMENT
F32, F16, Q8_0, Q6_K, BF16, I32 = 0, 1, 8, 14, 30, 26
# bytes per block, elements per block
BLOCK = {0: (4, 1), 1: (2, 1), 30: (2, 1), 8: (34, 32), 14: (210, 256), 39: (17, 32), 10: (84, 256), 11: (110, 256),
         12: (144, 256), 13: (176, 256), 16: (66, 256), 17: (74, 256), 18: (98, 256), 21: (110, 256), 22: (82, 256),
         23: (136, 256), 26: (4, 1), 43: (33, 32)}
TYPE_NAME = {0: "F32", 1: "F16", 30: "BF16", 8: "Q8_0", 14: "Q6_K", 39: "MXFP4", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
             13: "Q5_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 26: "I32",
             43: "MXFP8"}
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


# ------------------------------------------------------------------ GGUF header I/O (keeps every value's type)
class Header:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            magic, ver, nt, nkv = struct.unpack("<4sIQQ", f.read(24))
            if magic != b"GGUF" or ver != 3:
                raise SystemExit(f"{path}: not a GGUF v3 file")
            self.kv = {}                      # key -> (type, value); strings stay bytes, arrays are (elem type, list)
            for _ in range(nkv):
                k = self._str(f).decode("utf-8")
                t = self._u32(f)
                self.kv[k] = (t, self._val(f, t))
            self.tensors = []                 # (name, dims, type, offset)
            for _ in range(nt):
                name = self._str(f).decode("utf-8")
                nd = self._u32(f)
                dims = list(struct.unpack(f"<{nd}Q", f.read(8 * nd)))
                t, off = struct.unpack("<IQ", f.read(12))
                self.tensors.append((name, dims, t, off))
            end = f.tell()
        align = self.kv.get("general.alignment", (4, 32))[1]
        self.data_start = (end + align - 1) // align * align
        self.size = os.path.getsize(path)

    @staticmethod
    def _u32(f):
        return struct.unpack("<I", f.read(4))[0]

    @staticmethod
    def _str(f):
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n)

    def _val(self, f, t):
        if t == 8:
            return self._str(f)
        if t == 9:
            et, n = struct.unpack("<IQ", f.read(12))
            if et == 8:
                return et, [self._str(f) for _ in range(n)]
            fmt = SCALAR[et]
            return et, list(struct.unpack(f"<{n}{fmt}", f.read(struct.calcsize(fmt) * n)))
        fmt = SCALAR[t]
        return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]

    def get(self, k, default=None):
        v = self.kv.get(k)
        if v is None:
            return default
        t, x = v
        if t == 9:
            return [s.decode("utf-8") if isinstance(s, bytes) else s for s in x[1]]
        return x.decode("utf-8") if t == 8 else x


def kv_record(key, t, v):
    out = MD.pack_string(key) + struct.pack("<I", t)
    if t == 8:
        return out + MD.pack_string(v)
    if t == 9:
        et, items = v
        out += struct.pack("<IQ", et, len(items))
        if et == 8:
            return out + b"".join(MD.pack_string(s) for s in items)
        return out + struct.pack(f"<{len(items)}{SCALAR[et]}", *items)
    return out + struct.pack("<" + SCALAR[t], v)


def nbytes(t, dims):
    b, e = BLOCK[t]
    n = 1
    for d in dims:
        n *= d
    if dims[0] % e:
        raise ValueError(f"row of {dims[0]} is not whole {TYPE_NAME.get(t, t)} blocks")
    return n // e * b


def shard_paths(any_shard):
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", any_shard)
    if not m:
        return [any_shard]
    n = int(m.group(3))
    return [f"{m.group(1)}-{i:05d}-of-{n:05d}.gguf" for i in range(1, n + 1)]


# ------------------------------------------------------------------ dequantizers (ggml's arithmetic, float32)
def deq_q8_0(raw):
    b = np.frombuffer(raw, np.uint8).reshape(-1, 34)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    return (d * b[:, 2:].view(np.int8).astype(np.float32)).reshape(-1)   # ggml: qs[j] * d


def deq_q6_k(raw):
    b = np.frombuffer(raw, np.uint8).reshape(-1, 210)
    nb = b.shape[0]
    ql, qh = b[:, :128].astype(np.int16), b[:, 128:192].astype(np.int16)
    sc = b[:, 192:208].view(np.int8).astype(np.float32)
    d = b[:, 208:210].copy().view(np.float16).astype(np.float32)
    y = np.empty((nb, 256), np.float32)
    is_ = np.arange(32) // 16
    for n in range(2):                                   # ggml: two 128-value halves
        L, H, S = ql[:, 64 * n:64 * n + 64], qh[:, 32 * n:32 * n + 32], sc[:, 8 * n:8 * n + 8]
        q1 = ((L[:, :32] & 0xF) | (((H >> 0) & 3) << 4)) - 32
        q2 = ((L[:, 32:] & 0xF) | (((H >> 2) & 3) << 4)) - 32
        q3 = ((L[:, :32] >> 4) | (((H >> 4) & 3) << 4)) - 32
        q4 = ((L[:, 32:] >> 4) | (((H >> 6) & 3) << 4)) - 32
        for k, q in enumerate((q1, q2, q3, q4)):         # y = d * sc[is + 2k] * q, left to right as in C
            y[:, 128 * n + 32 * k:128 * n + 32 * k + 32] = d * S[:, is_ + 2 * k] * q.astype(np.float32)
    return y.reshape(-1)


def to_f16(v, name):
    h = v.astype(np.float16)
    if not np.isfinite(h).all():
        raise SystemExit(f"{name}: values outside the F16 range")
    return h


def bf16_f32(raw):
    return (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)


def q8_0(v):
    """floats -> Q8_0 blocks, ggml's quantize_row_q8_0_ref: d = amax / 127 and id = 1 / d in float32, d stored as F16,
    q = roundf(x * id) (half away from zero; decided in float64, where |x * id| + 0.5 is exact)"""
    x = v.astype(np.float32).reshape(-1, 32)
    d = np.abs(x).max(1) / np.float32(127)
    idv = np.zeros_like(d)
    np.divide(np.float32(1), d, out=idv, where=d != 0)
    x0 = (x * idv[:, None]).astype(np.float64)
    out = np.empty((x.shape[0], 34), np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = (np.sign(x0) * np.floor(np.abs(x0) + 0.5)).astype(np.int8).view(np.uint8)
    return out.tobytes()


def convert(kind, raw, name):
    if kind == "bf16>q8_0":   # Unsloth's UD-Q8_K_XL keeps the dense matrices in BF16
        return q8_0(bf16_f32(raw))
    if kind == "q8_0>f16":
        return to_f16(deq_q8_0(raw), name).tobytes()
    if kind == "q6_k>f16":
        return to_f16(deq_q6_k(raw), name).tobytes()
    if kind == "f32>f16":
        return to_f16(np.frombuffer(raw, np.float32), name).tobytes()
    if kind == "bf16>f16":
        return to_f16((np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32), name).tobytes()
    raise ValueError(kind)


# ------------------------------------------------------------------ Engram layout, as engram.py builds it
@functools.lru_cache(maxsize=None)   # every n-gram order restarts its search at engram_vocab_size
def is_prime(n):
    if n < 2 or n % 2 == 0:
        return n == 2
    r = int(n ** 0.5)
    for p in range(3, r + 1, 2):
        if n % p == 0:
            return False
    return True


def engram_layout(text, cvocab):
    seen, primes, mults = set(), [], []
    bound = (np.iinfo(np.int64).max // cvocab) // 2
    for layer in text["engram_layer_ids"]:
        for _ in range(text["engram_max_ngram_size"] - 1):
            cur = text["engram_vocab_size"] - 1
            for _ in range(text["engram_n_heads"]):
                cur += 1
                while not is_prime(cur) or cur in seen:
                    cur += 1
                seen.add(cur)
                primes.append(cur)
        rng = np.random.default_rng(10007 * layer)
        mults += (rng.integers(0, bound, size=text["engram_max_ngram_size"], dtype=np.int64) * 2 + 1).tolist()
    return primes, mults


# ------------------------------------------------------------------ the plan
def check_keys(src, text, cfg):
    """The file's llama.cpp keys against config.json: a converted file must describe the same model."""
    A = "deepseek41."
    eq = [
        ("block_count", text["num_hidden_layers"]), ("embedding_length", text["hidden_size"]),
        ("attention.head_count", text["num_attention_heads"]), ("attention.key_length", text["head_dim"]),
        ("rope.dimension_count", text["qk_rope_head_dim"]), ("attention.q_lora_rank", text["q_lora_rank"]),
        ("attention.output_lora_rank", text["o_lora_rank"]), ("attention.output_group_count", text["o_groups"]),
        ("attention.sliding_window", text["sliding_window"]), ("expert_count", text["n_routed_experts"]),
        ("expert_used_count", text["num_experts_per_tok"]), ("expert_feed_forward_length", text["moe_intermediate_size"]),
        ("expert_shared_count", text["n_shared_experts"]), ("hyper_connection.count", text["hc_mult"]),
        ("hyper_connection.sinkhorn_iterations", text["hc_sinkhorn_iters"]),
        ("attention.indexer.head_count", text["index_n_heads"]), ("attention.indexer.key_length", text["index_head_dim"]),
        ("attention.indexer.top_k", text["index_topk"]),
        ("attention.indexer.candidate_source_layer", text["candidate_source_layer_id"]),
        ("attention.indexer.candidate_top_k_blocks", text["candidate_topk_blocks"]),
        ("attention.indexer.candidate_block_size", text["candidate_block_size"]),
        ("attention.compress_ratios", text["compress_ratios"]),
        ("attention.kv_source_layers", text["kv_source_layer_ids"]),
        ("attention.indexer.source_layers", text["index_source_layer_ids"]),
        ("context_length", text["max_position_embeddings"]), ("rope.freq_base", text["rope_theta"]),
        ("attention.compress_rope_freq_base", text["compress_rope_theta"]),
        ("rope.scaling.factor", text["rope_scaling"]["factor"]),
        ("rope.scaling.original_context_length", text["rope_scaling"]["original_max_position_embeddings"]),
        ("rope.scaling.yarn_beta_fast", text["rope_scaling"]["beta_fast"]),
        ("rope.scaling.yarn_beta_slow", text["rope_scaling"]["beta_slow"]),
        ("expert_weights_scale", text["routed_scaling_factor"]), ("expert_weights_norm", text["norm_topk_prob"]),
        ("ple.layers", text["engram_layer_ids"]), ("ple.ngram_size", text["engram_max_ngram_size"]),
        ("ple.heads_per_ngram", text["engram_n_heads"]), ("embedding_length_per_layer_input", text["engram_head_dim"]),
        ("engram.vocab_size", text["engram_vocab_size"]), ("engram.pad_token_id", text["engram_pad_token_id"]),
        ("engram.compressed_vocab_size", text["engram_compressed_vocab_size"]),
    ]
    bad = []
    for k, want in eq:
        got = src.get(A + k)
        if isinstance(want, float) or isinstance(got, float):
            ok = got is not None and abs(float(got) - float(want)) <= 1e-6 * max(1.0, abs(float(want)))
        else:
            ok = got == want
        if not ok:
            bad.append(f"{A}{k} = {got}, config.json says {want}")
    for k, want in (("attention.layer_norm_rms_epsilon", text["rms_norm_eps"]), ("hyper_connection.epsilon", text["hc_eps"])):
        got = src.get(A + k)
        if got is None or abs(got - want) > 1e-6 * want:          # f32 round-off of 1e-20 / 1e-6
            bad.append(f"{A}{k} = {got}, config.json says {want}")
    for k in ("swiglu_clamp_exp", "swiglu_clamp_shexp"):
        if set(src.get(A + k) or [None]) != {float(text["swiglu_limit"])}:
            bad.append(f"{A}{k} is not swiglu_limit {text['swiglu_limit']} everywhere")
    if src.get("tokenizer.ggml.bos_token_id") != cfg["bos_token_id"] or src.get("tokenizer.ggml.eos_token_id") != cfg["eos_token_id"]:
        bad.append("bos/eos ids differ from config.json")
    if src.get("general.architecture") != "deepseek41":
        bad.append("general.architecture is not deepseek41")
    if src.get(A + "expert_gating_func") != 4:   # LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS
        bad.append("expert_gating_func is not sqrtsoftplus (4)")
    if bad:
        raise SystemExit("the GGUF does not match config.json:\n  " + "\n  ".join(bad))


def plan_tensors(src_t, text):
    """[(name, dims, type, source tensor or None, transform)] in the output order, after the ds41_check_tensors rules."""
    D, HC, NL = text["hidden_size"], text["hc_mult"], text["num_hidden_layers"]
    MIX, NH, DH = (2 + HC) * HC, text["num_attention_heads"], text["head_dim"]
    RQ, OG, OL = text["q_lora_rank"], text["o_groups"], text["o_lora_rank"]
    NE, FF, V = text["n_routed_experts"], text["moe_intermediate_size"], text["vocab_size"]
    IH, ID = text["index_n_heads"], text["index_head_dim"]
    NG, EH, ED = text["engram_max_ngram_size"], text["engram_n_heads"], text["engram_head_dim"]
    ratio, kvs, ixs = text["compress_ratios"], text["kv_source_layer_ids"], text["index_source_layer_ids"]
    out, used = [], set()

    def take(name, dims, want, src_name=None, kind=None):
        """want: the output type; the source type decides the transform (copy if equal)"""
        s = src_t.get(src_name or name)
        if s is None:
            raise SystemExit(f"missing source tensor {src_name or name}")
        _, sdims, st, _, _ = s
        if sdims != dims:
            raise SystemExit(f"{src_name or name}: shape {sdims}, expected {dims}")
        if kind is None:
            kind = "copy" if st == want else {(Q8_0, F16): "q8_0>f16", (Q6_K, F16): "q6_k>f16", (F32, F16): "f32>f16",
                                              (BF16, F16): "bf16>f16", (BF16, Q8_0): "bf16>q8_0"}.get((st, want))
            if kind is None:
                raise SystemExit(f"{src_name or name}: {TYPE_NAME.get(st, st)} cannot become {TYPE_NAME[want]}")
        used.add(src_name or name)
        out.append((name, dims, want, s, kind))

    take("token_embd.weight", [D, V], F16)
    take("output_norm.weight", [D], F32)
    take("output.weight", [D, V], Q8_0)
    for L in range(NL):
        b = f"blk.{L}."
        for site in ("attn", "ffn"):
            take(b + f"hc_{site}_fn.weight", [HC * D, MIX], F16)
            take(b + f"hc_{site}_base.weight", [MIX], F32)
            take(b + f"hc_{site}_scale.weight", [3], F32)
            take(b + f"{site}_norm.weight", [D], F32)
        take(b + "attn_sinks.weight", [NH], F32)
        take(b + "attn_q_a.weight", [D, RQ], Q8_0)
        take(b + "attn_q_b.weight", [RQ, NH * DH], Q8_0)
        take(b + "attn_q_a_norm.weight", [RQ], F32)
        take(b + "attn_kv.weight", [D, DH], Q8_0)
        take(b + "attn_kv_a_norm.weight", [DH], F32)
        take(b + "attn_output_a.weight", [NH * DH // OG, OG * OL], Q8_0)
        take(b + "attn_output_b.weight", [OG * OL, D], Q8_0)
        if L in kvs:
            take(b + "attn_compressor_kv.weight", [D, DH], F16)
            take(b + "attn_compressor_norm.weight", [DH], F32)
            take(b + "indexer.attn_k.weight", [DH, ID], F16)
            take(b + "indexer.k_norm.weight", [ID], F32)
            if ratio[L] > 1:
                take(b + "attn_compressor_gate.weight", [D, DH], F16)
        if L in ixs:
            take(b + "indexer.attn_q_b.weight", [RQ, IH * ID], F16)
            take(b + "indexer.proj.weight", [D, IH], F16)
        take(b + "ffn_gate_inp.weight", [D, NE], F32)
        take(b + "exp_probs_b.bias", [NE], F32)
        take(b + "exp_probs_b_vl.bias", [NE], F32)
        take(b + "ffn_gate_shexp.weight", [D, FF], Q8_0)
        take(b + "ffn_up_shexp.weight", [D, FF], Q8_0)
        take(b + "ffn_down_shexp.weight", [FF, D], Q8_0)
        if L in text["engram_layer_ids"]:
            take(b + "engram_q_norm.weight", [D, HC], F32, b + "engram_gate.weight")
            out.append((b + "engram_k_norm.weight", [D, HC], F32, None, "ones"))
            take(b + "engram_kv.weight", [(NG - 1) * EH * ED, (HC + 1) * D], F16)
        g, u, d = (src_t.get(b + f"ffn_{r}_exps.weight") for r in ("gate", "up", "down"))
        if not (g and u and d) or g[2] != u[2]:
            raise SystemExit(f"layer {L}: routed experts must be ffn_gate/up/down_exps with gate and up of one type")
        for r, dims, s in (("gate", [D, FF, NE], g), ("up", [D, FF, NE], u), ("down", [FF, D, NE], d)):
            take(b + f"ffn_{r}_exps.weight", dims, s[2])
    left = sorted(set(src_t) - used - {"per_layer_token_embd.weight"})
    if left:
        raise SystemExit("tensors the engine has no place for: " + ", ".join(left[:10]))
    if "per_layer_token_embd.weight" not in src_t:
        raise SystemExit("no per_layer_token_embd.weight (the Engram tables) in the source")
    return out


def build(args):
    paths = shard_paths(args.src)
    hdrs = [Header(p) for p in paths]
    src = next((h for h in hdrs if "general.architecture" in h.kv), None)
    if src is None:
        raise SystemExit("no shard carries the model's keys")
    src_t = {}
    for si, h in enumerate(hdrs):
        for name, dims, t, off in h.tensors:
            if t not in BLOCK:
                raise SystemExit(f"{name}: GGML type {t} is not handled here")
            n = nbytes(t, dims)
            if h.data_start + off + n > h.size:
                raise SystemExit(f"{name}: data runs past the end of {h.path}")
            src_t[name] = (si, dims, t, h.data_start + off, n)
    with open(args.hf_config, encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("model_type") != "deepseek_v41":
        raise SystemExit("--hf-config: not a DeepSeek V4.1 config.json")
    text = cfg["text_config"]
    check_keys(src, text, cfg)

    # Engram: the file's layout must be engram.py's (then the engine's own hash reproduces llama.cpp's row ids)
    A = "deepseek41."
    cvocab = text["engram_compressed_vocab_size"]
    primes, mults = engram_layout(text, cvocab)
    sizes, offs = src.get(A + "ple.head_vocab_sizes"), src.get(A + "ple.head_offsets")
    if sizes != primes:
        raise SystemExit("ple.head_vocab_sizes differ from engram.py's primes")
    if src.get(A + "ple.layer_multipliers") != mults:
        raise SystemExit("ple.layer_multipliers differ from engram.py's multipliers")
    if offs != [sum(sizes[:i]) for i in range(len(sizes))]:
        raise SystemExit("ple.head_offsets are not the running sum over every layer's heads")
    cols = (text["engram_max_ngram_size"] - 1) * text["engram_n_heads"]
    rows = [sum(primes[i * cols:(i + 1) * cols]) for i in range(len(text["engram_layer_ids"]))]
    if rows != text["engram_num_embeddings"]:
        raise SystemExit("Engram rows differ from config.json's engram_num_embeddings")
    token_map = src.get(A + "engram.token_map")
    if len(token_map) != text["vocab_size"] or max(token_map) + 1 != cvocab:
        raise SystemExit("engram.token_map does not cover the vocabulary / compressed vocabulary")
    pet = src_t["per_layer_token_embd.weight"]
    if pet[2] not in (Q6_K, 43) or pet[1] != [text["engram_head_dim"], sum(rows)]:   # 43: MXFP8 (UD-Q8_K_XL)
        raise SystemExit(f"per_layer_token_embd.weight is {TYPE_NAME.get(pet[2])} {pet[1]}, the reader takes Q6_K / MXFP8 "
                         f"[{text['engram_head_dim']}, {sum(rows)}]")

    plan = plan_tensors(src_t, text)

    # ---- metadata: v41_gguf_meta.metadata's records, the tokenizer and Engram tables taken from the source file
    name = args.name or os.path.basename(re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", paths[0])).removesuffix(".gguf")
    recs = [
        MD.kv_string("general.architecture", "deepseek41"),
        MD.kv_string("general.name", name),
        MD.kv_string("general.source.url", "llama.cpp GGUF: " + os.path.basename(paths[0])),
        MD.kv_u32("general.alignment", ALIGN),
        MD.kv_string("deepseek41.config", json.dumps(cfg, sort_keys=True, separators=(",", ":"))),
        MD.kv_string("deepseek41.engram.encoding", "q6_k_gguf"),
        MD.kv_u32_array("deepseek41.engram.layer_ids", text["engram_layer_ids"]),
        MD.kv_u32_array("deepseek41.engram.rows", rows),
        MD.kv_u32("deepseek41.engram.compressed_vocab_size", cvocab),
        MD.kv_u32("deepseek41.engram.pad_id", token_map[text["engram_pad_token_id"]]),
        MD.kv_u32_array("deepseek41.engram.token_map", token_map),
        MD.kv_u32_array("deepseek41.engram.primes", primes),
        MD.array_record("deepseek41.engram.multipliers", MD.GGUF_UINT64, mults),
    ]
    tok_keys = [k for k in src.kv if k.startswith("tokenizer.")]
    for k in tok_keys:
        recs.append(kv_record(k, *src.kv[k]))
    for key in ("vocab_size", "hidden_size", "moe_intermediate_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "head_dim", "qk_rope_head_dim", "q_lora_rank", "o_lora_rank", "o_groups",
                "n_routed_experts", "n_shared_experts", "num_experts_per_tok", "max_position_embeddings",
                "sliding_window", "index_n_heads", "index_head_dim", "index_topk", "candidate_source_layer_id",
                "candidate_topk_blocks", "candidate_block_size", "hc_mult", "hc_sinkhorn_iters", "rope_theta",
                "compress_rope_theta"):
        recs.append(MD.kv_u32(f"deepseek41.{key}", text[key]))
    for key in ("rms_norm_eps", "hc_eps", "swiglu_limit", "routed_scaling_factor"):
        recs.append(MD.kv_f32(f"deepseek41.{key}", text[key]))
    for key in ("scoring_func", "hidden_act", "topk_method"):
        recs.append(MD.kv_string(f"deepseek41.{key}", text[key]))
    recs.append(MD.kv_bool("deepseek41.norm_topk_prob", text["norm_topk_prob"]))
    recs.append(MD.kv_u32_array("deepseek41.compress_ratios", text["compress_ratios"][:text["num_hidden_layers"]]))
    for key in ("kv_source_layer_ids", "index_source_layer_ids"):
        recs.append(MD.kv_u32_array(f"deepseek41.{key}", text[key]))
    for key in ("factor", "beta_fast", "beta_slow", "original_max_position_embeddings"):
        recs.append(MD.kv_f32(f"deepseek41.rope_scaling.{key}", text["rope_scaling"][key]))
    tiers = {}
    for n, dims, t, s, kind in plan:
        if "_exps." in n:
            tiers[TYPE_NAME[t]] = tiers.get(TYPE_NAME[t], 0) + dims[2]
    recs.append(MD.kv_string("strata.v41.expert_source", "llama.cpp GGUF: " + os.path.basename(paths[0])))
    recs.append(MD.kv_string("strata.v41.expert_tiers", json.dumps(dict(sorted(tiers.items())))))
    recs.append(MD.kv_string("strata.v41.expert_layout", "uniform role per layer: ffn_<role>_exps.weight"))

    # ---- tensor infos and offsets
    infos, off, total = [], 0, 0
    for n, dims, t, s, kind in plan:
        nb = nbytes(t, dims)
        infos.append(MD.pack_string(n) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) +
                     struct.pack("<IQ", t, off))
        off = (off + nb + ALIGN - 1) // ALIGN * ALIGN
        total += nb
    head = struct.pack("<4sIQQ", b"GGUF", 3, len(plan), len(recs)) + b"".join(recs) + b"".join(infos)
    data_start = (len(head) + ALIGN - 1) // ALIGN * ALIGN
    return paths, hdrs, src, plan, head, data_start, off, total, tiers


def write_tokenizer(src, out_dir):
    tokens = src.get("tokenizer.ggml.tokens")
    types = src.get("tokenizer.ggml.token_type")
    merges = src.get("tokenizer.ggml.merges")
    os.makedirs(out_dir, exist_ok=True)
    # newline="\n": the same bytes as export_tokenizer.py writes on Linux (Windows text mode would write CRLF)
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({t: i for i, t in enumerate(tokens)}, f, ensure_ascii=False)
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(merges))
    with open(os.path.join(out_dir, "token_type.json"), "w", newline="\n") as f:
        json.dump(types, f)
    with open(os.path.join(out_dir, "chat_template.jinja"), "w", encoding="utf-8", newline="\n") as f:
        f.write("{# DeepSeek-V4.1: rendered by serve_ds41.py with DeepSeek's encoder (dsv41_encoding.py) #}\n")
    counts = {t: types.count(t) for t in sorted(set(types))}
    print(f"tokenizer: {len(tokens)} tokens (types {counts}), {len(merges)} merges -> {out_dir}")


def copy_stream(fin, at, n, fout, chunk=64 << 20):
    """n bytes from fin at `at` to fout's current position: a reader thread keeps two chunks ahead of the writes"""
    q = queue.Queue(maxsize=3)

    def reader():
        fin.seek(at)
        left = n
        while left:
            b = fin.read(min(chunk, left))
            if not b:
                q.put(None)
                return
            q.put(b)
            left -= len(b)
        q.put(b"")

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    done = 0
    while True:
        b = q.get()
        if b is None:
            raise SystemExit("short read")
        if not b:
            break
        fout.write(b)
        done += len(b)
    th.join()
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("--hf-config", required=True)
    ap.add_argument("--out")
    ap.add_argument("--name")
    ap.add_argument("--tokenizer-dir")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    paths, hdrs, src, plan, head, data_start, data_len, total, tiers = build(a)
    kinds = {}
    for n, dims, t, s, kind in plan:
        kinds[kind] = kinds.get(kind, 0) + 1
    print(f"{len(paths)} source file(s); {len(plan)} tensors {kinds}; experts {tiers}")
    print(f"output: {data_start + data_len} bytes ({(data_start + data_len) / 2**30:.2f} GiB), header {data_start}, "
          f"tensor data {total / 2**30:.2f} GiB")
    if a.tokenizer_dir:
        write_tokenizer(src, a.tokenizer_dir)
    if a.dry_run or not a.out:
        return
    part = a.out + ".part"
    t0, done = time.time(), 0
    fins = [open(p, "rb", buffering=0) for p in paths]
    try:
        with open(part, "wb", buffering=0) as fo:
            fo.write(head + b"\0" * (data_start - len(head)))
            pos = 0
            for i, (n, dims, t, s, kind) in enumerate(plan):
                nb = nbytes(t, dims)
                if kind == "copy":
                    got = copy_stream(fins[s[0]], s[3], s[4], fo)
                    if got != nb:
                        raise SystemExit(f"{n}: copied {got} of {nb} bytes")
                else:
                    if kind == "ones":
                        b = np.ones(nb // 4, np.float32).tobytes()
                    else:
                        f = fins[s[0]]
                        f.seek(s[3])
                        raw = f.read(s[4])
                        b = convert(kind, raw, n)
                    if len(b) != nb:
                        raise SystemExit(f"{n}: {len(b)} bytes, expected {nb}")
                    fo.write(b)
                pos += nb
                pad = (pos + ALIGN - 1) // ALIGN * ALIGN - pos
                fo.write(b"\0" * pad)
                pos += pad
                done += nb
                if kind != "copy" or nb > (1 << 30) or i == len(plan) - 1:
                    dt = time.time() - t0
                    print(f"[{i + 1}/{len(plan)}] {n} {TYPE_NAME[t]} ({kind}) {done / 2**30:.1f} GiB, "
                          f"{done / 2**30 / max(dt, 1e-9):.2f} GiB/s", flush=True)
            if pos != data_len:
                raise SystemExit(f"wrote {pos} data bytes, planned {data_len}")
    finally:
        for f in fins:
            f.close()
    os.replace(part, a.out)
    print(f"done: {a.out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
