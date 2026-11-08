"""A plain-torch (CUDA) port of Tencent's DSpine drafter (Qwen3.8-27B-DSpine-b16) for the offline acceptance experiment.

Follows dspine_mlx/model.py + bridge.py + propose.py (RefinementTail.greedy) of github.com/Tencent/DSpine: a 5-layer
DFlash-style block drafter over 5 target taps, causal inside the block, a code-bridge write after every layer (each row
gets a gated message from its predecessor's code), then the last write is redone serially over the top-16 head
candidates (a 16x16 transition table walked from the anchor).  Weights bf16, math fp32.  The target's embedding and
head are OUR GGUF's (Q6_K), passed in.
"""
import json, math
import torch
import torch.nn.functional as F
from safetensors import safe_open


def rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class DSpine:
    def __init__(self, path, embed_rows, head_bf16, dev="cuda"):
        cfg = json.load(open(path + "/config.json"))
        dc = cfg["dflash_config"]
        self.cfg = cfg
        self.H, self.L = cfg["hidden_size"], cfg["num_hidden_layers"]
        self.NH, self.NKV, self.HD = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.eps = cfg["rms_norm_eps"]
        self.mask_id = dc["mask_token_id"]
        self.taps = dc["target_layer_ids"]
        br = dc["selected_prefix_code_bridge"]
        self.C, self.R, self.beps = br["code_size"], br["rank"], br["eps"]
        self.K = cfg["code_bridge_second_pass"]["head_topk"]
        self.dev = dev
        self.w = {}
        with safe_open(path + "/model.safetensors", "pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if t.dtype == torch.int64:
                    assert int(t) == 1, (k, int(t))
                    continue
                self.w[k] = t.to(dev)                      # bf16
        b = "selected_prefix_bridge."
        assert not bool((self.w[b + "out.weight"] != 0).any()), "terminal projection must be zero"
        self.packed = torch.cat([self.w[b + "gate.weight"], self.w[b + "predecessor_gate.weight"], self.w[b + "prev.weight"]], 0).float()
        self.embed_rows = embed_rows                       # ids (cuda long) -> [n, H] fp32
        self.head = head_bf16                              # [V, H] bf16 on dev
        theta = cfg["rope_parameters"]["rope_theta"]
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.HD, 2, dtype=torch.float32, device=dev) / self.HD))
        self.Kc = None

    def W(self, k):
        return self.w[k].float()

    def lin(self, x, k):
        return x @ self.w[k].float().T

    def cos_sin(self, pos):
        fr = pos.float()[:, None] * self.inv_freq[None, :]
        e = torch.cat([fr, fr], -1)
        return e.cos(), e.sin()

    def set_context(self, feats):
        """feats [n, 5*H] fp32 (cuda): row j = the taps of position j, in target_layer_ids order."""
        h = rms(self.lin(feats, "fc.weight"), self.W("hidden_norm.weight"), self.eps)
        n = h.shape[0]
        cos, sin = self.cos_sin(torch.arange(n, device=self.dev))
        self.Kc, self.Vc = [], []
        for l in range(self.L):
            p = "layers.%d.self_attn." % l
            k = rms(self.lin(h, p + "k_proj.weight").view(n, self.NKV, self.HD), self.W(p + "k_norm.weight"), self.eps)
            k = k * cos[:, None] + rotate_half(k) * sin[:, None]
            v = self.lin(h, p + "v_proj.weight").view(n, self.NKV, self.HD)
            self.Kc.append(k.permute(1, 0, 2).contiguous())
            self.Vc.append(v.permute(1, 0, 2).contiguous())

    def bridge(self, out, ids, layer, next_norm_w, want_stash):
        b = "selected_prefix_bridge."
        T = out.shape[0]
        raw = out.to(torch.bfloat16).float()
        r = torch.sqrt(torch.clamp(raw.pow(2).mean(-1), min=1.1754943508222875e-38))       # [T]
        normalized = (raw / (r[:, None] + self.beps)).to(torch.bfloat16).float()
        rw = self.W(b + "read_weight") if layer == self.L - 1 else self.w[b + "early_read_weights"][layer].float()
        proj = normalized @ rw.T
        codes = (math.sqrt(self.C) * proj / (proj.norm(dim=-1, keepdim=True) + self.beps)).to(torch.bfloat16).float()
        anchor_code = (math.sqrt(self.C) * self.w[b + "codebook"][ids[0]].float()).to(torch.bfloat16).float()
        codes = torch.cat([anchor_code[None], codes[1:]], 0)
        packed = (codes @ self.packed.T).to(torch.bfloat16).float()                          # [T, 3R]
        q, s, v = packed[:, :self.R], packed[:, self.R:2 * self.R], packed[:, 2 * self.R:]
        gate = torch.sigmoid(q[1:] + s[:-1] + self.W(b + "gate.bias"))
        msg = torch.cat([torch.zeros_like(q[:1]), gate * v[:-1]], 0).to(torch.bfloat16).float()
        delta = msg @ self.w[b + "feature_write_weights"][layer].float().T
        not_anchor = torch.ones(T, 1, device=self.dev)
        not_anchor[0] = 0
        o = (raw + r[:, None] * delta * not_anchor).to(torch.bfloat16).float()
        normed = (o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + self.eps)).to(torch.bfloat16).float() * next_norm_w
        stash = dict(raw=raw, rms=r, q=q) if want_stash else None
        return normed.to(torch.bfloat16).float(), o, stash

    @torch.no_grad()
    def draft(self, s, anchor, bsz=16):
        """Anchor token at position s (context = positions 0..s-1); returns the bsz-1 drafted tokens and the plain top-1s."""
        ids = torch.tensor([anchor] + [self.mask_id] * (bsz - 1), device=self.dev)
        x = self.embed_rows(ids)
        pos = torch.arange(s, s + bsz, device=self.dev)
        cos, sin = self.cos_sin(pos)
        residual = x
        x_in = rms(x, self.W("layers.0.input_layernorm.weight"), self.eps)
        causal = torch.ones(bsz, bsz, dtype=torch.bool, device=self.dev).tril()
        rep = self.NH // self.NKV
        stash = None
        for l in range(self.L):
            p = "layers.%d." % l
            q = rms(self.lin(x_in, p + "self_attn.q_proj.weight").view(bsz, self.NH, self.HD), self.W(p + "self_attn.q_norm.weight"), self.eps)
            k = rms(self.lin(x_in, p + "self_attn.k_proj.weight").view(bsz, self.NKV, self.HD), self.W(p + "self_attn.k_norm.weight"), self.eps)
            v = self.lin(x_in, p + "self_attn.v_proj.weight").view(bsz, self.NKV, self.HD)
            q = q * cos[:, None] + rotate_half(q) * sin[:, None]
            k = k * cos[:, None] + rotate_half(k) * sin[:, None]
            K = torch.cat([self.Kc[l][:, :s], k.permute(1, 0, 2)], 1).repeat_interleave(rep, 0)   # [NH, s+b, HD]
            V = torch.cat([self.Vc[l][:, :s], v.permute(1, 0, 2)], 1).repeat_interleave(rep, 0)
            vis = torch.cat([torch.ones(bsz, s, dtype=torch.bool, device=self.dev), causal], 1)
            sc = torch.einsum("bhd,hkd->hbk", q, K) * self.HD ** -0.5
            att = torch.softmax(sc.masked_fill(~vis[None], float("-inf")), -1)
            o = torch.einsum("hbk,hkd->bhd", att, V).reshape(bsz, self.NH * self.HD)
            h = residual + self.lin(o, p + "self_attn.o_proj.weight")
            m = rms(h, self.W(p + "post_attention_layernorm.weight"), self.eps)
            m = (F.silu(self.lin(m, p + "mlp.gate_proj.weight")) * self.lin(m, p + "mlp.up_proj.weight"))
            out = h + self.lin(m, p + "mlp.down_proj.weight")
            nn_w = self.W("layers.%d.input_layernorm.weight" % (l + 1)) if l < self.L - 1 else self.W("norm.weight")
            x_in, residual, st = self.bridge(out, ids, l, nn_w, l == self.L - 1)
            if st is not None:
                stash = st
        rows = x_in[1:]                                                       # [n, H]
        logits = rows.to(torch.bfloat16) @ self.head.T
        top1 = logits.argmax(-1)
        cand = torch.topk(logits.float(), self.K, -1).indices                 # [n, K]
        return self.refine(rows, stash, anchor, cand), top1.tolist()

    def refine(self, rows, stash, anchor, cand):
        """RefinementTail.greedy: redo the last write with each candidate predecessor, walk the K x K table."""
        b = "selected_prefix_bridge."
        n = rows.shape[0]
        raw, r, q = stash["raw"][1:], stash["rms"][1:], stash["q"][1:]
        nw = self.W("norm.weight")
        hr = self.head[cand].float()                                          # [n, K, H]
        base = ((raw[:, None, :] * nw) * hr).sum(-1)                          # [n, K]
        sid = torch.cat([torch.full((1, self.K), anchor, device=self.dev, dtype=torch.long), cand[:-1]], 0)   # [n, K]
        e = (math.sqrt(self.C) * self.w[b + "codebook"][sid].float()).to(torch.bfloat16).float()             # [n, K, C]
        tvg = (e @ self.W(b + "predecessor_gate.weight").T).to(torch.bfloat16).float()
        tvv = (e @ self.W(b + "prev.weight").T).to(torch.bfloat16).float()
        mm = torch.sigmoid(q[:, None, :] + tvg + self.W(b + "gate.bias")) * tvv                              # [n, K(sender), R]
        tg = ((hr * nw).to(torch.bfloat16).float() @ self.w[b + "feature_write_weights"][self.L - 1].float()).to(torch.bfloat16).float()  # [n, K(cand), R]
        score = base[:, None, :] + r[:, None, None] * torch.matmul(mm, tg.transpose(1, 2))                   # [n, K, K]
        idx = score.argmax(-1)                                                # [n, K]
        j = int(idx[0, 0])
        path = []
        for i in range(n):
            path.append(int(cand[i, j]))
            if i + 1 < n:
                j = int(idx[i + 1, j])
        return path
