// src/program/inkling_vision.cpp - the inkling2v image encoder in-process (llama.cpp PR #32694's
// tools/mtmd/models/inkling2v.cpp + mtmd-image.cpp preprocessor, MIT, re-built on ggml-backend).
//
//   image file -> stb_image (RGB u8) -> (resize if over the token cap) -> pad to 40 px with -1 on the 0..255 scale ->
//   normalize (x/255 - 0.5)/0.5 -> ViT (23 blocks: LayerNorm, non-causal 16-head attention with axial 2-D RoPE,
//   GELU-erf MLP, layer-scale residuals, learned 32x32 position grid resized bilinearly) -> drop the CLS row ->
//   2x2 patch merge -> 4096 -> GELU-erf -> 2304 (the text model's width) -> SVE1.
//
// The weights are a normal mmproj GGUF (convert_hf_to_gguf.py --mmproj, projector_type inkling2v).  They sit in a
// no-alloc ggml context; on every picture they are uploaded to the chosen backend's buffer and that buffer is freed
// afterwards, so a text-only session keeps no VRAM for the tower.

#include "inkling_vision.h"

#include "ggml.h"
#include "ggml-alloc.h"

#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/stat.h>
#ifdef _WIN32
#include <io.h>                 // _open/_close/_read: the fd API, though MSVC has no unistd.h
#include <stdint.h>
#include <sys/types.h>          // declares off_t itself: read before the macro below, not again after it
typedef long long ssize_t;      // POSIX ssize_t, MSVC has none
#define off_t int64_t           // MSVC's off_t is 32-bit; the seek below casts a 64-bit offset through it
// pread as seek + read: one loop in encode() reads the projector, tensor by tensor, so the two calls cannot interleave
static inline long long pread_win(int fd, void* buf, size_t n, int64_t off) {
    if (_lseeki64(fd, off, SEEK_SET) < 0) return -1;
    return _read(fd, buf, (unsigned) n);
}
#define pread(fd, buf, n, off) pread_win((fd), (buf), (n), (off))
static inline int open_readonly(const char* path) { return _open(path, O_RDONLY | _O_BINARY); }   // text mode would translate CRLF into the file
static inline int close_fd(int fd) { return _close(fd); }
#else
#include <unistd.h>
static inline int open_readonly(const char* path) { return ::open(path, O_RDONLY); }
static inline int close_fd(int fd) { return ::close(fd); }
#endif
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb/stb_image.h"

namespace inkling_vision {
namespace {

constexpr int kDefaultMaxTokens = 4096;   // mtmd hparams.set_limit_image_tokens(1, 4096) for inkling2v
constexpr int kAlign = 40;                // patch (20) * merge (2)
constexpr float kPad = -1.0f;             // the pad pixel on the 0..255 scale (normalized: -1.007843)

struct Layer {
    ggml_tensor *ln1_w = nullptr, *ln1_b = nullptr;
    ggml_tensor *ln2_w = nullptr, *ln2_b = nullptr;
    ggml_tensor *ls1 = nullptr, *ls2 = nullptr;
    ggml_tensor *q_w = nullptr, *q_b = nullptr;
    ggml_tensor *k_w = nullptr, *k_b = nullptr;
    ggml_tensor *v_w = nullptr, *v_b = nullptr;
    ggml_tensor *o_w = nullptr, *o_b = nullptr;
    ggml_tensor *up_w = nullptr, *up_b = nullptr;
    ggml_tensor *down_w = nullptr, *down_b = nullptr;
};

}   // namespace

struct Encoder {
    std::string path;
    int fd = -1;
    size_t data_off = 0;
    bool use_gpu = false;
    ggml_backend_t gpu = nullptr, cpu = nullptr, dev = nullptr;

    gguf_context* gctx = nullptr;     // keys + tensor metadata
    ggml_context* wctx = nullptr;     // the weight tensors (no-alloc)
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_backend_sched_t sched = nullptr;

    int n_embd = 0, n_head = 0, d_head = 0, n_layer = 0, patch = 0, merge = 0, grid = 0, n_out = 0;
    float eps = 1e-5f;

    ggml_tensor *patch_w = nullptr, *cls = nullptr, *cls_pos = nullptr, *pos_embd = nullptr;
    ggml_tensor *pre_w = nullptr, *pre_b = nullptr, *mm1 = nullptr, *mm2 = nullptr;
    std::vector<Layer> L;

    std::vector<uint8_t> graph_buf;   // scratch for one graph's tensors
};

namespace {

ggml_tensor* T(Encoder& e, const std::string& name, bool need = true) {
    ggml_tensor* t = ggml_get_tensor(e.wctx, name.c_str());
    if (!t && need) {
        std::string err = "the mmproj has no tensor " + name;
        throw std::runtime_error(err);
    }
    return t;
}

uint32_t key_u32(Encoder& e, const char* k, uint32_t dflt) {
    const int id = gguf_find_key(e.gctx, k);
    if (id < 0) return dflt;
    return gguf_get_val_u32(e.gctx, id);
}

float key_f32(Encoder& e, const char* k, float dflt) {
    const int id = gguf_find_key(e.gctx, k);
    if (id < 0) return dflt;
    return gguf_get_val_f32(e.gctx, id);
}

// the reference's build_norm (NORM_TYPE_NORMAL: LayerNorm + optional weight/bias)
ggml_tensor* layernorm(ggml_context* c, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, float eps) {
    x = ggml_norm(c, x, eps);
    if (w) x = ggml_mul(c, x, w);
    if (b) x = ggml_add(c, x, b);
    return x;
}

// build_rope_2d (mtmd-graph.cpp): rotate the first half of the head dims by the width position and the second half by
// the height position, both with freq_base and the same even/odd frequency layout as llama.cpp's rope.
ggml_tensor* rope_2d(ggml_context* c, ggml_tensor* x, ggml_tensor* pos_w, ggml_tensor* pos_h, float theta) {
    const int64_t n_dim = x->ne[0];
    x = ggml_rope_ext(c, x, pos_w, nullptr, (int) (n_dim / 2), 0, 0, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    x = ggml_rope_ext(c, x, pos_h, nullptr, (int) (n_dim / 2), 0, 0, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    return ggml_rope_set_offset(x, (int) (n_dim / 2));
}

// resize_position_embeddings (bilinear), when the picture's patch grid is not the learned 32x32 one
ggml_tensor* resized_positions(ggml_context* c, ggml_tensor* pos, int n_embd, int n_per_side, int cols, int rows) {
    if (cols == n_per_side && rows == n_per_side) return pos;
    pos = ggml_reshape_3d(c, pos, n_embd, n_per_side, n_per_side);
    pos = ggml_permute(c, pos, 2, 0, 1, 3);
    pos = ggml_interpolate(c, pos, cols, rows, n_embd, 1, GGML_SCALE_MODE_BILINEAR);
    pos = ggml_permute(c, pos, 1, 2, 0, 3);
    return ggml_cont_2d(c, pos, n_embd, cols * rows);
}

// build_patch_merge_permute (2x2 unshuffle): [n_embd, w*h] -> [4*n_embd, (w/2)*(h/2)]
ggml_tensor* patch_merge(ggml_context* c, ggml_tensor* cur, int n_embd, int width, int height, int scale) {
    const int64_t pad_w = (width + scale - 1) / scale * scale - width;
    const int64_t pad_h = (height + scale - 1) / scale * scale - height;
    cur = ggml_reshape_3d(c, cur, n_embd, width, height);
    if (pad_w || pad_h) {
        cur = ggml_pad(c, cur, 0, (int) pad_w, (int) pad_h, 0);
        width += (int) pad_w;
        height += (int) pad_h;
    }
    cur = ggml_reshape_3d(c, cur, (int64_t) n_embd * scale, width / scale, height);
    cur = ggml_permute(c, cur, 0, 2, 1, 3);
    cur = ggml_cont_3d(c, cur, (int64_t) n_embd * scale * scale, height / scale, width / scale);
    cur = ggml_permute(c, cur, 0, 2, 1, 3);
    return ggml_cont_2d(c, cur, cur->ne[0], cur->ne[1] * cur->ne[2]);
}

// Flash attention in the tower: XOLOTL_VISION_FA=1 always, =0 never (the plain path's score tensor may not fit),
// default = only when the f32 scores of one layer would pass 512 MiB (n_pos > ~2,900, i.e. pictures over ~725 tokens).
// Smaller pictures keep the plain softmax path, so their rows stay byte-identical to the gated reference dumps.
bool use_flash_attn(int64_t n_pos, int n_head) {
    if (const char* s = std::getenv("XOLOTL_VISION_FA")) return std::atoi(s) != 0;
    return (double) n_pos * (double) n_pos * n_head * 4.0 > 512.0 * 1024 * 1024;
}

// the graph for one picture: [W, H] pixels -> [n_out, rows]
ggml_tensor* build_graph(Encoder& e, ggml_context* c, ggml_cgraph* gf, int W, int H, ggml_tensor** inp,
                         ggml_tensor** pos_w_in, ggml_tensor** pos_h_in, int* rows_out) {
    const int pw = W / e.patch, ph = H / e.patch;      // patches per side
    const int n_pos = pw * ph + 1;                     // + the CLS row
    const int64_t E = e.n_embd;
    const float kq_scale = 1.0f / std::sqrt((float) e.d_head);
    const bool fa = use_flash_attn(n_pos, e.n_head);

    *inp = ggml_new_tensor_4d(c, GGML_TYPE_F32, W, H, 3, 1);
    ggml_set_name(*inp, "inp_raw");
    ggml_set_input(*inp);

    ggml_tensor* cur = ggml_conv_2d(c, e.patch_w, *inp, e.patch, e.patch, 0, 0, 1, 1);
    cur = ggml_reshape_3d(c, cur, (int64_t) pw * ph, E, 1);
    cur = ggml_cont(c, ggml_transpose(c, cur));                       // [n_embd, n_patches, 1]
    cur = ggml_add(c, cur, resized_positions(c, e.pos_embd, (int) E, e.grid, pw, ph));
    {
        ggml_tensor* cls = ggml_add(c, e.cls, e.cls_pos);              // [n_embd]
        cur = ggml_concat(c, ggml_reshape_2d(c, cls, E, 1), ggml_reshape_2d(c, cur, E, (int64_t) pw * ph), 1);
    }
    cur = layernorm(c, cur, e.pre_w, e.pre_b, e.eps);

    *pos_w_in = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_pos);
    ggml_set_name(*pos_w_in, "pos_w");
    ggml_set_input(*pos_w_in);
    *pos_h_in = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_pos);
    ggml_set_name(*pos_h_in, "pos_h");
    ggml_set_input(*pos_h_in);

    for (int il = 0; il < e.n_layer; ++il) {
        const Layer& L = e.L[il];
        ggml_tensor* t = layernorm(c, cur, L.ln1_w, L.ln1_b, e.eps);
        ggml_tensor* q = ggml_mul_mat(c, L.q_w, t);
        if (L.q_b) q = ggml_add(c, q, L.q_b);
        ggml_tensor* k = ggml_mul_mat(c, L.k_w, t);
        if (L.k_b) k = ggml_add(c, k, L.k_b);
        ggml_tensor* v = ggml_mul_mat(c, L.v_w, t);
        if (L.v_b) v = ggml_add(c, v, L.v_b);
        q = ggml_reshape_4d(c, q, e.d_head, e.n_head, n_pos, 1);
        k = ggml_reshape_4d(c, k, e.d_head, e.n_head, n_pos, 1);
        v = ggml_reshape_4d(c, v, e.d_head, e.n_head, n_pos, 1);
        q = rope_2d(c, q, *pos_w_in, *pos_h_in, 10000.0f);
        k = rope_2d(c, k, *pos_w_in, *pos_h_in, 10000.0f);
        ggml_tensor* qp = ggml_permute(c, q, 0, 2, 1, 3);               // [d_head, n_pos, n_head]
        ggml_tensor* kp = ggml_permute(c, k, 0, 2, 1, 3);
        ggml_tensor* attn = nullptr;
        if (fa) {
            // flash attention: no [n_pos, n_pos, n_head] score tensor (a 3840x2160 screenshot at the 4096-token cap
            // has 16,385 positions -> 16.6 GiB of f32 scores on the plain path). K/V in f16 as ggml-cuda's FA wants.
            ggml_tensor* kf = ggml_cast(c, kp, GGML_TYPE_F16);
            ggml_tensor* vf = ggml_cast(c, ggml_permute(c, v, 0, 2, 1, 3), GGML_TYPE_F16);
            ggml_tensor* o = ggml_flash_attn_ext(c, qp, kf, vf, nullptr, kq_scale, 0.0f, 0.0f);   // [d_head, n_head, n_pos]
            ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
            attn = ggml_reshape_2d(c, o, (int64_t) e.d_head * e.n_head, n_pos);
        } else {
            ggml_tensor* vp = ggml_cont(c, ggml_permute(c, v, 1, 2, 0, 3)); // [n_pos, d_head, n_head]
            ggml_tensor* kq = ggml_mul_mat(c, kp, qp);
            kq = ggml_soft_max_ext(c, kq, nullptr, kq_scale, 0.0f);
            ggml_tensor* kqv = ggml_mul_mat(c, vp, kq);                    // [d_head, n_pos, n_head]
            attn = ggml_cont_2d(c, ggml_permute(c, kqv, 0, 2, 1, 3), (int64_t) e.d_head * e.n_head, n_pos);
        }
        attn = ggml_mul_mat(c, L.o_w, attn);
        if (L.o_b) attn = ggml_add(c, attn, L.o_b);
        if (L.ls1) attn = ggml_mul(c, attn, L.ls1);
        cur = ggml_add(c, cur, attn);

        ggml_tensor* f = layernorm(c, cur, L.ln2_w, L.ln2_b, e.eps);
        f = ggml_mul_mat(c, L.up_w, f);
        if (L.up_b) f = ggml_add(c, f, L.up_b);
        f = ggml_gelu_erf(c, f);
        f = ggml_mul_mat(c, L.down_w, f);
        if (L.down_b) f = ggml_add(c, f, L.down_b);
        if (L.ls2) f = ggml_mul(c, f, L.ls2);
        cur = ggml_add(c, cur, f);
    }

    // drop the CLS row (row 0), merge 2x2 patch blocks, project to the text model's width
    const int64_t n_patches = (int64_t) pw * ph;
    cur = ggml_view_2d(c, cur, E, n_patches, cur->nb[1], cur->nb[1]);
    cur = patch_merge(c, cur, (int) E, pw, ph, e.merge);
    cur = ggml_mul_mat(c, e.mm1, cur);
    cur = ggml_gelu_erf(c, cur);
    cur = ggml_mul_mat(c, e.mm2, cur);
    ggml_set_name(cur, "projected");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    *rows_out = (int) cur->ne[1];
    return cur;
}

// ---- preprocessing (mtmd_image_preprocessor_inkling2v)

struct Img {
    int w = 0, h = 0;
    std::vector<uint8_t> rgb;   // w*h*3
};

// Pillow-style separable resize (bicubic, a = -0.5, antialias), in float: only for pictures over the token cap.
// The reference resizes in fixed point on the u8 pixels; this rounds the same way, to float pixels.
struct Dim {
    int out = 0, ksize = 0;
    std::vector<int> lo, hi;
    std::vector<float> w;
};

Dim coeffs(int in, int out) {
    Dim d;
    d.out = out;
    const double scale = (double) in / out;
    const double filterscale = std::max(1.0, scale);
    const double support = 2.0 * filterscale;           // bicubic
    d.ksize = (int) std::ceil(support) * 2 + 1;
    d.lo.resize(out); d.hi.resize(out); d.w.assign((size_t) out * d.ksize, 0.0f);
    for (int xx = 0; xx < out; ++xx) {
        const double center = (xx + 0.5) * scale;
        int lo = (int) std::floor(center - support + 0.5);
        int hi = (int) std::ceil(center + support - 0.5);
        lo = std::max(lo, 0);
        hi = std::min(hi, in);
        double sum = 0.0;
        std::vector<double> kk((size_t) (hi - lo));
        for (int x = lo; x < hi; ++x) {
            const double dd = std::fabs((x - center + 0.5) / filterscale);
            constexpr double a = -0.5;
            double v = 0.0;
            if (dd < 1.0) v = ((a + 2.0) * dd - (a + 3.0)) * dd * dd + 1.0;
            else if (dd < 2.0) v = (((dd - 5.0) * dd + 8.0) * dd - 4.0) * a;
            kk[(size_t) (x - lo)] = v;
            sum += v;
        }
        d.lo[xx] = lo; d.hi[xx] = hi;
        for (int x = lo; x < hi; ++x)
            d.w[(size_t) xx * d.ksize + (x - lo)] = (float) (kk[(size_t) (x - lo)] / (sum != 0.0 ? sum : 1.0));
    }
    return d;
}

void resize_bicubic(const Img& src, Img& dst, int tw, int th) {
    dst.w = tw; dst.h = th;
    dst.rgb.assign((size_t) tw * th * 3, 0);
    const Dim xd = coeffs(src.w, tw), yd = coeffs(src.h, th);
    std::vector<float> tmp((size_t) tw * src.h * 3, 0.0f);
    for (int y = 0; y < src.h; ++y)
        for (int x = 0; x < tw; ++x)
            for (int c = 0; c < 3; ++c) {
                float s = 0.0f;
                for (int xx = xd.lo[x]; xx < xd.hi[x]; ++xx)
                    s += xd.w[(size_t) x * xd.ksize + (xx - xd.lo[x])] * src.rgb[(size_t) (y * src.w + xx) * 3 + c];
                tmp[((size_t) y * tw + x) * 3 + c] = s;
            }
    for (int y = 0; y < th; ++y)
        for (int x = 0; x < tw; ++x)
            for (int c = 0; c < 3; ++c) {
                float s = 0.0f;
                for (int yy = yd.lo[y]; yy < yd.hi[y]; ++yy)
                    s += yd.w[(size_t) y * yd.ksize + (yy - yd.lo[y])] * tmp[((size_t) yy * tw + x) * 3 + c];
                dst.rgb[((size_t) y * tw + x) * 3 + c] = (uint8_t) std::min(255.0f, std::max(0.0f, std::round(s)));
            }
}

// decode + the size rule of mtmd_image_preprocessor_inkling2v: keep the pixels; over the token cap, shrink with the
// aspect kept and the sides on the 40-px grid (mtmd's calc_size_preserved_ratio)
bool preprocess(const std::string& image, int max_tokens, Img& img, std::string& err) {
    int w = 0, h = 0, n = 0;
    unsigned char* pix = stbi_load(image.c_str(), &w, &h, &n, 3);
    if (!pix) {
        err = "cannot read the image " + image + " (" + (stbi_failure_reason() ? stbi_failure_reason() : "unknown") + ")";
        return false;
    }
    img.w = w; img.h = h;
    img.rgb.assign(pix, pix + (size_t) w * h * 3);
    stbi_image_free(pix);
    if (w <= 0 || h <= 0) { err = "the image has no pixels"; return false; }

    auto align = [](int v) { return (v + kAlign - 1) / kAlign * kAlign; };
    const int cap = max_tokens > 0 ? max_tokens : kDefaultMaxTokens;
    if ((align(w) / kAlign) * (align(h) / kAlign) > cap) {
        const double max_pixels = (double) cap * kAlign * kAlign;
        const double beta = std::sqrt((double) w * h / max_pixels);
        const int tw = std::max(kAlign, (int) (std::floor(w / beta / kAlign) * kAlign));
        const int th = std::max(kAlign, (int) (std::floor(h / beta / kAlign) * kAlign));
        Img small;
        resize_bicubic(img, small, tw, th);
        img = std::move(small);
    }
    return true;
}

}   // namespace

Encoder* load(const std::string& mmproj, ggml_backend_t gpu, ggml_backend_t cpu, bool use_gpu, std::string& err) {
    Encoder* e = new Encoder();
    e->path = mmproj;
    e->gpu = gpu;
    e->cpu = cpu;
    e->use_gpu = use_gpu && gpu != nullptr;
    e->dev = e->use_gpu ? gpu : cpu;
    gguf_init_params ip{};
    ip.no_alloc = true;
    ip.ctx = &e->wctx;
    e->gctx = gguf_init_from_file(mmproj.c_str(), ip);
    if (!e->gctx || !e->wctx) { err = "cannot read the mmproj " + mmproj; free(e); return nullptr; }
    const int pt = gguf_find_key(e->gctx, "clip.vision.projector_type");
    if (pt < 0 || std::strcmp(gguf_get_val_str(e->gctx, pt), "inkling2v") != 0) {
        err = "the mmproj is not an Inkling-2 vision tower (clip.vision.projector_type must be inkling2v)";
        free(e);
        return nullptr;
    }
    e->n_embd = (int) key_u32(*e, "clip.vision.embedding_length", 1024);
    e->n_head = (int) key_u32(*e, "clip.vision.attention.head_count", 16);
    e->n_layer = (int) key_u32(*e, "clip.vision.block_count", 23);
    e->patch = (int) key_u32(*e, "clip.vision.patch_size", 20);
    e->merge = (int) key_u32(*e, "clip.vision.spatial_merge_size", 2);
    e->n_out = (int) key_u32(*e, "clip.vision.projection_dim", 2304);
    e->eps = key_f32(*e, "clip.vision.attention.layer_norm_epsilon", 1e-5f);
    const int image_size = (int) key_u32(*e, "clip.vision.image_size", 640);
    e->grid = image_size / (e->patch > 0 ? e->patch : 20);
    if (e->n_embd <= 0 || e->n_head <= 0 || e->n_embd % e->n_head != 0) { err = "bad vision head count"; free(e); return nullptr; }
    e->d_head = e->n_embd / e->n_head;
    try {
        e->patch_w = T(*e, "v.patch_embd.weight");
        e->cls = T(*e, "v.class_embd");
        e->cls_pos = T(*e, "v.class_position_embd.weight");
        e->pos_embd = T(*e, "v.position_embd.weight");
        e->pre_w = T(*e, "v.pre_ln.weight");
        e->pre_b = T(*e, "v.pre_ln.bias");
        e->mm1 = T(*e, "mm.1.weight");
        e->mm2 = T(*e, "mm.2.weight");
        e->L.resize((size_t) e->n_layer);
        for (int il = 0; il < e->n_layer; ++il) {
            const std::string p = "v.blk." + std::to_string(il) + ".";
            Layer& L = e->L[(size_t) il];
            L.ln1_w = T(*e, p + "ln1.weight"); L.ln1_b = T(*e, p + "ln1.bias");
            L.ln2_w = T(*e, p + "ln2.weight"); L.ln2_b = T(*e, p + "ln2.bias");
            L.ls1 = T(*e, p + "ls1.weight", false); L.ls2 = T(*e, p + "ls2.weight", false);
            L.q_w = T(*e, p + "attn_q.weight"); L.q_b = T(*e, p + "attn_q.bias", false);
            L.k_w = T(*e, p + "attn_k.weight"); L.k_b = T(*e, p + "attn_k.bias", false);
            L.v_w = T(*e, p + "attn_v.weight"); L.v_b = T(*e, p + "attn_v.bias", false);
            L.o_w = T(*e, p + "attn_out.weight"); L.o_b = T(*e, p + "attn_out.bias", false);
            L.up_w = T(*e, p + "ffn_up.weight"); L.up_b = T(*e, p + "ffn_up.bias", false);
            L.down_w = T(*e, p + "ffn_down.weight"); L.down_b = T(*e, p + "ffn_down.bias", false);
        }
    } catch (const std::exception& ex) {
        err = ex.what();
        free(e);
        return nullptr;
    }
    e->fd = open_readonly(mmproj.c_str());
    if (e->fd < 0) { err = "cannot open the mmproj " + mmproj; free(e); return nullptr; }
    e->data_off = gguf_get_data_offset(e->gctx);
    ggml_backend_t bes[2] = {e->dev, e->cpu};
    e->sched = ggml_backend_sched_new(bes, nullptr, e->dev == e->cpu ? 1 : 2, 8192, false, true);
    e->graph_buf.resize((size_t) 96 << 20);   // graph tensors for the largest picture (4096 patches)
    return e;
}

// drop the uploaded weights: the tensors keep their metadata in the no-alloc context, so their data pointers have
// to be cleared for a later picture to re-allocate them (ggml's allocator skips tensors that already have data)
static void release_weights(Encoder& e) {
    if (!e.wbuf) return;
    ggml_backend_buffer_free(e.wbuf);
    e.wbuf = nullptr;
    for (ggml_tensor* t = ggml_get_first_tensor(e.wctx); t; t = ggml_get_next_tensor(e.wctx, t)) {
        t->data = nullptr;
        t->buffer = nullptr;
    }
}

void free(Encoder* e) {
    if (!e) return;
    if (e->sched) ggml_backend_sched_free(e->sched);
    release_weights(*e);
    if (e->fd >= 0) close_fd(e->fd);
    if (e->gctx) gguf_free(e->gctx);
    if (e->wctx) ggml_free(e->wctx);
    delete e;
}

int encode(Encoder& e, const std::string& image, const std::string& out, int max_tokens, std::string& err) {
    Img img;
    int rows = 0;
    if (!preprocess(image, max_tokens, img, err)) return 0;
    auto align = [](int v) { return (v + kAlign - 1) / kAlign * kAlign; };
    const int W = align(img.w), H = align(img.h);   // the 40-px grid, padded with -1 on the 0..255 scale
    rows = (W / kAlign) * (H / kAlign);
    if (rows <= 0) { err = "the image produced no tokens"; return 0; }

    // ---- the weights of this picture (freed after)
    if (!e.wbuf) {
        e.wbuf = ggml_backend_alloc_ctx_tensors(e.wctx, e.dev);
        if (!e.wbuf) { err = "cannot allocate the vision weights on the chosen backend"; return 0; }
        std::vector<uint8_t> tmp;
        for (ggml_tensor* t = ggml_get_first_tensor(e.wctx); t; t = ggml_get_next_tensor(e.wctx, t)) {
            const size_t nb = ggml_nbytes(t);
            const int tid = gguf_find_tensor(e.gctx, t->name);
            if (tid < 0) { ggml_backend_buffer_free(e.wbuf); e.wbuf = nullptr; err = std::string("mmproj tensor offset missing: ") + t->name; return 0; }
            const size_t off = e.data_off + gguf_get_tensor_offset(e.gctx, tid);
            tmp.resize(nb);
            if (::pread(e.fd, tmp.data(), nb, (off_t) off) != (ssize_t) nb) {
                ggml_backend_buffer_free(e.wbuf); e.wbuf = nullptr;
                err = std::string("short read of the mmproj tensor ") + t->name;
                return 0;
            }
            ggml_backend_tensor_set(t, tmp.data(), 0, nb);
        }
    }

    // ---- the graph
    ggml_init_params gp{e.graph_buf.size(), e.graph_buf.data(), true};   // tensors only: the scheduler allocates their data
    ggml_context* ctx = ggml_init(gp);
    if (!ctx) { err = "cannot create the vision graph context"; return 0; }
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_tensor *inp = nullptr, *pos_w = nullptr, *pos_h = nullptr;
    int n_rows = 0;
    ggml_tensor* outt = build_graph(e, ctx, gf, W, H, &inp, &pos_w, &pos_h, &n_rows);
    ggml_backend_sched_reset(e.sched);
    if (!ggml_backend_sched_alloc_graph(e.sched, gf)) {
        ggml_free(ctx);
        err = "cannot allocate the vision graph (no VRAM for this picture size)";
        return 0;
    }

    // ---- inputs: the pixel buffer (x fastest, then y, then channel) and the 2-D RoPE positions
    {
        // the graph's input is [W, H, 3]: x fastest, then y, then the channel (planar, as mtmd de-interleaves the
        // encoder's RGB buffer into)
        const float pad = (kPad - 127.5f) / 127.5f;
        std::vector<float> px((size_t) W * H * 3, pad);
        const size_t n = (size_t) W * H;
        for (int y = 0; y < img.h; ++y)
            for (int x = 0; x < img.w; ++x)
                for (int c = 0; c < 3; ++c)
                    px[(size_t) c * n + (size_t) y * W + x] = ((float) img.rgb[((size_t) y * img.w + x) * 3 + c] - 127.5f) / 127.5f;
        ggml_backend_tensor_set(inp, px.data(), 0, px.size() * sizeof(float));
    }
    {
        const int pw = W / e.patch, ph = H / e.patch;
        std::vector<int32_t> w_pos((size_t) pw * ph + 1, 0), h_pos((size_t) pw * ph + 1, 0);
        for (int y = 0; y < ph; ++y)
            for (int x = 0; x < pw; ++x) {
                w_pos[(size_t) y * pw + x + 1] = x + 1;
                h_pos[(size_t) y * pw + x + 1] = y + 1;
            }
        ggml_backend_tensor_set(pos_w, w_pos.data(), 0, w_pos.size() * sizeof(int32_t));
        ggml_backend_tensor_set(pos_h, h_pos.data(), 0, h_pos.size() * sizeof(int32_t));
    }
    if (ggml_backend_sched_graph_compute(e.sched, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        err = "the vision graph failed to compute";
        return 0;
    }
    rows = n_rows;
    std::vector<float> rows_f((size_t) rows * e.n_out);
    ggml_backend_tensor_get(outt, rows_f.data(), 0, rows_f.size() * sizeof(float));

    // ---- release the weights' device buffer and the graph's scratch
    ggml_backend_sched_reset(e.sched);
    ggml_free(ctx);
    release_weights(e);

    // ---- the SVE1 record
    FILE* f = std::fopen(out.c_str(), "wb");
    if (!f) { err = "cannot write the embeddings file " + out; return 0; }
    const int32_t hd[5] = {0x31455653, rows, rows, 1, e.n_out};
    bool ok = std::fwrite(hd, sizeof(int32_t), 5, f) == 5;
    ok = ok && std::fwrite(rows_f.data(), sizeof(float), rows_f.size(), f) == rows_f.size();
    std::fclose(f);
    if (!ok) { err = "short write of the embeddings file " + out; return 0; }
    return rows;
}

}   // namespace inkling_vision
