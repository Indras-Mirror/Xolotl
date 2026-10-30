// Included INSIDE Engine. The immutable Model and activation arena belong to
// this Engine; recurrent/KV/verification state belongs to each group's Engine.
// Unlike a sequence of complete forward() calls, every layer projects/routes
// all owners together before visiting the next layer.
struct BatchGroup {
    Engine* state = nullptr;
    int first = 0, count = 0;
    int64_t pos = 0;
    bool verify = false;
    int heads = 1;
    bool active = true;  // latched false only after a completed stream boundary
};

// Decode arithmetic is intentional even when four verify windows total >8
// rows. The row count of a batch is not the length of one prompt chunk.
void batch_apply(const Mat& w, const void* in_q8, float* out, int rows) {
    const size_t row_bytes = K::native_q8_1_bytes(w.n_in, 1);
    for (int first = 0; first < rows; first += kGemmT) {
        const int count = std::min(kGemmT, rows - first);
        K::native_mmvq(w.type, w.d, (const char*) in_q8 + (size_t) first * row_bytes,
                      out + (size_t) first * w.n_out, w.n_in, w.n_out, count, st);
    }
}

void batch_mm(const Mat& w, const float* in, float* out, int rows) {
    quant(in, w.n_in, rows, q8);
    batch_apply(w, q8, out, rows);
}

void batch_swiglu_quantize(const float* g, const float* u, void* dst, int width, int rows) {
    const size_t row_bytes = K::native_q8_1_bytes(width, 1);
    for (int first = 0; first < rows; first += kGemmT) {
        const int count = std::min(kGemmT, rows - first);
        K::native_swiglu_quantize_q8_1(g + (size_t) first * width, u + (size_t) first * width,
            (char*) dst + (size_t) first * row_bytes, width, count, st);
    }
}

bool batch_groups_valid(const std::vector<BatchGroup>& groups, int& rows) const {
    rows = 0;
    if (groups.empty() || m->hybrid) return false;
    for (size_t i = 0; i < groups.size(); ++i) {
        const auto& b = groups[i];
        if (!b.active || !b.state || b.state->m != m || b.state->ctx != ctx || b.first != rows || b.count < 1 || b.count > kVerify ||
            b.pos < 0 || b.pos > b.state->ctx - b.count || b.heads < 0 || b.heads > b.count)
            return false;
        // The arena may itself be owner zero. A later self-owner would require
        // an overlapping D2D move when scattering its hidden rows to row zero.
        if (b.state == this && b.first != 0) return false;
        for (size_t j = 0; j < i; ++j) if (groups[j].state == b.state) return false;
        rows += b.count;
        if (rows > tmax) return false;
    }
    return true;
}

bool batch_layer(const Layer& L, size_t li, const std::vector<BatchGroup>& groups, int T,
                 const float* wn, float* hout, bool mtp = false) {
    if (mtp)
        for (const auto& b : groups) if (b.verify) return false;
    const auto& g = m->g;
    const int N = (int) g.n_embd, Kp = (int) m->top_k;
    const int FS = (int) g.n_ff_shexp, C = (int) g.ssm_conv_channels;
    const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads;
    const int ZV = (int) g.ssm_value_dim;
    const int H = (int) g.n_head, HKV = (int) g.n_head_kv, HD = (int) g.head_dim;
    if (!L.attn) {
        quant(h, N, T, xq);
        batch_apply(L.qkv, xq, qkv, T);
        batch_apply(L.zg, xq, z, T);
        k_gdn_ab<<<dim3(rows8(2 * HV), T), kThreads, 0, st>>>(
            L.alpha, L.beta, h, L.dt, L.ssm_a, alpha, beta, N, HV);
        for (const auto& b : groups) {
            if (!b.active) continue;
            Engine& s = *b.state;
            const size_t c = (size_t) b.first * C, hv = (size_t) b.first * HV;
            const size_t zv = (size_t) b.first * ZV;
            k_conv_seq<<<C / 128, 128, 0, st>>>(s.conv[li], qkv + c, L.conv, hc + c,
                                               C, 2 * HK, m->eps, b.count, !b.verify);
            k_gdn_seq<<<HV, dim3(128, 4), 0, st>>>(s.S[li], hc + c, C, alpha + hv, beta + hv,
                z + zv, L.ssm_norm, m->eps, gdn_y + zv, HK, HV, b.count, !b.verify);
            // commit() runs later in the owner's stream. Never leave its inputs
            // in the shared arena: the next layer/window overwrites that arena.
            if (b.verify) {
                CK(cudaMemcpyAsync(s.v_qkv[li], qkv + c, (size_t) b.count * C * 4,
                                   cudaMemcpyDeviceToDevice, st));
                CK(cudaMemcpyAsync(s.v_hc[li], hc + c, (size_t) b.count * C * 4,
                                   cudaMemcpyDeviceToDevice, st));
                CK(cudaMemcpyAsync(s.v_alpha[li], alpha + hv, (size_t) b.count * HV * 4,
                                   cudaMemcpyDeviceToDevice, st));
                CK(cudaMemcpyAsync(s.v_beta[li], beta + hv, (size_t) b.count * HV * 4,
                                   cudaMemcpyDeviceToDevice, st));
            }
        }
        batch_mm(L.ssm_out, gdn_y, y, T);
    } else {
        quant(h, N, T, xq);
        batch_apply(L.wq, xq, qg, T);
        batch_apply(L.wk, xq, kx, T);
        batch_apply(L.wv, xq, vx, T);
        for (const auto& b : groups) {
            if (!b.active) continue;
            Engine& s = *b.state;
            const int32_t* dp = mtp ? s.pos0_m : s.pos0;
            const size_t qo = (size_t) b.first * H * HD;
            const size_t ko = (size_t) b.first * HKV * HD;
            s.resident_before_write(b.pos);
            k_attn_prep<<<dim3(H + HKV, b.count), kThreads, 0, st>>>(
                qg + 2 * qo, kx + ko, vx + ko, L.q_norm, L.k_norm, q + qo, gate + qo,
                s.kc[li], s.vc[li], H, HKV, HD, m->n_rot, dp, s.ctx, m->rope_base, m->eps,
                m->has_mrope && !mrope_off() ? s.mrope : nullptr);
            // The partial scratch is reused between groups in stream order. Its
            // row indices are LOCAL to this group; offset Q/gate/output instead.
            const int rows_per = att_rows / H;
            if (rows_per < 1) return false;
            for (int t0 = 0; t0 < b.count; t0 += rows_per) {
                const int nt = std::min(rows_per, b.count - t0);
                const int nc = (int) std::min<int64_t>(n_chunk,
                    (b.pos + t0 + nt + kChunkCells - 1) / kChunkCells);
                // t0 is local to this owner's Q/gate/output slice. Partial
                // row indices restart at zero for each subdivision, exactly
                // as in layer(), so four-row scratch also supports 5..8 rows.
                attn_part(H / HKV, dim3(nc, nt * HKV), q + qo, s.kc[li], s.vc[li],
                          dp, t0, H, HKV, s.ctx, 1.0f / sqrtf((float) HD), n_chunk);
                k_attn_merge<<<nt * H, HD, 0, st>>>(pacc, pm, pl, gate + qo, dp, t0,
                                                  H, n_chunk, att + qo);
            }
        }
        batch_mm(L.wo, att, y, T);
    }
    k_add_rms<<<T, kThreads, 0, st>>>(x, y, L.post_norm, h, N, m->eps);
    // Same warp-per-output arithmetic as solo decode/verify, independently of
    // aggregate T. Grouping experts shares their weight loads across owners.
    k_gemv_f32<<<dim3(rows8((int) g.n_expert), T), kThreads, 0, st>>>(
        L.router, h, logits_r, N, (int) g.n_expert);
    K::native_router_topk(logits_r, ids, ewts, (int) g.n_expert, Kp, T, st);
    const int64_t entries = (int64_t) T * Kp;
    k_group<<<1, kThreads, 0, st>>>(ids, (int) entries, Kp, (int) g.n_expert, L.blobs,
        L.stride, grp_ptr, grp_start, n_groups, ent_dst, ent_tok);
    quant(h, N, T, xq);
    K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
        std::min<int64_t>(entries, g.n_expert), entries, xq, escr, eparts, st);
    batch_apply(L.sh_gate, xq, sg, T);
    batch_apply(L.sh_up, xq, su, T);
    batch_swiglu_quantize(sg, su, q8, FS, T);
    batch_apply(L.sh_down, q8, sy, T);
    k_gemv_f32<<<dim3(rows8(1), T), kThreads, 0, st>>>(L.sh_gate_inp, h, sgate, N, 1);
    k_moe_out<<<T, kThreads, 0, st>>>(x, eparts, nullptr, ewts, Kp, sy, sgate, 1,
                                    wn, hout, N, m->eps);
    return true;
}

// Caller stages every owner's token/pos buffers and fences their streams before
// entry; it must wait on this stream before owner commit/sampling/reuse. State
// mutation and cancellation publication remain scheduler-owned transactions.
bool batch_forward(const std::vector<BatchGroup>& input, const float* const* img = nullptr,
                   const std::function<void(Engine*)>& cancel_notify = {}) {
    std::vector<BatchGroup> groups(input);
    int T = 0;
    if (!batch_groups_valid(groups, T)) return false;
    if (cancel_notify)
        for (const auto& b : groups) if (b.state == this) return false;
    // Only verify windows can abandon a partial layer stack: their recurrent
    // state is read-only and KV above logical pos is speculative. Nonverify
    // work must finish its transaction (or use the prefill rollback path).
    auto boundary = [&](cudaStream_t stream) {
        CK(cudaStreamSynchronize(stream));
        bool any = false;
        for (auto& b : groups) {
            if (b.active && b.verify && b.state->cancel && b.state->cancel->load()) {
                b.active = false;
                // No later mixer/scatter accesses this owner's buffers. The
                // scheduler may wake it with an aborted result now; Engine and
                // immutable Model lifetimes must still span this whole call.
                if (cancel_notify) cancel_notify(b.state);
            }
            any = any || b.active;
        }
        return any;
    };
    // A singleton takes the established numerical and graph-free span path.
    if (groups.size() == 1) {
        const auto& b = groups[0];
        for (size_t begin = 0; begin < m->layers.size(); begin += kVerifyLayersPerGraph) {
            const size_t end = std::min(m->layers.size(), begin + kVerifyLayersPerGraph);
            if (!b.state->forward_span(b.count, b.pos, b.heads, false, b.verify,
                                      begin, end, img, false)) return false;
            if (!boundary(b.state->st)) break;
        }
        return true;
    }
    const int N = (int) m->g.n_embd;
    for (const auto& b : groups)
        CK(cudaMemcpyAsync(tok + b.first, b.state->tok, (size_t) b.count * sizeof(int32_t),
                           cudaMemcpyDeviceToDevice, st));
    K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, tok, T, N, x, st);
    put_image_rows(img, T, x);
    k_rms<<<T, kThreads, 0, st>>>(x, m->layers[0].attn_norm, h, N, m->eps);
    for (size_t li = 0; li < m->layers.size(); ++li) {
        const bool last = li + 1 == m->layers.size();
        if (!batch_layer(m->layers[li], li, groups, T,
                         last ? m->out_norm : m->layers[li + 1].attn_norm,
                         last ? hmain : h)) return false;
        if ((li + 1) % kVerifyLayersPerGraph == 0 || last)
            if (!boundary(st)) return true;
    }
    for (const auto& b : groups) {
        if (!b.active) continue;
        Engine& s = *b.state;
        const float* out = hmain + (size_t) b.first * N;
        if (b.heads) head_rows(out + (size_t) (b.count - b.heads) * N, b.heads, s.logits, s.pick);
        if (s.hmain != out)
            CK(cudaMemcpyAsync(s.hmain, out, (size_t) b.count * N * 4,
                               cudaMemcpyDeviceToDevice, st));
    }
    return true;
}
