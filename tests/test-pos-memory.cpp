// Mode-W memory cells (LLAMA_POS_MEMORY, include/llama.h) on the generated test models (test-llama-archs -o DIR):
//
//   test-pos-memory MODEL.gguf [MODEL.gguf ...]   memory semantics, for each model
//   test-pos-memory --dump OUT MODEL.gguf          logits of a decode with no memory (chunked prompt with SWA eviction,
//                                                  seq_cp, partial seq_rm, state save and restore), so that a build
//                                                  before LLAMA_POS_MEMORY and one after can be compared byte for byte
//
// The checks use the public API only. Masks are read through the eval callback (the attn_inp_kq_mask inputs of the
// last ubatch): memory must add exactly its own cells to every mask, the sliding-window one included.

#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#ifndef LLAMA_POS_MEMORY
#define LLAMA_POS_MEMORY ((llama_pos) -2) // a build before the sentinel: --dump never uses it
#endif

namespace {

struct mask_probe {
    bool on = false;
    std::map<const ggml_tensor *, int> kept; // per mask input: kept cells in the row of the last token
};

bool probe_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * probe = (mask_probe *) user_data;
    if (ask) {
        return probe->on;
    }
    for (int k = 0; k < GGML_MAX_SRC; ++k) {
        const ggml_tensor * m = t->src[k];
        if (!m || strncmp(m->name, "attn_inp_kq_mask", 16) != 0 || probe->kept.count(m) || !m->data) {
            continue;
        }
        const int64_t n_kv = m->ne[0];
        const int64_t row  = m->ne[1] - 1;
        int kept = 0;
        for (int64_t j = 0; j < n_kv; ++j) {
            float v;
            if (m->type == GGML_TYPE_F16) {
                v = ggml_fp16_to_fp32(((const ggml_fp16_t *) m->data)[row*n_kv + j]);
            } else {
                v = ((const float *) m->data)[row*n_kv + j];
            }
            kept += v == 0.0f;
        }
        probe->kept[m] = kept;
    }
    return true;
}

struct ctx_opts {
    uint32_t n_ctx      = 1024;
    uint32_t n_batch    = 256;
    uint32_t n_ubatch   = 64;
    uint32_t n_seq_max  = 1;
    uint32_t n_rs_seq   = 0;
    bool     swa_full   = true;
    bool     kv_unified = true;
};

struct runner {
    llama_model * model = nullptr;
    int n_vocab = 0;
    int n_swa   = 0;
    bool spans  = false; // native state spans (LFM)
    bool recr   = false;
    mask_probe probe;

    llama_context * ctx(const ctx_opts & o) {
        auto p = llama_context_default_params();
        p.n_ctx       = o.n_ctx;
        p.n_batch     = o.n_batch;
        p.n_ubatch    = o.n_ubatch;
        p.n_seq_max   = o.n_seq_max;
        p.n_rs_seq    = o.n_rs_seq;
        p.swa_full    = o.swa_full;
        p.kv_unified  = o.kv_unified;
        p.n_threads   = 4;
        p.n_threads_batch = 4;
        p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        p.cb_eval           = probe_cb;
        p.cb_eval_user_data = &probe;
        return llama_init_from_model(model, p);
    }
};

int g_n_vocab = 0; // of the model under test

std::vector<llama_token> text(int n, int seed) {
    std::mt19937 rng(seed);
    std::vector<llama_token> out(n);
    for (auto & t : out) {
        t = 1 + (llama_token) (rng() % (uint32_t) std::min(200, g_n_vocab - 1));
    }
    return out;
}

// decode tokens at pos0.. (or all at LLAMA_POS_MEMORY) in chunks of `chunk`; returns the last chunk's decode status
int decode(llama_context * ctx, const std::vector<llama_token> & toks, llama_pos pos0, llama_seq_id seq,
           bool memory = false, int chunk = 0, std::vector<float> * logits = nullptr, int n_vocab = 0) {
    if (chunk <= 0) {
        chunk = (int) toks.size();
    }
    for (size_t i0 = 0; i0 < toks.size(); i0 += chunk) {
        const int n = (int) std::min<size_t>(chunk, toks.size() - i0);
        llama_batch b = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) {
            b.token[i]     = toks[i0 + i];
            b.pos[i]       = memory ? LLAMA_POS_MEMORY : pos0 + (llama_pos) (i0 + i);
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = seq;
            b.logits[i]    = !memory && i0 + i + 1 == toks.size();
        }
        b.n_tokens = n;
        const int rc = llama_decode(ctx, b);
        llama_batch_free(b);
        if (rc != 0) {
            return rc;
        }
        if (logits && !memory && i0 + n == toks.size()) {
            const float * l = llama_get_logits_ith(ctx, -1);
            logits->assign(l, l + n_vocab);
        }
    }
    return 0;
}

double max_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) {
        return INFINITY;
    }
    double d = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d = std::max(d, (double) std::fabs(a[i] - b[i]));
    }
    return d;
}

// memory tokens: virtual span tokens on LFM (stored K/V registered, convolution rows zero or random), else vocabulary
std::vector<llama_token> memory_tokens(runner & r, llama_context * ctx, int m, bool random_conv, int seed) {
    if (!r.spans) {
        return text(m, 1000 + seed);
    }
    const int n_layer = llama_model_n_layer(r.model);
    const int n_embd  = llama_model_n_embd(r.model);
    std::mt19937 rng(seed), rng_conv(seed + 1); // the same K/V whatever the convolution rows
    std::normal_distribution<float> dist(0.0f, 1.0f);
    // k/v rows are n_embd_k_gqa(il) wide, at most 2*n_embd for these models: the buffers are sized generously
    static std::vector<std::vector<float>> keep;
    std::vector<llama_state_span_layer> layers(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        keep.emplace_back((size_t) m*n_embd*2);
        for (auto & x : keep.back()) { x = dist(rng); }
        keep.emplace_back((size_t) m*n_embd*2);
        for (auto & x : keep.back()) { x = dist(rng); }
        keep.emplace_back((size_t) m*n_embd, 0.0f);
        if (random_conv) {
            for (auto & x : keep.back()) { x = dist(rng_conv); }
        }
        layers[il] = { keep[keep.size() - 3].data(), keep[keep.size() - 2].data(), keep.back().data() };
    }
    const llama_token first = llama_state_span_register(ctx, m, layers.data(), n_layer);
    std::vector<llama_token> out;
    if (first == LLAMA_TOKEN_NULL) {
        return out;
    }
    for (int i = 0; i < m; ++i) {
        out.push_back(first + i);
    }
    return out;
}

int g_failures = 0;

void check(bool ok, const char * what, const std::string & detail = "") {
    printf("  %s  %s%s%s\n", ok ? "PASS" : "FAIL", what, detail.empty() ? "" : ": ", detail.c_str());
    g_failures += !ok;
}

std::string fmt(const char * f, double a, double b = 0) {
    char buf[256];
    snprintf(buf, sizeof(buf), f, a, b);
    return buf;
}

// kept cells per mask for one probed decode of `tok` at `pos` in `seq`
std::vector<int> probe_masks(runner & r, llama_context * ctx, llama_token tok, llama_pos pos, llama_seq_id seq,
                             std::vector<float> * logits = nullptr) {
    r.probe.kept.clear();
    r.probe.on = true;
    const int rc = decode(ctx, { tok }, pos, seq, false, 0, logits, r.n_vocab);
    r.probe.on = false;
    std::vector<int> out;
    if (rc != 0) {
        return out;
    }
    for (const auto & [_, kept] : r.probe.kept) {
        out.push_back(kept);
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::string counts(const std::vector<int> & v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        s += (i ? " " : "") + std::to_string(v[i]);
    }
    return s + "]";
}

void test_model(const char * path) {
    auto mp = llama_model_default_params();
    runner r;
    r.model = llama_model_load_from_file(path, mp);
    if (!r.model) {
        check(false, "load", path);
        return;
    }
    r.n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(r.model));
    g_n_vocab = r.n_vocab;
    r.n_swa   = llama_model_n_swa(r.model);
    r.recr    = llama_model_is_recurrent(r.model) || llama_model_is_hybrid(r.model);
    char desc[256];
    llama_model_desc(r.model, desc, sizeof(desc));
    printf("%s (%s): n_swa %d, recurrent %d\n", path, desc, r.n_swa, (int) r.recr);
    if (r.n_swa <= 0) {
        check(false, "the test model has a sliding window");
        llama_model_free(r.model);
        return;
    }

    const int M      = 4;
    const int N_long = std::max(3*r.n_swa + 5, 400); // past the SWA cache (n_swa + n_ubatch, padded to 256)
    const auto T     = text(N_long + 1, 7);
    const std::vector<llama_token> T_long(T.begin(), T.begin() + N_long);
    const uint32_t rs = r.recr ? 8 : 0; // recurrent snapshots: a partial seq_rm of up to 8 tokens
    r.spans = r.recr; // the LFM test models: memory is a registered state span

    // 1. a batch is memory or positioned tokens, never both; memory is not on the timeline
    {
        ctx_opts o; o.n_rs_seq = rs;
        auto * ctx = r.ctx(o);
        const auto mem = memory_tokens(r, ctx, M, false, 1);
        llama_batch b = llama_batch_init(2, 0, 1);
        b.n_tokens = 2;
        b.token[0] = mem.empty() ? 1 : mem[0]; b.pos[0] = LLAMA_POS_MEMORY;
        b.token[1] = T[0];                     b.pos[1] = 0;
        for (int i = 0; i < 2; ++i) { b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = i == 1; }
        check(llama_decode(ctx, b) != 0, "a batch mixing memory and positioned tokens is refused");
        llama_batch_free(b);

        auto * m = llama_get_memory(ctx);
        check(decode(ctx, mem, 0, 0, true) == 0, "a memory batch decodes");
        check(llama_memory_seq_pos_min(m, 0) == -1 && llama_memory_seq_pos_max(m, 0) == -1,
              "memory alone leaves the sequence's positions empty",
              fmt("min %.0f max %.0f", llama_memory_seq_pos_min(m, 0), llama_memory_seq_pos_max(m, 0)));
        check(decode(ctx, std::vector<llama_token>(T.begin(), T.begin() + 10), 0, 0) == 0 &&
              llama_memory_seq_pos_max(m, 0) == 9, "positioned tokens after memory start at 0");
        const auto mem2 = memory_tokens(r, ctx, 2, false, 2);
        check(decode(ctx, mem2, 0, 0, true) == 0 && llama_memory_seq_pos_max(m, 0) == 9,
              "memory after positioned tokens leaves the last position");
        check(decode(ctx, { T[10] }, 10, 0) == 0, "the next positioned token continues at 10");
        llama_free(ctx);
    }

    // 2. memory is visible to every query, the sliding window included, with and without SWA eviction; the evicting
    //    cache equals the full one
    std::vector<float> l_full;
    for (bool swa_full : { true, false }) {
        ctx_opts o; o.swa_full = swa_full; o.n_ubatch = 16; o.n_batch = 16; o.n_rs_seq = rs;
        auto * ctx  = r.ctx(o);
        auto * ctx0 = r.ctx(o);
        const auto mem = memory_tokens(r, ctx, M, false, 3);
        std::vector<float> l_mem, l_none;
        const bool ok = decode(ctx, mem, 0, 0, true) == 0 && decode(ctx, T_long, 0, 0, false, 16) == 0 &&
                        decode(ctx0, T_long, 0, 0, false, 16) == 0;
        const auto k_mem  = probe_masks(r, ctx,  T[N_long], N_long, 0, &l_mem);
        const auto k_none = probe_masks(r, ctx0, T[N_long], N_long, 0, &l_none);
        bool adds = ok && !k_mem.empty() && k_mem.size() == k_none.size() && k_mem.size() >= 2;
        for (size_t i = 0; adds && i < k_mem.size(); ++i) {
            adds = k_mem[i] - k_none[i] == M;
        }
        check(adds, swa_full ? "memory adds its cells to every mask (full SWA cache)" : "memory adds its cells to every mask after SWA eviction",
              "with " + counts(k_mem) + " without " + counts(k_none));
        check(std::isfinite(max_diff(l_mem, l_none)) && max_diff(l_mem, l_none) > 1e-6, "memory changes the logits", fmt("max |d| %.3g", max_diff(l_mem, l_none)));
        if (swa_full) {
            l_full = l_mem;
        } else {
            const double d = max_diff(l_mem, l_full);
            check(d < 1e-4, "the evicting SWA cache equals the full one with memory", fmt("max |d| %.3g", d));
        }
        llama_free(ctx0);

        // 3. a state saved and restored keeps the memory (state_write exempts it from the SWA filter)
        if (!swa_full) {
            auto * ctx2 = r.ctx(o);
            const auto mem2 = memory_tokens(r, ctx2, M, false, 3);
            decode(ctx2, mem2, 0, 0, true);
            decode(ctx2, T_long, 0, 0, false, 16);
            std::vector<uint8_t> state(llama_state_seq_get_size(ctx2, 0));
            const size_t n = llama_state_seq_get_data(ctx2, state.data(), state.size(), 0);
            llama_memory_seq_rm(llama_get_memory(ctx2), 0, -1, -1);
            const size_t m = llama_state_seq_set_data(ctx2, state.data(), n, 0);
            std::vector<float> l_restored;
            const auto k_restored = probe_masks(r, ctx2, T[N_long], N_long, 0, &l_restored);
            check(n > 0 && m == n && k_restored == k_mem && max_diff(l_restored, l_mem) < 1e-4,
                  "a saved and restored sequence keeps its memory",
                  counts(k_restored) + fmt(" max |d| %.3g", max_diff(l_restored, l_mem)));
            llama_free(ctx2);
        }
        llama_free(ctx);
    }

    // 4. seq_rm: a whole-sequence clear removes memory, [LLAMA_POS_MEMORY, 0) removes memory alone, a partial
    //    removal keeps it
    {
        ctx_opts o; o.n_rs_seq = rs;
        const std::vector<llama_token> T20(T.begin(), T.begin() + 20);
        std::vector<float> l_text, l_cleared, l_mem, l_again;
        auto * ctx0 = r.ctx(o);
        decode(ctx0, std::vector<llama_token>(T.begin(), T.begin() + 10), 0, 0, false, 0, &l_text, r.n_vocab);
        llama_free(ctx0);

        auto * ctx = r.ctx(o);
        auto * m = llama_get_memory(ctx);
        const auto mem = memory_tokens(r, ctx, M, false, 4);
        decode(ctx, mem, 0, 0, true);
        decode(ctx, T20, 0, 0, false, 0, &l_mem, r.n_vocab);
        check(llama_memory_seq_rm(m, 0, 14, -1) && llama_memory_seq_pos_max(m, 0) == 13 &&
              decode(ctx, std::vector<llama_token>(T.begin() + 14, T.begin() + 20), 14, 0, false, 0, &l_again, r.n_vocab) == 0 &&
              max_diff(l_again, l_mem) < 1e-5, "a partial seq_rm keeps memory", fmt("max |d| %.3g", max_diff(l_again, l_mem)));

        const auto with = probe_masks(r, ctx, T[20], 20, 0);
        llama_memory_seq_rm(m, 0, 20, -1);
        check(llama_memory_seq_rm(m, 0, LLAMA_POS_MEMORY, 0) && llama_memory_seq_pos_max(m, 0) == 19,
              "seq_rm [LLAMA_POS_MEMORY, 0) keeps the positioned tokens");
        const auto without = probe_masks(r, ctx, T[20], 20, 0);
        bool gone = with.size() == without.size() && !with.empty();
        for (size_t i = 0; gone && i < with.size(); ++i) {
            gone = with[i] - without[i] == M;
        }
        check(gone, "seq_rm [LLAMA_POS_MEMORY, 0) removes the memory", counts(with) + " -> " + counts(without));

        const auto mem2 = memory_tokens(r, ctx, M, false, 5);
        decode(ctx, mem2, 0, 0, true);
        llama_memory_seq_rm(m, 0, -1, -1);
        decode(ctx, std::vector<llama_token>(T.begin(), T.begin() + 10), 0, 0, false, 0, &l_cleared, r.n_vocab);
        check(max_diff(l_cleared, l_text) == 0.0, "a whole-sequence seq_rm removes memory (then equals a fresh decode)",
              fmt("max |d| %.3g", max_diff(l_cleared, l_text)));
        llama_free(ctx);
    }

    // 5. seq_cp from the start copies memory (a whole-sequence copy on hybrids, whose recurrent state cannot be copied
    //    in part; a prefix copy elsewhere)
    {
        ctx_opts o; o.n_seq_max = 2; o.kv_unified = true; o.n_rs_seq = rs;
        auto * ctx = r.ctx(o);
        auto * m = llama_get_memory(ctx);
        const auto mem = memory_tokens(r, ctx, M, false, 6);
        std::vector<float> l0, l1;
        decode(ctx, mem, 0, 0, true);
        decode(ctx, std::vector<llama_token>(T.begin(), T.begin() + 20), 0, 0, false, 0, &l0, r.n_vocab);
        if (r.recr) {
            llama_memory_seq_cp(m, 0, 1, -1, -1);
            const auto k0 = probe_masks(r, ctx, T[20], 20, 0, &l0);
            const auto k1 = probe_masks(r, ctx, T[20], 20, 1, &l1);
            check(k0 == k1 && max_diff(l0, l1) < 1e-5, "a whole-sequence seq_cp copies memory",
                  counts(k0) + " " + counts(k1) + fmt(" max |d| %.3g", max_diff(l0, l1)));
        } else {
            llama_memory_seq_cp(m, 0, 1, -1, 10);
            check(decode(ctx, std::vector<llama_token>(T.begin() + 10, T.begin() + 20), 10, 1, false, 0, &l1, r.n_vocab) == 0 &&
                  max_diff(l0, l1) < 1e-5, "a prefix seq_cp copies memory", fmt("max |d| %.3g", max_diff(l0, l1)));
        }
        llama_free(ctx);
    }

    // 6. LFM: memory is RoPE-free (keys rotated at 0) and leaves the convolution state: stored state at
    //    LLAMA_POS_MEMORY equals the same state at position 0 (zero convolution rows) in the window, whatever its
    //    convolution rows
    if (r.spans) {
        ctx_opts o; o.n_rs_seq = rs;
        const std::vector<llama_token> T10(T.begin(), T.begin() + 10);
        std::vector<float> l_pos0, l_mem, l_mem_conv, l_pos0_conv;
        auto run = [&](bool at_memory, bool random_conv, std::vector<float> & out) {
            auto * ctx = r.ctx(o);
            const auto mem = memory_tokens(r, ctx, M, random_conv, 8); // same K/V draws for every arm
            int rc = mem.empty() ? -100 : 0;
            if (rc == 0 && at_memory) {
                rc = decode(ctx, mem, 0, 0, true);
            } else if (rc == 0) {
                llama_batch b = llama_batch_init(M, 0, 1);
                for (int i = 0; i < M; ++i) {
                    b.token[i] = mem[i]; b.pos[i] = 0; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = false;
                }
                b.n_tokens = M;
                rc = llama_decode(ctx, b);
                llama_batch_free(b);
            }
            if (rc == 0) {
                rc = decode(ctx, T10, 1, 0, false, 0, &out, r.n_vocab);
            }
            llama_free(ctx);
            return rc;
        };
        const int rc = run(false, false, l_pos0) | run(true, false, l_mem) | run(true, true, l_mem_conv) | run(false, true, l_pos0_conv);
        check(rc == 0 && max_diff(l_mem, l_pos0) == 0.0, "memory equals the state at position 0: unrotated, visible",
              fmt("max |d| %.3g", max_diff(l_mem, l_pos0)));
        check(rc == 0 && max_diff(l_mem_conv, l_mem) == 0.0, "memory leaves the convolution state as it was",
              fmt("max |d| %.3g", max_diff(l_mem_conv, l_mem)));
        check(rc == 0 && std::isfinite(max_diff(l_pos0_conv, l_pos0)) && max_diff(l_pos0_conv, l_pos0) > 1e-6, "(control) positioned state's convolution rows matter",
              fmt("max |d| %.3g", max_diff(l_pos0_conv, l_pos0)));
    }

    llama_model_free(r.model);
}

int dump(const char * out, const char * path) {
    auto mp = llama_model_default_params();
    runner r;
    r.model = llama_model_load_from_file(path, mp);
    if (!r.model) {
        return 1;
    }
    r.n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(r.model));
    g_n_vocab = r.n_vocab;
    r.n_swa   = llama_model_n_swa(r.model);
    const bool recr = llama_model_is_recurrent(r.model) || llama_model_is_hybrid(r.model);
    const int N = std::max(3*r.n_swa + 5, 400);
    const auto T = text(N + 8, 11);
    std::vector<float> all, l;
    auto add = [&]() { all.insert(all.end(), l.begin(), l.end()); };

    // a single sequence through an evicting SWA cache: chunked prompt, partial seq_rm, state save and restore
    ctx_opts o; o.swa_full = false; o.n_ubatch = 16; o.n_batch = 16;
    o.n_rs_seq = recr ? 8 : 0;
    auto * ctx = r.ctx(o);
    auto * m = llama_get_memory(ctx);
    int rc = 0;
    for (int i0 = 0; i0 < N; i0 += 7) {
        const int n = std::min(7, N - i0);
        rc |= decode(ctx, std::vector<llama_token>(T.begin() + i0, T.begin() + i0 + n), i0, 0, false, 0, &l, r.n_vocab); add();
    }
    rc |= !llama_memory_seq_rm(m, 0, N - 5, -1); // within the 8 recurrent snapshots
    rc |= decode(ctx, std::vector<llama_token>(T.begin() + N - 5, T.begin() + N + 1), N - 5, 0, false, 0, &l, r.n_vocab); add();
    std::vector<uint8_t> state(llama_state_seq_get_size(ctx, 0));
    const size_t n = llama_state_seq_get_data(ctx, state.data(), state.size(), 0);
    rc |= !llama_memory_seq_rm(m, 0, -1, -1);
    rc |= llama_state_seq_set_data(ctx, state.data(), n, 0) != n;
    rc |= decode(ctx, { T[N + 1] }, N + 1, 0, false, 0, &l, r.n_vocab); add();
    llama_free(ctx);

    // two sequences sharing a prefix (seq_cp; a full SWA cache: shared cells are never evicted)
    ctx_opts o2; o2.n_seq_max = 2; o2.kv_unified = true;
    ctx = r.ctx(o2);
    m = llama_get_memory(ctx);
    rc |= decode(ctx, std::vector<llama_token>(T.begin(), T.begin() + N), 0, 0, false, 64, &l, r.n_vocab); add();
    llama_memory_seq_cp(m, 0, 1, -1, -1);
    rc |= decode(ctx, { T[N] }, N, 1, false, 0, &l, r.n_vocab); add();
    rc |= decode(ctx, { T[N + 2] }, N, 0, false, 0, &l, r.n_vocab); add();
    llama_free(ctx);
    if (rc != 0) {
        fprintf(stderr, "dump: a step failed\n");
        return 1;
    }
    llama_model_free(r.model);

    FILE * f = fopen(out, "wb");
    if (!f) {
        return 1;
    }
    fwrite(all.data(), sizeof(float), all.size(), f);
    fclose(f);
    printf("%s: %zu logits\n", out, all.size());
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fputs(text, stderr);
        }
    }, nullptr);
    if (argc >= 4 && strcmp(argv[1], "--dump") == 0) {
        return dump(argv[2], argv[3]);
    }
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.gguf [MODEL.gguf ...] | --dump OUT MODEL.gguf\n", argv[0]);
        return 2;
    }
    for (int i = 1; i < argc; ++i) {
        test_model(argv[i]);
    }
    printf("%s (%d failures)\n", g_failures ? "FAILED" : "OK", g_failures);
    llama_backend_free();
    return g_failures ? 1 : 0;
}
