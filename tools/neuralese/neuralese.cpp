#include "neuralese.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "llama-ext.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>

struct nz_heads {
    ggml_context * wctx = nullptr;
    gguf_context * gctx = nullptr;
    int n_threads = 4;

    int32_t n_embd     = 0;
    int32_t n_vocab    = 0;
    int32_t cutoff     = 0;
    int32_t max_length = 0;
    int32_t pos_dim    = 0;
    float   eps        = 1e-5f;
    float   tau        = 1.0f;
    float   gate       = 1.0f;
    llama_token open_token  = -1;
    llama_token close_token = -1;
    std::string dialect;

    ggml_tensor * interface_w     = nullptr;
    ggml_tensor * fb_readout_norm = nullptr;
    ggml_tensor * fb_readout      = nullptr; // [n_embd, n_vocab]
    ggml_tensor * fb_table_t      = nullptr; // [n_vocab, n_embd]: the mixture table, transposed
    ggml_tensor * fb_mlp_norm     = nullptr;
    ggml_tensor * fb_mlp_in_w     = nullptr;
    ggml_tensor * fb_mlp_in_b     = nullptr;
    ggml_tensor * fb_mlp_out_w    = nullptr;
    ggml_tensor * fb_mlp_out_b    = nullptr;
    ggml_tensor * stop_norm       = nullptr;
    ggml_tensor * stop_position   = nullptr; // [pos_dim, max_length + 1]
    ggml_tensor * stop_in_w       = nullptr;
    ggml_tensor * stop_in_b       = nullptr;
    ggml_tensor * stop_out_w      = nullptr;
    ggml_tensor * stop_out_b      = nullptr;
    ggml_tensor * ct_norm         = nullptr;
    ggml_tensor * ct_proj_w       = nullptr;
    ggml_tensor * ct_proj_b       = nullptr;
    ggml_tensor * ct_sigma_w      = nullptr;
    ggml_tensor * ct_sigma_b      = nullptr;
};

static ggml_tensor * get_tensor(nz_heads * h, const char * name) {
    ggml_tensor * t = ggml_get_tensor(h->wctx, name);
    if (!t) {
        fprintf(stderr, "%s: missing tensor %s\n", __func__, name);
    } else if (t->type != GGML_TYPE_F32) {
        fprintf(stderr, "%s: tensor %s must be F32\n", __func__, name);
        return nullptr;
    }
    return t;
}

static bool get_u32(const gguf_context * g, const char * key, uint32_t & out) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) {
        fprintf(stderr, "nz_heads_load: missing key %s\n", key);
        return false;
    }
    out = gguf_get_val_u32(g, id);
    return true;
}

static bool get_f32(const gguf_context * g, const char * key, float & out) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) {
        fprintf(stderr, "nz_heads_load: missing key %s\n", key);
        return false;
    }
    out = gguf_get_val_f32(g, id);
    return true;
}

nz_heads * nz_heads_load(const char * path, const nz_heads_params & params) {
    auto * h = new nz_heads();
    h->n_threads = params.n_threads;
    gguf_init_params gp = { /*.no_alloc =*/ false, /*.ctx =*/ &h->wctx };
    h->gctx = gguf_init_from_file(path, gp);
    if (!h->gctx) {
        fprintf(stderr, "%s: failed to read %s\n", __func__, path);
        delete h;
        return nullptr;
    }
    uint32_t n_embd = 0, cutoff = 0, max_length = 0, open_id = 0, close_id = 0;
    bool ok = get_u32(h->gctx, "neuralese.n_embd", n_embd) &&
              get_u32(h->gctx, "neuralese.cutoff", cutoff) &&
              get_u32(h->gctx, "neuralese.max_length", max_length) &&
              get_u32(h->gctx, "neuralese.open_token_id", open_id) &&
              get_u32(h->gctx, "neuralese.close_token_id", close_id) &&
              get_f32(h->gctx, "neuralese.norm_eps", h->eps) &&
              get_f32(h->gctx, "neuralese.feedback.tau", h->tau);
    const int64_t did = gguf_find_key(h->gctx, "neuralese.dialect");
    h->dialect = did >= 0 ? gguf_get_val_str(h->gctx, did) : "";
    h->n_embd = (int32_t) n_embd;
    h->cutoff = (int32_t) cutoff;
    h->max_length = (int32_t) max_length;
    h->open_token = (llama_token) open_id;
    h->close_token = (llama_token) close_id;

    struct { ggml_tensor ** dst; const char * name; } tensors[] = {
        { &h->interface_w,     "nz.interface.weight" },
        { &h->fb_readout_norm, "nz.feedback.readout_norm.weight" },
        { &h->fb_readout,      "nz.feedback.readout.weight" },
        { &h->fb_table_t,      "nz.feedback.table_t" },
        { &h->fb_mlp_norm,     "nz.feedback.mlp_norm.weight" },
        { &h->fb_mlp_in_w,     "nz.feedback.mlp_in.weight" },
        { &h->fb_mlp_in_b,     "nz.feedback.mlp_in.bias" },
        { &h->fb_mlp_out_w,    "nz.feedback.mlp_out.weight" },
        { &h->fb_mlp_out_b,    "nz.feedback.mlp_out.bias" },
        { &h->stop_norm,       "nz.stop.norm.weight" },
        { &h->stop_position,   "nz.stop.position.weight" },
        { &h->stop_in_w,       "nz.stop.mlp_in.weight" },
        { &h->stop_in_b,       "nz.stop.mlp_in.bias" },
        { &h->stop_out_w,      "nz.stop.mlp_out.weight" },
        { &h->stop_out_b,      "nz.stop.mlp_out.bias" },
        { &h->ct_norm,         "nz.content.norm.weight" },
        { &h->ct_proj_w,       "nz.content.proj.weight" },
        { &h->ct_proj_b,       "nz.content.proj.bias" },
        { &h->ct_sigma_w,      "nz.content.log_sigma.weight" },
        { &h->ct_sigma_b,      "nz.content.log_sigma.bias" },
    };
    for (auto & t : tensors) {
        *t.dst = ok ? get_tensor(h, t.name) : nullptr;
        ok = ok && *t.dst != nullptr;
    }
    ggml_tensor * gate = ok ? get_tensor(h, "nz.feedback.gate") : nullptr;
    ok = ok && gate != nullptr;
    if (!ok) {
        nz_heads_free(h);
        return nullptr;
    }
    h->gate    = ((const float *) gate->data)[0];
    h->n_vocab = (int32_t) h->fb_readout->ne[1];
    h->pos_dim = (int32_t) h->stop_position->ne[0];
    return h;
}

void nz_heads_free(nz_heads * heads) {
    if (!heads) {
        return;
    }
    if (heads->gctx) {
        gguf_free(heads->gctx);
    }
    if (heads->wctx) {
        ggml_free(heads->wctx);
    }
    delete heads;
}

int32_t      nz_heads_n_embd     (const nz_heads * h) { return h->n_embd; }
int32_t      nz_heads_cutoff     (const nz_heads * h) { return h->cutoff; }
int32_t      nz_heads_max_length (const nz_heads * h) { return h->max_length; }
llama_token  nz_heads_open_token (const nz_heads * h) { return h->open_token; }
llama_token  nz_heads_close_token(const nz_heads * h) { return h->close_token; }
const char * nz_heads_dialect    (const nz_heads * h) { return h->dialect.c_str(); }

//
// head computation: small CPU graphs over the projector weights
//

namespace {

struct scratch {
    ggml_context * ctx = nullptr;
    explicit scratch(size_t bytes) {
        ggml_init_params p = { bytes, nullptr, false };
        ctx = ggml_init(p);
    }
    ~scratch() { ggml_free(ctx); }
};

ggml_tensor * rms(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    return b ? ggml_add(ctx, y, b) : y;
}

ggml_tensor * input(ggml_context * ctx, const float * data, int64_t n_embd, int64_t n) {
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n);
    memcpy(t->data, data, ggml_nbytes(t));
    return t;
}

void compute(ggml_context * ctx, ggml_tensor * out, int n_threads) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);
}

size_t scratch_bytes(const nz_heads * h, int64_t n) {
    return (size_t) 64 * 1024 * 1024 + (size_t) n * (h->n_vocab + 16 * h->n_embd) * sizeof(float) * 4;
}

} // namespace

void nz_interface(const nz_heads * h, const float * in, float * out, int32_t n) {
    if (n <= 0) {
        return;
    }
    scratch s(scratch_bytes(h, n));
    ggml_tensor * y = rms(s.ctx, input(s.ctx, in, h->n_embd, n), h->interface_w, h->eps);
    compute(s.ctx, y, h->n_threads);
    memcpy(out, y->data, ggml_nbytes(y));
}

void nz_feedback(const nz_heads * h, const float * h_cut, float * sketch) {
    scratch s(scratch_bytes(h, 1));
    ggml_context * ctx = s.ctx;
    ggml_tensor * x = input(ctx, h_cut, h->n_embd, 1);
    // vocabulary mixture: E^T softmax(R(norm(h)) / tau)
    ggml_tensor * logits  = linear(ctx, rms(ctx, x, h->fb_readout_norm, h->eps), h->fb_readout, nullptr);
    ggml_tensor * weights = ggml_soft_max_ext(ctx, logits, nullptr, 1.0f / h->tau, 0.0f);
    ggml_tensor * mixture = ggml_mul_mat(ctx, h->fb_table_t, weights);
    // residual MLP branch
    ggml_tensor * m = rms(ctx, x, h->fb_mlp_norm, h->eps);
    m = ggml_gelu_erf(ctx, linear(ctx, m, h->fb_mlp_in_w, h->fb_mlp_in_b));
    m = linear(ctx, m, h->fb_mlp_out_w, h->fb_mlp_out_b);
    ggml_tensor * y = ggml_add(ctx, mixture, ggml_scale(ctx, m, h->gate));
    y = rms(ctx, y, h->interface_w, h->eps);
    compute(ctx, y, h->n_threads);
    memcpy(sketch, y->data, ggml_nbytes(y));
}

float nz_stop_logit(const nz_heads * h, const float * h_cut, int32_t count) {
    scratch s(scratch_bytes(h, 1));
    ggml_context * ctx = s.ctx;
    const int32_t row = std::min(count, h->max_length);
    ggml_tensor * x = rms(ctx, input(ctx, h_cut, h->n_embd, 1), h->stop_norm, h->eps);
    ggml_tensor * p = ggml_view_2d(ctx, h->stop_position, h->pos_dim, 1, h->stop_position->nb[1],
                                   (size_t) row * h->stop_position->nb[1]);
    ggml_tensor * f = ggml_concat(ctx, x, p, 0);
    f = ggml_gelu_erf(ctx, linear(ctx, f, h->stop_in_w, h->stop_in_b));
    f = linear(ctx, f, h->stop_out_w, h->stop_out_b);
    compute(ctx, f, h->n_threads);
    return ((const float *) f->data)[0];
}

void nz_content(const nz_heads * h, const float * sketches, const float * h_final, int32_t n,
                float * mean, float * log_sigma) {
    if (n <= 0) {
        return;
    }
    scratch s(scratch_bytes(h, n));
    ggml_context * ctx = s.ctx;
    ggml_tensor * hf = rms(ctx, input(ctx, h_final, h->n_embd, n), h->ct_norm, h->eps);
    ggml_tensor * mu = ggml_add(ctx, input(ctx, sketches, h->n_embd, n), linear(ctx, hf, h->ct_proj_w, h->ct_proj_b));
    ggml_tensor * ls = ggml_clamp(ctx, linear(ctx, hf, h->ct_sigma_w, h->ct_sigma_b), -9.0f, 2.0f);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, mu);
    ggml_build_forward_expand(gf, ls);
    ggml_graph_compute_with_ctx(ctx, gf, h->n_threads);
    memcpy(mean, mu->data, ggml_nbytes(mu));
    memcpy(log_sigma, ls->data, ggml_nbytes(ls));
}

//
// decoding helpers
//

static int32_t decode_embd(llama_context * ctx, const float * rows, int32_t n, int32_t n_embd,
                           llama_pos pos, llama_seq_id seq, bool logits_last) {
    llama_batch batch = llama_batch_init(n, n_embd, 1);
    batch.n_tokens = n;
    memcpy(batch.embd, rows, (size_t) n * n_embd * sizeof(float));
    for (int32_t i = 0; i < n; ++i) {
        batch.pos[i]       = pos + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = seq;
        batch.logits[i]    = logits_last && i == n - 1;
    }
    const int32_t status = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return status;
}

static int32_t decode_token(llama_context * ctx, llama_token token, llama_pos pos, llama_seq_id seq) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.n_tokens     = 1;
    batch.token[0]     = token;
    batch.pos[0]       = pos;
    batch.n_seq_id[0]  = 1;
    batch.seq_id[0][0] = seq;
    batch.logits[0]    = 1;
    const int32_t status = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return status;
}

int32_t nz_read(llama_context * ctx, const nz_heads * h, const float * payload, int32_t n,
                llama_pos pos, llama_seq_id seq) {
    if (n <= 0) {
        return 0;
    }
    std::vector<float> normed((size_t) n * h->n_embd);
    nz_interface(h, payload, normed.data(), n);
    return decode_embd(ctx, normed.data(), n, h->n_embd, pos, seq, false);
}

int32_t nz_open(llama_context * ctx, const nz_heads * h, llama_pos pos, llama_seq_id seq, float * h_cut) {
    llama_set_layer_range(ctx, 0, -1);
    llama_set_embeddings_layer_inp(ctx, (uint32_t) h->cutoff, true);
    const int32_t status = decode_token(ctx, h->open_token, pos, seq);
    if (status != 0) {
        return status;
    }
    memcpy(h_cut, llama_get_embeddings_layer_inp(ctx, (uint32_t) h->cutoff), (size_t) h->n_embd * sizeof(float));
    return 0;
}

int32_t nz_write(llama_context * ctx, const nz_heads * h, llama_seq_id seq, llama_pos pos_first,
                 const float * h_cut, const nz_write_params & params, nz_write_result & r) {
    const int32_t d       = h->n_embd;
    const int32_t k       = h->cutoff;
    const int32_t n_layer = llama_model_n_layer(llama_get_model(ctx));
    const int32_t max_len = params.max_length < 0 ? h->max_length : std::min(params.max_length, h->max_length);
    llama_memory_t mem    = llama_get_memory(ctx);

    r = nz_write_result();
    std::vector<float> state(h_cut, h_cut + d);
    std::vector<float> sketch(d);

    // sketch: layers [0, k) on a scratch copy of the sequence, one position at a time
    llama_memory_seq_rm(mem, params.seq_sketch, -1, -1);
    llama_memory_seq_cp(mem, seq, params.seq_sketch, -1, -1);
    llama_set_embeddings_layer_inp(ctx, (uint32_t) k, true);
    int32_t status = 0;
    int32_t count  = 0;
    while (true) {
        if (count > 0 || params.allow_empty) {
            const float logit = nz_stop_logit(h, state.data(), count);
            if (count > 0) {
                r.stop_logits.push_back(logit);
            }
            if (logit > 0.0f) {
                break;
            }
        }
        if (count == max_len) {
            r.truncated = true;
            break;
        }
        nz_feedback(h, state.data(), sketch.data());
        llama_set_layer_range(ctx, 0, k);
        status = decode_embd(ctx, sketch.data(), 1, d, pos_first + count, params.seq_sketch, false);
        llama_set_layer_range(ctx, 0, -1);
        if (status != 0) {
            break;
        }
        const float * h_k = llama_get_embeddings_layer_inp(ctx, (uint32_t) k);
        r.sketches.insert(r.sketches.end(), sketch.begin(), sketch.end());
        r.shallow.insert(r.shallow.end(), h_k, h_k + d);
        state.assign(h_k, h_k + d);
        count++;
    }
    llama_memory_seq_rm(mem, params.seq_sketch, -1, -1);
    r.n = count;
    if (status != 0 || count == 0) {
        return status;
    }

    // complete: layers [k, n_layer) over the shallow residuals, from the block start
    llama_memory_seq_rm(mem, params.seq_complete, -1, -1);
    llama_memory_seq_cp(mem, seq, params.seq_complete, -1, -1);
    llama_set_embeddings_layer_inp(ctx, (uint32_t) n_layer, true);
    llama_set_layer_range(ctx, k, -1);
    status = decode_embd(ctx, r.shallow.data(), count, d, pos_first, params.seq_complete, false);
    llama_set_layer_range(ctx, 0, -1);
    if (status == 0) {
        const float * h_d = llama_get_embeddings_layer_inp(ctx, (uint32_t) n_layer);
        r.final_h.assign(h_d, h_d + (size_t) count * d);
    }
    llama_set_embeddings_layer_inp(ctx, (uint32_t) n_layer, false);
    llama_memory_seq_rm(mem, params.seq_complete, -1, -1);
    if (status != 0) {
        return status;
    }

    // project: payload mean and scale; z = mu + rms(mu) * tau * sigma * eps
    r.mean.resize((size_t) count * d);
    r.log_sigma.resize((size_t) count * d);
    nz_content(h, r.sketches.data(), r.final_h.data(), count, r.mean.data(), r.log_sigma.data());
    r.payload = r.mean;
    if (params.temperature > 0.0f) {
        r.noise.resize((size_t) count * d);
        if (params.noise) {
            std::copy(params.noise, params.noise + r.noise.size(), r.noise.begin());
        } else {
            std::mt19937 rng(params.seed);
            std::normal_distribution<float> normal(0.0f, 1.0f);
            for (auto & e : r.noise) {
                e = normal(rng);
            }
        }
        for (int32_t i = 0; i < count; ++i) {
            const float * mu = r.mean.data() + (size_t) i * d;
            double sq = 0.0;
            for (int32_t j = 0; j < d; ++j) {
                sq += (double) mu[j] * mu[j];
            }
            const float scale = (float) std::sqrt(sq / d + 1e-6);
            for (int32_t j = 0; j < d; ++j) {
                const size_t idx = (size_t) i * d + j;
                r.payload[idx] = mu[j] + scale * params.temperature * std::exp(r.log_sigma[idx]) * r.noise[idx];
            }
        }
    }
    return 0;
}

int32_t nz_read_back(llama_context * ctx, const nz_heads * h, llama_seq_id seq, llama_pos pos_first,
                     const nz_write_result & r) {
    const int32_t status = nz_read(ctx, h, r.payload.data(), r.n, pos_first, seq);
    if (status != 0) {
        return status;
    }
    return decode_token(ctx, h->close_token, pos_first + r.n, seq);
}
