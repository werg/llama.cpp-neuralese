#include "neuralese.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "llama-ext.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

struct nz_heads {
    ggml_context * wctx = nullptr;
    gguf_context * gctx = nullptr;
    int n_threads = 4;

    // the heads run on one ggml backend: the GPU when requested and available, else the CPU
    ggml_backend_t         backend = nullptr;
    ggml_backend_buffer_t  wbuf    = nullptr;
    mutable ggml_gallocr_t galloc  = nullptr;

    int32_t n_embd     = 0;
    int32_t n_vocab    = 0;
    int32_t cutoff     = 0;
    int32_t max_length = 0;
    int32_t pos_dim    = 0;
    float   eps        = 1e-5f;
    float   tau        = 1.0f;
    float   read_adapter_eps = 1e-5f;
    float   gate       = 1.0f;
    llama_token open_token  = -1;
    llama_token close_token = -1;
    std::string dialect;
    // stop source: false = the sketch's shallow residual h_k decides during the recurrence; true ("final") = the
    // completed residual h_D at each written position decides (the reference's StopHead on final states)
    bool stop_final    = false;
    bool stop_counts   = true;  // false: the count is not an input; position row 0 stands in for it

    ggml_tensor * interface_w     = nullptr;
    ggml_tensor * read_adapter_norm = nullptr;
    ggml_tensor * read_adapter_proj_w = nullptr;
    ggml_tensor * read_adapter_proj_b = nullptr;
    bool read_adapter = false;
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

    // direct-v1 profile (neuralese.profile "direct-v1"; natlang plans/neuralese/NEURALESE_DIRECT_V1.md): no read
    // markers and no interface norm (a block's vectors are read as input embeddings), and the writer runs the full model:
    // the value at each position is softmax(logits(h))·E plus the causal conv stack's correction over the last final
    // states, and the next position reads the greedy token's embedding.
    bool direct       = false;
    bool read_markers = true;
    struct conv_block {
        std::vector<float> norm, taps, up_w, up_b, down_w, down_b; // taps [kernel][d], up_w [hidden][d], down_w [d][hidden]
    };
    std::vector<conv_block> stack;
    int32_t stack_kernel = 0;
    int32_t stack_hidden = 0;
    float   stack_eps    = 0.0f;
    std::vector<llama_token> stop_ids;  // the readout's stop tokens (the reference engine's stop_ids)
    ggml_tensor * table_t = nullptr;    // [n_vocab, n_embd]: the input embedding matrix, transposed, for the mixture
};

// Matrices that only enter ggml_mul_mat may be stored as F16 or Q8_0 (smaller projector files, e.g. for the browser);
// every other tensor is read as F32.
static bool matmul_weight(const std::string & name) {
    static const char * names[] = { "nz.feedback.readout.weight", "nz.feedback.table_t", "nz.feedback.mlp_in.weight",
        "nz.feedback.mlp_out.weight", "nz.stop.mlp_in.weight", "nz.stop.mlp_out.weight", "nz.content.proj.weight",
        "nz.content.log_sigma.weight", "nz.direct.table_t" };
    for (const char * n : names) if (name == n) return true;
    return false;
}

static ggml_tensor * get_tensor(nz_heads * h, const char * name) {
    ggml_tensor * t = ggml_get_tensor(h->wctx, name);
    if (!t) {
        fprintf(stderr, "%s: missing tensor %s\n", __func__, name);
    } else if (t->type != GGML_TYPE_F32 &&
               !(matmul_weight(name) && (t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_Q8_0))) {
        fprintf(stderr, "%s: tensor %s must be F32%s\n", __func__, name, matmul_weight(name) ? ", F16 or Q8_0" : "");
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

// Allocate the projector's tensors on the heads' backend and read them from the file.
static bool load_weights(nz_heads * h, const char * path, const nz_heads_params & params) {
    if (params.use_gpu) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (dev) {
            h->backend = ggml_backend_dev_init(dev, nullptr);
        }
    }
    if (!h->backend) {
        h->backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(h->backend, h->n_threads);
        if (params.threadpool) ggml_backend_cpu_set_threadpool(h->backend, params.threadpool);
    }
    h->wbuf = ggml_backend_alloc_ctx_tensors(h->wctx, h->backend);
    FILE * fp = h->wbuf ? fopen(path, "rb") : nullptr;
    if (!fp) {
        fprintf(stderr, "nz_heads_load: failed to allocate or read the projector weights\n");
        return false;
    }
    std::vector<uint8_t> buf;
    const size_t data_offset = gguf_get_data_offset(h->gctx);
    for (int64_t i = 0; i < gguf_get_n_tensors(h->gctx); ++i) {
        ggml_tensor * t = ggml_get_tensor(h->wctx, gguf_get_tensor_name(h->gctx, i));
        const size_t n = ggml_nbytes(t);
        buf.resize(n);
        if (fseek(fp, (long) (data_offset + gguf_get_tensor_offset(h->gctx, i)), SEEK_SET) != 0 ||
            fread(buf.data(), 1, n, fp) != n) {
            fprintf(stderr, "nz_heads_load: short read for %s\n", ggml_get_name(t));
            fclose(fp);
            return false;
        }
        ggml_backend_tensor_set(t, buf.data(), 0, n);
    }
    fclose(fp);
    h->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(h->backend));
    return true;
}

// The direct-v1 projector: the conv stack (kept on the host; a few positions per step) and the mixture table.
static bool load_direct(nz_heads * h, const char * path, const nz_heads_params & params) {
    uint32_t blocks = 0, kernel = 0, hidden = 0;
    if (!get_u32(h->gctx, "neuralese.direct.blocks", blocks) || !get_u32(h->gctx, "neuralese.direct.kernel", kernel) ||
        !get_u32(h->gctx, "neuralese.direct.hidden", hidden) || !get_f32(h->gctx, "neuralese.direct.norm_eps", h->stack_eps)) {
        return false;
    }
    const int64_t fid = gguf_find_key(h->gctx, "neuralese.direct.feedback");
    if (fid < 0 || gguf_get_kv_type(h->gctx, fid) != GGUF_TYPE_STRING || std::string(gguf_get_val_str(h->gctx, fid)) != "token") {
        fprintf(stderr, "nz_heads_load: direct-v1 needs neuralese.direct.feedback \"token\"\n");
        return false;
    }
    if (gguf_find_key(h->gctx, "neuralese.direct.token_choice") >= 0 ||
        ggml_get_tensor(h->wctx, "nz.token_choice.blk.0.norm.weight") != nullptr) {
        // The reference applies the TokenChoiceProjection to every text and write readout; not ported yet.
        fprintf(stderr, "nz_heads_load: direct-v1 TokenChoiceProjection is not supported by this runtime yet\n");
        return false;
    }
    const int64_t sid = gguf_find_key(h->gctx, "neuralese.direct.stop_ids");
    if (sid < 0 || gguf_get_kv_type(h->gctx, sid) != GGUF_TYPE_ARRAY || gguf_get_arr_type(h->gctx, sid) != GGUF_TYPE_INT32 ||
        gguf_get_arr_n(h->gctx, sid) == 0) {
        fprintf(stderr, "nz_heads_load: direct-v1 needs neuralese.direct.stop_ids (int32 array)\n");
        return false;
    }
    const int32_t * ids = (const int32_t *) gguf_get_arr_data(h->gctx, sid);
    h->stop_ids.assign(ids, ids + gguf_get_arr_n(h->gctx, sid));
    if (blocks < 1 || kernel < 1 || hidden < 1 || !std::isfinite(h->stack_eps) || h->stack_eps <= 0.0f) {
        fprintf(stderr, "nz_heads_load: invalid direct-v1 stack geometry\n");
        return false;
    }
    h->stack_kernel = (int32_t) kernel;
    h->stack_hidden = (int32_t) hidden;
    h->table_t = get_tensor(h, "nz.direct.table_t");
    if (!h->table_t || h->table_t->ne[1] != h->n_embd) {
        fprintf(stderr, "nz_heads_load: nz.direct.table_t missing or not [n_vocab, n_embd]\n");
        return false;
    }
    h->n_vocab = (int32_t) h->table_t->ne[0];
    if (!load_weights(h, path, params)) {
        return false;
    }
    const int64_t d = h->n_embd;
    auto host = [&](const std::string & name, int64_t ne0, int64_t ne1, std::vector<float> & out) {
        ggml_tensor * t = get_tensor(h, name.c_str());
        if (!t || t->type != GGML_TYPE_F32 || t->ne[0] != ne0 || t->ne[1] != ne1 || ggml_nelements(t) != ne0 * ne1) {
            fprintf(stderr, "nz_heads_load: %s missing or not F32 [%lld, %lld]\n", name.c_str(), (long long) ne1, (long long) ne0);
            return false;
        }
        out.resize((size_t) ne0 * ne1);
        ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
        for (float v : out) if (!std::isfinite(v)) { fprintf(stderr, "nz_heads_load: %s is not finite\n", name.c_str()); return false; }
        return true;
    };
    h->stack.resize(blocks);
    for (uint32_t b = 0; b < blocks; ++b) {
        const std::string p = "nz.direct.blk." + std::to_string(b) + ".";
        auto & blk = h->stack[b];
        if (!host(p + "norm.weight", d, 1, blk.norm) || !host(p + "taps", d, kernel, blk.taps) ||
            !host(p + "up.weight", d, hidden, blk.up_w) || !host(p + "up.bias", hidden, 1, blk.up_b) ||
            !host(p + "down.weight", hidden, d, blk.down_w) || !host(p + "down.bias", d, 1, blk.down_b)) {
            return false;
        }
    }
    return true;
}

nz_heads * nz_heads_load(const char * path, const nz_heads_params & params) {
    auto * h = new nz_heads();
    h->n_threads = params.n_threads;
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &h->wctx };
    h->gctx = gguf_init_from_file(path, gp);
    if (!h->gctx) {
        fprintf(stderr, "%s: failed to read %s\n", __func__, path);
        delete h;
        return nullptr;
    }
    uint32_t n_embd = 0, cutoff = 0, max_length = 0, open_id = 0, close_id = 0;
    const int64_t profile_id = gguf_find_key(h->gctx, "neuralese.profile");
    const std::string profile = profile_id >= 0 && gguf_get_kv_type(h->gctx, profile_id) == GGUF_TYPE_STRING
        ? gguf_get_val_str(h->gctx, profile_id) : "legacy-rms-v1";
    if (profile != "legacy-rms-v1" && profile != "direct-v1") {
        fprintf(stderr, "%s: unsupported neuralese.profile %s\n", __func__, profile.c_str());
        nz_heads_free(h);
        return nullptr;
    }
    h->direct = profile == "direct-v1";
    h->read_markers = !h->direct;
    bool ok = get_u32(h->gctx, "neuralese.n_embd", n_embd) &&
              get_u32(h->gctx, "neuralese.cutoff", cutoff) &&
              get_u32(h->gctx, "neuralese.max_length", max_length) &&
              get_u32(h->gctx, "neuralese.open_token_id", open_id) &&
              get_u32(h->gctx, "neuralese.close_token_id", close_id) &&
              get_f32(h->gctx, "neuralese.norm_eps", h->eps) &&
              (h->direct || get_f32(h->gctx, "neuralese.feedback.tau", h->tau));
    if (ok && h->direct) {
        h->n_embd = (int32_t) n_embd;
        h->cutoff = (int32_t) cutoff;
        h->max_length = (int32_t) max_length;
        h->open_token = (llama_token) open_id;
        h->close_token = (llama_token) close_id;
        const int64_t did = gguf_find_key(h->gctx, "neuralese.dialect");
        h->dialect = did >= 0 ? gguf_get_val_str(h->gctx, did) : "";
        if (!load_direct(h, path, params)) {
            nz_heads_free(h);
            return nullptr;
        }
        return h;
    }
    const int64_t did = gguf_find_key(h->gctx, "neuralese.dialect");
    h->dialect = did >= 0 ? gguf_get_val_str(h->gctx, did) : "";
    // Optional (projectors exported before stop sources existed use the shallow source with the count).
    const int64_t sid = gguf_find_key(h->gctx, "neuralese.stop_source");
    const std::string stop_source = sid >= 0 ? gguf_get_val_str(h->gctx, sid) : "shallow";
    if (stop_source != "shallow" && stop_source != "final") {
        fprintf(stderr, "%s: unknown stop source %s\n", __func__, stop_source.c_str());
        ok = false;
    }
    h->stop_final = stop_source == "final";
    const int64_t pid = gguf_find_key(h->gctx, "neuralese.stop_position");
    h->stop_counts = pid < 0 || gguf_get_val_u32(h->gctx, pid) != 0;
    h->n_embd = (int32_t) n_embd;
    h->cutoff = (int32_t) cutoff;
    h->max_length = (int32_t) max_length;
    h->open_token = (llama_token) open_id;
    h->close_token = (llama_token) close_id;

    const int64_t read_adapter_id = gguf_find_key(h->gctx, "neuralese.read_adapter");
    if (read_adapter_id >= 0) {
        const char * read_adapter_name = gguf_get_kv_type(h->gctx, read_adapter_id) == GGUF_TYPE_STRING
            ? gguf_get_val_str(h->gctx, read_adapter_id) : nullptr;
        if (read_adapter_name == nullptr || std::string(read_adapter_name) != "full-residual-v1") {
            fprintf(stderr, "%s: unsupported neuralese.read_adapter\n", __func__);
            ok = false;
        } else {
            h->read_adapter = true;
            const int64_t eps_id = gguf_find_key(h->gctx, "neuralese.read_adapter.norm_eps");
            if (eps_id < 0 || gguf_get_kv_type(h->gctx, eps_id) != GGUF_TYPE_FLOAT32) {
                fprintf(stderr, "%s: missing or invalid neuralese.read_adapter.norm_eps\n", __func__);
                ok = false;
            } else {
                h->read_adapter_eps = gguf_get_val_f32(h->gctx, eps_id);
                if (!std::isfinite(h->read_adapter_eps) || h->read_adapter_eps <= 0.0f) {
                    fprintf(stderr, "%s: invalid reader adapter norm epsilon\n", __func__);
                    ok = false;
                }
            }
        }
    } else if (gguf_find_key(h->gctx, "neuralese.read_adapter.norm_eps") >= 0 ||
               ggml_get_tensor(h->wctx, "nz.read_adapter.norm.weight") != nullptr ||
               ggml_get_tensor(h->wctx, "nz.read_adapter.proj.weight") != nullptr ||
               ggml_get_tensor(h->wctx, "nz.read_adapter.proj.bias") != nullptr) {
        fprintf(stderr, "%s: reader adapter tensors need neuralese.read_adapter metadata\n", __func__);
        ok = false;
    }

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
    if (ok && h->read_adapter) {
        h->read_adapter_norm = get_tensor(h, "nz.read_adapter.norm.weight");
        h->read_adapter_proj_w = get_tensor(h, "nz.read_adapter.proj.weight");
        h->read_adapter_proj_b = get_tensor(h, "nz.read_adapter.proj.bias");
        ok = h->read_adapter_norm && h->read_adapter_proj_w && h->read_adapter_proj_b;
        if (ok && (ggml_n_dims(h->read_adapter_norm) != 1 || h->read_adapter_norm->ne[0] != h->n_embd ||
                   ggml_n_dims(h->read_adapter_proj_w) != 2 || h->read_adapter_proj_w->ne[0] != h->n_embd ||
                   h->read_adapter_proj_w->ne[1] != h->n_embd || ggml_n_dims(h->read_adapter_proj_b) != 1 ||
                   h->read_adapter_proj_b->ne[0] != h->n_embd)) {
            fprintf(stderr, "%s: reader adapter tensor shape mismatch\n", __func__);
            ok = false;
        }
    }
    ggml_tensor * gate = ok ? get_tensor(h, "nz.feedback.gate") : nullptr;
    ok = ok && gate != nullptr;
    if (!ok) {
        nz_heads_free(h);
        return nullptr;
    }

    if (!load_weights(h, path, params)) {
        nz_heads_free(h);
        return nullptr;
    }
    ggml_backend_tensor_get(gate, &h->gate, 0, sizeof(float));
    h->n_vocab = (int32_t) h->fb_readout->ne[1];
    h->pos_dim = (int32_t) h->stop_position->ne[0];
    return h;
}

void nz_heads_free(nz_heads * heads) {
    if (!heads) {
        return;
    }
    if (heads->galloc) {
        ggml_gallocr_free(heads->galloc);
    }
    if (heads->wbuf) {
        ggml_backend_buffer_free(heads->wbuf);
    }
    if (heads->backend) {
        ggml_backend_free(heads->backend);
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
bool         nz_heads_direct     (const nz_heads * h) { return h->direct; }
bool         nz_heads_read_markers(const nz_heads * h) { return h->read_markers; }

//
// head computation: small graphs over the projector weights, on the heads' backend
//

namespace {

// one graph: inputs are copied in after allocation, outputs are read back after compute
struct graph_run {
    ggml_context * ctx = nullptr;
    std::vector<std::pair<ggml_tensor *, const float *>> inputs;

    graph_run() {
        ggml_init_params p = { ggml_tensor_overhead() * 256 + ggml_graph_overhead(), nullptr, /*no_alloc =*/ true };
        ctx = ggml_init(p);
    }
    ~graph_run() { ggml_free(ctx); }

    ggml_tensor * input(const float * data, int64_t n_embd, int64_t n) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n);
        ggml_set_input(t);
        inputs.emplace_back(t, data);
        return t;
    }

    bool compute(const nz_heads * h, std::initializer_list<ggml_tensor *> outputs) {
        ggml_cgraph * gf = ggml_new_graph(ctx);
        for (ggml_tensor * out : outputs) {
            ggml_set_output(out);
            ggml_build_forward_expand(gf, out);
        }
        if (!ggml_gallocr_alloc_graph(h->galloc, gf)) {
            fprintf(stderr, "nz: failed to allocate a head graph\n");
            return false;
        }
        for (auto & [t, data] : inputs) {
            ggml_backend_tensor_set(t, data, 0, ggml_nbytes(t));
        }
        return ggml_backend_graph_compute(h->backend, gf) == GGML_STATUS_SUCCESS;
    }
};

ggml_tensor * rms(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    return b ? ggml_add(ctx, y, b) : y;
}

void get(ggml_tensor * t, float * out) {
    ggml_backend_tensor_get(t, out, 0, ggml_nbytes(t));
}

} // namespace

void nz_interface(const nz_heads * h, const float * in, float * out, int32_t n) {
    if (n <= 0) {
        return;
    }
    graph_run g;
    ggml_tensor * y = rms(g.ctx, g.input(in, h->n_embd, n), h->interface_w, h->eps);
    if (g.compute(h, { y })) {
        get(y, out);
    }
}

void nz_feedback(const nz_heads * h, const float * h_cut, float * sketch) {
    graph_run g;
    ggml_context * ctx = g.ctx;
    ggml_tensor * x = g.input(h_cut, h->n_embd, 1);
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
    if (g.compute(h, { y })) {
        get(y, sketch);
    }
}

float nz_stop_logit(const nz_heads * h, const float * h_cut, int32_t count) {
    graph_run g;
    ggml_context * ctx = g.ctx;
    const int32_t row = h->stop_counts ? std::min(count, h->max_length) : 0;
    ggml_tensor * x = rms(ctx, g.input(h_cut, h->n_embd, 1), h->stop_norm, h->eps);
    ggml_tensor * p = ggml_view_2d(ctx, h->stop_position, h->pos_dim, 1, h->stop_position->nb[1],
                                   (size_t) row * h->stop_position->nb[1]);
    ggml_tensor * f = ggml_concat(ctx, x, p, 0);
    f = ggml_gelu_erf(ctx, linear(ctx, f, h->stop_in_w, h->stop_in_b));
    f = linear(ctx, f, h->stop_out_w, h->stop_out_b);
    float logit = 0.0f;
    if (g.compute(h, { f })) {
        get(f, &logit);
    }
    return logit;
}

void nz_content(const nz_heads * h, const float * sketches, const float * h_final, int32_t n,
                float * mean, float * log_sigma) {
    if (n <= 0) {
        return;
    }
    graph_run g;
    ggml_context * ctx = g.ctx;
    ggml_tensor * hf = rms(ctx, g.input(h_final, h->n_embd, n), h->ct_norm, h->eps);
    ggml_tensor * mu = ggml_add(ctx, g.input(sketches, h->n_embd, n), linear(ctx, hf, h->ct_proj_w, h->ct_proj_b));
    ggml_tensor * ls = ggml_clamp(ctx, linear(ctx, hf, h->ct_sigma_w, h->ct_sigma_b), -9.0f, 2.0f);
    if (g.compute(h, { mu, ls })) {
        get(mu, mean);
        get(ls, log_sigma);
    }
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
    if (h->direct) {
        // raw identity: the vectors are input embeddings
        return decode_embd(ctx, payload, n, h->n_embd, pos, seq, false);
    }
    std::vector<float> normed((size_t) n * h->n_embd);
    nz_interface(h, payload, normed.data(), n);
    if (h->read_adapter) {
        graph_run g;
        ggml_tensor * x = g.input(normed.data(), h->n_embd, n);
        ggml_tensor * update = linear(g.ctx, rms(g.ctx, x, h->read_adapter_norm, h->read_adapter_eps),
                                      h->read_adapter_proj_w, h->read_adapter_proj_b);
        ggml_tensor * y = ggml_add(g.ctx, x, update);
        if (!g.compute(h, { y })) {
            return -1;
        }
        get(y, normed.data());
    }
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
    const int32_t hint = params.length > 0 ? std::min(params.length, max_len) : 0;
    if (hint > 0 && params.passes > 0) {
        // Block-wise: every position runs through layers [0, k) at once and the sketches are refined by fixed-point
        // iteration, sketch[j] = feedback(h_k[j - 1]), sketch[0] = feedback(h_cut). The first p positions are exact
        // after p passes (causality); passes >= length reproduces the sequential write.
        const int32_t passes = std::min(params.passes, hint);
        std::vector<float> first(d), inputs((size_t) hint * d), shallow((size_t) hint * d);
        nz_feedback(h, h_cut, first.data());
        for (int32_t j = 0; j < hint; ++j) std::copy(first.begin(), first.end(), inputs.begin() + (size_t) j * d);
        for (int32_t p = 1; p <= passes && status == 0; ++p) {
            llama_memory_seq_rm(mem, params.seq_sketch, -1, -1);
            llama_memory_seq_cp(mem, seq, params.seq_sketch, -1, -1);
            llama_set_layer_range(ctx, 0, k);
            status = decode_embd(ctx, inputs.data(), hint, d, pos_first, params.seq_sketch, false);
            llama_set_layer_range(ctx, 0, -1);
            if (status != 0) break;
            const float * h_k = llama_get_embeddings_layer_inp(ctx, (uint32_t) k);
            std::copy(h_k, h_k + (size_t) hint * d, shallow.begin());
            r.passes = p;
            if (p == passes) break;
            std::vector<float> refined(inputs.size());
            std::copy(first.begin(), first.end(), refined.begin());
            for (int32_t j = 1; j < hint; ++j) nz_feedback(h, shallow.data() + (size_t) (j - 1) * d, refined.data() + (size_t) j * d);
            if (refined == inputs) break;
            inputs.swap(refined);
        }
        r.sketches = inputs;
        r.shallow = shallow;
        count = hint;
    } else if (hint > 0) {
        // A size hint, position by position: no stop decisions.
        llama_set_embeddings_layer_inp(ctx, (uint32_t) k, true);
        while (count < hint) {
            nz_feedback(h, state.data(), sketch.data());
            llama_set_layer_range(ctx, 0, k);
            status = decode_embd(ctx, sketch.data(), 1, d, pos_first + count, params.seq_sketch, false);
            llama_set_layer_range(ctx, 0, -1);
            if (status != 0) break;
            const float * h_k = llama_get_embeddings_layer_inp(ctx, (uint32_t) k);
            r.sketches.insert(r.sketches.end(), sketch.begin(), sketch.end());
            r.shallow.insert(r.shallow.end(), h_k, h_k + d);
            state.assign(h_k, h_k + d);
            count++;
        }
    } else if (h->stop_final) {
        // Final stop source: after each sketch step, the new shallow residual is completed through layers [k, n)
        // on its own scratch sequence, and the stop head reads the completed residual h_D with the count. By
        // causality h_D at a position depends only on positions up to it, so deciding here equals the reference's
        // lookahead chunks. The hard maximum truncates without a decision; an empty block is not possible.
        if (params.allow_empty) {
            llama_memory_seq_rm(mem, params.seq_sketch, -1, -1);
            return -1;
        }
        llama_memory_seq_rm(mem, params.seq_complete, -1, -1);
        llama_memory_seq_cp(mem, seq, params.seq_complete, -1, -1);
        while (count < max_len) {
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
            llama_set_embeddings_layer_inp(ctx, (uint32_t) n_layer, true);
            llama_set_layer_range(ctx, k, -1);
            status = decode_embd(ctx, state.data(), 1, d, pos_first + count, params.seq_complete, false);
            llama_set_layer_range(ctx, 0, -1);
            if (status != 0) {
                break;
            }
            std::vector<float> h_final(llama_get_embeddings_layer_inp(ctx, (uint32_t) n_layer),
                                       llama_get_embeddings_layer_inp(ctx, (uint32_t) n_layer) + d);
            llama_set_embeddings_layer_inp(ctx, (uint32_t) n_layer, false);
            count++;
            if (count == max_len) {
                r.truncated = true;
                break;
            }
            const float logit = nz_stop_logit(h, h_final.data(), count);
            r.stop_logits.push_back(logit);
            if (logit > 0.0f) {
                break;
            }
        }
        llama_set_embeddings_layer_inp(ctx, (uint32_t) n_layer, false);
        llama_memory_seq_rm(mem, params.seq_complete, -1, -1);
    }
    while (hint == 0 && !h->stop_final) {
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
    if (h->direct) {
        // no close marker: the payload is read and decoding resumes from the logits at its last vector
        if (r.n <= 0) return -1;
        return decode_embd(ctx, r.payload.data(), r.n, h->n_embd, pos_first, seq, true);
    }
    const int32_t status = nz_read(ctx, h, r.payload.data(), r.n, pos_first, seq);
    if (status != 0) {
        return status;
    }
    return decode_token(ctx, h->close_token, pos_first + r.n, seq);
}

//
// direct-v1 writer
//

namespace {

// The causal conv stack over `rows` (n final states, oldest first; positions before the window read zero) at the last
// row: CausalConvStack in natlang_neuralese/model/neuralese_projection.py, blocks of RMS norm, depthwise causal conv
// and a SiLU MLP added to the residual.
std::vector<float> stack_last(const nz_heads * h, const std::vector<float> & rows, int32_t n) {
    const int32_t d = h->n_embd, hid = h->stack_hidden, kernel = h->stack_kernel;
    std::vector<float> x(rows.begin(), rows.begin() + (size_t) n * d), y((size_t) n * d), mixed(d), a(hid);
    for (const auto & blk : h->stack) {
        for (int32_t t = 0; t < n; ++t) {
            const float * xt = x.data() + (size_t) t * d;
            double sq = 0.0;
            for (int32_t j = 0; j < d; ++j) sq += (double) xt[j] * xt[j];
            const float inv = 1.0f / std::sqrt((float) (sq / d) + h->stack_eps);
            for (int32_t j = 0; j < d; ++j) y[(size_t) t * d + j] = xt[j] * inv * blk.norm[j];
        }
        std::vector<float> next(x);
        for (int32_t t = 0; t < n; ++t) {
            std::fill(mixed.begin(), mixed.end(), 0.0f);
            for (int32_t k = 0; k < kernel && k <= t; ++k) {
                const float * yt = y.data() + (size_t) (t - k) * d;
                const float * tap = blk.taps.data() + (size_t) k * d;
                for (int32_t j = 0; j < d; ++j) mixed[j] += yt[j] * tap[j];
            }
            for (int32_t i = 0; i < hid; ++i) {
                const float * w = blk.up_w.data() + (size_t) i * d;
                double acc = blk.up_b[i];
                for (int32_t j = 0; j < d; ++j) acc += (double) w[j] * mixed[j];
                const float v = (float) acc;
                a[i] = v / (1.0f + std::exp(-v));
            }
            float * out = next.data() + (size_t) t * d;
            for (int32_t j = 0; j < d; ++j) {
                const float * w = blk.down_w.data() + (size_t) j * hid;
                double acc = blk.down_b[j];
                for (int32_t i = 0; i < hid; ++i) acc += (double) w[i] * a[i];
                out[j] += (float) acc;
            }
        }
        x.swap(next);
    }
    return std::vector<float>(x.end() - d, x.end());
}

} // namespace

int32_t nz_write_direct(llama_context * ctx, const nz_heads * h, llama_seq_id seq, llama_pos pos_first,
                        const float * top, const float * top_logits, const nz_write_params & params,
                        nz_write_result & r) {
    if (!h->direct) {
        return -1;
    }
    const int32_t d       = h->n_embd;
    const int32_t n_vocab = h->n_vocab;
    const int32_t n_layer = llama_model_n_layer(llama_get_model(ctx));
    const int32_t max_len = params.max_length < 0 ? h->max_length : std::min(params.max_length, h->max_length);
    const int32_t hint    = params.length > 0 ? std::min(params.length, max_len) : 0;
    // the receptive field of the stack: the current and (kernel - 1) earlier positions per block
    const int32_t window  = 1 + (int32_t) h->stack.size() * (h->stack_kernel - 1);
    llama_memory_t mem    = llama_get_memory(ctx);
    if (params.temperature > 0.0f || max_len < 1 || params.allow_empty) {
        return -1; // direct values have no trained stochastic payload; a direct block is never empty
    }
    if (llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx))) != n_vocab) {
        return -1;
    }

    r = nz_write_result();
    std::vector<float> history(top, top + d);   // final states, oldest first, at most `window` rows
    std::vector<float> logits(top_logits, top_logits + n_vocab);
    // the greedy tokens run on a scratch copy of the sequence (a recurrent state cannot be rolled back)
    llama_memory_seq_rm(mem, params.seq_sketch, -1, -1);
    llama_memory_seq_cp(mem, seq, params.seq_sketch, -1, -1);
    llama_set_embeddings_layer_inp(ctx, (uint32_t) n_layer, true);
    int32_t status = 0, count = 0;
    while (true) {
        // the readout at the newest state: greedy token and stop log-odds (the reference's DirectWriter.decide)
        int32_t token = 0;
        float best = logits[0], top_logit = logits[0];
        for (int32_t v = 1; v < n_vocab; ++v) {
            if (logits[v] > best) { best = logits[v]; token = v; }
        }
        top_logit = best;
        double z = 0.0;
        for (int32_t v = 0; v < n_vocab; ++v) z += std::exp((double) logits[v] - top_logit);
        const double log_z = top_logit + std::log(z);
        double stop_mass = 0.0;
        for (llama_token t : h->stop_ids) {
            if (t >= 0 && t < n_vocab) stop_mass += std::exp((double) logits[t] - log_z);
        }
        const double stop_logp = std::log(stop_mass);
        const double stop_logit = stop_logp - std::log1p(-std::min(std::exp(stop_logp), 1.0 - 1e-7));
        if (count > 0) {
            r.stop_logits.push_back((float) stop_logit);
        }
        bool stop = count > 0 && std::find(h->stop_ids.begin(), h->stop_ids.end(), token) != h->stop_ids.end();
        if (hint > 0) {
            stop = count == hint;
        }
        if (stop || count == max_len) {
            r.truncated = !stop && hint == 0;
            break;
        }
        // value = softmax(logits)·E + (stack(history) - h)[last]
        std::vector<float> value(d);
        {
            graph_run g;
            ggml_tensor * l = g.input(logits.data(), n_vocab, 1);
            ggml_tensor * mixture = ggml_mul_mat(g.ctx, h->table_t, ggml_soft_max(g.ctx, l));
            if (!g.compute(h, { mixture })) { status = -1; break; }
            get(mixture, value.data());
        }
        const int32_t rows = (int32_t) (history.size() / d);
        const std::vector<float> stacked = stack_last(h, history, rows);
        const float * last = history.data() + (size_t) (rows - 1) * d;
        for (int32_t j = 0; j < d; ++j) value[j] += stacked[j] - last[j];
        // the next position reads the greedy token
        status = decode_token(ctx, token, pos_first + count, params.seq_sketch);
        if (status != 0) break;
        const float * l = llama_get_logits_ith(ctx, -1);
        logits.assign(l, l + n_vocab);
        const float * state = llama_get_embeddings_layer_inp(ctx, (uint32_t) n_layer);
        history.insert(history.end(), state, state + d);
        if ((int32_t) (history.size() / d) > window) history.erase(history.begin(), history.begin() + d);
        r.payload.insert(r.payload.end(), value.begin(), value.end());
        r.final_h.insert(r.final_h.end(), state, state + d);
        count++;
    }
    llama_memory_seq_rm(mem, params.seq_sketch, -1, -1);
    r.n = count;
    r.mean = r.payload;
    r.log_sigma.assign(r.payload.size(), 0.0f);
    return status;
}
