#include "native-state.h"

#include "gguf.h"
#include "ggml.h"
#include "log.h"

#include <cstring>

static bool get_str(const gguf_context * g, const char * key, std::string & out) {
    const int64_t i = gguf_find_key(g, key);
    if (i < 0 || gguf_get_kv_type(g, i) != GGUF_TYPE_STRING) {
        return false;
    }
    out = gguf_get_val_str(g, i);
    return true;
}

static bool get_i32(const gguf_context * g, const char * key, int32_t & out) {
    const int64_t i = gguf_find_key(g, key);
    if (i < 0) {
        return false;
    }
    switch (gguf_get_kv_type(g, i)) {
        case GGUF_TYPE_INT32:  out = gguf_get_val_i32(g, i); return true;
        case GGUF_TYPE_UINT32: out = (int32_t) gguf_get_val_u32(g, i); return true;
        default: return false;
    }
}

bool common_native_state_load(const std::string & path, int32_t n_layer, common_native_state & out) {
    ggml_context * ctx = nullptr;
    gguf_init_params params = { /*.no_alloc =*/ false, /*.ctx =*/ &ctx };
    gguf_context * g = gguf_init_from_file(path.c_str(), params);
    if (!g) {
        LOG_ERR("%s: %s is not a GGUF file\n", __func__, path.c_str());
        return false;
    }
    bool ok = true;
    std::string type;
    int32_t open = -1, close = -1;
    ok &= get_str(g, "general.type", type) && type == "natlang.native_state";
    ok &= get_str(g, "natlang.state.id", out.id);
    ok &= get_str(g, "natlang.state.text_sha256", out.text_sha256);
    ok &= get_str(g, "natlang.state.architecture", out.architecture);
    ok &= get_str(g, "natlang.state.checkpoint_sha256", out.checkpoint_sha256);
    ok &= get_str(g, "natlang.state.tokenizer_fingerprint", out.tokenizer_fingerprint);
    ok &= get_str(g, "natlang.state.embedding_weight_sha256", out.embedding_weight_sha256);
    ok &= get_i32(g, "natlang.state.n_tokens", out.n_tokens) && out.n_tokens > 0;
    ok &= get_i32(g, "natlang.state.consume_open", open) && get_i32(g, "natlang.state.consume_close", close);
    if (!ok) {
        LOG_ERR("%s: %s is not a natlang native state (metadata)\n", __func__, path.c_str());
        gguf_free(g); ggml_free(ctx);
        return false;
    }
    out.consume_open = open; out.consume_close = close;
    out.k.assign(n_layer, {}); out.v.assign(n_layer, {}); out.conv.assign(n_layer, {});
    int64_t found = 0;
    for (int32_t il = 0; il < n_layer; ++il) {
        for (const char * name : { "state_k", "state_v", "state_conv" }) {
            const std::string tn = "blk." + std::to_string(il) + "." + name;
            ggml_tensor * t = ggml_get_tensor(ctx, tn.c_str());
            if (!t) {
                continue;
            }
            if (t->type != GGML_TYPE_F32 || ggml_n_dims(t) != 2 || t->ne[1] != out.n_tokens) {
                LOG_ERR("%s: %s: %s must be F32 [width, n_tokens]\n", __func__, path.c_str(), tn.c_str());
                gguf_free(g); ggml_free(ctx);
                return false;
            }
            auto & dst = !strcmp(name, "state_k") ? out.k[il] : !strcmp(name, "state_v") ? out.v[il] : out.conv[il];
            const float * src = (const float *) t->data;
            dst.assign(src, src + ggml_nelements(t));
            ++found;
        }
    }
    if ((int64_t) gguf_get_n_tensors(g) != found) {
        LOG_ERR("%s: %s holds tensors outside %d layers\n", __func__, path.c_str(), n_layer);
        ok = false;
    }
    gguf_free(g); ggml_free(ctx);
    return ok;
}

bool common_native_state_register(llama_context * ctx, common_native_state & state) {
    std::vector<llama_state_span_layer> layers(state.k.size());
    for (size_t il = 0; il < layers.size(); ++il) {
        layers[il].k    = state.k[il].empty()    ? nullptr : state.k[il].data();
        layers[il].v    = state.v[il].empty()    ? nullptr : state.v[il].data();
        layers[il].conv = state.conv[il].empty() ? nullptr : state.conv[il].data();
    }
    state.first = llama_state_span_register(ctx, state.n_tokens, layers.data(), (int32_t) layers.size());
    return state.first != LLAMA_TOKEN_NULL;
}

std::vector<llama_token> common_native_state_tokens(const common_native_state & state) {
    std::vector<llama_token> out;
    out.reserve(state.n_tokens + 2);
    out.push_back(state.consume_open);
    for (int32_t i = 0; i < state.n_tokens; ++i) {
        out.push_back(state.first + i);
    }
    out.push_back(state.consume_close);
    return out;
}
