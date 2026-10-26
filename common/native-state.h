#pragma once

#include "llama.h"

#include <string>
#include <vector>

// A natlang native state span exported as GGUF (natlang tune soft states, `natlang_neuralese.tune.soft_export`):
//   general.type = "natlang.native_state"
//   natlang.state.{id, text_sha256, architecture, checkpoint_sha256, tokenizer_fingerprint, embedding_weight_sha256}
//   natlang.state.{n_tokens, consume_open, consume_close}
//   tensors blk.{il}.state_k / blk.{il}.state_v (attention layers; F32 [n_embd_k_gqa, n_tokens]: the key after the key
//   norm and before RoPE, the value) and blk.{il}.state_conv (short-convolution layers; F32 [n_embd, n_tokens]: B*x)
struct common_native_state {
    std::string id;
    std::string text_sha256;
    std::string architecture;
    std::string checkpoint_sha256;
    std::string tokenizer_fingerprint;
    std::string embedding_weight_sha256;
    int32_t     n_tokens      = 0;
    llama_token consume_open  = LLAMA_TOKEN_NULL;
    llama_token consume_close = LLAMA_TOKEN_NULL;
    std::vector<std::vector<float>> k, v, conv; // per layer; empty where absent

    // registered virtual tokens (common_native_state_register)
    llama_token first = LLAMA_TOKEN_NULL;
};

// load a state file for a model with n_layer layers; false (with a logged reason) on any malformation
bool common_native_state_load(const std::string & path, int32_t n_layer, common_native_state & out);

// register the state's span with the context; sets out.first; false if the context refuses it
bool common_native_state_register(llama_context * ctx, common_native_state & state);

// the tokens that consume the state: the consume-open marker, the virtual span, the consume-close marker
std::vector<llama_token> common_native_state_tokens(const common_native_state & state);
