#pragma once

// natlang native states on llama-server: the content-addressed native-state artifacts natlang's reference server
// stores and exchanges (natlang_neuralese/serve/store.py, model/state_block.py), so a natlang runtime reads and writes
// the same states through either server.
//
// An artifact is a safetensors body: token_ids I64 [n], positions I64 [n], embeddings [n, n_embd] (the producing
// tokens' input embeddings before the producer adapter), and per layer either layers.{i}.raw_k / layers.{i}.v
// [n, n_head_kv, head_dim] (attention keys after the key norm and before RoPE, values) or layers.{i}.gated_inputs
// [n, n_embd] (short convolution B*x). Its metadata (schema neuralese.native-state/1, text, architecture,
// tokenizer_fingerprint, checkpoint_sha256, embedding_weight_sha256, layer_kinds) and every tensor's bytes determine its
// ID, nzs1_<base32 sha256> (native_state_id in store.py).

#include "llama.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// the model identity a native state carries and a server declares (natlang.identity.* in a natlang GGUF)
struct native_identity {
    std::string architecture;
    std::string checkpoint_sha256;
    std::string tokenizer_fingerprint;
    std::string embedding_weight_sha256;

    bool empty() const { return checkpoint_sha256.empty(); }
    bool operator==(const native_identity & other) const {
        return architecture == other.architecture && checkpoint_sha256 == other.checkpoint_sha256 &&
               tokenizer_fingerprint == other.tokenizer_fingerprint && embedding_weight_sha256 == other.embedding_weight_sha256;
    }
    nlohmann::json to_json() const;
};

// read natlang.identity.* from a model; empty when the GGUF has none
native_identity native_identity_from_model(const llama_model * model);

struct native_state_tensor {
    std::string          dtype; // F16, BF16, F32, F64, I64
    std::vector<int64_t> shape;
    std::string          data;  // little-endian bytes
};

struct native_state_block {
    nlohmann::json                             metadata; // without "id"; sorted keys (canonical)
    std::map<std::string, native_state_tensor> tensors;  // sorted by name

    std::string text() const;
    native_identity identity() const;
    int64_t n_tokens() const;
};

// the content ID (store.py native_state_id)
std::string native_state_id(const native_state_block & block);

// the safetensors body (store.py encode_native_state)
std::string native_state_encode(const native_state_block & block, const std::string & id);

// decode and verify a body: a well-formed native-state/1 layout whose ID matches its content; throws
// std::invalid_argument with the reason
native_state_block native_state_decode(const std::string & body, std::string & id);

// what GET .../meta answers: {"id": ..., **metadata}
nlohmann::json native_state_meta(const std::string & id, const native_state_block & block);

// one state's per-layer rows in the fork's span layout (F32 [n][width]); throws std::invalid_argument when the block
// does not fit the model's layer layout
struct native_state_rows {
    int32_t n_tokens = 0;
    std::vector<std::vector<float>> k, v, conv;
};
native_state_rows native_state_to_rows(const native_state_block & block, int32_t n_layer);

// the stored states of one server: artifacts by ID, and the virtual token spans of those registered with the context
class native_state_store {
public:
    struct entry {
        std::string        body;   // the encoded artifact
        native_state_block block;
    };

    // store a verified artifact; returns false when it was already stored
    bool put(const std::string & id, const std::string & body, native_state_block block);

    std::shared_ptr<const entry> get(const std::string & id) const;
    bool has(const std::string & id) const;

    // record a span registration (main thread) and look one up (any thread)
    void set_span(const std::string & id, llama_token first);
    llama_token span(const std::string & id) const;

    size_t size() const;

private:
    mutable std::mutex m;
    std::map<std::string, std::shared_ptr<const entry>> entries;
    std::map<std::string, llama_token> spans; // registered span starts (exact-identity states only)
};

// how a request's native_state part is consumed: the tokens it expands to and the receipt mode
struct native_state_use {
    std::vector<llama_token> tokens;
    std::string              mode; // exact-native-state | best-effort-text-retokenization
};

// every native state a server can consume: --native-state files (registered at startup) and stored artifacts. A stored
// state of this server's identity is consumed exactly through its registered span; another model's state is consumed
// best effort, by re-encoding its text (as the reference server does until semantic adaptation is trained).
class native_state_registry {
public:
    native_state_store store;
    native_identity    identity;  // this server's; empty when the GGUF declares none
    llama_token        consume_open  = LLAMA_TOKEN_NULL;
    llama_token        consume_close = LLAMA_TOKEN_NULL;
    const llama_vocab * vocab = nullptr;
    int32_t            context = 0; // tokens per slot

    void add_static(const std::string & id, std::vector<llama_token> tokens);
    bool known(const std::string & id) const;
    // throws std::invalid_argument("neuralese-unknown-native-state: <id>")
    native_state_use resolve(const std::string & id) const;

private:
    mutable std::mutex m;
    std::map<std::string, std::vector<llama_token>> statics;
};
