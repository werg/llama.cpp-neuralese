#pragma once

// natlang RL actor capture: llama-server as an actor whose rollouts natlang's native whole-graph RL learner trains on.
//
// The capture is the reference native engine's x_natlang_rl_capture (natlang_neuralese/serve/engine.py), consumed by
// rl/native_replay.py compile_replay: the exact prompt token IDs with per-token embedding roles and the native-state
// imports at their own-token boundaries, every sampled token with its log-probability under the actual sampling law
// (softmax of logits / temperature with masked tokens excluded; no truncation sampler), the behaviour top-k of that
// distribution (common/sampling.py natlang.behavior-top-k/1: k x int32 ID and float16 log-probability, base64), the
// producer write spans with their own-input coordinates, and the policy's native identity. A JOLT+ teacher attempt
// inserts a privileged block at its declared own-token index and marks it privileged.

#include "server-common.h"
#include "server-native-state.h"

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

constexpr int RL_DEFAULT_TOP_K = 128;  // common/sampling.py DEFAULT_BEHAVIOR_TOP_K
constexpr int RL_MAX_TOP_K     = 4096; // MAX_BEHAVIOR_TOP_K

// what a request asks for (x_natlang_* fields of a completion request)
struct rl_request {
    bool capture   = false; // x_natlang_rl_capture: return the capture
    bool recording = false; // x_natlang_native_recording: structural capture (greedy decoding allowed)
    bool recovery  = false; // x_natlang_native_protocol_recovery: a native marker failure answers its capture
    int  top_k     = -1;    // x_natlang_rl_top_k: -1 default (RL_DEFAULT_TOP_K for a sampled capture), 0 none
    json privileged;        // x_natlang_privileged_block {insert_at, token_ids}, or null

    // filled when the prompt is tokenized: the capture's prompt segments and own-token count
    json    prompt;
    int32_t prompt_own = 0;

    // the request records its inputs and samples (a capture, or recovery that may answer one)
    bool trace() const { return capture || recording || recovery; }
    bool emits() const { return capture || recording; }
};

// parse the x_natlang_* RL fields of a request body into `out`; throws std::invalid_argument
void rl_request_parse(const json & data, rl_request & out);

// The sampling law of a traced request. Truncation samplers, penalties, grammars and the reasoning budget are removed
// so the sampler draws from softmax(logits / temperature) over the tokens the logit biases leave; returns the banned
// tokens (-inf biases, including the vocabulary's suppressed tokens). Throws std::invalid_argument for a finite bias
// or another setting that changes the law.
std::vector<llama_token> rl_sampling_law(const llama_vocab * vocab, common_params_sampling & sampling);

// log-probability of `tok` under softmax(logits / temp) with `banned` excluded, and (k > 0) the top-k receipt
// {ids, logps} of that distribution
double rl_law_logp(const float * logits, int32_t n_vocab, float temp, const std::vector<llama_token> & banned,
                   llama_token tok, int k, json * top_k);

// the same from the backend draw (llama_sampler_init_dist_capture): the drawn token's probability and the device top-k
// (probabilities and IDs, unordered); the receipt is sorted by decreasing probability, then ID
double rl_backend_logp(float p_sampled, const float * probs, const llama_token * ids, uint32_t n,
                       int32_t n_vocab, int k, json * top_k);

// the behaviour top-k declaration (common/sampling.py behavior_top_k_declaration)
json rl_top_k_declaration(int k);

// Insert a privileged block's tokens at own-token index insert_at (virtual native-state tokens, at or above n_vocab,
// are not own tokens). Returns the index of the first inserted token. Throws std::invalid_argument when the point is
// outside the prompt or on a native-state import boundary (rl/privileged_context.py insert_into_prompt_segments).
size_t rl_insert_privileged(llama_tokens & tokens, const json & block, int32_t n_vocab);

// The capture's prompt segments: token runs with per-token producing flags (all false: a prompt does not produce) and
// native_state parts for each run of virtual tokens, named by the exact-mode import receipts in order. Tokens in
// [privileged_begin, privileged_end) form their own segment marked privileged. Sets n_own to the own-token count.
json rl_prompt_segments(const llama_tokens & tokens, const json & imports, const native_state_registry & registry,
                        int32_t n_vocab, size_t privileged_begin, size_t privileged_end, int32_t & n_own);

// How an actor's weights relate to the checkpoint its identity names (rl/actors.py export): kind exact (F32/BF16
// file) or quantized, the GGUF file type as format, and the sha256 of the GGUF file.
struct rl_actor_export {
    std::string kind;
    std::string format;
    std::string sha256;

    bool empty() const { return sha256.empty(); }
    json to_json() const { return { { "kind", kind }, { "format", format }, { "sha256", sha256 } }; }
};

// Bind the running executable and loaded llama/ggml libraries; the image and
// system/CUDA libraries require their own independently verified custody.
json rl_actor_runtime_from_process();

rl_actor_export rl_actor_export_from_model(const llama_model * model, const std::string & path);

// the hex sha256 of a file's bytes; empty when it cannot be read
std::string rl_file_sha256(const std::string & path);
