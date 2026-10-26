#pragma once

#include "llama.h"

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

// Native state spans (llama_state_span_register): exact per-layer stored state imported from outside the model, for
// virtual tokens that follow the vocabulary. A virtual token in a batch does not compute what the model stores for it:
// attention layers store the registered key (after the key norm, before RoPE, rotated at the token's position) and
// value, and short-convolution layers store the registered gated input B*x. Nothing else a decode persists depends on
// the virtual token's own embedding or residual stream, so the import is exact.

// virtual token IDs accepted in batches: [n_vocab, n_vocab + LLAMA_STATE_SPAN_MAX_TOKENS)
constexpr llama_token LLAMA_STATE_SPAN_MAX_TOKENS = 1 << 24;

struct llama_state_span {
    llama_token first = 0;
    int32_t     n     = 0;
    // per layer, row-major [n][width]; empty where the layer stores no such state
    std::vector<std::vector<float>> k;
    std::vector<std::vector<float>> v;
    std::vector<std::vector<float>> conv;
    std::vector<int64_t> width_kv; // n_embd_k_gqa(il) for attention layers
    int64_t width_conv = 0;        // n_embd for short-convolution layers
};

struct llama_state_spans {
    llama_token base = 0; // n_vocab: the first virtual token ID
    llama_token next = 0; // the first unallocated virtual token ID
    std::map<llama_token, std::unique_ptr<llama_state_span>> spans; // by first virtual token

    // the span holding virtual token t and its row, or nullptr
    const llama_state_span * find(llama_token t, int32_t & row) const {
        if (t < base) {
            return nullptr;
        }
        auto it = spans.upper_bound(t);
        if (it == spans.begin()) {
            return nullptr;
        }
        --it;
        if (t >= it->first + it->second->n) {
            return nullptr;
        }
        row = t - it->first;
        return it->second.get();
    }
};
