#pragma once

// Neuralese: soft-token read and write ports for causal LMs.
//
// A Neuralese block is an ordered sequence of embedding-width vectors placed between two
// control tokens. Reading a block splices its vectors into the embedding stream. Writing a
// block runs one procedure:
//   1. sketch: layers [0, k) run position by position from the open marker; a stop head on the
//      shallow residual decides when to close, and a feedback projection turns each shallow
//      residual into the next sketch input
//   2. complete: layers [k, n_layer) run once, causally, over the collected shallow residuals
//   3. project: a content projection turns the completed residuals into the payload
//      (mean + temperature-gated noise)
//   4. read back: the payload and the close marker are prefilled through the full model
//
// The heads (feedback projection, stop head, content projection, interface norm) live in a
// sidecar GGUF ("Neuralese projector"), next to an unchanged model GGUF, like mtmd's mmproj.
// The procedure uses llama_set_layer_range() and llama_set_embeddings_layer_inp() from
// llama-ext.h, so the model architecture must support layer ranges.
//
// Context requirements: kv_unified = true and n_seq_max >= 3 (the writer works on two scratch
// sequences that share the prefix with the main sequence).

#include "llama.h"

#include <cstdint>
#include <vector>

struct nz_heads;

struct nz_heads_params {
    int  n_threads = 4;     // CPU threads when the heads run on the CPU
    bool use_gpu   = false; // run the heads on the first GPU backend, if there is one
};

// load a Neuralese projector GGUF; returns nullptr on failure
nz_heads * nz_heads_load(const char * path, const nz_heads_params & params);
void       nz_heads_free(nz_heads * heads);

int32_t      nz_heads_n_embd    (const nz_heads * heads);
int32_t      nz_heads_cutoff    (const nz_heads * heads); // k: layers in the sketch generator
int32_t      nz_heads_max_length(const nz_heads * heads); // the heads' maximum block length
llama_token  nz_heads_open_token (const nz_heads * heads);
llama_token  nz_heads_close_token(const nz_heads * heads);
const char * nz_heads_dialect   (const nz_heads * heads);

// interface norm, applied to every vector entering the read port: n rows of n_embd floats
void nz_interface(const nz_heads * heads, const float * in, float * out, int32_t n);

// head primitives (n rows of n_embd floats each)
void  nz_feedback(const nz_heads * heads, const float * h_cut, float * sketch);           // one row
float nz_stop_logit(const nz_heads * heads, const float * h_cut, int32_t count);          // one row
void  nz_content(const nz_heads * heads, const float * sketches, const float * h_final, int32_t n,
                 float * mean, float * log_sigma);

// Read port: decode the interface-normed payload as embeddings at positions pos, pos+1, ...
// of sequence seq. Returns the llama_decode status.
int32_t nz_read(llama_context * ctx, const nz_heads * heads, const float * payload, int32_t n,
                llama_pos pos, llama_seq_id seq);

struct nz_write_params {
    int32_t      max_length   = -1;    // hard maximum (<= heads max); -1: the heads' maximum
    float        temperature  = 0.0f;  // Neuralese temperature; 0 delivers the mean
    const float * noise       = nullptr; // optional caller noise eps [max_length * n_embd]; else seeded
    uint32_t     seed         = 0;
    bool         allow_empty  = false;
    llama_seq_id seq_sketch   = 1;     // scratch sequence for the sketch recurrence
    llama_seq_id seq_complete = 2;     // scratch sequence for the completion pass
};

struct nz_write_result {
    int32_t n         = 0;      // vectors written
    bool    truncated = false;  // the hard maximum closed the block
    std::vector<float> payload;   // [n * n_embd] delivered vectors
    std::vector<float> mean;      // [n * n_embd]
    std::vector<float> log_sigma; // [n * n_embd]
    std::vector<float> noise;     // [n * n_embd] eps used (empty at temperature 0)
    std::vector<float> sketches;  // [n * n_embd]
    std::vector<float> shallow;   // [n * n_embd] residuals after k layers
    std::vector<float> final_h;   // [n * n_embd] residuals after all layers
    std::vector<float> stop_logits; // [n] stop logit evaluated after j+1 vectors
};

// Open a block: decode the open marker at pos on seq through the full model and return the
// shallow residual h_k at the marker (n_embd floats). Logits at the marker are available as
// llama_get_logits_ith(ctx, -1) afterwards.
int32_t nz_open(llama_context * ctx, const nz_heads * heads, llama_pos pos, llama_seq_id seq, float * h_cut);

// Write a block after nz_open. pos_first is the position of the first payload vector
// (the open marker's position + 1). The main sequence is left at the open marker.
int32_t nz_write(llama_context * ctx, const nz_heads * heads, llama_seq_id seq, llama_pos pos_first,
                 const float * h_cut, const nz_write_params & params, nz_write_result & result);

// Read back a written block: payload at pos_first.., then the close marker with logits.
int32_t nz_read_back(llama_context * ctx, const nz_heads * heads, llama_seq_id seq, llama_pos pos_first,
                     const nz_write_result & result);
