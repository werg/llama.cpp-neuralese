#pragma once

#include "server-common.h"

#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// guided generation: the reply checks behind the natlang `guidance` request field
// a port of natlang's training/neuralese/natlang_neuralese/serve/guidance.py, the reference
//
// tool calls: call names must be offered tools, checked when the `(` arrives
// eval code: inside an `eval(code=...)` string, each completed line is checked for repetition, redeclaration (a
// const/let/var name declared again at the same indentation in one block) and TypeScript syntax (tree-sitter), the
// unfinished last line for a run (one chunk repeated `run` times at its end)
// a rejection names the reply byte offset where the offending line or name starts; the slot rolls back there, bans
// the token it chose first and samples again, at most `retries` times per offset, then accepts what is there

struct server_guide_settings {
    bool require_call = false;
    std::vector<std::string> tools; // allowed call names, empty: any
    int32_t repeat  = 3;            // occurrences of one line that reject it (0: off)
    int32_t run     = 4;            // repetitions of one chunk at the end of the unfinished line that reject it (0: off)
    bool    syntax  = true;
    int32_t retries = 4;

    // `true` or an object of settings; `offered` are the request's tool names. throws std::invalid_argument
    static server_guide_settings from_json(const json & value, const std::vector<std::string> & offered, bool require_call);
    json to_json() const;
};

struct server_guide_verdict {
    std::string reason; // "unknown-tool", "repetition", "redeclaration", "syntax"
    size_t at = 0;      // reply byte offset to roll back to
};

struct server_guide {
    server_guide_settings settings;

    // the next verdict for `reply`, the whole reply so far; checks already passed are not repeated
    std::optional<server_guide_verdict> check(const std::string & reply);

    // the reply was cut back to `offset`: what follows is checked again
    void rewind(size_t offset);

    // the retries at `at` are used up: what is there stands
    void accept(size_t at);

private:
    std::set<std::tuple<std::string, size_t, bool>> checked; // ("line", offset, final) and ("at", offset, false)
    std::set<size_t> names_checked;
};

// (start, end) byte offsets of tree-sitter errors and missing tokens in TypeScript `code`
std::vector<std::pair<size_t, size_t>> server_guide_syntax_errors(const std::string & code);
