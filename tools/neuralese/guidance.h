#pragma once

// Guided generation: the reply checks behind the `guidance` request field. A port of natlang's
// training/neuralese/natlang_neuralese/serve/guidance.py (the reference; this file follows it).
//
// Envelope: with require_call the reply is forced to open a tool call; call names must be allowed (checked at `(`).
// Eval code: inside an `eval(code=...)` string, every completed line is checked for repetition and for TypeScript
// syntax (tree-sitter: an error or missing token inside the completed text that does not reach its end). A rejection
// names the reply offset where the offending line (or name) starts; the server rolls back there, bans the token it
// chose first and samples again, at most `retries` times per point, then accepts what is there.

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

struct nz_guide_settings {
    bool require_call = false;
    std::vector<std::string> tools; // allowed call names; empty: any
    int32_t repeat  = 3;            // occurrences of one line that reject it (0: off)
    bool    syntax  = true;
    int32_t retries = 4;
};

struct nz_guide_verdict {
    std::string reason; // "unknown-tool", "repetition", "syntax"
    size_t at = 0;      // reply offset to roll back to
};

struct nz_guide {
    nz_guide_settings settings;
    std::set<std::tuple<std::string, size_t, bool>> checked; // ("line", offset, final) and ("at", offset, false)
    std::set<size_t> names_checked;

    std::string forced_prefix() const;
    std::optional<nz_guide_verdict> check(const std::string & reply);
    void rewind(size_t offset);
    void accept(size_t at);
};

// (start, end) byte offsets of tree-sitter errors and missing tokens in TypeScript `code`.
std::vector<std::pair<size_t, size_t>> nz_syntax_errors(const std::string & code);
