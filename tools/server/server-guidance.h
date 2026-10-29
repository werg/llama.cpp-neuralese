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
// two tool-call forms (`call_format`):
//   "pythonic" (LFM2): `[name(code='...')]`, string values quoted and backslash-escaped
//   "xml" (LFM3, Qwen3-Coder family): `<tool_call>\n<function=NAME>\n<parameter=K>\nVALUE\n</parameter>\n</function>\n</tool_call>`,
//     values raw. The value boundary is the template's exact one: VALUE starts after `<parameter=K>\n` and ends at the
//     first `\n</parameter>\n`, as upstream's qwen3-coder parser and natlang's serve/chat.py (an empty value is
//     `<parameter=K>\n\n</parameter>\n`, as the template renders ""); a `\n</parameter>` not followed by `\n` stays in the
//     value; a reply that ends in a proper prefix of `\n</parameter>\n` holds that suffix back until it resolves. No
//     escapes are decoded.
// tool calls: call names must be offered tools, checked when the `(` (pythonic) or the `>` of `<function=NAME>` (xml)
// arrives
// eval code: inside an `eval(code=...)` string or an eval call's `code` parameter, each completed line is checked for
// repetition, redeclaration (a
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
    std::string call_format = "pythonic"; // "pythonic" or "xml"
    // the generation prompt opened a reasoning block, which the first of these tags closes: reasoning is not an action,
    // so it is neither checked nor rolled back, and it can stream while the rest of the reply is held back
    bool reasoning_open = false;
    std::vector<std::string> reasoning_end;

    // `true` or an object of settings; `offered` are the request's tool names. throws std::invalid_argument
    // `default_format` applies when the settings name no "call_format" (server_guide_call_format of the chat template)
    static server_guide_settings from_json(const json & value, const std::vector<std::string> & offered, bool require_call,
                                           const std::string & default_format = "pythonic");
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

    // the reply offset where the action begins: 0 without an open reasoning block, just after its closing tag, or
    // npos while the reply is still reasoning
    size_t action_start(const std::string & reply) const;

    // the reply was cut back to `offset`: what follows is checked again
    void rewind(size_t offset);

    // the retries at `at` are used up: what is there stands
    void accept(size_t at);

private:
    std::set<std::tuple<std::string, size_t, bool>> checked; // ("line", offset, final) and ("at", offset, false)
    std::set<size_t> names_checked;
};

// the call form a chat template renders: "xml" for `<tool_call>` / `<function=` / `<parameter=` templates, else "pythonic"
std::string server_guide_call_format(const std::string & template_source);

// (start, end) byte offsets of tree-sitter errors and missing tokens in TypeScript `code`
std::vector<std::pair<size_t, size_t>> server_guide_syntax_errors(const std::string & code);
