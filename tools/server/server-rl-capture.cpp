#include "server-rl-capture.h"

#include "ggml.h"
extern "C" {
#include "hash/sha256/sha256.h"
}

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>

void rl_request_parse(const json & data, rl_request & out) {
    auto flag = [&](const char * key, bool & value) {
        if (!data.contains(key) || data.at(key).is_null()) {
            return;
        }
        if (!data.at(key).is_boolean()) {
            throw std::invalid_argument(std::string(key) + " must be boolean");
        }
        value = data.at(key).get<bool>();
    };
    flag("x_natlang_rl_capture", out.capture);
    flag("x_natlang_native_recording", out.recording);
    flag("x_natlang_native_protocol_recovery", out.recovery);
    if (data.contains("x_natlang_rl_top_k") && !data.at("x_natlang_rl_top_k").is_null()) {
        const json & k = data.at("x_natlang_rl_top_k");
        if (!k.is_number_integer() || k.get<int64_t>() < 0 || k.get<int64_t>() > RL_MAX_TOP_K) {
            throw std::invalid_argument("x_natlang_rl_top_k must be 0 or an integer in [1, " + std::to_string(RL_MAX_TOP_K) + "]");
        }
        if (!out.capture) {
            throw std::invalid_argument("x_natlang_rl_top_k records sampler top-k log-probs inside x_natlang_rl_capture; request both");
        }
        out.top_k = (int) k.get<int64_t>();
    }
    if (data.contains("x_natlang_privileged_block") && !data.at("x_natlang_privileged_block").is_null()) {
        const json & block = data.at("x_natlang_privileged_block");
        bool ok = block.is_object() && block.size() == 2 && block.contains("insert_at") && block.contains("token_ids") &&
                  block.at("insert_at").is_number_integer() && block.at("insert_at").get<int64_t>() >= 0 &&
                  block.at("token_ids").is_array() && !block.at("token_ids").empty() && out.capture;
        if (ok) {
            for (const auto & t : block.at("token_ids")) {
                ok = ok && t.is_number_integer() && t.get<int64_t>() >= 0;
            }
        }
        if (!ok) {
            throw std::invalid_argument("x_natlang_privileged_block is {insert_at, token_ids} of a captured JOLT+ teacher attempt");
        }
        out.privileged = block;
    }
    for (const char * key : { "x_natlang_forced_message", "x_natlang_native_state_modes" }) {
        if (data.contains(key) && !data.at(key).is_null()) {
            throw std::invalid_argument(std::string("unsupported-request-field: ") + key + " is served by the native reference server only");
        }
    }
    if (data.contains("x_natlang_adapters") && !data.at("x_natlang_adapters").is_null() &&
            !(data.at("x_natlang_adapters").is_array() && data.at("x_natlang_adapters").empty())) {
        throw std::invalid_argument("natlang adapter IDs are not served here; select LoRA adapters with \"lora\"");
    }
}

std::vector<llama_token> rl_sampling_law(const llama_vocab * vocab, common_params_sampling & s) {
    if (s.grammar.type == COMMON_GRAMMAR_TYPE_USER && !s.grammar.empty()) {
        throw std::invalid_argument("native-rl-sampling-law: a grammar changes the sampling law of a captured rollout");
    }
    if (s.grammar.type == COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT && !s.grammar.empty()) {
        throw std::invalid_argument("native-rl-sampling-law: a response format changes the sampling law of a captured rollout");
    }
    if (s.mirostat != 0) {
        throw std::invalid_argument("native-rl-sampling-law: mirostat changes the sampling law of a captured rollout");
    }
    std::set<llama_token> banned;
    for (const auto & bias : s.logit_bias) {
        if (bias.bias == -INFINITY) {
            banned.insert(bias.token);
        } else if (bias.bias != 0.0f) {
            throw std::invalid_argument("native-rl-sampling-law: a finite logit bias changes the sampling law of a captured rollout");
        }
    }
    int32_t n_suppress = 0;
    const llama_token * suppress = llama_vocab_get_suppress_tokens(vocab, &n_suppress);
    for (int32_t i = 0; i < n_suppress; ++i) {
        banned.insert(suppress[i]);
    }
    // the native engine's law: temperature only, then the categorical draw (common_sampler_init adds dist)
    s.samplers       = { COMMON_SAMPLER_TYPE_TEMPERATURE };
    s.dynatemp_range = 0.0f;
    s.grammar        = {};
    s.grammar_lazy   = false;
    s.grammar_triggers.clear();
    s.generation_prompt.clear();
    s.reasoning_budget_tokens = -1;
    s.reasoning_budget_start.clear();
    s.reasoning_budget_end.clear();
    s.reasoning_budget_forced.clear();
    s.reasoning_control = false;
    s.backend_sampling  = false;
    return std::vector<llama_token>(banned.begin(), banned.end());
}

static std::string base64(const unsigned char * data, size_t n) {
    static const char * chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((n + 2) / 3 * 4);
    for (size_t i = 0; i < n; i += 3) {
        const uint32_t a = data[i], b = i + 1 < n ? data[i + 1] : 0, c = i + 2 < n ? data[i + 2] : 0;
        const uint32_t v = (a << 16) | (b << 8) | c;
        out += chars[(v >> 18) & 63];
        out += chars[(v >> 12) & 63];
        out += i + 1 < n ? chars[(v >> 6) & 63] : '=';
        out += i + 2 < n ? chars[v & 63] : '=';
    }
    return out;
}

double rl_law_logp(const float * logits, int32_t n_vocab, float temp, const std::vector<llama_token> & banned,
                   llama_token tok, int k, json * top_k) {
    std::vector<char> masked(n_vocab, 0);
    for (const llama_token t : banned) {
        if (t >= 0 && t < n_vocab) {
            masked[t] = 1;
        }
    }
    // FP32 logits / T (common/sampling.py masked_logits), normalized in double
    double max = -INFINITY;
    for (int32_t i = 0; i < n_vocab; ++i) {
        if (!masked[i]) {
            max = std::max(max, (double) (logits[i] / temp));
        }
    }
    double sum = 0.0;
    for (int32_t i = 0; i < n_vocab; ++i) {
        if (!masked[i]) {
            sum += std::exp((double) (logits[i] / temp) - max);
        }
    }
    const double lse = max + std::log(sum);
    const double logp = masked[tok] ? -INFINITY : (double) (logits[tok] / temp) - lse;
    if (top_k != nullptr && k > 0) {
        std::vector<int32_t> ids;
        ids.reserve(n_vocab);
        for (int32_t i = 0; i < n_vocab; ++i) {
            if (!masked[i]) {
                ids.push_back(i);
            }
        }
        const size_t n = std::min<size_t>(k, ids.size());
        std::partial_sort(ids.begin(), ids.begin() + n, ids.end(), [&](int32_t a, int32_t b) {
            return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
        });
        std::vector<unsigned char> id_bytes(n * 4), lp_bytes(n * 2);
        for (size_t i = 0; i < n; ++i) {
            const uint32_t id = (uint32_t) ids[i];
            const float lp = (float) ((double) (logits[ids[i]] / temp) - lse);
            const uint16_t h = ggml_fp32_to_fp16(std::min(lp, 0.0f));
            for (int b = 0; b < 4; ++b) {
                id_bytes[i * 4 + b] = (unsigned char) ((id >> (8 * b)) & 0xff);
            }
            lp_bytes[i * 2]     = (unsigned char) (h & 0xff);
            lp_bytes[i * 2 + 1] = (unsigned char) (h >> 8);
        }
        *top_k = { { "ids", base64(id_bytes.data(), id_bytes.size()) }, { "logps", base64(lp_bytes.data(), lp_bytes.size()) } };
    }
    return logp;
}

json rl_top_k_declaration(int k) {
    return { { "schema", "natlang.behavior-top-k/1" }, { "k", k }, { "distribution", "post-processing-normalized" },
             { "ids", "int32-le-base64" }, { "logps", "float16-le-base64" } };
}

size_t rl_insert_privileged(llama_tokens & tokens, const json & block, int32_t n_vocab) {
    const int64_t at = block.at("insert_at").get<int64_t>();
    llama_tokens inserted;
    for (const auto & t : block.at("token_ids")) {
        const int64_t id = t.get<int64_t>();
        if (id >= n_vocab) {
            throw std::invalid_argument("bad-request: a privileged block token is outside the vocabulary");
        }
        inserted.push_back((llama_token) id);
    }
    // the first index after `at` own tokens
    int64_t own = 0;
    size_t p = 0;
    while (own < at && p < tokens.size()) {
        if (tokens[p] < n_vocab) {
            own++;
        }
        p++;
    }
    if (own != at) {
        throw std::invalid_argument("bad-request: privileged insertion point is outside the rendered prompt");
    }
    if ((p < tokens.size() && tokens[p] >= n_vocab) || (p > 0 && tokens[p - 1] >= n_vocab)) {
        throw std::invalid_argument("bad-request: a privileged block cannot share a native-state import boundary");
    }
    tokens.insert(tokens.begin() + p, inserted.begin(), inserted.end());
    return p;
}

json rl_prompt_segments(const llama_tokens & tokens, const json & imports, const native_state_registry & registry,
                        int32_t n_vocab, size_t privileged_begin, size_t privileged_end, int32_t & n_own) {
    json segments = json::array();
    std::vector<std::string> exact;
    if (imports.is_array()) {
        for (const auto & receipt : imports) {
            if (receipt.value("mode", std::string()) == "exact-native-state") {
                exact.push_back(receipt.at("id").get<std::string>());
            }
        }
    }
    size_t next_import = 0;
    n_own = 0;
    json run = json::array();
    bool run_privileged = false;
    auto flush = [&]() {
        if (run.empty()) {
            return;
        }
        json flags = json::array();
        for (size_t i = 0; i < run.size(); ++i) {
            flags.push_back(false);
        }
        json segment = { { "kind", "tokens" }, { "token_ids", run }, { "producing", flags } };
        if (run_privileged) {
            segment["privileged"] = true;
        }
        segments.push_back(segment);
        run = json::array();
    };
    for (size_t i = 0; i < tokens.size();) {
        const llama_token t = tokens[i];
        // an exact import of an empty state: its markers with nothing between them
        if (i > 0 && tokens[i - 1] == registry.consume_open && t == registry.consume_close && next_import < exact.size()) {
            const auto stored = registry.store.get(exact[next_import]);
            if (stored && stored->block.n_tokens() == 0) {
                flush();
                json segment = { { "kind", "native_state" }, { "id", exact[next_import++] },
                                 { "native_identity", json::parse(stored->block.identity().to_json().dump()) } };
                segments.push_back(segment);
            }
        }
        if (t >= n_vocab) {
            flush();
            size_t end = i;
            while (end < tokens.size() && tokens[end] >= n_vocab) {
                end++;
            }
            if (next_import >= exact.size()) {
                throw std::invalid_argument("native-rl-capture: a native-state span has no import receipt");
            }
            const std::string & id = exact[next_import++];
            const auto use = registry.resolve(id);
            if (use.tokens.size() != end - i + 2 || use.tokens[1] != t) {
                throw std::invalid_argument("native-rl-capture: native-state span " + id + " differs from its registration");
            }
            const auto stored = registry.store.get(id);
            const native_identity identity = stored ? stored->block.identity() : registry.identity;
            json segment = { { "kind", "native_state" }, { "id", id },
                             { "native_identity", json::parse(identity.to_json().dump()) } };
            if (stored && stored->block.metadata.contains("derivation") && !stored->block.metadata.at("derivation").is_null()) {
                segment["state_derivation"] = json::parse(stored->block.metadata.at("derivation").dump());
            }
            segments.push_back(segment);
            i = end;
            continue;
        }
        const bool privileged = i >= privileged_begin && i < privileged_end;
        if (privileged != run_privileged) {
            flush();
            run_privileged = privileged;
        }
        run.push_back(t);
        n_own++;
        i++;
    }
    flush();
    if (next_import != exact.size()) {
        throw std::invalid_argument("native-rl-capture: an exact native-state import has no span in the prompt");
    }
    return segments;
}

std::string rl_file_sha256(const std::string & path) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        return "";
    }
    sha256_t ctx;
    sha256_init(&ctx);
    std::vector<unsigned char> buf(1 << 22);
    size_t n;
    while ((n = std::fread(buf.data(), 1, buf.size(), f)) > 0) {
        sha256_update(&ctx, buf.data(), n);
    }
    const bool ok = !std::ferror(f);
    std::fclose(f);
    if (!ok) {
        return "";
    }
    unsigned char digest[32];
    sha256_final(&ctx, digest);
    static const char * hex = "0123456789abcdef";
    std::string out;
    for (unsigned char c : digest) {
        out += hex[c >> 4];
        out += hex[c & 15];
    }
    return out;
}

rl_actor_export rl_actor_export_from_model(const llama_model * model, const std::string & path) {
    rl_actor_export out;
    const llama_ftype ftype = llama_model_ftype(model);
    // llama_ftype_name may carry a "(guessed) " prefix; the format is the plain type name, e.g. gguf-bf16, gguf-nvfp4
    std::string name = llama_ftype_name(ftype);
    const std::string guessed = "(guessed) ";
    if (name.rfind(guessed, 0) == 0) {
        name = name.substr(guessed.size());
    }
    if (name.rfind("all ", 0) == 0) {
        name = name.substr(4);
    }
    for (auto & c : name) {
        c = c == ' ' ? '_' : (char) std::tolower((unsigned char) c);
    }
    out.format = "gguf-" + name;
    // F32 and BF16 files carry the identity's own embedding-path dtype (export/native_gguf.py); others round weights
    out.kind   = (name == "f32" || name == "bf16") ? "exact" : "quantized";
    out.sha256 = rl_file_sha256(path);
    return out;
}
