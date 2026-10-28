// llama-state-span-dump: logits of a token sequence that consumes natlang native state spans (forward-parity checks
// against the reference engine). The input JSON is a list whose numbers are token IDs and whose strings are native
// state GGUF files (common/native-state.h); each state expands to its consume-open marker, its virtual span and its
// consume-close marker. An object {"produce": [ids]} decodes its tokens as a producer span (llama_set_seq_producing: the
// producer adapter applies and their state is captured). Writes F32 logits [n_out, n_vocab] for expanded positions
// >= --from to --out, and with --capture PREFIX the captured state: PREFIX.json (token count and per-kind widths) and
// PREFIX.f32 (the embeddings, then per layer key, value and gated input where present, each [n][width]).
//
//   llama-state-span-dump -m model.gguf --input seq.json --from N --out logits.f32 [--chunk K] [--f32-cache] [--fa on|off|auto]
//                         [--capture PREFIX] [-ngl N]

#include "llama.h"
#include "native-state.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    std::string model_path, input_path, out_path, capture_prefix;
    int32_t from = 0, chunk = 0, threads = 8, n_gpu_layers = 0;
    bool f32_cache = false;
    std::string fa = "auto";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "-m") model_path = next();
        else if (a == "--input") input_path = next();
        else if (a == "--out") out_path = next();
        else if (a == "--from") from = std::stoi(next());
        else if (a == "--chunk") chunk = std::stoi(next());
        else if (a == "-t") threads = std::stoi(next());
        else if (a == "--f32-cache") f32_cache = true;
        else if (a == "--fa") fa = next();
        else if (a == "--capture") capture_prefix = next();
        else if (a == "-ngl") n_gpu_layers = std::stoi(next());
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) return 1;
    auto cparams = llama_context_default_params();
    cparams.n_ctx = 4096; cparams.n_batch = 4096; cparams.n_ubatch = 4096;
    cparams.n_threads = cparams.n_threads_batch = threads;
    if (f32_cache) { cparams.type_k = GGML_TYPE_F32; cparams.type_v = GGML_TYPE_F32; }
    if (fa == "on") cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    else if (fa == "off") cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    else if (fa != "auto") { fprintf(stderr, "--fa takes on, off or auto\n"); return 2; }
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) return 1;

    std::vector<common_native_state> states;
    states.reserve(64);
    std::vector<llama_token> tokens;
    std::vector<bool> producing; // per token
    const auto input = nlohmann::json::parse(std::ifstream(input_path));
    for (const auto & item : input) {
        if (item.is_number_integer()) { tokens.push_back(item.get<int>()); producing.push_back(false); continue; }
        if (item.is_object()) {
            for (const auto & t : item.at("produce")) { tokens.push_back(t.get<int>()); producing.push_back(true); }
            continue;
        }
        states.emplace_back();
        auto & st = states.back();
        if (!common_native_state_load(item.get<std::string>(), llama_model_n_layer(model), st)) return 1;
        if (!common_native_state_register(ctx, st)) return 1;
        for (auto t : common_native_state_tokens(st)) { tokens.push_back(t); producing.push_back(false); }
    }
    const int32_t n = (int32_t) tokens.size();
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int32_t step = chunk > 0 ? chunk : n;
    std::vector<float> logits;
    for (int32_t start = 0; start < n;) {
        // one decode never mixes producing and other tokens: the flag is per sequence
        int32_t m = std::min(step, n - start);
        for (int32_t i = 1; i < m; ++i) if (producing[start + i] != producing[start]) { m = i; break; }
        llama_set_seq_producing(ctx, 0, producing[start]);
        llama_batch batch = llama_batch_init(m, 0, 1);
        for (int32_t i = 0; i < m; ++i) {
            batch.token[i] = tokens[start + i]; batch.pos[i] = start + i;
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = start + i >= from;
        }
        batch.n_tokens = m;
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed at %d\n", start); return 1; }
        for (int32_t i = 0; i < m; ++i) if (start + i >= from) {
            const float * row = llama_get_logits_ith(ctx, i);
            logits.insert(logits.end(), row, row + n_vocab);
        }
        llama_batch_free(batch);
        start += m;
    }
    llama_set_seq_producing(ctx, 0, false);
    if (!capture_prefix.empty()) {
        const int32_t n_cap = llama_state_capture_n_tokens(ctx, 0);
        nlohmann::json meta = { {"n_tokens", n_cap}, {"producer_adapter", llama_model_has_producer_adapter(model)} };
        std::vector<float> all, part;
        auto add = [&](int32_t il, llama_state_capture_kind kind) {
            const int64_t width = llama_state_capture_get(ctx, 0, il, kind, nullptr, 0);
            if (width > 0) {
                part.resize((size_t) width*n_cap);
                llama_state_capture_get(ctx, 0, il, kind, part.data(), (int64_t) part.size());
                all.insert(all.end(), part.begin(), part.end());
            }
            return width;
        };
        meta["embd_width"] = add(-1, LLAMA_STATE_CAPTURE_EMBD);
        for (int32_t il = 0; il < llama_model_n_layer(model); ++il) {
            const int64_t k = add(il, LLAMA_STATE_CAPTURE_K), v = add(il, LLAMA_STATE_CAPTURE_V), c = add(il, LLAMA_STATE_CAPTURE_CONV);
            meta["layers"].push_back({ {"k", k}, {"v", v}, {"conv", c} });
        }
        std::ofstream(capture_prefix + ".json") << meta.dump();
        FILE * cf = fopen((capture_prefix + ".f32").c_str(), "wb");
        fwrite(all.data(), sizeof(float), all.size(), cf);
        fclose(cf);
    }
    FILE * f = fopen(out_path.c_str(), "wb");
    fwrite(logits.data(), sizeof(float), logits.size(), f);
    fclose(f);
    fprintf(stderr, "positions %d, wrote %zu rows of %d\n", n, logits.size() / n_vocab, n_vocab);
    llama_free(ctx); llama_model_free(model);
    return 0;
}
