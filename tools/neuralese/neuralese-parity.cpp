// llama-neuralese-parity: run a Neuralese read or write case and dump the results, for
// comparison against a reference implementation.
//
// Case and result files are sequences of named arrays:
//   [int32 name_len][name bytes][int32 dtype: 0 = int32, 1 = float32][int32 count][data]
//
// read case:  mode=0, prefix (tokens ending with the open marker), payload (n * n_embd),
//             suffix (tokens starting with the close marker), steps
//   result:   logits (at the last suffix token), greedy (tokens)
// write case: mode=1, prefix (tokens ending with the open marker), max_length, steps,
//             temperature (optional), noise (optional)
//   result:   h_cut, n, truncated, sketches, shallow, final, mean, log_sigma, payload,
//             stop_logits, logits (at the close marker after read back), greedy

#include "neuralese.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct array {
    int32_t dtype = 0;
    std::vector<int32_t> i;
    std::vector<float>   f;
};

static bool read_arrays(const char * path, std::map<std::string, array> & out) {
    FILE * fp = fopen(path, "rb");
    if (!fp) {
        return false;
    }
    while (true) {
        int32_t len = 0;
        if (fread(&len, 4, 1, fp) != 1) {
            break;
        }
        std::string name(len, '\0');
        int32_t dtype = 0, count = 0;
        if (fread(name.data(), 1, len, fp) != (size_t) len || fread(&dtype, 4, 1, fp) != 1 || fread(&count, 4, 1, fp) != 1) {
            fclose(fp);
            return false;
        }
        array a;
        a.dtype = dtype;
        bool ok = true;
        if (dtype == 0) {
            a.i.resize(count);
            ok = fread(a.i.data(), 4, count, fp) == (size_t) count;
        } else {
            a.f.resize(count);
            ok = fread(a.f.data(), 4, count, fp) == (size_t) count;
        }
        if (!ok) {
            fclose(fp);
            return false;
        }
        out[name] = std::move(a);
    }
    fclose(fp);
    return true;
}

static void write_i32(FILE * fp, const char * name, const std::vector<int32_t> & v) {
    const int32_t len = (int32_t) strlen(name), dtype = 0, count = (int32_t) v.size();
    fwrite(&len, 4, 1, fp); fwrite(name, 1, len, fp); fwrite(&dtype, 4, 1, fp); fwrite(&count, 4, 1, fp);
    fwrite(v.data(), 4, v.size(), fp);
}

static void write_f32(FILE * fp, const char * name, const float * v, size_t n) {
    const int32_t len = (int32_t) strlen(name), dtype = 1, count = (int32_t) n;
    fwrite(&len, 4, 1, fp); fwrite(name, 1, len, fp); fwrite(&dtype, 4, 1, fp); fwrite(&count, 4, 1, fp);
    fwrite(v, 4, n, fp);
}

static int32_t decode_tokens(llama_context * ctx, const std::vector<int32_t> & tokens, llama_pos pos, bool logits_last) {
    if (tokens.empty()) {
        return 0;
    }
    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    batch.n_tokens = (int32_t) tokens.size();
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = pos + (llama_pos) i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = logits_last && i + 1 == tokens.size();
    }
    const int32_t status = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return status;
}

static std::vector<int32_t> greedy(llama_context * ctx, llama_pos pos, int32_t steps, int32_t n_vocab) {
    std::vector<int32_t> out;
    for (int32_t s = 0; s < steps; ++s) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        int32_t best = 0;
        for (int32_t v = 1; v < n_vocab; ++v) {
            if (logits[v] > logits[best]) {
                best = v;
            }
        }
        out.push_back(best);
        if (decode_tokens(ctx, { best }, pos++, true) != 0) {
            break;
        }
    }
    return out;
}

int main(int argc, char ** argv) {
    std::string model_path, heads_path, case_path, out_path;
    int n_threads = 8;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "-m") model_path = argv[i + 1];
        else if (a == "--nz") heads_path = argv[i + 1];
        else if (a == "--case") case_path = argv[i + 1];
        else if (a == "--out") out_path = argv[i + 1];
        else if (a == "-t") n_threads = std::stoi(argv[i + 1]);
    }
    if (model_path.empty() || heads_path.empty() || case_path.empty() || out_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf --nz heads.gguf --case case.bin --out result.bin [-t threads]\n", argv[0]);
        return 1;
    }
    std::map<std::string, array> c;
    if (!read_arrays(case_path.c_str(), c)) {
        fprintf(stderr, "failed to read %s\n", case_path.c_str());
        return 1;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        return 1;
    }
    nz_heads_params hp;
    hp.n_threads = n_threads;
    nz_heads * heads = nz_heads_load(heads_path.c_str(), hp);
    if (!heads) {
        return 1;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_seq_max = 3;
    cp.kv_unified = true;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        return 1;
    }
    const int32_t n_embd  = nz_heads_n_embd(heads);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int32_t steps   = c.count("steps") ? c["steps"].i[0] : 8;
    FILE * fp = fopen(out_path.c_str(), "wb");

    const auto & prefix = c["prefix"].i;
    if (c["mode"].i[0] == 0) {
        const auto & payload = c["payload"].f;
        const auto & suffix  = c["suffix"].i;
        const int32_t n = (int32_t) (payload.size() / n_embd);
        llama_pos pos = 0;
        if (decode_tokens(ctx, prefix, pos, false) != 0) return 2;
        pos += (llama_pos) prefix.size();
        if (nz_read(ctx, heads, payload.data(), n, pos, 0) != 0) return 2;
        pos += n;
        if (decode_tokens(ctx, suffix, pos, true) != 0) return 2;
        pos += (llama_pos) suffix.size();
        write_f32(fp, "logits", llama_get_logits_ith(ctx, -1), n_vocab);
        write_i32(fp, "greedy", greedy(ctx, pos, steps, n_vocab));
    } else {
        std::vector<int32_t> head(prefix.begin(), prefix.end() - 1);
        llama_pos pos = 0;
        if (decode_tokens(ctx, head, pos, false) != 0) return 2;
        pos += (llama_pos) head.size();
        std::vector<float> h_cut(n_embd);
        if (nz_open(ctx, heads, pos, 0, h_cut.data()) != 0) return 2;
        pos += 1;
        nz_write_params wp;
        wp.max_length = c.count("max_length") ? c["max_length"].i[0] : -1;
        wp.temperature = c.count("temperature") ? c["temperature"].f[0] : 0.0f;
        wp.noise = c.count("noise") ? c["noise"].f.data() : nullptr;
        nz_write_result r;
        if (nz_write(ctx, heads, 0, pos, h_cut.data(), wp, r) != 0) return 3;
        if (nz_read_back(ctx, heads, 0, pos, r) != 0) return 4;
        pos += r.n + 1;
        write_f32(fp, "h_cut", h_cut.data(), h_cut.size());
        write_i32(fp, "n", { r.n });
        write_i32(fp, "truncated", { r.truncated ? 1 : 0 });
        write_f32(fp, "sketches", r.sketches.data(), r.sketches.size());
        write_f32(fp, "shallow", r.shallow.data(), r.shallow.size());
        write_f32(fp, "final", r.final_h.data(), r.final_h.size());
        write_f32(fp, "mean", r.mean.data(), r.mean.size());
        write_f32(fp, "log_sigma", r.log_sigma.data(), r.log_sigma.size());
        write_f32(fp, "payload", r.payload.data(), r.payload.size());
        write_f32(fp, "stop_logits", r.stop_logits.data(), r.stop_logits.size());
        write_f32(fp, "logits", llama_get_logits_ith(ctx, -1), n_vocab);
        write_i32(fp, "greedy", greedy(ctx, pos, steps, n_vocab));
    }
    fclose(fp);
    llama_free(ctx);
    nz_heads_free(heads);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
