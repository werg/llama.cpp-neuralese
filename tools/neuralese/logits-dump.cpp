// llama-logits-dump: logits of every position for given token ids, for forward-parity checks against a reference
// implementation (natlang scripts/mellum_gguf_parity.py).
//
//   llama-logits-dump -m MODEL.gguf --ids IDS.i32 --out LOGITS.f32 [--ubatch N] [--greedy N] [-t THREADS] [-ngl N]
//   llama-logits-dump -m MODEL.gguf --tokenize TEXT.txt --ids-out IDS.i32      (special tokens parsed, no BOS added)
//
// IDS.i32 is raw little-endian int32 token ids; LOGITS.f32 is raw float32 [n_tokens][n_vocab]. With --greedy N the
// greedy continuation of the prompt (N tokens) is printed as a JSON line on stdout. Weights are memory-mapped and
// not repacked (no extra buffer types), so a CPU run's resident weight memory is reclaimable page cache.

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<char> read_file(const char * path) {
    std::vector<char> data;
    FILE * fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
    char buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) {
        data.insert(data.end(), buf, buf + n);
    }
    fclose(fp);
    return data;
}

static void write_file(const char * path, const void * data, size_t size) {
    FILE * fp = fopen(path, "wb");
    if (!fp || fwrite(data, 1, size, fp) != size) {
        fprintf(stderr, "cannot write %s\n", path);
        exit(1);
    }
    fclose(fp);
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr, * ids_path = nullptr, * out_path = nullptr;
    const char * text_path = nullptr, * ids_out_path = nullptr;
    int ubatch = 0, greedy = 0, threads = 0, ngl = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", a.c_str());
                exit(1);
            }
            return argv[++i];
        };
        if (a == "-m") model_path = next();
        else if (a == "--ids") ids_path = next();
        else if (a == "--out") out_path = next();
        else if (a == "--tokenize") text_path = next();
        else if (a == "--ids-out") ids_out_path = next();
        else if (a == "--ubatch") ubatch = atoi(next());
        else if (a == "--greedy") greedy = atoi(next());
        else if (a == "-t") threads = atoi(next());
        else if (a == "-ngl") ngl = atoi(next());
        else {
            fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 1;
        }
    }
    if (!model_path || (!ids_path && !text_path)) {
        fprintf(stderr, "usage: %s -m MODEL --ids IDS.i32 --out LOGITS.f32 [--ubatch N] [--greedy N] [-t N] [-ngl N]\n"
                        "       %s -m MODEL --tokenize TEXT --ids-out IDS.i32\n", argv[0], argv[0]);
        return 1;
    }

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers    = ngl;
    mparams.use_extra_bufts = false;
    mparams.vocab_only      = text_path != nullptr;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "cannot load %s\n", model_path);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (text_path) {
        std::vector<char> text = read_file(text_path);
        std::vector<llama_token> ids(text.size() + 16);
        int n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), ids.data(), (int32_t) ids.size(),
                               /*add_special*/ false, /*parse_special*/ true);
        if (n < 0) {
            fprintf(stderr, "tokenize failed\n");
            return 1;
        }
        write_file(ids_out_path, ids.data(), n * sizeof(llama_token));
        printf("{\"n_tokens\": %d}\n", n);
        llama_model_free(model);
        return 0;
    }

    std::vector<char> raw = read_file(ids_path);
    std::vector<llama_token> ids(raw.size() / 4);
    memcpy(ids.data(), raw.data(), ids.size() * 4);
    const int n = (int) ids.size();
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = n + greedy + 16;
    cparams.n_batch  = n;
    cparams.n_ubatch = ubatch > 0 ? ubatch : n;
    cparams.n_seq_max = 1;
    if (threads > 0) {
        cparams.n_threads = cparams.n_threads_batch = threads;
    }
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "cannot create context\n");
        return 1;
    }

    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        batch.token[i] = ids[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;
    }
    batch.n_tokens = n;
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }
    std::vector<float> logits((size_t) n * n_vocab);
    for (int i = 0; i < n; i++) {
        memcpy(logits.data() + (size_t) i * n_vocab, llama_get_logits_ith(ctx, i), n_vocab * sizeof(float));
    }
    if (out_path) {
        write_file(out_path, logits.data(), logits.size() * sizeof(float));
    }

    auto argmax = [&](const float * row) {
        int best = 0;
        for (int v = 1; v < n_vocab; v++) {
            if (row[v] > row[best]) best = v;
        }
        return best;
    };
    printf("{\"n_tokens\": %d, \"n_vocab\": %d, \"greedy\": [", n, n_vocab);
    llama_token tok = argmax(logits.data() + (size_t) (n - 1) * n_vocab);
    for (int g = 0; g < greedy; g++) {
        printf("%s%d", g ? ", " : "", tok);
        if (g + 1 == greedy || llama_vocab_is_eog(vocab, tok)) {
            break;
        }
        llama_batch one = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx, one) != 0) {
            fprintf(stderr, "decode failed\n");
            return 1;
        }
        tok = argmax(llama_get_logits_ith(ctx, -1));
    }
    printf("]}\n");

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
