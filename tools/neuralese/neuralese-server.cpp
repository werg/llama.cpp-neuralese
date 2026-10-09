// llama-neuralese-server: an OpenAI-style chat server with Neuralese soft-token blocks.
//
// It speaks the same wire protocol as the PyTorch reference server (natlang, training/neuralese,
// `python -m natlang_neuralese.serve`), so clients written against one work against the other:
//
//   POST /v1/chat/completions          chat completion; message content and tool-call arguments may be
//                                      arrays of {"type":"text","text":...} and {"type":"neuralese","id":...}
//   GET  /health, /v1/health, /v1/models
//   GET  /v1/neuralese/info            dialect, width, maximum block length, cutoff
//   PUT  /v1/neuralese/blocks/{id}     store a block (one-tensor safetensors body); the ID is checked
//   GET  /v1/neuralese/blocks/{id}     fetch a block; .../meta for its metadata
//   POST /v1/neuralese/blocks/{id}/pin, .../unpin   pins count per owner (x-natlang-owner)
//   POST /v1/neuralese/collect         {"referenced": [...]}: with an owner, it now holds exactly those (and its pinned
//                                      blocks), and what it released is dropped unless another owner holds or pins it;
//                                      without one, drops the unreferenced blocks no owner holds or pins
//   POST /v1/neuralese/render          the rendered prompt of {"messages", "tools"?} (blocks as "<block>"), for conformance
//   POST /v1/neuralese/encode          text into a block in one forward pass through the port: {"text", "type"?, "context"?}
//   POST /v1/neuralese/write           the write procedure at a write site: {"messages", "prefix"?, "tools"?}
//   POST /v1/neuralese/digest          the digest operator: {"name", "type", "value", "instructions", "system"?, "window"?}
//   POST /v1/neuralese/grad            501: gradient sessions live in the reference server
//
// Request fields beyond OpenAI's: neuralese_temperature (0, the default, delivers the payload mean),
// guidance (guidance.h: envelope and eval-code line checks with rollback; x_natlang_guidance.rejections in the reply),
// neuralese_max_length (capped by --max-block), neuralese_length (an optional size hint: exactly that many vectors, no
// stop decision) and neuralese_passes (with a hint, block-wise in that many parallel passes), neuralese_template (template readout: the reply forced to a call cut at
// its value, {"call", "arguments"?, "argument"?, "value": "write" | "decode"}), and the test hook x_natlang_forced (a plan of text
// strings and {"neuralese": "write"} items that replaces sampling).
//
// Blocks are rendered as <open> + one embedding position per vector + <close>. Literal occurrences of
// the control tokens' text inside message content or tool arguments are tokenized as text, never as the
// control tokens. When the model samples the open token, the write procedure in neuralese.h runs:
// shallow sketch with the stop head, upper-layer completion, content projection, read-back.
//
// Requests are served one at a time (a mutex around the model); continuous batching across requests is
// left to an integration into llama-server.

#include "neuralese-service.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <string>

int main(int argc, char ** argv) {
    nz_service_options o;
    std::string host = "127.0.0.1";
    int port = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "-m") o.model_path = argv[i + 1];
        else if (a == "--nz") o.heads_path = argv[i + 1];
        else if (a == "--host") host = argv[i + 1];
        else if (a == "--port") port = std::stoi(argv[i + 1]);
        else if (a == "-t") o.n_threads = std::stoi(argv[i + 1]);
        else if (a == "-ngl") o.n_gpu_layers = std::stoi(argv[i + 1]);
        else if (a == "-c") o.n_ctx = std::stoi(argv[i + 1]);
        else if (a == "--max-block") o.max_block = std::stoi(argv[i + 1]);
        else if (a == "--dialect") o.dialect = argv[i + 1];
        else if (a == "--alias") o.alias = argv[i + 1];
    }
    if (o.model_path.empty() || o.heads_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf --nz neuralese.gguf [--host H] [--port P (0: any)] [-t threads]\n"
                        "       [-ngl layers] [-c ctx] [--max-block N] [--dialect D] [--alias name]\n", argv[0]);
        return 1;
    }
    std::string error;
    nz_service * service = nz_service_load(o, error);
    if (!service) {
        fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }

    httplib::Server server;
    auto handler = [service](const char * method) {
        return [service, method](const httplib::Request & req, httplib::Response & res) {
            const nz_response r = nz_service_handle(service, method, req.path, req.body, req.get_header_value("x-natlang-owner"));
            res.status = r.status;
            res.set_content(r.body, r.content_type);
        };
    };
    server.Get(".*", handler("GET"));
    server.Put(".*", handler("PUT"));
    server.Post(".*", handler("POST"));

    int bound = port;
    if (port == 0) {
        bound = server.bind_to_any_port(host);
    } else if (!server.bind_to_port(host, port)) {
        bound = -1;
    }
    if (bound <= 0) {
        fprintf(stderr, "failed to bind %s:%d\n", host.c_str(), port);
        return 1;
    }
    nlohmann::ordered_json hello = nlohmann::ordered_json::parse(nz_service_hello(service));
    nlohmann::ordered_json line = { { "listening", "http://" + host + ":" + std::to_string(bound) } };
    for (auto & [key, value] : hello.items()) line[key] = value;
    printf("%s\n", line.dump().c_str());
    fflush(stdout);
    server.listen_after_bind();
    nz_service_free(service);
    return 0;
}
