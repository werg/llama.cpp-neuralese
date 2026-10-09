#pragma once

// The Neuralese service: the server protocol of llama-neuralese-server (neuralese-server.cpp's header comment lists
// it) without a transport, for the HTTP server and the WebAssembly build.

#include <functional>
#include <string>

struct nz_service_options {
    std::string model_path;
    std::string heads_path;
    std::string dialect;                      // empty: the heads' dialect
    std::string alias = "natlang-neuralese";  // the model name /v1/models reports
    int n_threads    = 8;
    int n_gpu_layers = 0;
    int n_ctx        = 8192;
    int max_block    = 64;
    bool use_mmap    = true;                  // false where the file system cannot map files (WORKERFS in a browser)
};

struct nz_response {
    int status = 200;
    std::string body;
    std::string content_type = "application/json";
};

struct nz_service;

// nullptr on failure, with the reason in `error`
nz_service * nz_service_load(const nz_service_options & options, std::string & error);
void         nz_service_free(nz_service * service);
// {"dialect", "cutoff"} as JSON text
std::string  nz_service_hello(const nz_service * service);
// Receives each server-sent event's data (a JSON chunk, an {"error": …} event, and finally "[DONE]") of a streamed
// response as it is produced; the transport frames it ("data: …\n\n").
using nz_event_sink = std::function<void(const std::string & data)>;

// Whether the request is a streamed chat completion (POST /v1/chat/completions with "stream": true): its answer is
// 200 text/event-stream, and every failure after the request is accepted is an error event in the stream.
bool         nz_service_streams(const std::string & method, const std::string & path, const std::string & body);
// One request: GET/PUT/POST with the path, the raw body and the requesting owner (`x-natlang-owner`; "" for none).
// Requests are served one at a time. A streamed request sends its events through `events` while it runs and answers
// with an empty body; without `events` its answer's body is the whole event stream.
nz_response  nz_service_handle(nz_service * service, const std::string & method, const std::string & path, const std::string & body,
                               const std::string & owner = "", const nz_event_sink & events = nullptr);
