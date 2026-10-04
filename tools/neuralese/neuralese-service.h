#pragma once

// The Neuralese service: the server protocol of llama-neuralese-server (neuralese-server.cpp's header comment lists
// it) without a transport, for the HTTP server and the WebAssembly build.

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
// One request: GET/PUT/POST with the path and the raw body. Requests are served one at a time.
nz_response  nz_service_handle(nz_service * service, const std::string & method, const std::string & path, const std::string & body);
