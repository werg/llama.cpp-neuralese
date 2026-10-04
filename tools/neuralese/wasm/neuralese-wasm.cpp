// The Neuralese service for WebAssembly: natlang's browser runtime (and Node) load this module and send it requests of
// the llama-neuralese-server protocol as function calls. Model and heads GGUF files are read from the Emscripten file
// system (the JS side writes or mounts them first).
//
//   nzw_load(options JSON) -> 0, or 1 with nzw_error()        {"model", "heads", "n_ctx"?, "max_block"?, "threads"?, ...}
//   nzw_handle(method, path, body, length) -> status            then nzw_body() / nzw_body_length() / nzw_content_type()

#include "../neuralese-service.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>

namespace {
nz_service * service = nullptr;
std::string error;
nz_response last;
}

extern "C" {

int nzw_load(const char * options_json) {
    const auto o = nlohmann::json::parse(options_json, nullptr, false);
    if (o.is_discarded() || !o.is_object()) {
        error = "options are not a JSON object";
        return 1;
    }
    nz_service_options options;
    options.model_path = o.value("model", "");
    options.heads_path = o.value("heads", "");
    options.dialect = o.value("dialect", "");
    options.alias = o.value("alias", "natlang-neuralese");
    options.n_threads = o.value("threads", 1);
    options.n_ctx = o.value("n_ctx", 8192);
    options.max_block = o.value("max_block", 64);
    if (service) nz_service_free(service);
    service = nz_service_load(options, error);
    return service ? 0 : 1;
}

const char * nzw_error() { return error.c_str(); }

const char * nzw_hello() {
    static std::string hello;
    hello = service ? nz_service_hello(service) : "{}";
    return hello.c_str();
}

int nzw_handle(const char * method, const char * path, const uint8_t * body, int length) {
    if (!service) {
        last = { 503, R"({"error":{"code":"not-loaded","message":"call nzw_load first"}})", "application/json" };
        return last.status;
    }
    last = nz_service_handle(service, method, path, std::string(reinterpret_cast<const char *>(body), (size_t) length));
    return last.status;
}

const uint8_t * nzw_body() { return reinterpret_cast<const uint8_t *>(last.body.data()); }
int nzw_body_length() { return (int) last.body.size(); }
const char * nzw_content_type() { return last.content_type.c_str(); }

void nzw_unload() {
    nz_service_free(service);
    service = nullptr;
}

}
