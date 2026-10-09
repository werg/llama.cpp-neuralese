// The Neuralese service: the engine behind llama-neuralese-server, without a transport. `nz_service_handle` answers one
// request of the server protocol (method, path, body) and is shared by the HTTP server (neuralese-server.cpp) and the
// WebAssembly build for the browser runtime (neuralese-wasm.cpp).
//
// llama-neuralese-server: an OpenAI-style chat server with Neuralese soft-token blocks.
//
// It speaks the same wire protocol as the PyTorch reference server (natlang, training/neuralese,
// `python -m natlang_neuralese.serve`), so clients written against one work against the other:
//
//   POST /v1/chat/completions          chat completion; message content and tool-call arguments may be
//                                      arrays of {"type":"text","text":...} and {"type":"neuralese","id":...}
//   GET  /health, /v1/health, /v1/models
//   GET  /v1/neuralese/info            dialect, width, maximum block length, cutoff, capabilities
//   PUT  /v1/neuralese/blocks/{id}     store a block (one-tensor safetensors body); the ID is checked
//   GET  /v1/neuralese/blocks/{id}     fetch a block; .../meta for its metadata
//   POST /v1/neuralese/blocks/{id}/pin, .../unpin   pins count per owner (x-natlang-owner)
//   POST /v1/neuralese/collect         {"referenced": [...]}: with an owner, it now holds exactly those (and its pinned
//                                      blocks), and what it released is dropped unless another owner holds or pins it;
//                                      without one, drops the unreferenced blocks no owner holds or pins
//   POST /v1/neuralese/render          the rendered prompt of {"messages", "tools"?} (blocks as "<block>"), for conformance
//   POST /v1/neuralese/encode          text into a block in one forward pass through the port: {"text", "type"?, "context"?}
//   POST /v1/neuralese/write           the write procedure at a write site: {"messages", "prefix"?, "tools"?}
//   POST /v1/neuralese/view            the builtin view's Neuralese instance: {"value", "instructions"?, "system"?, "window"?};
//                                      every write the template write of view's body
//   POST /v1/neuralese/embed           a block initialised from text token embeddings: {"text", "type"?}
//   POST /v1/neuralese/grad, optim, adapters; GET /v1/neuralese/adapters/{id}/lora
//                                      501 neuralese-<capability>-unavailable: served by the reference server
//
// `"stream": true` on chat completions answers server-sent chat.completion.chunk events (the reference's format).
//
// Request fields beyond OpenAI's: neuralese_temperature (0, the default, delivers the payload mean),
// guidance (guidance.h: envelope and eval-code line checks with rollback; x_natlang_guidance.rejections in the reply),
// neuralese_max_length (capped by --max-block), neuralese_length (an optional size hint: exactly that many vectors, no
// stop decision) and neuralese_passes (with a hint, block-wise in that many parallel passes), neuralese_template (template readout: the reply forced to a call cut at
// its value, {"call", "arguments"?, "argument"?, "argument_path"?, "value": "write" | "decode", "value_type"?}; value_type
// "unknown" writes the value unquoted, typed Neuralese<unknown>), and the test hook x_natlang_forced (a plan of text
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
#include "neuralese.h"
#include "guidance.h"
#include "llama-ext.h"

#include "chat.h"
#include "common.h"
#include "llama.h"

#include "ggml.h"

#include <nlohmann/json.hpp>
#include "hash/hash.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

//
// errors
//

// The write site of the builtin view's Neuralese instance (natlang training/neuralese view.py; pinned by
// tests/fixtures/view-site.json): view's body as the system text, and the reply forced by template readout to
// return_result with the value written (VIEW_TEMPLATE, offered as VIEW_TOOLS).
const char * VIEW_INSTRUCTIONS = "Write a view of value: a shorter form of it that a reader uses in its place.\n\nWithout instructions, the view is faithful: keep everything needed to reproduce value exactly, its content, its structure and its exact details (names, numbers, identifiers, code), and drop only what repeats or can be restated more briefly without loss.\n\nWith instructions, they say what the view is for: keep what that purpose needs, including how value is organised and where in it the rest can be found, and leave out what the purpose does not need.";
const char * VIEW_TOOLS = R"json([{"type":"function","function":{"name":"return_result","description":"Finish the call: value is the result.","parameters":{"type":"object","properties":{"status":{"type":"string","enum":["success"]},"value":{"type":"string"}},"required":["status","value"]}}}])json";
const char * VIEW_TEMPLATE = R"json({"call":"return_result","arguments":{"status":"success"},"argument":"value","value":"write"})json";
const int32_t VIEW_SITE_MARGIN = 1024;

struct request_error : std::runtime_error {
    std::string code;
    int status;
    request_error(std::string code, const std::string & detail, int status = 400)
        : std::runtime_error(code + ": " + detail), code(std::move(code)), status(status) {}
};


//
// content addressing: nz1_ + base32(sha256("natlang.neuralese-block/1\0{dialect}\0{L}x{W}\0{dtype}\0" + bytes))
//

std::string base32_lower(const std::vector<uint8_t> & data) {
    static const char * alphabet = "abcdefghijklmnopqrstuvwxyz234567";
    std::string out;
    uint32_t buffer = 0;
    int bits = 0;
    for (uint8_t byte : data) {
        buffer = (buffer << 8) | byte;
        bits += 8;
        while (bits >= 5) {
            out.push_back(alphabet[(buffer >> (bits - 5)) & 31]);
            bits -= 5;
        }
    }
    if (bits > 0) {
        out.push_back(alphabet[(buffer << (5 - bits)) & 31]);
    }
    return out;
}

std::string content_id(const std::string & dialect, int64_t length, int64_t width, const std::string & dtype,
                       const uint8_t * data, size_t n) {
    std::string head = "natlang.neuralese-block/1";
    head.push_back('\0');
    head += dialect;
    head.push_back('\0');
    head += std::to_string(length) + "x" + std::to_string(width);
    head.push_back('\0');
    head += dtype;
    head.push_back('\0');
    std::string buf = head;
    buf.append((const char *) data, n);
    const std::string hex = hash_sha256_hex(buf.data(), buf.size());
    std::vector<uint8_t> digest(hex.size() / 2);
    for (size_t i = 0; i < digest.size(); ++i) {
        digest[i] = (uint8_t) std::stoi(hex.substr(2 * i, 2), nullptr, 16);
    }
    return "nz1_" + base32_lower(digest);
}

//
// tensor store
//

struct block {
    std::string id;
    std::string dialect;
    int64_t length = 0;
    int64_t width  = 0;
    std::vector<float> payload; // [length * width], f32
    // Preserve uploaded representation for content-addressed GETs. Inference always uses payload above.
    std::string wire_dtype = "f32";
    std::vector<uint8_t> wire_payload;
    std::string type;
    json producer = json::object();
    bool truncated = false;

    json meta() const {
        json m = { { "id", id }, { "dialect", dialect }, { "length", length }, { "width", width }, { "dtype", wire_dtype } };
        if (!type.empty()) {
            m["type"] = type;
        }
        if (!producer.empty()) {
            m["producer"] = producer;
        }
        if (truncated) {
            m["truncated"] = true;
        }
        return m;
    }
};

block make_block(const float * rows, int64_t length, int64_t width, const std::string & dialect,
                 json producer = json::object(), bool truncated = false) {
    block b;
    b.dialect = dialect;
    b.length = length;
    b.width = width;
    b.payload.assign(rows, rows + length * width);
    b.producer = std::move(producer);
    b.truncated = truncated;
    b.id = content_id(dialect, length, width, "f32", (const uint8_t *) b.payload.data(), b.payload.size() * sizeof(float));
    return b;
}

std::string encode_block(const block & b) {
    const bool canonical = b.wire_dtype == "f32" && b.wire_payload.empty();
    const size_t nbytes = canonical ? b.payload.size() * sizeof(float) : b.wire_payload.size();
    const char * data = canonical ? (const char *) b.payload.data() : (const char *) b.wire_payload.data();
    const char * st_dtype = b.wire_dtype == "f16" ? "F16" : b.wire_dtype == "bf16" ? "BF16" : "F32";
    // the metadata string has sorted keys, like the reference server's
    json header = {
        { "__metadata__", { { "natlang.block", nlohmann::json::parse(b.meta().dump()).dump() } } },
        { "payload", { { "dtype", st_dtype }, { "shape", { b.length, b.width } }, { "data_offsets", { 0, nbytes } } } },
    };
    std::string raw = header.dump();
    raw.append((8 - raw.size() % 8) % 8, ' ');
    std::string out(8, '\0');
    const uint64_t size = raw.size();
    memcpy(out.data(), &size, 8);
    out += raw;
    if (nbytes) out.append(data, nbytes);
    return out;
}

block decode_block(const std::string & body) {
    try {
    if (body.size() < 8) throw request_error("neuralese-bad-block", "not a safetensors body");
    uint64_t size = 0;
    memcpy(&size, body.data(), 8);
    if (size > body.size() - 8) throw request_error("neuralese-bad-block", "header exceeds the body");
    json header = json::parse(body.data() + 8, body.data() + 8 + (size_t) size);
    if (!header.is_object()) throw request_error("neuralese-bad-block", "header must be an object");
    json meta = json::object();
    if (header.contains("__metadata__")) {
        const json & metadata = header["__metadata__"];
        if (!metadata.is_object()) throw request_error("neuralese-bad-block", "metadata must be an object");
        if (metadata.contains("natlang.block")) {
            if (!metadata["natlang.block"].is_string()) throw request_error("neuralese-bad-block", "natlang.block metadata must be a JSON string");
            meta = json::parse(metadata["natlang.block"].get<std::string>());
            if (!meta.is_object()) throw request_error("neuralese-bad-block", "natlang.block metadata must decode to an object");
        }
    }
    if (!header.contains("payload") || !header["payload"].is_object())
        throw request_error("neuralese-bad-block", "a block body needs a payload tensor object");
    const json & entry = header["payload"];
    if (!entry.contains("dtype") || !entry["dtype"].is_string() || !entry.contains("shape") ||
        !entry["shape"].is_array() || entry["shape"].size() != 2 || !entry.contains("data_offsets") ||
        !entry["data_offsets"].is_array() || entry["data_offsets"].size() != 2)
        throw request_error("neuralese-bad-block", "payload needs a dtype, rank-2 shape, and two data offsets");
    const std::string st_dtype = entry["dtype"].get<std::string>();
    std::string dtype;
    size_t itemsize = 0;
    if (st_dtype == "F32") { dtype = "f32"; itemsize = 4; }
    else if (st_dtype == "F16") { dtype = "f16"; itemsize = 2; }
    else if (st_dtype == "BF16") { dtype = "bf16"; itemsize = 2; }
    else {
        throw request_error("neuralese-bad-block", "unsupported block dtype " + st_dtype);
    }
    if (!entry["shape"][0].is_number_integer() || !entry["shape"][1].is_number_integer() ||
        !entry["data_offsets"][0].is_number_integer() || !entry["data_offsets"][1].is_number_integer())
        throw request_error("neuralese-bad-block", "shape and data offsets must be integers");
    const int64_t length = entry["shape"][0].get<int64_t>();
    const int64_t width  = entry["shape"][1].get<int64_t>();
    if (length < 0 || width <= 0 || length > std::numeric_limits<int32_t>::max() ||
        width > std::numeric_limits<int32_t>::max() ||
        (length != 0 && (uint64_t) width > std::numeric_limits<uint64_t>::max() / (uint64_t) length))
        throw request_error("neuralese-bad-block", "shape dimensions are out of range");
    const uint64_t elements = (uint64_t) length * (uint64_t) width;
    if (elements > std::numeric_limits<size_t>::max() / itemsize)
        throw request_error("neuralese-bad-block", "payload size overflows this host");
    const size_t expected_bytes = (size_t) elements * itemsize;
    const size_t header_bytes = (size_t) size;
    const size_t data_body_size = body.size() - 8 - header_bytes;
    const size_t start = entry["data_offsets"][0].get<size_t>();
    const size_t end   = entry["data_offsets"][1].get<size_t>();
    if (end < start || end > data_body_size || end - start != expected_bytes) {
        throw request_error("neuralese-bad-block", "block byte count does not match its shape");
    }
    const uint8_t * data = (const uint8_t *) body.data() + 8 + header_bytes + start;
    if (!meta.contains("dialect") || !meta["dialect"].is_string() || meta["dialect"].get<std::string>().empty()) {
        throw request_error("neuralese-bad-block", "block metadata needs a dialect");
    }
    block b;
    b.dialect = meta["dialect"].get<std::string>();
    b.length = length;
    b.width = width;
    b.wire_dtype = dtype;
    if (dtype != "f32") b.wire_payload.assign(data, data + expected_bytes);
    b.id = content_id(b.dialect, length, width, dtype, data, end - start);
    if (meta.contains("id") && meta["id"].is_string() && meta["id"].get<std::string>() != b.id) {
        throw request_error("neuralese-bad-block", "block ID " + meta["id"].get<std::string>() +
                            " does not match its content (" + b.id + ")");
    }
    b.payload.resize((size_t) elements);
    for (size_t i = 0; i < b.payload.size(); ++i) {
        if (dtype == "f32") {
            memcpy(&b.payload[i], data + 4 * i, 4);
        } else if (dtype == "f16") {
            ggml_fp16_t h;
            memcpy(&h, data + 2 * i, 2);
            b.payload[i] = ggml_fp16_to_fp32(h);
        } else {
            uint16_t bits;
            memcpy(&bits, data + 2 * i, 2);
            const uint32_t f = (uint32_t) bits << 16;
            memcpy(&b.payload[i], &f, 4);
        }
    }
    if (meta.contains("type") && meta["type"].is_string()) {
        b.type = meta["type"].get<std::string>();
    }
    if (meta.contains("producer") && meta["producer"].is_object()) {
        b.producer = meta["producer"];
    }
    b.truncated = meta.value("truncated", false);
    return b;
    } catch (const request_error &) {
        throw;
    } catch (const nlohmann::json::exception & e) {
        throw request_error("neuralese-bad-block", e.what());
    }
}

// Owner-scoped holds and pins (the reference's serve/store.py `TensorStore`, without its persistent spill): an owner
// (`x-natlang-owner`) holds the blocks it uploads and every block ID the service names to it, and releases them by
// collecting; a block is dropped only when no owner holds or pins it. A collection without an owner drops only blocks
// that no owner holds or pins. Anonymous pins belong to the owner "".
class tensor_store {
public:
    block put(block b, const std::string & owner = "") {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!owner.empty()) holds_[b.id].insert(owner);
        auto it = blocks_.find(b.id);
        if (it != blocks_.end()) {
            if (it->second.type.empty() && !b.type.empty()) {
                it->second.type = b.type;
            }
            return it->second;
        }
        blocks_[b.id] = b;
        return b;
    }
    bool get(const std::string & id, block & out) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = blocks_.find(id);
        if (it == blocks_.end()) {
            return false;
        }
        out = it->second;
        return true;
    }
    // `owner` holds every stored block among `ids`
    void hold(const std::string & owner, const std::set<std::string> & ids) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto & id : ids) {
            if (blocks_.count(id)) holds_[id].insert(owner);
        }
    }
    void pin(const std::string & id, const std::string & owner = "") {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!blocks_.count(id)) {
            throw request_error("neuralese-unknown-block", id, 404);
        }
        pins_[id][owner]++;
        if (!owner.empty()) holds_[id].insert(owner);
    }
    void unpin(const std::string & id, const std::string & owner = "") {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = pins_.find(id);
        if (it == pins_.end()) return;
        auto by = it->second.find(owner);
        if (by != it->second.end() && --by->second <= 0) it->second.erase(by);
        if (it->second.empty()) pins_.erase(it);
    }
    std::vector<std::string> collect(const std::set<std::string> & referenced, const std::string & owner = "") {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::string> removed;
        if (owner.empty()) {
            for (const auto & [id, _] : blocks_) {
                if (!referenced.count(id) && !holds_.count(id) && !pins_.count(id)) removed.push_back(id);
            }
        } else {
            std::vector<std::string> released;
            for (const auto & [id, owners] : holds_) {
                const auto pinned = pins_.find(id);
                if (owners.count(owner) && !referenced.count(id) && (pinned == pins_.end() || !pinned->second.count(owner))) {
                    released.push_back(id);
                }
            }
            for (const auto & id : referenced) {
                if (blocks_.count(id)) holds_[id].insert(owner);
            }
            for (const auto & id : released) {
                auto it = holds_.find(id);
                it->second.erase(owner);
                if (it->second.empty()) {
                    holds_.erase(it);
                    if (!pins_.count(id)) removed.push_back(id);
                }
            }
        }
        for (const auto & id : removed) blocks_.erase(id);
        return removed;
    }

private:
    std::mutex mutex_;
    std::map<std::string, block> blocks_;
    std::map<std::string, std::set<std::string>> holds_;
    std::map<std::string, std::map<std::string, int>> pins_;
};

//
// placeholders: private-use characters stand in for blocks (U+E010 n U+E011) and for escaped control-token
// text (U+E012 n U+E013) while the chat template runs
//

const std::string PH_OPEN  = "\xEE\x80\x90"; // U+E010
const std::string PH_CLOSE = "\xEE\x80\x91"; // U+E011
const std::string ES_OPEN  = "\xEE\x80\x92"; // U+E012
const std::string ES_CLOSE = "\xEE\x80\x93"; // U+E013

std::string placeholder(size_t i) { return PH_OPEN + std::to_string(i) + PH_CLOSE; }

bool is_parts(const json & v) {
    if (!v.is_array() || v.empty()) {
        return false;
    }
    for (const auto & p : v) {
        if (!p.is_object() || !p.contains("type")) {
            return false;
        }
        const std::string t = p["type"].get<std::string>();
        if (t != "text" && t != "neuralese") {
            return false;
        }
    }
    return true;
}

struct rendered {
    // a segment is template text, an escaped literal (text that must not become control tokens), or a block
    struct segment {
        enum kind_t { TEXT, LITERAL, BLOCK } kind;
        std::string text;
        size_t index = 0;
    };
    std::vector<segment> segments;
    std::vector<std::string> blocks;
};

class renderer {
public:
    // `block_type(id)`: the stored block's type, so a block typed Neuralese<unknown> defaults to value_type "unknown"
    renderer(const std::vector<std::string> & escaped, std::function<std::string(const std::string &)> block_type = nullptr)
        : escaped_(escaped), block_type_(std::move(block_type)) {
        escaped_.insert(escaped_.end(), { PH_OPEN, PH_CLOSE, ES_OPEN, ES_CLOSE });
        std::stable_sort(escaped_.begin(), escaped_.end(), [](const std::string & a, const std::string & b) {
            return a.size() > b.size();
        });
    }

    std::string flatten(const json & parts, std::vector<std::string> & blocks) {
        std::string out;
        for (const auto & part : parts) {
            if (part["type"] == "text") {
                out += escape(part.contains("text") && part["text"].is_string() ? part["text"].get<std::string>() : "");
            } else {
                if (!part.contains("id") || !part["id"].is_string() || part["id"].get<std::string>().rfind("nz1_", 0) != 0) {
                    throw request_error("neuralese-bad-part", "invalid block part " + part.dump());
                }
                const std::string id = part["id"].get<std::string>();
                // A part's value_type, else the stored block's: "unknown" renders unquoted as a whole argument value
                std::string value_type = part.contains("value_type") && part["value_type"].is_string()
                    ? part["value_type"].get<std::string>() : "";
                if (value_type.empty() && block_type_ && block_type_(id) == "Neuralese<unknown>") value_type = "unknown";
                out += placeholder(blocks.size());
                value_types_.push_back(value_type);
                blocks.push_back(id);
            }
        }
        return out;
    }

    // A tool-call argument value that is exactly one block typed "unknown": it renders unquoted, in native value
    // syntax (the reference's render_messages `native_values`).
    void note_native(const std::string & value) {
        if (value.size() <= PH_OPEN.size() + PH_CLOSE.size() || value.compare(0, PH_OPEN.size(), PH_OPEN) != 0 ||
            value.compare(value.size() - PH_CLOSE.size(), PH_CLOSE.size(), PH_CLOSE) != 0) return;
        const std::string number = value.substr(PH_OPEN.size(), value.size() - PH_OPEN.size() - PH_CLOSE.size());
        if (number.empty() || number.size() > 9 || number.find_first_not_of("0123456789") != std::string::npos) return;
        const size_t index = (size_t) std::stoull(number);
        if (index < value_types_.size() && value_types_[index] == "unknown") native_.insert(value);
    }

    // Every string in a JSON value that is exactly a native block marker.
    void note_native_strings(const json & v) {
        if (v.is_string()) note_native(v.get<std::string>());
        else if (v.is_structured()) for (const auto & item : v) note_native_strings(item);
    }

    // The rendered template text with each native block's quotes removed (`"m"` or `'m'` becomes `m`).
    std::string unquote_native(std::string text) const {
        for (const auto & marker : native_) {
            for (const char quote : { '"', '\'' }) {
                const std::string quoted = std::string(1, quote) + marker + std::string(1, quote);
                size_t at = 0;
                while ((at = text.find(quoted, at)) != std::string::npos) {
                    text.replace(at, quoted.size(), marker);
                    at += marker.size();
                }
            }
        }
        return text;
    }

    // control-token text inside content becomes an escape placeholder
    std::string escape(const std::string & text) {
        // Scan original bytes once: never escape placeholders inserted by this scan.
        std::string out;
        size_t pos = 0;
        while (pos < text.size()) {
            bool matched = false;
            for (size_t i = 0; i < escaped_.size(); ++i) {
                const std::string & needle = escaped_[i];
                if (!needle.empty() && text.compare(pos, needle.size(), needle) == 0) {
                    out += ES_OPEN + std::to_string(i) + ES_CLOSE;
                    pos += needle.size();
                    matched = true;
                    break;
                }
            }
            if (!matched) out.push_back(text[pos++]);
        }
        return out;
    }

    std::string escape_json_strings(const json & v, std::vector<std::string> & blocks) {
        // escape every string inside a JSON value (tool arguments as an object), returned as JSON text; a value that
        // is a part list holding a block (as replies return a written argument: "value": [{"type": "neuralese", …}])
        // becomes that block inside the value's string
        std::function<json(const json &)> walk = [&](const json & x) -> json {
            if (is_parts(x)) {
                for (const auto & part : x) {
                    if (part["type"] == "neuralese") {
                        const std::string value = flatten(x, blocks);
                        note_native(value);
                        return value;
                    }
                }
            }
            if (x.is_string()) return escape(x.get<std::string>());
            if (x.is_array()) { json a = json::array(); for (const auto & e : x) a.push_back(walk(e)); return a; }
            if (x.is_object()) { json o = json::object(); for (auto it = x.begin(); it != x.end(); ++it) o[escape(it.key())] = walk(it.value()); return o; }
            return x;
        };
        return walk(v).dump();
    }

    std::string unescape_index(size_t i) const {
        if (i >= escaped_.size()) throw request_error("neuralese-render", "invalid escaped-text placeholder");
        return escaped_[i];
    }

private:
    std::vector<std::string> escaped_;
    std::function<std::string(const std::string &)> block_type_;
    std::vector<std::string> value_types_;  // per block, in order: "unknown", "string" or "" (unset)
    std::set<std::string> native_;
};

// split rendered template text at placeholders
rendered split_rendered(const std::string & text, const std::vector<std::string> & blocks, const renderer & r) {
    rendered out;
    out.blocks = blocks;
    size_t pos = 0;
    std::string run;
    auto flush = [&]() {
        if (!run.empty()) {
            out.segments.push_back({ rendered::segment::TEXT, run, 0 });
            run.clear();
        }
    };
    while (pos < text.size()) {
        if (text.compare(pos, PH_OPEN.size(), PH_OPEN) == 0 || text.compare(pos, ES_OPEN.size(), ES_OPEN) == 0) {
            const bool is_block = text.compare(pos, PH_OPEN.size(), PH_OPEN) == 0;
            const std::string & close = is_block ? PH_CLOSE : ES_CLOSE;
            const size_t start = pos + PH_OPEN.size();
            const size_t end = text.find(close, start);
            if (end != std::string::npos) {
                const std::string number = text.substr(start, end - start);
                if (number.empty() || number.find_first_not_of("0123456789") != std::string::npos)
                    throw request_error("neuralese-render", "invalid placeholder index");
                size_t index;
                try { index = (size_t) std::stoull(number); }
                catch (const std::exception &) { throw request_error("neuralese-render", "placeholder index out of range"); }
                if (is_block && index >= blocks.size())
                    throw request_error("neuralese-render", "invalid block placeholder");
                flush();
                if (is_block) {
                    out.segments.push_back({ rendered::segment::BLOCK, "", index });
                } else {
                    out.segments.push_back({ rendered::segment::LITERAL, r.unescape_index(index), 0 });
                }
                pos = end + close.size();
                continue;
            }
        }
        run.push_back(text[pos++]);
    }
    flush();
    std::vector<size_t> seen;
    for (const auto & s : out.segments) {
        if (s.kind == rendered::segment::BLOCK) {
            seen.push_back(s.index);
        }
    }
    if (seen.size() != blocks.size())
        throw request_error("neuralese-render", "the chat template dropped or duplicated a block");
    std::sort(seen.begin(), seen.end());
    for (size_t i = 0; i < seen.size(); ++i) {
        if (seen[i] != i) {
            throw request_error("neuralese-render", "the chat template dropped or duplicated a block");
        }
    }
    if (seen.size() != blocks.size()) {
        throw request_error("neuralese-render", "the chat template dropped or duplicated a block");
    }
    return out;
}

//
// Pythonic tool calls: [name(arg=literal, ...), ...]
//

class pythonic_parser {
public:
    explicit pythonic_parser(const std::string & s) : s_(s) {}

    std::vector<std::pair<std::string, json>> calls() {
        std::vector<std::pair<std::string, json>> out;
        skip();
        bool bracketed = peek() == '[';
        if (bracketed) {
            ++i_;
        }
        while (true) {
            skip();
            if (bracketed && peek() == ']') {
                ++i_;
                break;
            }
            out.push_back(call());
            skip();
            if (peek() == ',') {
                ++i_;
                continue;
            }
            if (bracketed) {
                expect(']');
            }
            break;
        }
        skip();
        if (i_ != s_.size()) {
            fail("trailing text");
        }
        return out;
    }

private:
    const std::string & s_;
    size_t i_ = 0;

    [[noreturn]] void fail(const std::string & what) { throw std::runtime_error("pythonic: " + what); }
    char peek() const { return i_ < s_.size() ? s_[i_] : '\0'; }
    void skip() { while (i_ < s_.size() && isspace((unsigned char) s_[i_])) ++i_; }
    void expect(char c) { skip(); if (peek() != c) fail(std::string("expected ") + c); ++i_; }

    std::string ident() {
        skip();
        size_t start = i_;
        while (i_ < s_.size() && (isalnum((unsigned char) s_[i_]) || s_[i_] == '_' || s_[i_] == '.')) ++i_;
        if (start == i_) fail("expected a name");
        return s_.substr(start, i_ - start);
    }

    std::pair<std::string, json> call() {
        std::string name = ident();
        expect('(');
        json args = json::object();
        skip();
        if (peek() == ')') {
            ++i_;
            return { name, args };
        }
        while (true) {
            std::string key = ident();
            expect('=');
            args[key] = value();
            skip();
            if (peek() == ',') {
                ++i_;
                skip();
                if (peek() == ')') { ++i_; break; }
                continue;
            }
            expect(')');
            break;
        }
        return { name, args };
    }

    static void utf8(std::string & out, uint32_t cp) {
        if (cp < 0x80) out.push_back((char) cp);
        else if (cp < 0x800) { out.push_back((char) (0xC0 | (cp >> 6))); out.push_back((char) (0x80 | (cp & 0x3F))); }
        else if (cp < 0x10000) { out.push_back((char) (0xE0 | (cp >> 12))); out.push_back((char) (0x80 | ((cp >> 6) & 0x3F))); out.push_back((char) (0x80 | (cp & 0x3F))); }
        else { out.push_back((char) (0xF0 | (cp >> 18))); out.push_back((char) (0x80 | ((cp >> 12) & 0x3F))); out.push_back((char) (0x80 | ((cp >> 6) & 0x3F))); out.push_back((char) (0x80 | (cp & 0x3F))); }
    }

    json string() {
        const char q = s_[i_];
        bool triple = s_.compare(i_, 3, std::string(3, q)) == 0;
        i_ += triple ? 3 : 1;
        std::string out;
        while (true) {
            if (i_ >= s_.size()) fail("unterminated string");
            char c = s_[i_];
            if (triple ? s_.compare(i_, 3, std::string(3, q)) == 0 : c == q) {
                i_ += triple ? 3 : 1;
                break;
            }
            if (c == '\\') {
                char e = i_ + 1 < s_.size() ? s_[i_ + 1] : '\0';
                i_ += 2;
                switch (e) {
                    case 'n': out.push_back('\n'); break;
                    case 't': out.push_back('\t'); break;
                    case 'r': out.push_back('\r'); break;
                    case '0': out.push_back('\0'); break;
                    case '\\': out.push_back('\\'); break;
                    case '\'': out.push_back('\''); break;
                    case '"': out.push_back('"'); break;
                    case '\n': break;
                    case 'x': utf8(out, (uint32_t) std::stoul(s_.substr(i_, 2), nullptr, 16)); i_ += 2; break;
                    case 'u': utf8(out, (uint32_t) std::stoul(s_.substr(i_, 4), nullptr, 16)); i_ += 4; break;
                    case 'U': utf8(out, (uint32_t) std::stoul(s_.substr(i_, 8), nullptr, 16)); i_ += 8; break;
                    default: out.push_back('\\'); out.push_back(e); break;
                }
                continue;
            }
            out.push_back(c);
            ++i_;
        }
        return out;
    }

    json value() {
        skip();
        char c = peek();
        if (c == '\'' || c == '"') return string();
        if (c == '[' || c == '(') {
            const char close = c == '[' ? ']' : ')';
            ++i_;
            json a = json::array();
            skip();
            while (peek() != close) {
                a.push_back(value());
                skip();
                if (peek() == ',') { ++i_; skip(); }
                else if (peek() != close) fail("expected , in sequence");
            }
            ++i_;
            return a;
        }
        if (c == '{') {
            ++i_;
            json o = json::object();
            skip();
            while (peek() != '}') {
                json k = value();
                expect(':');
                o[k.is_string() ? k.get<std::string>() : k.dump()] = value();
                skip();
                if (peek() == ',') { ++i_; skip(); }
                else if (peek() != '}') fail("expected , in dict");
            }
            ++i_;
            return o;
        }
        if (s_.compare(i_, 4, "True") == 0) { i_ += 4; return true; }
        if (s_.compare(i_, 5, "False") == 0) { i_ += 5; return false; }
        if (s_.compare(i_, 4, "None") == 0) { i_ += 4; return nullptr; }
        // JSON's literals too: the template renders nested values as JSON (value={"a": false}), and models write them
        if (s_.compare(i_, 4, "true") == 0) { i_ += 4; return true; }
        if (s_.compare(i_, 5, "false") == 0) { i_ += 5; return false; }
        if (s_.compare(i_, 4, "null") == 0) { i_ += 4; return nullptr; }
        size_t start = i_;
        if (c == '-' || c == '+') ++i_;
        bool is_float = false;
        while (i_ < s_.size() && (isdigit((unsigned char) s_[i_]) || s_[i_] == '.' || s_[i_] == 'e' || s_[i_] == 'E' ||
                                  ((s_[i_] == '-' || s_[i_] == '+') && (s_[i_ - 1] == 'e' || s_[i_ - 1] == 'E')))) {
            if (s_[i_] == '.' || s_[i_] == 'e' || s_[i_] == 'E') is_float = true;
            ++i_;
        }
        if (start == i_) fail("unsupported value");
        const std::string num = s_.substr(start, i_ - start);
        if (is_float) return std::stod(num);
        return std::stoll(num);
    }
};

// A string with block placeholders -> a part array; a string without them stays a string. Placeholder-like text that
// names no written block (private-use characters the model wrote) stays text, as in the reference's chat._to_parts.
json to_parts(const std::string & text, const std::vector<std::string> & ids) {
    std::vector<std::pair<size_t, size_t>> found; // (start, end) of placeholders that name a written block
    std::vector<size_t> indices;
    bool any = false;
    for (size_t pos = text.find(PH_OPEN); pos != std::string::npos; pos = text.find(PH_OPEN, pos + PH_OPEN.size())) {
        const size_t start = pos + PH_OPEN.size();
        size_t end = start;
        while (end < text.size() && isdigit((unsigned char) text[end])) ++end;
        if (end == start || text.compare(end, PH_CLOSE.size(), PH_CLOSE) != 0) continue;
        any = true;
        const std::string number = text.substr(start, end - start);
        if (number.size() > 9 || (size_t) std::stoul(number) >= ids.size()) continue;
        found.push_back({ pos, end + PH_CLOSE.size() });
        indices.push_back((size_t) std::stoul(number));
    }
    if (!any) return text;
    json parts = json::array();
    size_t last = 0;
    for (size_t i = 0; i < found.size(); ++i) {
        if (found[i].first > last) parts.push_back({ { "type", "text" }, { "text", text.substr(last, found[i].first - last) } });
        parts.push_back({ { "type", "neuralese" }, { "id", ids[indices[i]] } });
        last = found[i].second;
    }
    if (last < text.size()) parts.push_back({ { "type", "text" }, { "text", text.substr(last) } });
    return parts;
}

json restore(const json & v, const std::vector<std::string> & ids) {
    if (v.is_string()) return to_parts(v.get<std::string>(), ids);
    if (v.is_array()) { json a = json::array(); for (const auto & e : v) a.push_back(restore(e, ids)); return a; }
    if (v.is_object()) { json o = json::object(); for (auto it = v.begin(); it != v.end(); ++it) o[it.key()] = restore(it.value(), ids); return o; }
    return v;
}

std::string strip(const std::string & s) {
    size_t a = 0, b = s.size();
    while (a < b && isspace((unsigned char) s[a])) ++a;
    while (b > a && isspace((unsigned char) s[b - 1])) --b;
    return s.substr(a, b - a);
}

// A Qwen-family `<tool_call>` body, {"name": …, "arguments": {…}} (arguments may be JSON text) -> [(name, arguments)].
// Throws on anything else (the reference's chat.parse_json_call).
std::vector<std::pair<std::string, json>> parse_json_call(const std::string & body) {
    const json call = json::parse(body);
    if (!call.is_object() || !call.contains("name") || !call["name"].is_string()) throw std::runtime_error("a tool call names its function");
    json arguments = call.contains("arguments") ? call["arguments"] : json::object();
    if (arguments.is_string()) arguments = json::parse(arguments.get<std::string>());
    if (!arguments.is_object()) throw std::runtime_error("tool call arguments are an object");
    return { { call["name"].get<std::string>(), arguments } };
}

std::vector<std::pair<std::string, json>> parse_pythonic_calls(const std::string & body) {
    return pythonic_parser(body).calls();
}

// Tool-call markup in model output: `open` body `close`, the body the shortest run up to the next `close`
// (with `trim`, surrounding whitespace removed: the reference's `<tool_call>\s*(.*?)\s*</tool_call>`).
struct call_span { size_t start, end; std::string body; };

std::vector<call_span> call_spans(const std::string & text, const std::string & open, const std::string & close, bool trim) {
    std::vector<call_span> out;
    size_t last = 0;
    while (true) {
        const size_t a = text.find(open, last);
        const size_t b = a == std::string::npos ? std::string::npos : text.find(close, a + open.size());
        if (b == std::string::npos) break;
        const std::string body = text.substr(a + open.size(), b - a - open.size());
        out.push_back({ a, b + close.size(), trim ? strip(body) : body });
        last = b + close.size();
    }
    return out;
}

using call_parser = std::vector<std::pair<std::string, json>> (*)(const std::string &);

bool parses(call_parser parse, const std::string & body) {
    try { parse(body); return true; } catch (const std::exception &) { return false; }
}

// `text` without the markup spans whose bodies parse; markup that does not parse stays visible (the runtime treats
// leaked markup as a malformed call).
std::string without_calls(const std::string & text, const std::string & open, const std::string & close, bool trim, call_parser parse) {
    std::string out;
    size_t last = 0;
    for (const auto & span : call_spans(text, open, close, trim)) {
        if (!parses(parse, span.body)) continue;
        out += text.substr(last, span.start - last);
        last = span.end;
    }
    return out + text.substr(last);
}

// Model output (with block placeholders) -> an assistant message with parts where blocks are (the reference's
// chat.build_message): thinking in `<think>…</think>`, or everything before a lone `</think>` (thinking templates open
// the block in the generation prompt); Pythonic calls `<|tool_call_start|>[…]<|tool_call_end|>` (LFM2) and then JSON
// calls `<tool_call>{…}</tool_call>` (Qwen-family templates: Maple, Mellum), numbered in that order.
json build_message(std::string text, const std::vector<std::string> & ids, const std::string & prefix) {
    for (const char * marker : { "<|im_end|>", "<|endoftext|>" }) {
        size_t p;
        while ((p = text.find(marker)) != std::string::npos) text.erase(p, strlen(marker));
    }
    std::string reasoning;
    const size_t t0 = text.find("<think>");
    const size_t t1 = t0 == std::string::npos ? std::string::npos : text.find("</think>", t0);
    if (t1 != std::string::npos) {
        reasoning = strip(text.substr(t0 + 7, t1 - t0 - 7));
        text = text.substr(0, t0) + text.substr(t1 + 8);
    } else if (const size_t close = text.find("</think>"); close != std::string::npos) {
        reasoning = strip(text.substr(0, close));
        text = text.substr(close + 8);
    }
    static const std::string py_open = "<|tool_call_start|>", py_close = "<|tool_call_end|>";
    static const std::string js_open = "<tool_call>", js_close = "</tool_call>";
    const struct { const std::string & open, & close; bool trim; call_parser parse; } syntaxes[] = {
        { py_open, py_close, false, parse_pythonic_calls }, { js_open, js_close, true, parse_json_call } };
    json tool_calls = json::array();
    for (const auto & syntax : syntaxes) {
        for (const auto & span : call_spans(text, syntax.open, syntax.close, syntax.trim)) {
            std::vector<std::pair<std::string, json>> calls;
            try { calls = syntax.parse(span.body); } catch (const std::exception &) { continue; }
            for (auto & [name, args] : calls) {
                tool_calls.push_back({ { "id", prefix + "_" + std::to_string(tool_calls.size()) }, { "type", "function" },
                                       { "function", { { "name", name }, { "arguments", restore(args, ids).dump() } } } });
            }
        }
    }
    std::string content = text;
    for (const auto & syntax : syntaxes) content = without_calls(content, syntax.open, syntax.close, syntax.trim, syntax.parse);
    content = strip(content);
    json message = { { "role", "assistant" }, { "content", content.empty() ? json(nullptr) : to_parts(content, ids) } };
    if (!reasoning.empty()) message["reasoning_content"] = reasoning;
    if (!tool_calls.empty()) message["tool_calls"] = tool_calls;
    return message;
}

// Whether `text` ends on a whole UTF-8 character (not inside a multi-byte sequence).
bool utf8_complete(const std::string & text) {
    size_t i = text.size(), back = 0;
    while (i > 0 && back < 4 && ((unsigned char) text[i - 1] & 0xC0) == 0x80) { --i; ++back; }
    if (i == 0) return back == 0;
    const unsigned char lead = (unsigned char) text[i - 1];
    const size_t need = lead < 0x80 ? 0 : (lead >> 5) == 0x6 ? 1 : (lead >> 4) == 0xE ? 2 : (lead >> 3) == 0x1E ? 3 : back;
    return back >= need;
}

uint32_t derive_seed(const std::string & what) {
    const std::string hex = hash_sha256_hex(what.data(), what.size());
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= (uint64_t) std::stoul(hex.substr(2 * i, 2), nullptr, 16) << (8 * i);
    }
    return (uint32_t) (v % (1ull << 31));
}

//
// the engine: one model, one request at a time
//

struct engine {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    nz_heads * heads = nullptr;
    common_chat_templates_ptr templates;
    tensor_store store;
    std::string dialect;
    std::string model_name = "natlang-neuralese";
    int32_t max_block = 64;
    int32_t n_batch = 512;
    int32_t width = 0;
    std::mutex mutex;
    // Weight adapters (natlang model/tiny_adapters.py) as LoRAs, by adapter block ID: a `tiny`/`xs` adapter is exactly
    // a rank-r LoRA (natlang export/adapters.py), loaded with PUT /v1/neuralese/adapters/{id}/lora.
    std::map<std::string, llama_adapter_lora *> loras;

    // The adapters a request binds (`[{"id", "scale"}]`), set on the context for the whole request (every request
    // sets its own, none by default). An adapter given as a code through a projection is decoded by the reference
    // server only, and one that was never loaded fails rather than answer with the base model.
    void bind_adapters(const json & body, const char * key) {
        std::vector<llama_adapter_lora *> bound;
        std::vector<float> scales;
        if (key && body.contains(key) && body[key].is_array()) {
            for (const auto & entry : body[key]) {
                if (!entry.is_object() || !entry.contains("id") || !entry["id"].is_string())
                    throw request_error("neuralese-adapters-projection-unavailable",
                                        "adapters decoded through a projection are served by the reference server", 501);
                const std::string id = entry["id"].get<std::string>();
                const auto found = loras.find(id);
                if (found == loras.end())
                    throw request_error("neuralese-adapter-not-loaded", id + " has no LoRA on this server (PUT /v1/neuralese/adapters/" + id + "/lora)", 409);
                bound.push_back(found->second);
                scales.push_back(entry.value("scale", 1.0f));
            }
        }
        if (llama_set_adapters_lora(ctx, bound.data(), bound.size(), scales.data()) != 0)
            throw request_error("neuralese-adapters", "the context refused the adapters", 500);
    }

    json load_lora(const std::string & id, const std::string & gguf) {
        std::lock_guard<std::mutex> lock(mutex);
        FILE * file = fmemopen((void *) gguf.data(), gguf.size(), "rb");
        llama_adapter_lora * adapter = llama_adapter_lora_init_from_file_ptr(model, file);
        if (file) fclose(file);
        if (!adapter) throw request_error("neuralese-adapter-lora", "not a LoRA for this model", 400);
        const auto found = loras.find(id);
        if (found != loras.end()) {
            llama_set_adapters_lora(ctx, nullptr, 0, nullptr);
            llama_adapter_lora_free(found->second);
        }
        loras[id] = adapter;
        return { { "id", id }, { "loaded", true } };
    }
    std::atomic<uint64_t> next_id { 0 };
    std::vector<std::string> escaped; // control-token text that must stay text inside content

    int decode_tokens(const std::vector<llama_token> & tokens, llama_pos pos, bool logits_last, llama_seq_id seq = 0) {
        for (size_t off = 0; off < tokens.size(); off += (size_t) n_batch) {
            const int32_t n = (int32_t) std::min<size_t>(n_batch, tokens.size() - off);
            llama_batch batch = llama_batch_init(n, 0, 1);
            batch.n_tokens = n;
            for (int32_t i = 0; i < n; ++i) {
                batch.token[i] = tokens[off + i];
                batch.pos[i] = pos + (llama_pos) (off + i);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = seq;
                batch.logits[i] = logits_last && off + i + 1 == tokens.size();
            }
            const int status = llama_decode(ctx, batch);
            llama_batch_free(batch);
            if (status != 0) {
                return status;
            }
        }
        return 0;
    }

    block lookup(const std::string & id) {
        block b;
        if (!store.get(id, b)) {
            throw request_error("neuralese-unknown-block", id + " is not in this server's store; PUT it first");
        }
        if (b.dialect != dialect) {
            throw request_error("neuralese-dialect-mismatch", id + " is in " + b.dialect + ", this server speaks " + dialect);
        }
        if (b.width != width) {
            throw request_error("neuralese-bad-block", id + " has width " + std::to_string(b.width) + ", this server reads " + std::to_string(width));
        }
        return b;
    }

    // The stored type of a block ("" when absent or untyped).
    std::string block_type(const std::string & id) {
        block b;
        return store.get(id, b) ? b.type : std::string();
    }

    rendered render(const json & messages, const json & tools, bool add_generation_prompt = true) {
        renderer r(escaped, [this](const std::string & id) { return block_type(id); });
        std::vector<std::string> blocks;
        json plain = json::array();
        for (const auto & m : messages) {
            json message = m;
            if (message.contains("content")) {
                if (is_parts(message["content"])) {
                    message["content"] = r.flatten(message["content"], blocks);
                } else if (message["content"].is_string()) {
                    message["content"] = r.escape(message["content"].get<std::string>());
                }
            }
            if (message.contains("tool_calls") && message["tool_calls"].is_array()) {
                for (auto & call : message["tool_calls"]) {
                    json & fn = call.contains("function") ? call["function"] : call;
                    json & arguments = fn["arguments"];
                    std::string text;
                    if (is_parts(arguments)) {
                        text = r.flatten(arguments, blocks);
                        const json parsed = json::parse(text, nullptr, false);
                        if (!parsed.is_discarded()) r.note_native_strings(parsed);
                    } else if (arguments.is_string()) {
                        text = arguments.get<std::string>();
                        json parsed = text.empty() ? json::object() : json::parse(text, nullptr, false);
                        if (parsed.is_discarded()) {
                            throw request_error("tool-arguments", "tool-call arguments are not JSON");
                        }
                        text = r.escape_json_strings(parsed, blocks);
                    } else {
                        text = r.escape_json_strings(arguments, blocks);
                    }
                    if (json::parse(text, nullptr, false).is_discarded()) {
                        throw request_error("tool-arguments", "tool-call arguments are not JSON");
                    }
                    arguments = text;
                }
            }
            plain.push_back(message);
        }
        common_chat_templates_inputs inputs;
        inputs.messages = common_chat_msgs_parse_oaicompat(common_json::parse(plain.dump()));
        if (tools.is_array() && !tools.empty()) {
            inputs.tools = common_chat_tools_parse_oaicompat(common_json::parse(tools.dump()));
        }
        inputs.add_generation_prompt = add_generation_prompt;
        inputs.use_jinja = true;
        const common_chat_params params = common_chat_templates_apply(templates.get(), inputs);
        return split_rendered(r.unquote_native(params.prompt), blocks, r);
    }

    // Prefill a rendered prompt (text, literal text and read blocks) on sequence 0 from position 0, with logits at
    // its last position. Returns the next position.
    llama_pos prefill(const rendered & prompt) {
        std::vector<block> inputs;
        for (const auto & id : prompt.blocks) {
            inputs.push_back(lookup(id));
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_pos pos = 0;
        const llama_token open_id = nz_heads_open_token(heads), close_id = nz_heads_close_token(heads);
        std::vector<llama_token> pending;
        auto flush = [&](bool logits) {
            if (pending.empty()) return;
            // Add the model's BOS once, including prompts that begin with a soft block.
            const llama_token bos = llama_vocab_bos(vocab);
            if (pos == 0 && llama_vocab_get_add_bos(vocab) && pending.front() != bos) {
                pending.insert(pending.begin(), bos);
            }
            if (decode_tokens(pending, pos, logits) != 0) throw request_error("decode-failed", "prefill", 500);
            pos += (llama_pos) pending.size();
            pending.clear();
        };
        for (size_t s = 0; s < prompt.segments.size(); ++s) {
            const auto & seg = prompt.segments[s];
            if (seg.kind == rendered::segment::TEXT) {
                auto ids = common_tokenize(vocab, seg.text, false, true);
                pending.insert(pending.end(), ids.begin(), ids.end());
            } else if (seg.kind == rendered::segment::LITERAL) {
                auto ids = common_tokenize(vocab, seg.text, false, false);
                pending.insert(pending.end(), ids.begin(), ids.end());
            } else {
                pending.push_back(open_id);
                flush(false);
                const block & b = inputs[seg.index];
                if (nz_read(ctx, heads, b.payload.data(), (int32_t) b.length, pos, 0) != 0) {
                    throw request_error("decode-failed", "read port", 500);
                }
                pos += (llama_pos) b.length;
                pending.push_back(close_id);
            }
        }
        if (pending.empty()) {
            throw request_error("empty-prompt", "the prompt rendered to nothing");
        }
        flush(true);
        return pos;
    }

    // `value` with the value at `path` (from key `at` on) replaced: object keys are strings, list indices integers,
    // and every key must exist (the reference's chat._replace_value_path).
    static json replace_value_path(const json & value, const json & path, size_t at, const json & replacement) {
        if (at == path.size()) return replacement;
        const json & key = path[at];
        if (value.is_object() && key.is_string() && value.contains(key.get<std::string>())) {
            json out = value;
            out[key.get<std::string>()] = replace_value_path(value[key.get<std::string>()], path, at + 1, replacement);
            return out;
        }
        if (value.is_array() && key.is_number_integer() && key.get<int64_t>() >= 0 && key.get<int64_t>() < (int64_t) value.size()) {
            json out = value;
            out[(size_t) key.get<int64_t>()] = replace_value_path(value[(size_t) key.get<int64_t>()], path, at + 1, replacement);
            return out;
        }
        throw request_error("neuralese-template", "the selected write value path is absent or ambiguous");
    }

    // The value at `path` (the reference's chat._at_value_path).
    static json at_value_path(json value, const json & path) {
        for (const auto & key : path) {
            if (value.is_object() && key.is_string() && value.contains(key.get<std::string>())) {
                value = json(value[key.get<std::string>()]);
            } else if (value.is_array() && key.is_number_integer() && key.get<int64_t>() >= 0 && key.get<int64_t>() < (int64_t) value.size()) {
                value = json(value[(size_t) key.get<int64_t>()]);
            } else {
                throw request_error("neuralese-template", "the selected write value path is absent or ambiguous");
            }
        }
        return value;
    }

    // The model's own rendering of an assistant reply that calls `name` with `arguments` and then `argument` (or the
    // value at `argument_path` inside the arguments, a list of keys that starts at `argument`), cut at that value (the
    // reference's chat.call_reply): with `quoted` the prefix ends with the opening quote and the suffix runs from the
    // closing quote through the end of the turn; without, the prefix ends where the value starts.
    std::pair<std::string, std::string> call_reply(const std::string & name, const json & arguments, const std::string & argument,
                                                   bool quoted, const json & argument_path = json(nullptr)) {
        static const std::string VALUE = "natlangValue7f3a9c";
        json args = arguments.is_object() ? arguments : json::object();
        if (argument_path.is_null()) {
            args[argument] = VALUE;
        } else {
            if (argument_path.empty() || !argument_path[0].is_string() || argument_path[0].get<std::string>() != argument)
                throw request_error("neuralese-template", "write path must begin at the declared argument");
            args = replace_value_path(args, argument_path, 0, VALUE);
        }
        const json user = { { "role", "user" }, { "content", "x" } };
        const json call = { { "role", "assistant" }, { "content", "" }, { "tool_calls", json::array({ { { "type", "function" },
            { "function", { { "name", name }, { "arguments", args.dump() } } } } }) } };
        auto apply = [&](const json & messages, bool generation) {
            common_chat_templates_inputs inputs;
            inputs.messages = common_chat_msgs_parse_oaicompat(common_json::parse(messages.dump()));
            inputs.add_generation_prompt = generation;
            inputs.use_jinja = true;
            return common_chat_templates_apply(templates.get(), inputs).prompt;
        };
        const std::string prompt = apply(json::array({ user }), true), full = apply(json::array({ user, call }), false);
        const size_t at = full.compare(0, prompt.size(), prompt) == 0 ? full.find(VALUE, prompt.size()) : std::string::npos;
        if (at == std::string::npos) throw request_error("neuralese-template", "the chat template renders the call's reply differently");
        std::string prefix = full.substr(prompt.size(), at - prompt.size()), suffix = full.substr(at + VALUE.size());
        if (quoted) return { prefix, suffix };
        if (prefix.empty() || (prefix.back() != '\'' && prefix.back() != '"')) {
            throw request_error("neuralese-template", "the chat template does not quote string arguments");
        }
        prefix.pop_back();
        return { prefix, "" };
    }

    // A written value's cut (the reference's chat.write_reply): value_type "string" sits inside the quoted string;
    // "unknown" sits unquoted, in native value syntax, so the quotes are cut from both sides.
    std::pair<std::string, std::string> write_reply(const std::string & name, const json & arguments, const std::string & argument,
                                                    const std::string & value_type, const json & argument_path) {
        auto [prefix, suffix] = call_reply(name, arguments, argument, true, argument_path);
        if (value_type == "string") return { prefix, suffix };
        if (value_type != "unknown") throw request_error("neuralese-template", "write value_type must be string or unknown");
        if (prefix.empty() || (prefix.back() != '"' && prefix.back() != '\'') || suffix.empty() || suffix.front() != prefix.back())
            throw request_error("neuralese-template", "the chat template does not quote string arguments");
        return { prefix.substr(0, prefix.size() - 1), suffix.substr(1) };
    }

    // log softmax of logits at token: logit[token] - logsumexp(logits)
    float token_logprob(const float * logits, llama_token token) const {
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        float top = logits[0];
        for (int32_t i = 1; i < n_vocab; ++i) top = std::max(top, logits[i]);
        double sum = 0.0;
        for (int32_t i = 0; i < n_vocab; ++i) sum += std::exp((double) logits[i] - top);
        return logits[token] - top - (float) std::log(sum);
    }

    // Decision readout (POST /v1/neuralese/decide): each option's log-probability as the whole assistant reply
    // after the messages. The prompt is prefilled once; tokens every option shares at the start of the reply run
    // once; each option's own tokens run on a copy of the sequence. Matches the reference's
    // GradSession.decision_logprobs: the reply is cut from the full rendering after the prompt, through its last
    // end-of-message marker.
    json decide(const json & body) {
        std::lock_guard<std::mutex> lock(mutex);
        prompt_cache.valid = false;
        return decide_locked(body, false);
    }

    // Batched decision readout (POST /v1/natlang/score and /v1/neuralese/decide_many, natlang
    // plans/BATCHED_EXECUTION.md): {items: [{messages, continuations|options, tools?, adapters?}]} ->
    // {results: [{log_probs, tokens} | {error}]}. Each result equals decide on that item alone; consecutive items
    // with the same rendered prompt and adapters share one prefill (sequence 0 keeps the bare prompt; an item's
    // shared reply head runs in sequence 2). An item that fails returns {error} and fails alone.
    json decide_many(const json & body) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!body.contains("items") || !body["items"].is_array()) throw request_error("neuralese-decision", "items must be a list");
        const json & items = body["items"];
        json results = json::array();
        for (size_t i = 0; i < items.size(); ++i) results.push_back(nullptr);
        // Items run grouped by (adapters, messages, tools), as the reference server sorts them, so every item with the
        // same prompt shares one prefill wherever it sits in the request; results keep the request's order.
        std::vector<std::pair<std::string, size_t>> order;
        for (size_t i = 0; i < items.size(); ++i) {
            const json & item = items[i];
            std::string key;
            if (item.is_object()) {
                const json adapters = item.contains("adapters") ? item["adapters"] : (body.contains("adapters") ? body["adapters"] : json(nullptr));
                key = adapters.dump() + '\x1f' + item.value("messages", json::array()).dump() + '\x1f' +
                      (item.contains("tools") ? item["tools"].dump() : std::string("null"));
            }
            order.emplace_back(key, i);
        }
        std::stable_sort(order.begin(), order.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
        prompt_cache.valid = false;
        for (const auto & entry : order) {
            const json & item = items[entry.second];
            try {
                if (!item.is_object()) throw request_error("neuralese-decision", "an item is an object");
                json request = item;
                if (!request.contains("options") && request.contains("continuations")) request["options"] = request["continuations"];
                if (!request.contains("adapters") && body.contains("adapters")) request["adapters"] = body["adapters"];
                results[entry.second] = decide_locked(request, true);
            } catch (const request_error & error) {
                // "code: detail", as the reference server reports a request error
                prompt_cache.valid = false;
                results[entry.second] = { { "error", std::string(error.what()).substr(0, 500) } };
            } catch (const std::exception & error) {
                prompt_cache.valid = false;
                results[entry.second] = { { "error", ("internal: " + std::string(error.what())).substr(0, 500) } };
            }
        }
        prompt_cache.valid = false;
        return { { "results", results } };
    }

    struct {
        bool valid = false;
        std::string key;
        llama_pos pos = 0;
        std::vector<float> last;
    } prompt_cache;

    json decide_locked(const json & body, bool reuse_prompt) {
        bind_adapters(body, "adapters");
        const json messages = body.value("messages", json::array());
        const json tools = body.contains("tools") ? body["tools"] : json(nullptr);
        const json options = body.value("options", json::array());
        if (!options.is_array() || options.empty()) throw request_error("neuralese-decision", "a decision needs at least one option");
        const rendered prompt = render(messages, tools);
        std::string prompt_text;
        for (const auto & seg : prompt.segments) prompt_text += seg.text;
        std::vector<std::vector<llama_token>> rests;
        for (const auto & option : options) {
            if (!option.is_string()) throw request_error("neuralese-decision", "options are reply texts");
            json full_messages = messages;
            full_messages.push_back({ { "role", "assistant" }, { "content", option } });
            const rendered full = render(full_messages, tools, false);
            if (full.blocks != prompt.blocks) throw request_error("neuralese-decision", "options rendered different prompts");
            std::string full_text;
            for (const auto & seg : full.segments) full_text += seg.text;
            if (full_text.compare(0, prompt_text.size(), prompt_text) != 0) {
                throw request_error("neuralese-decision", "the template renders the reply's prompt differently");
            }
            std::string rest = full_text.substr(prompt_text.size());
            const size_t end = rest.rfind("<|im_end|>");
            if (end != std::string::npos) rest = rest.substr(0, end + std::string("<|im_end|>").size());
            rests.push_back(common_tokenize(vocab, rest, false, true));
        }
        size_t shared = 0;
        if (rests.size() > 1) {
            while (true) {
                bool same = true;
                for (const auto & r : rests) same = same && r.size() > shared && r[shared] == rests[0][shared];
                if (!same) break;
                ++shared;
            }
        }
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        std::string key = body.contains("adapters") ? body["adapters"].dump() : std::string("null");
        key += '\x1f' + prompt_text;
        for (const auto & id : prompt.blocks) key += '\x1f' + id;
        llama_pos pos;
        std::vector<float> last;
        if (reuse_prompt && prompt_cache.valid && prompt_cache.key == key) {
            pos = prompt_cache.pos;
            last = prompt_cache.last;
        } else {
            prompt_cache.valid = false;
            pos = prefill(prompt);
            last.assign(llama_get_logits_ith(ctx, -1), llama_get_logits_ith(ctx, -1) + n_vocab);
            prompt_cache.key = key;
            prompt_cache.pos = pos;
            prompt_cache.last = last;
            prompt_cache.valid = reuse_prompt;
        }
        llama_memory_t mem = llama_get_memory(ctx);
        llama_seq_id source = 0;  // the bare prompt stays in sequence 0 for the next item
        llama_memory_seq_rm(mem, 2, -1, -1);
        if (shared) {
            llama_memory_seq_cp(mem, 0, 2, -1, -1);
            std::vector<llama_token> head(rests[0].begin(), rests[0].begin() + (long) shared);
            if (decode_tokens(head, pos, true, 2) != 0) throw request_error("decode-failed", "decision prefix", 500);
            pos += (llama_pos) shared;
            last.assign(llama_get_logits_ith(ctx, -1), llama_get_logits_ith(ctx, -1) + n_vocab);
            source = 2;
        }
        json log_probs = json::array(), counts = json::array();
        for (const auto & r : rests) {
            const std::vector<llama_token> own(r.begin() + (long) shared, r.end());
            double total = 0.0;
            if (!own.empty()) {
                total += token_logprob(last.data(), own[0]);
                if (own.size() > 1) {
                    llama_memory_seq_rm(mem, 1, -1, -1);
                    llama_memory_seq_cp(mem, source, 1, -1, -1);
                    llama_batch batch = llama_batch_init((int32_t) own.size(), 0, 1);
                    batch.n_tokens = (int32_t) own.size() - 1;
                    for (int32_t i = 0; i < batch.n_tokens; ++i) {
                        batch.token[i] = own[i];
                        batch.pos[i] = pos + i;
                        batch.n_seq_id[i] = 1;
                        batch.seq_id[i][0] = 1;
                        batch.logits[i] = 1;
                    }
                    const int status = llama_decode(ctx, batch);
                    llama_batch_free(batch);
                    if (status != 0) throw request_error("decode-failed", "decision option", 500);
                    for (size_t i = 1; i < own.size(); ++i) total += token_logprob(llama_get_logits_ith(ctx, (int32_t) i - 1), own[i]);
                    llama_memory_seq_rm(mem, 1, -1, -1);
                }
            }
            log_probs.push_back(total);
            counts.push_back(own.size());
        }
        llama_memory_seq_rm(mem, 2, -1, -1);
        return { { "log_probs", log_probs }, { "tokens", counts } };
    }

    std::string next_request_id() { return "chatcmpl-" + std::to_string(next_id++); }

    // A chat completion. `on_delta`, when given, receives the output as it is produced (streaming): {"text": …} for
    // decoded text (whole UTF-8 characters) and {"neuralese": meta} for each written block, as the reference engine's
    // `on_delta` listener.
    json generate(const json & body, const std::function<void(const json &)> & on_delta = nullptr, std::string request_id = "") {
        std::lock_guard<std::mutex> lock(mutex);
        bind_adapters(body, "x_natlang_adapters");
        if (request_id.empty()) request_id = next_request_id();
        const json messages = body.value("messages", json::array());
        const json tools = body.contains("tools") ? body["tools"] : json(nullptr);
        const int32_t max_tokens = body.contains("max_tokens") && body["max_tokens"].is_number()
            ? body["max_tokens"].get<int32_t>()
            : (body.contains("max_completion_tokens") && body["max_completion_tokens"].is_number()
                   ? body["max_completion_tokens"].get<int32_t>() : 512);
        const float temperature = body.contains("temperature") && body["temperature"].is_number()
            ? body["temperature"].get<float>() : 0.0f;
        const float nz_temperature = body.contains("neuralese_temperature") && body["neuralese_temperature"].is_number()
            ? body["neuralese_temperature"].get<float>() : 0.0f;
        int32_t nz_max = max_block;
        if (body.contains("neuralese_max_length") && body["neuralese_max_length"].is_number()) {
            nz_max = std::min(max_block, body["neuralese_max_length"].get<int32_t>());
        }
        const int32_t nz_length = body.contains("neuralese_length") && body["neuralese_length"].is_number_integer()
            ? body["neuralese_length"].get<int32_t>() : 0;
        const int32_t nz_passes = body.contains("neuralese_passes") && body["neuralese_passes"].is_number_integer()
            ? body["neuralese_passes"].get<int32_t>() : 0;
        const std::string seed_key = body.contains("seed") && !body["seed"].is_null() ? body["seed"].dump() : request_id;

        const rendered prompt = render(messages, tools);
        llama_pos pos = prefill(prompt);
        const llama_token open_id = nz_heads_open_token(heads), close_id = nz_heads_close_token(heads);
        const int32_t prompt_positions = pos;

        // forced plan (test hook)
        struct plan_item { bool write; llama_token token; };
        bool forced = body.contains("x_natlang_forced") && body["x_natlang_forced"].is_array();
        std::vector<plan_item> plan;
        if (forced) {
            for (const auto & item : body["x_natlang_forced"]) {
                if (item.is_string()) {
                    for (llama_token t : common_tokenize(vocab, item.get<std::string>(), false, true)) plan.push_back({ false, t });
                } else if (item.is_object() && item.value("neuralese", "") == "write") {
                    plan.push_back({ true, open_id });
                } else {
                    throw request_error("forced-plan", "unsupported forced item " + item.dump());
                }
            }
        }
        // Template readout: the reply is forced to a call cut at its value argument (call_reply); "write" writes the
        // value as a block and closes the call, "decode" decodes from the value on. The test hook's plan replaces it.
        bool free_after = false;
        // A template write of value_type "unknown": the block is typed Neuralese<unknown> and its parts in the reply
        // carry value_type "unknown" (the value sits unquoted in the model's cache).
        const json templ = body.contains("neuralese_template") && body["neuralese_template"].is_object() ? body["neuralese_template"] : json(nullptr);
        const bool unknown_value = !templ.is_null() && templ.contains("value_type") && templ["value_type"] == "unknown";
        const bool unknown_write = unknown_value && templ.value("value", json(nullptr)) == "write";
        if (!forced && !templ.is_null()) {
            const json & t = templ;
            const std::string mode = t.contains("value") && t["value"].is_string() ? t["value"].get<std::string>() : "";
            if (!t.contains("call") || !t["call"].is_string() || (mode != "write" && mode != "decode")) {
                throw request_error("neuralese-template", "a template needs a call name and value 'write' or 'decode'");
            }
            const json argument_path = t.contains("argument_path") ? t["argument_path"] : json(nullptr);
            if (!argument_path.is_null()) {
                bool keys = argument_path.is_array();
                if (keys) for (const auto & key : argument_path) keys = keys && (key.is_string() || key.is_number_integer());
                if (!keys) throw request_error("neuralese-template", "argument_path must be a list of string or integer keys");
            }
            const std::string argument = t.contains("argument") && t["argument"].is_string() && !t["argument"].get<std::string>().empty()
                ? t["argument"].get<std::string>() : "value";
            const json arguments = t.contains("arguments") && t["arguments"].is_object() ? t["arguments"] : json::object();
            std::string prefix, suffix;
            if (mode == "decode") {
                std::tie(prefix, suffix) = call_reply(t["call"].get<std::string>(), arguments, argument, false, argument_path);
            } else {
                if (t.contains("value_type") && !t["value_type"].is_string())
                    throw request_error("neuralese-template", "write value_type must be string or unknown");
                const std::string value_type = t.contains("value_type") ? t["value_type"].get<std::string>() : "string";
                std::tie(prefix, suffix) = write_reply(t["call"].get<std::string>(), arguments, argument, value_type, argument_path);
            }
            auto prefix_tokens = common_tokenize(vocab, prefix, false, true);
            if (mode == "decode") {
                // A text cut at value= can split the trained =[ or ={ token.
                // Force only the shared token prefix; decode the boundary normally.
                const std::vector<std::string> continuations = {
                    "[]", "{}", "\"x\"", "'x'", "true", "false", "null", "-1",
                    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
                };
                for (const auto & continuation : continuations) {
                    const auto other = common_tokenize(vocab, prefix + continuation, false, true);
                    size_t n = 0;
                    while (n < std::min(prefix_tokens.size(), other.size()) && prefix_tokens[n] == other[n]) ++n;
                    prefix_tokens.resize(n);
                }
                if (prefix_tokens.empty()) throw request_error("neuralese-template", "chat template has no stable decode prefix");
            }
            for (llama_token tok : prefix_tokens) plan.push_back({ false, tok });
            if (mode == "write") {
                plan.push_back({ true, open_id });
                for (llama_token tok : common_tokenize(vocab, suffix, false, true)) plan.push_back({ false, tok });
            }
            forced = true;
            free_after = mode == "decode";
        }
        // Guidance (guidance.h): envelope and line checks with rollback. Snapshots of sequence 0 live on the spare
        // sequences 3..6, at most four, as in the reference.
        std::optional<nz_guide> guide;
        if (body.contains("guidance") && ((body["guidance"].is_boolean() && body["guidance"].get<bool>()) || body["guidance"].is_object())) {
            const json g = body["guidance"].is_object() ? body["guidance"] : json::object();
            nz_guide_settings settings;
            settings.require_call = g.value("require_call", body.value("tool_choice", json("auto")) == json("required"));
            if (g.contains("tools") && g["tools"].is_array() && !g["tools"].empty()) {
                for (const auto & n : g["tools"]) settings.tools.push_back(n.get<std::string>());
            } else if (tools.is_array()) {
                for (const auto & t : tools) {
                    const json & fn = t.contains("function") ? t["function"] : t;
                    if (fn.contains("name") && fn["name"].is_string()) settings.tools.push_back(fn["name"].get<std::string>());
                }
            }
            settings.repeat = g.value("repeat", 3);
            settings.syntax = g.value("syntax", true);
            settings.retries = g.value("retries", 4);
            settings.run = g.value("run", 4);
            guide = nz_guide { settings, {}, {} };
            const std::string prefix = guide->forced_prefix();
            if (!prefix.empty() && !forced) {
                for (llama_token tok : common_tokenize(vocab, prefix, false, true)) plan.push_back({ false, tok });
                forced = true;
                free_after = true;
            }
        }
        size_t plan_at = 0;

        // Streaming: text goes out in whole UTF-8 characters (a token may end inside one).
        std::string unsent;
        auto flush_text = [&]() {
            if (on_delta && !unsent.empty()) on_delta({ { "text", unsent } });
            unsent.clear();
        };
        auto emit_text = [&](llama_token token) {
            if (!on_delta) return;
            unsent += common_token_to_piece(vocab, token, true);
            if (utf8_complete(unsent)) flush_text();
        };
        auto emit_block = [&](const block & b) {
            if (on_delta) on_delta({ { "neuralese", b.meta() } });
        };

        llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
        if (temperature > 0.0f) {
            llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(derive_seed("text\x1f" + seed_key)));
        } else {
            llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
        }

        struct item { bool is_block; llama_token token; size_t block_index; };
        std::vector<item> items;
        std::vector<block> written;
        int32_t generated = 0;
        std::string finish = "stop";
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        llama_memory_t mem = llama_get_memory(ctx);
        struct snapshot { size_t count; llama_seq_id seq; std::vector<float> logits; llama_pos pos; int32_t generated; };
        std::vector<snapshot> snapshots;
        std::vector<llama_seq_id> spare { 3, 4, 5, 6 };
        std::map<size_t, std::set<llama_token>> bans;
        std::map<size_t, int32_t> retries;
        std::vector<std::string> pieces;
        json rejections = json::array();
        std::vector<float> cur;
        auto current_logits = [&]() { const float * l = llama_get_logits_ith(ctx, -1); cur.assign(l, l + n_vocab); };
        auto take_snapshot = [&]() {
            if (spare.empty()) {
                spare.push_back(snapshots.front().seq);
                snapshots.erase(snapshots.begin());
            }
            const llama_seq_id s = spare.back();
            spare.pop_back();
            llama_memory_seq_rm(mem, s, -1, -1);
            llama_memory_seq_cp(mem, 0, s, -1, -1);
            snapshots.push_back({ items.size(), s, cur, pos, generated });
        };
        if (guide) {
            current_logits();
            take_snapshot();
        }
        // After a token: the reply checks; on a rejection roll back to the offending line and ban its first token.
        auto guide_step = [&]() {
            std::string reply;
            std::vector<size_t> starts;
            for (const auto & piece : pieces) {
                starts.push_back(reply.size());
                reply += piece;
            }
            const auto verdict = guide->check(reply);
            if (!verdict) {
                const auto ends = [&](const std::string & tail) {
                    return reply.size() >= tail.size() && reply.compare(reply.size() - tail.size(), tail.size(), tail) == 0;
                };
                if (ends("\\n") || ends("\n") || ends("[")) take_snapshot();
                return;
            }
            const size_t at = verdict->at;
            size_t target = 0;
            for (size_t i = 0; i < starts.size(); ++i) if (starts[i] <= at) target = i;
            const int32_t tries = retries.count(at) ? retries[at] : 0;
            const snapshot * usable = nullptr;
            for (const auto & snap : snapshots) if (snap.count <= target) usable = &snap;
            bool blocked = !usable;
            if (usable) for (size_t i = usable->count; i < target; ++i) blocked = blocked || items[i].is_block;
            if (tries >= guide->settings.retries || blocked) {
                guide->accept(at);
                rejections.push_back({ { "reason", verdict->reason }, { "offset", at }, { "accepted", true } });
                return;
            }
            retries[at] = tries + 1;
            rejections.push_back({ { "reason", verdict->reason }, { "offset", at } });
            bans[target].insert(items[target].token);
            llama_memory_seq_rm(mem, 0, -1, -1);
            llama_memory_seq_cp(mem, usable->seq, 0, -1, -1);
            cur = usable->logits;
            pos = usable->pos;
            generated = usable->generated;
            std::vector<llama_token> between;
            for (size_t i = usable->count; i < target; ++i) between.push_back(items[i].token);
            if (!between.empty()) {
                if (decode_tokens(between, pos, true) != 0) throw request_error("decode-failed", "guidance rollback", 500);
                current_logits();
                pos += (llama_pos) between.size();
                generated += (int32_t) between.size();
            }
            guide->rewind(starts[target]);
            items.resize(target);
            pieces.resize(target);
            for (auto it = snapshots.begin(); it != snapshots.end();) {
                if (it->count > target) { spare.push_back(it->seq); it = snapshots.erase(it); } else ++it;
            }
            for (auto it = bans.begin(); it != bans.end();) it = it->first > target ? bans.erase(it) : std::next(it);
        };
        while (true) {
            if (generated >= max_tokens) { finish = "length"; break; }
            llama_token token;
            if (forced && free_after && plan_at >= plan.size()) forced = false;
            if (forced) {
                if (plan_at >= plan.size()) break;
                token = plan[plan_at++].token;
            } else if (guide) {
                std::vector<llama_token_data> candidates((size_t) n_vocab);
                const auto & banned = bans[items.size()];
                for (int32_t i = 0; i < n_vocab; ++i) {
                    candidates[i] = { i, banned.count(i) ? -INFINITY : cur[i], 0.0f };
                }
                llama_token_data_array array { candidates.data(), candidates.size(), -1, false };
                llama_sampler_apply(sampler, &array);
                token = array.data[array.selected].id;
            } else {
                token = llama_sampler_sample(sampler, ctx, -1);
            }
            if (llama_vocab_is_eog(vocab, token)) break;
            if (token == open_id) {
                const int32_t remaining = max_tokens - generated;
                // Do not emit a partial opening marker. A complete soft block needs open + >=1 payload + close.
                if (remaining < 3) { finish = "length"; break; }
                std::vector<float> h_cut(width);
                if (nz_open(ctx, heads, pos, 0, h_cut.data()) != 0) throw request_error("decode-failed", "open", 500);
                const llama_pos first = pos + 1;
                nz_write_params wp;
                wp.max_length = std::min(nz_max, remaining - 2);
                wp.temperature = nz_temperature;
                wp.length = nz_length > 0 ? std::min(nz_length, wp.max_length) : 0;
                wp.passes = wp.length > 0 ? nz_passes : 0;
                const size_t index = written.size();
                wp.seed = derive_seed("neuralese\x1f" + seed_key + "\x1f" + std::to_string(index));
                nz_write_result r;
                if (nz_write(ctx, heads, 0, first, h_cut.data(), wp, r) != 0) throw request_error("decode-failed", "write", 500);
                if (nz_read_back(ctx, heads, 0, first, r) != 0) throw request_error("decode-failed", "read back", 500);
                json record = { { "kind", "write" }, { "request", request_id }, { "index", index },
                                { "cutoff", nz_heads_cutoff(heads) }, { "temperature", nz_temperature },
                                { "seed", wp.seed }, { "rng", "mt19937-normal" }, { "stop_logits", r.stop_logits } };
                if (wp.length > 0) {
                    record["length_hint"] = wp.length;
                    record["passes"] = r.passes;
                }
                std::vector<block> parts;
                if (r.n > 0) {
                    parts.push_back(make_block(r.mean.data(), r.n, width, dialect, { { "kind", "payload-mean" } }));
                    parts.push_back(make_block(r.log_sigma.data(), r.n, width, dialect, { { "kind", "payload-log-sigma" } }));
                    record["mean"] = parts[0].id;
                    record["log_sigma"] = parts[1].id;
                }
                // The payload first: at temperature 0 it is its mean, and the store keeps the first block of an ID, which
                // must be the one carrying the write record.
                block made = make_block(r.payload.data(), r.n, width, dialect, record, r.truncated);
                if (unknown_value) made.type = "Neuralese<unknown>";
                block b = store.put(made);
                for (const auto & part : parts) store.put(part);
                written.push_back(b);
                items.push_back({ true, 0, index });
                if (guide) {
                    pieces.push_back("__nz");
                    current_logits();
                }
                emit_block(b);
                pos = first + r.n + 1;
                generated += r.n + 2;
                // a forced plan's own close marker is written by the procedure instead
                if (forced && plan_at < plan.size() && !plan[plan_at].write && plan[plan_at].token == close_id) ++plan_at;
                continue;
            }
            if (decode_tokens({ token }, pos, true) != 0) throw request_error("decode-failed", "decode", 500);
            llama_sampler_accept(sampler, token);
            pos += 1;
            generated += 1;
            items.push_back({ false, token, 0 });
            if (guide) {
                current_logits();
                pieces.push_back(common_token_to_piece(vocab, token, true));
                guide_step();
            }
            // as the reference: after the reply checks (text streamed before a rollback is not retracted)
            emit_text(token);
        }
        llama_sampler_free(sampler);
        flush_text();
        for (const auto & snap : snapshots) llama_memory_seq_rm(mem, snap.seq, -1, -1);

        // output text with block placeholders
        std::string text;
        std::vector<std::string> ids;
        for (const auto & it : items) {
            if (it.is_block) {
                // A value written unquoted (value_type "unknown") is parsed as the string the placeholder is.
                text += unknown_write ? "\"" + placeholder(ids.size()) + "\"" : placeholder(ids.size());
                ids.push_back(written[it.block_index].id);
            } else {
                text += common_token_to_piece(vocab, it.token, true);
            }
        }
        json message = build_message(text, ids, request_id);
        if (unknown_write && message.contains("tool_calls")) {
            // The written value's parts carry value_type "unknown" (the reference engine's `_response`).
            std::function<void(json &)> mark = [&](json & v) {
                if (v.is_array()) { for (auto & item : v) mark(item); }
                else if (v.is_object()) {
                    if (v.value("type", json(nullptr)) == "neuralese") v["value_type"] = "unknown";
                    else for (auto & item : v) mark(item);
                }
            };
            const std::string argument = templ.contains("argument") && templ["argument"].is_string() && !templ["argument"].get<std::string>().empty()
                ? templ["argument"].get<std::string>() : "value";
            for (auto & call : message["tool_calls"]) {
                json arguments = json::parse(call["function"]["arguments"].get<std::string>());
                if (templ.contains("argument_path") && !templ["argument_path"].is_null()) {
                    // mark the value in place: walk the path through the arguments
                    at_value_path(arguments, templ["argument_path"]); // absent: neuralese-template
                    json * at = &arguments;
                    for (const auto & key : templ["argument_path"]) at = key.is_string() ? &(*at)[key.get<std::string>()] : &(*at)[(size_t) key.get<int64_t>()];
                    mark(*at);
                } else if (arguments.is_object() && arguments.contains(argument)) {
                    mark(arguments[argument]);
                }
                call["function"]["arguments"] = arguments.dump();
            }
        }
        if (finish == "stop" && message.contains("tool_calls")) finish = "tool_calls";
        json metas = json::array();
        for (const auto & b : written) metas.push_back(b.meta());
        json response = {
            { "id", request_id }, { "object", "chat.completion" }, { "created", (int64_t) time(nullptr) },
            { "model", model_name },
            { "choices", json::array({ { { "index", 0 }, { "message", message }, { "finish_reason", finish } } }) },
            { "usage", { { "prompt_tokens", prompt_positions }, { "completion_tokens", generated },
                         { "total_tokens", prompt_positions + generated } } },
            { "neuralese", { { "dialect", dialect }, { "blocks", metas } } },
        };
        if (guide) response["x_natlang_guidance"] = { { "rejections", rejections } };
        return response;
    }

    // Decode embedding rows on seq from pos through the current layer range, in batches, collecting the residual
    // entering layer `lid` at every row (the caller enables its extraction).
    std::vector<float> run_rows(const std::vector<float> & rows, llama_pos pos, llama_seq_id seq, uint32_t lid) {
        const int32_t n = (int32_t) (rows.size() / (size_t) width);
        std::vector<float> out;
        out.reserve(rows.size());
        for (int32_t off = 0; off < n; off += n_batch) {
            const int32_t m = std::min(n_batch, n - off);
            llama_batch batch = llama_batch_init(m, width, 1);
            batch.n_tokens = m;
            memcpy(batch.embd, rows.data() + (size_t) off * width, (size_t) m * width * sizeof(float));
            for (int32_t i = 0; i < m; ++i) {
                batch.pos[i] = pos + off + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = seq;
                batch.logits[i] = 0;
            }
            const int status = llama_decode(ctx, batch);
            llama_batch_free(batch);
            if (status != 0) throw request_error("decode-failed", "encode", 500);
            const float * h = llama_get_embeddings_layer_inp(ctx, lid);
            out.insert(out.end(), h, h + (size_t) m * width);
        }
        return out;
    }

    std::vector<float> tok_embd; // the token embedding matrix [n_vocab * width], loaded on the first encode

    // Encode (POST /v1/neuralese/encode): text into a block in one forward pass through the port, as the reference's
    // grad.encode_text. The write site is the rendered `context` (or BOS), then the open marker; the text's token
    // embeddings, interface-normed, are supplied at the sketch positions (S3 phase A), layers [0, k) and then [k, n)
    // run over them, and the content projection gives one payload vector per token (the mean). No stop decision.
    // The token embedding matrix, loaded once (`code`: the error code of the endpoint that needs it).
    void load_tok_embd(const char * code) {
        if (!tok_embd.empty()) return;
        tok_embd.resize(llama_model_get_tok_embd(model, nullptr));
        if (tok_embd.empty() || llama_model_get_tok_embd(model, tok_embd.data()) != tok_embd.size() ||
            tok_embd.size() % (size_t) width != 0) {
            tok_embd.clear();
            throw request_error(code, "the token embedding matrix is unavailable", 500);
        }
    }

    static std::string text_field(const json & body, const char * key) {
        return body.contains(key) && body[key].is_string() ? body[key].get<std::string>() : std::string();
    }

    // Embed (POST /v1/neuralese/embed): a block initialised from text, the token embeddings of `text` (runtime
    // registration, no writer), as the reference's grad.embed_text. Producer {"kind": "text-init", "text"}.
    json embed(const json & body) {
        std::lock_guard<std::mutex> lock(mutex);
        const std::string text = text_field(body, "text");
        const std::vector<llama_token> ids = common_tokenize(vocab, text, false, true);
        if (ids.empty()) throw request_error("neuralese-embed", "empty text");
        load_tok_embd("neuralese-embed");
        const int32_t n = (int32_t) ids.size();
        std::vector<float> rows((size_t) n * width);
        for (int32_t i = 0; i < n; ++i) {
            if (ids[i] < 0 || (size_t) (ids[i] + 1) * width > tok_embd.size()) throw request_error("neuralese-embed", "token outside the embedding matrix", 500);
            std::copy_n(tok_embd.data() + (size_t) ids[i] * width, width, rows.data() + (size_t) i * width);
        }
        block b = make_block(rows.data(), n, width, dialect, { { "kind", "text-init" }, { "text", text } });
        if (body.contains("type") && body["type"].is_string()) b.type = body["type"].get<std::string>();
        return store.put(b).meta();
    }

    json encode(const json & body) {
        std::lock_guard<std::mutex> lock(mutex);
        bind_adapters(body, nullptr);
        const std::string text = text_field(body, "text");
        const std::vector<llama_token> ids = common_tokenize(vocab, text, false, true);
        if (ids.empty()) throw request_error("neuralese-encode", "empty text");
        load_tok_embd("neuralese-encode");
        llama_pos pos = 0;
        const json context = body.contains("context") ? body["context"] : json(nullptr);
        if (context.is_array() && !context.empty()) {
            const rendered prompt = render(context, json(nullptr));
            if (!prompt.blocks.empty()) throw request_error("neuralese-encode", "the encoding context may not hold blocks");
            pos = prefill(prompt);
        } else {
            llama_memory_clear(llama_get_memory(ctx), true);
            const llama_token bos = llama_vocab_bos(vocab);
            if (bos != LLAMA_TOKEN_NULL) {
                if (decode_tokens({ bos }, 0, false) != 0) throw request_error("decode-failed", "encode prefix", 500);
                pos = 1;
            }
        }
        std::vector<float> h_cut(width);
        if (nz_open(ctx, heads, pos, 0, h_cut.data()) != 0) throw request_error("decode-failed", "open", 500);
        const llama_pos first = pos + 1;
        const int32_t n = (int32_t) ids.size();
        std::vector<float> embedded((size_t) n * width), inputs((size_t) n * width);
        for (int32_t i = 0; i < n; ++i) {
            std::copy_n(tok_embd.data() + (size_t) ids[i] * width, width, embedded.data() + (size_t) i * width);
        }
        nz_interface(heads, embedded.data(), inputs.data(), n);
        const int32_t k = nz_heads_cutoff(heads), n_layer = llama_model_n_layer(model);
        llama_memory_t mem = llama_get_memory(ctx);
        for (llama_seq_id s : { 1, 2 }) {
            llama_memory_seq_rm(mem, s, -1, -1);
            llama_memory_seq_cp(mem, 0, s, -1, -1);
        }
        std::vector<float> shallow, final_h;
        try {
            llama_set_embeddings_layer_inp(ctx, (uint32_t) k, true);
            llama_set_layer_range(ctx, 0, k);
            shallow = run_rows(inputs, first, 1, (uint32_t) k);
            llama_set_embeddings_layer_inp(ctx, (uint32_t) k, false);
            llama_set_embeddings_layer_inp(ctx, (uint32_t) n_layer, true);
            llama_set_layer_range(ctx, k, -1);
            final_h = run_rows(shallow, first, 2, (uint32_t) n_layer);
        } catch (...) {
            llama_set_layer_range(ctx, 0, -1);
            llama_set_embeddings_layer_inp(ctx, (uint32_t) n_layer, false);
            for (llama_seq_id s : { 1, 2 }) llama_memory_seq_rm(mem, s, -1, -1);
            throw;
        }
        llama_set_layer_range(ctx, 0, -1);
        llama_set_embeddings_layer_inp(ctx, (uint32_t) n_layer, false);
        for (llama_seq_id s : { 1, 2 }) llama_memory_seq_rm(mem, s, -1, -1);
        std::vector<float> mean((size_t) n * width), log_sigma((size_t) n * width);
        nz_content(heads, inputs.data(), final_h.data(), n, mean.data(), log_sigma.data());
        block b = make_block(mean.data(), n, width, dialect, { { "kind", "text-encode" }, { "text", text } });
        if (body.contains("type") && body["type"].is_string()) b.type = body["type"].get<std::string>();
        return store.put(b).meta();
    }

    // Write (POST /v1/neuralese/write): the write procedure at a write site. The reply is forced to `prefix` and then
    // the open marker; the stop head decides the length.
    json write(const json & body) {
        const std::string prefix = body.value("prefix", "");
        json forced = json::array();
        if (!prefix.empty()) forced.push_back(prefix);
        forced.push_back({ { "neuralese", "write" } });
        json request = { { "messages", body.value("messages", json::array()) }, { "x_natlang_forced", forced },
                         { "max_tokens", max_block + (int32_t) common_tokenize(vocab, prefix, false, true).size() + 8 } };
        if (body.contains("tools")) request["tools"] = body["tools"];
        if (body.contains("neuralese_temperature")) request["neuralese_temperature"] = body["neuralese_temperature"];
        if (body.contains("length")) request["neuralese_length"] = body["length"];
        if (body.contains("passes")) request["neuralese_passes"] = body["passes"];
        const json blocks = generate(request)["neuralese"]["blocks"];
        if (blocks.empty()) throw request_error("neuralese-write", "the write produced no block", 500);
        return blocks[0];
    }

    // View (POST /v1/neuralese/view): the builtin view's Neuralese instance by its plan, as the reference's view.py
    // (DECISIONS.md 2026-10-09). Every write is the template write of view's body at its site: the reply forced to
    // return_result with the value written. A value that fits one write site's window is viewed there; a longer one is
    // split into token chunks, each viewed at a part site, and a combine site reads the part views and writes the final
    // one. Without instructions the view is faithful.
    json view(const json & body) {
        const std::string value = body.value("value", "");
        const std::string instructions = body.contains("instructions") && body["instructions"].is_string() ? body["instructions"].get<std::string>() : "";
        const json system = body.contains("system") && !body["system"].is_null()
            && !(body["system"].is_string() && body["system"].get<std::string>().empty()) ? body["system"] : json(VIEW_INSTRUCTIONS);
        // window: the model's context (as served: the smaller of its training context and this server's -c) less the
        // site's own text, the tool, the forced reply, the block and a margin
        const int32_t context = std::min<int32_t>(llama_model_n_ctx_train(model), (int32_t) llama_n_ctx(ctx));
        const int32_t overhead = (int32_t) common_tokenize(vocab, instructions, false, true).size() + max_block + VIEW_SITE_MARGIN;
        int32_t window = std::max(256, context - overhead);
        if (body.contains("window") && body["window"].is_number_integer() && body["window"].get<int32_t>() > 0) {
            window = std::min(window, body["window"].get<int32_t>());
        }
        const std::string purpose = instructions.empty() ? "" : "instructions:\n" + instructions + "\n\n";
        auto site = [&](const json & content) {
            return json::array({ { { "role", "system" }, { "content", system } }, { { "role", "user" }, { "content", content } } });
        };
        static const json tools = json::parse(VIEW_TOOLS), templ = json::parse(VIEW_TEMPLATE);
        auto write_at = [&](const json & messages) {
            const json request = { { "messages", messages }, { "tools", tools }, { "neuralese_template", templ },
                                   { "max_tokens", max_block + 64 } };
            const json blocks = generate(request)["neuralese"]["blocks"];
            if (blocks.empty()) throw request_error("neuralese-view", "a view write produced no block", 500);
            return blocks[0]["id"].get<std::string>();
        };
        const std::vector<llama_token> ids = common_tokenize(vocab, value, false, true);
        std::string id;
        size_t parts = 1;
        if ((int32_t) ids.size() <= window) {
            id = write_at(site(purpose + "value:\n" + value));
        } else {
            std::vector<std::string> blocks;
            const size_t count = (ids.size() + (size_t) window - 1) / (size_t) window;
            for (size_t i = 0; i < count; ++i) {
                const std::vector<llama_token> chunk(ids.begin() + (long) (i * window),
                                                     ids.begin() + (long) std::min(ids.size(), (i + 1) * (size_t) window));
                blocks.push_back(write_at(site(purpose + "value, part " + std::to_string(i + 1) + " of " + std::to_string(count)
                                               + ":\n" + common_detokenize(vocab, chunk, true))));
            }
            json content = json::array({ { { "type", "text" }, { "text", purpose + "Views of the " + std::to_string(count)
                                                                    + " parts of value, in order:\n" } } });
            for (size_t i = 0; i < blocks.size(); ++i) {
                content.push_back({ { "type", "text" }, { "text", "Part " + std::to_string(i + 1) + ": " } });
                content.push_back({ { "type", "neuralese" }, { "id", blocks[i] } });
                content.push_back({ { "type", "text" }, { "text", "\n" } });
            }
            id = write_at(site(content));
            parts = blocks.size();
        }
        json meta = lookup(id).meta();
        meta["parts"] = parts;
        meta["window"] = window;
        return meta;
    }
};


// The capabilities this server serves (spec/NEURALESE_PORT.md "Capabilities"); the reference server reports its own.
json capabilities() {
    return json::array({ "adapters.lora-load", "chat", "chat.stream", "decide", "embed", "encode", "guidance.check",
                         "parts.value-type", "render", "score", "store", "store.owners", "template",
                         "template.argument-path", "template.value-type", "view", "write" });
}

// `neuralese-<capability with dots as dashes>-unavailable`.
std::string unavailable_code(const std::string & capability) {
    std::string code = capability;
    std::replace(code.begin(), code.end(), '.', '-');
    return "neuralese-" + code + "-unavailable";
}

json error_body(const std::string & code, const std::string & detail) {
    return { { "error", { { "code", code }, { "message", detail } } } };
}

// A streamed chat completion (`"stream": true`), event by event as the reference's http.py `_stream`: a role delta;
// text `content` deltas until a tool call opens (call markup, `<|tool_call_start|>` or `<tool_call>`, is held back);
// per written block a `[{"type": "neuralese", "id"}]` content delta with `neuralese.block` set to its meta; the
// parsed calls as one `tool_calls` delta; a final chunk with `finish_reason`, `usage`, `neuralese`,
// `x_natlang_guidance` (when guidance was on) and `x_natlang_message`, the complete parsed message. An error after the
// first event is an `{"error": …}` event. `event` receives each event's JSON; the caller ends the stream ([DONE]).
void stream_chat(engine & e, const json & body, const std::function<void(const json &)> & event) {
    const std::string request_id = e.next_request_id();
    const int64_t created = (int64_t) time(nullptr);
    auto chunk = [&](const json & delta, const json & finish = json(nullptr), const json & extra = json::object()) {
        json value = { { "id", request_id }, { "object", "chat.completion.chunk" }, { "created", created }, { "model", e.model_name },
                       { "choices", json::array({ { { "index", 0 }, { "delta", delta }, { "finish_reason", finish } } }) } };
        for (auto it = extra.begin(); it != extra.end(); ++it) value[it.key()] = it.value();
        return value;
    };
    event(chunk({ { "role", "assistant" } }));
    static const std::vector<std::string> markers = { "<|tool_call_start|>", "<tool_call>" };
    std::string streamed, pending;
    bool in_call = false;
    auto on_delta = [&](const json & item) {
        if (in_call) return;
        if (item.contains("text")) {
            pending += item["text"].get<std::string>();
            size_t at = std::string::npos;
            for (const auto & marker : markers) at = std::min(at, pending.find(marker));
            std::string out;
            if (at != std::string::npos) {
                out = pending.substr(0, at);
                pending.clear();
                in_call = true;
            } else {
                // hold back a tail that may be the start of call markup
                size_t keep = 0;
                for (const auto & marker : markers) {
                    for (size_t n = 1; n < marker.size(); ++n) {
                        if (pending.size() >= n && pending.compare(pending.size() - n, n, marker, 0, n) == 0) keep = std::max(keep, n);
                    }
                }
                out = pending.substr(0, pending.size() - keep);
                pending = pending.substr(pending.size() - keep);
            }
            if (!out.empty()) {
                streamed += out;
                event(chunk({ { "content", out } }));
            }
        } else {
            const json & meta = item["neuralese"];
            event(chunk({ { "content", json::array({ { { "type", "neuralese" }, { "id", meta["id"] } } }) } }, json(nullptr),
                        { { "neuralese", { { "block", meta } } } }));
        }
    };
    try {
        const json response = e.generate(body, on_delta, request_id);
        const json & choice = response["choices"][0];
        const json & message = choice["message"];
        const std::string sent = strip(streamed);
        if (message["content"].is_string() && message["content"].get<std::string>().compare(0, sent.size(), sent) == 0 &&
            message["content"].get<std::string>().size() > sent.size()) {
            event(chunk({ { "content", message["content"].get<std::string>().substr(sent.size()) } }));
        } else if (!in_call && !pending.empty()) {
            event(chunk({ { "content", pending } }));
        }
        if (message.contains("tool_calls")) {
            json calls = json::array();
            size_t index = 0;
            for (const auto & call : message["tool_calls"]) {
                const json & arguments = call["function"]["arguments"];
                calls.push_back({ { "index", index++ }, { "id", call["id"] }, { "type", "function" },
                                  { "function", { { "name", call["function"]["name"] },
                                                  { "arguments", arguments.is_string() ? arguments.get<std::string>() : arguments.dump() } } } });
            }
            event(chunk({ { "tool_calls", calls } }));
        }
        json extra = { { "x_natlang_message", message }, { "usage", response["usage"] }, { "neuralese", response["neuralese"] } };
        if (response.contains("x_natlang_guidance")) extra["x_natlang_guidance"] = response["x_natlang_guidance"];
        event(chunk(json::object(), choice["finish_reason"], extra));
    } catch (const request_error & err) {
        event(error_body(err.code, err.what()));
    } catch (const std::exception & err) {
        event(error_body("internal", err.what()));
    }
}

// Whether a request is a streamed chat completion: POST /v1/chat/completions with a JSON object body whose "stream"
// is true.
bool streams(const std::string & method, const std::string & path, const json & body) {
    return method == "POST" && path == "/v1/chat/completions" && body.is_object() && body.contains("stream") &&
           body["stream"].is_boolean() && body["stream"].get<bool>();
}

} // namespace

struct nz_service {
    engine e;
    int n_gpu_layers = 0;
};

nz_service * nz_service_load(const nz_service_options & o, std::string & error) {
    static bool initialised = false;
    if (!initialised) {
        llama_backend_init();
        llama_log_set([](ggml_log_level level, const char * text, void *) {
            if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
        }, nullptr);
        initialised = true;
    }
    auto * s = new nz_service();
    engine & e = s->e;
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = o.n_gpu_layers;
    s->n_gpu_layers = o.n_gpu_layers;
    if (!o.use_mmap) mp.load_mode = LLAMA_LOAD_MODE_NONE;
    e.model = llama_model_load_from_file(o.model_path.c_str(), mp);
    if (!e.model) { error = "could not load the model " + o.model_path; delete s; return nullptr; }
    e.vocab = llama_model_get_vocab(e.model);
    nz_heads_params hp;
    hp.n_threads = o.n_threads;
    hp.use_gpu = o.n_gpu_layers > 0;
    e.heads = nz_heads_load(o.heads_path.c_str(), hp);
    if (!e.heads) { error = "could not load the Neuralese heads " + o.heads_path; llama_model_free(e.model); delete s; return nullptr; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = o.n_ctx;
    cp.n_batch = e.n_batch;
    cp.n_ubatch = e.n_batch;
    cp.n_seq_max = 7; // 0 main, 1-2 the writer's scratch, 3-6 guidance snapshots
    cp.kv_unified = true;
    cp.n_threads = o.n_threads;
    cp.n_threads_batch = o.n_threads;
    e.ctx = llama_init_from_model(e.model, cp);
    if (!e.ctx) { error = "could not create the context"; nz_heads_free(e.heads); llama_model_free(e.model); delete s; return nullptr; }
    e.templates = common_chat_templates_init(e.model, "");
    e.dialect = o.dialect.empty() ? nz_heads_dialect(e.heads) : o.dialect;
    e.max_block = std::min(o.max_block, nz_heads_max_length(e.heads));
    e.width = nz_heads_n_embd(e.heads);
    e.model_name = o.alias;
    for (llama_token t : { nz_heads_open_token(e.heads), nz_heads_close_token(e.heads) }) {
        e.escaped.push_back(common_token_to_piece(e.vocab, t, true));
    }
    return s;
}

void nz_service_free(nz_service * s) {
    if (!s) return;
    llama_free(s->e.ctx);
    nz_heads_free(s->e.heads);
    llama_model_free(s->e.model);
    delete s;
}

std::string nz_service_hello(const nz_service * s) {
    json devices = json::array();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        devices.push_back({ { "name", ggml_backend_dev_name(dev) }, { "description", ggml_backend_dev_description(dev) },
                            { "gpu", ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU } });
    }
    return json({ { "dialect", s->e.dialect }, { "cutoff", nz_heads_cutoff(s->e.heads) }, { "devices", devices },
                  { "gpu_layers", s->n_gpu_layers } }).dump();
}

// Every block ID a JSON value names (as a whole string anywhere in it).
static void block_ids(const json & v, std::set<std::string> & out) {
    static const std::regex id(R"(^nz1_[a-z2-7]+$)");
    if (v.is_string()) {
        const auto & text = v.get_ref<const std::string &>();
        if (text.rfind("nz1_", 0) == 0 && std::regex_match(text, id)) out.insert(text);
    } else if (v.is_structured()) {
        for (const auto & item : v) block_ids(item, out);
    }
}

bool nz_service_streams(const std::string & method, const std::string & path, const std::string & body_text) {
    return streams(method, path, json::parse(body_text.empty() ? "{}" : body_text, nullptr, false));
}

nz_response nz_service_handle(nz_service * s, const std::string & method, const std::string & path, const std::string & body_text,
                              const std::string & owner, const nz_event_sink & events) {
    engine & e = s->e;
    static const std::regex block_path(R"(^/v1/neuralese/blocks/(nz1_[a-z2-7]+)(/meta|/pin|/unpin)?$)");
    // The requesting owner holds every block the service names to it.
    auto ok = [&](int status, const json & v) {
        if (!owner.empty() && status < 400) {
            std::set<std::string> ids;
            block_ids(v, ids);
            if (!ids.empty()) e.store.hold(owner, ids);
        }
        return nz_response { status, v.dump(), "application/json" };
    };
    auto fail = [&](int status, const std::string & code, const std::string & detail) { return ok(status, error_body(code, detail)); };
    // A capability this server does not serve (spec/NEURALESE_PORT.md "Capabilities"): 501 with its code, never 404.
    auto unavailable = [&](const std::string & capability) {
        return fail(501, unavailable_code(capability), capability + " is not served by this server (see capabilities in /v1/neuralese/info)");
    };
    try {
        if (method == "GET") {
            if (path == "/health" || path == "/v1/health") return ok(200, { { "status", "ok" } });
            if (path == "/v1/models") {
                return ok(200, { { "object", "list" }, { "data", json::array({ { { "id", e.model_name }, { "object", "model" } } }) } });
            }
            if (path == "/v1/neuralese/info") {
                // The reference's fields, with this server's values: no gradient sessions (grad_order 0), no adapter
                // kinds applied directly (adapters are loaded as LoRAs: capability adapters.lora-load), no projections.
                return ok(200, { { "dialects", json::array({ e.dialect }) }, { "width", e.width }, { "dtype", "f32" },
                                 { "store", { { "owners", true }, { "persistent", false } } },
                                 { "max_block_length", e.max_block }, { "grad", false }, { "grad_order", 0 }, { "stream", true },
                                 { "cutoff", nz_heads_cutoff(e.heads) }, { "adapters", json::array() }, { "projections", json::object() },
                                 { "server", "llama.cpp" }, { "capabilities", capabilities() } });
            }
            std::smatch m;
            if (path.rfind("/v1/neuralese/blocks/", 0) == 0) {
                if (!std::regex_match(path, m, block_path) || (m[2].matched && m[2].str() != "/meta")) return fail(404, "not-found", path);
                block b;
                if (!e.store.get(m[1].str(), b)) return fail(404, "neuralese-unknown-block", m[1].str());
                if (m[2].matched) return ok(200, b.meta());
                if (!owner.empty()) e.store.hold(owner, { b.id });
                return nz_response { 200, encode_block(b), "application/octet-stream" };
            }
            static const std::regex lora_get(R"(^/v1/neuralese/adapters/(nz1_[a-z2-7]+)/lora$)");
            if (std::regex_match(path, m, lora_get)) return unavailable("adapters.lora-export");
            return fail(404, "not-found", path);
        }
        if (method == "PUT") {
            static const std::regex lora_path(R"(^/v1/neuralese/adapters/(nz1_[a-z2-7]+)/lora$)");
            std::smatch m;
            if (std::regex_match(path, m, lora_path)) return ok(201, e.load_lora(m[1].str(), body_text));
            if (!std::regex_match(path, m, block_path) || m[2].matched) return fail(404, "not-found", path);
            block b = decode_block(body_text);
            if (b.id != m[1].str()) return fail(400, "neuralese-id-mismatch", "content hashes to " + b.id);
            return ok(201, e.store.put(b, owner).meta());
        }
        if (method != "POST") return fail(405, "method-not-allowed", method);
        if (path.rfind("/v1/neuralese/blocks/", 0) == 0) {
            std::smatch m;
            if (!std::regex_match(path, m, block_path) || !m[2].matched || m[2].str() == "/meta") return fail(404, "not-found", path);
            if (m[2].str() == "/pin") e.store.pin(m[1].str(), owner); else e.store.unpin(m[1].str(), owner);
            return ok(200, { { "ok", true } });
        }
        if (path == "/v1/neuralese/grad") return unavailable("grad");
        if (path == "/v1/neuralese/optim") return unavailable("optim");
        if (path == "/v1/neuralese/adapters") return unavailable("adapters.create");
        json body = json::parse(body_text.empty() ? "{}" : body_text, nullptr, false);
        // A malformed body is refused before anything acts on it: a collect that read it as "nothing referenced"
        // would drop every unpinned block.
        if (body.is_discarded() || !body.is_object()) return fail(400, "bad-json", "request body is not a JSON object");
        if (path == "/v1/neuralese/collect") {
            std::set<std::string> referenced;
            if (body.contains("referenced") && !body["referenced"].is_null()) {
                if (!body["referenced"].is_array()) return fail(400, "bad-json", "referenced must be a list of block IDs");
                for (const auto & id : body["referenced"]) {
                    if (!id.is_string()) return fail(400, "bad-json", "referenced must be a list of block IDs");
                    referenced.insert(id.get<std::string>());
                }
            }
            return nz_response { 200, json({ { "removed", e.store.collect(referenced, owner) } }).dump(), "application/json" };
        }
        if (streams(method, path, body)) {
            // Server-sent events: through `events` as they are produced, else collected into one event-stream body.
            std::string collected;
            auto send = [&](const std::string & data) {
                if (events) events(data);
                else collected += "data: " + data + "\n\n";
            };
            stream_chat(e, body, [&](const json & value) {
                // the requesting owner holds every block an event names
                if (!owner.empty() && !value.contains("error")) {
                    std::set<std::string> ids;
                    block_ids(value, ids);
                    if (!ids.empty()) e.store.hold(owner, ids);
                }
                send(value.dump());
            });
            send("[DONE]");
            return nz_response { 200, collected, "text/event-stream" };
        }
        if (path == "/v1/chat/completions") return ok(200, e.generate(body));
        if (path == "/v1/neuralese/embed") return ok(201, e.embed(body));
        if (path == "/v1/neuralese/decide") return ok(200, e.decide(body));
        if (path == "/v1/natlang/score" || path == "/v1/neuralese/decide_many") return ok(200, e.decide_many(body));
        if (path == "/v1/neuralese/encode") return ok(201, e.encode(body));
        if (path == "/v1/neuralese/write") return ok(201, e.write(body));
        if (path == "/v1/neuralese/view") return ok(201, e.view(body));
        if (path == "/v1/neuralese/render") {
            const rendered r = e.render(body.value("messages", json::array()), body.contains("tools") ? body["tools"] : json(nullptr));
            std::string prompt;
            for (const auto & seg : r.segments) prompt += seg.kind == rendered::segment::BLOCK ? "<block>" : seg.text;
            // as prefill feeds it: the model's BOS once at the start
            const std::string bos = common_token_to_piece(e.vocab, llama_vocab_bos(e.vocab), true);
            if (llama_vocab_get_add_bos(e.vocab) && prompt.compare(0, bos.size(), bos) != 0) prompt = bos + prompt;
            return ok(201, { { "prompt", prompt } });
        }
        if (path == "/v1/neuralese/guidance/check") {
            // The first rejection of `reply` checked prefix by prefix, as during generation (for conformance).
            nz_guide guide;
            const json g = body.value("guidance", json::object());
            for (const auto & n : g.value("tools", json::array())) guide.settings.tools.push_back(n.get<std::string>());
            guide.settings.repeat = g.value("repeat", 3);
            guide.settings.syntax = g.value("syntax", true);
            guide.settings.run = g.value("run", 4);
            const std::string reply = body.value("reply", "");
            for (size_t end = 1; end <= reply.size(); ++end) {
                const auto verdict = guide.check(reply.substr(0, end));
                if (verdict) return ok(201, { { "reason", verdict->reason }, { "offset", verdict->at }, { "end", end } });
            }
            return ok(201, { { "reason", nullptr } });
        }
        return fail(404, "not-found", path);
    } catch (const request_error & err) {
        return fail(err.status, err.code, err.what());
    } catch (const std::exception & err) {
        return fail(500, "internal", err.what());
    }
}
