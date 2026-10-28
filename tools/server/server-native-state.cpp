#include "server-native-state.h"

#include "ggml.h"
extern "C" {
#include "hash/sha256/sha256.h"
}

#include <algorithm>
#include <cstring>
#include <stdexcept>

using ojson = nlohmann::json; // std::map objects: sorted keys, as json.dumps(sort_keys=True)

static const char * NATIVE_HEADER = "natlang.native-state/1";
static const char * NATIVE_SCHEMA = "neuralese.native-state/1";

ojson native_identity::to_json() const {
    return { { "architecture", architecture }, { "checkpoint_sha256", checkpoint_sha256 },
             { "tokenizer_fingerprint", tokenizer_fingerprint }, { "embedding_weight_sha256", embedding_weight_sha256 } };
}

native_identity native_identity_from_model(const llama_model * model) {
    auto get = [&](const char * key) {
        char buf[256];
        const int32_t n = llama_model_meta_val_str(model, key, buf, sizeof(buf));
        return n > 0 && n < (int32_t) sizeof(buf) ? std::string(buf, n) : std::string();
    };
    native_identity out;
    out.architecture            = get("natlang.identity.architecture");
    out.checkpoint_sha256       = get("natlang.identity.checkpoint_sha256");
    out.tokenizer_fingerprint   = get("natlang.identity.tokenizer_fingerprint");
    out.embedding_weight_sha256 = get("natlang.identity.embedding_weight_sha256");
    if (out.architecture.empty() || out.tokenizer_fingerprint.empty() || out.embedding_weight_sha256.empty()) {
        return {};
    }
    return out;
}

std::string native_state_block::text() const {
    return metadata.value("text", std::string());
}

native_identity native_state_block::identity() const {
    native_identity out;
    out.architecture            = metadata.value("architecture", std::string());
    out.checkpoint_sha256       = metadata.value("checkpoint_sha256", std::string());
    out.tokenizer_fingerprint   = metadata.value("tokenizer_fingerprint", std::string());
    out.embedding_weight_sha256 = metadata.value("embedding_weight_sha256", std::string());
    return out;
}

int64_t native_state_block::n_tokens() const {
    auto it = tensors.find("token_ids");
    return it == tensors.end() || it->second.shape.empty() ? 0 : it->second.shape[0];
}

static std::string canonical(const ojson & value) {
    // compact separators, non-ASCII as UTF-8 (json.dumps(ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    return value.dump(-1, ' ', false, ojson::error_handler_t::strict);
}

static void put_u32(std::string & out, uint32_t v) { for (int i = 0; i < 4; ++i) out.push_back((char) ((v >> (8*i)) & 0xff)); }
static void put_u64(std::string & out, uint64_t v) { for (int i = 0; i < 8; ++i) out.push_back((char) ((v >> (8*i)) & 0xff)); }

static std::string base32_lower(const unsigned char * data, size_t n) {
    static const char * alphabet = "abcdefghijklmnopqrstuvwxyz234567";
    std::string out;
    uint32_t buffer = 0;
    int bits = 0;
    for (size_t i = 0; i < n; ++i) {
        buffer = (buffer << 8) | data[i];
        bits += 8;
        while (bits >= 5) {
            out.push_back(alphabet[(buffer >> (bits - 5)) & 31]);
            bits -= 5;
        }
    }
    if (bits > 0) {
        out.push_back(alphabet[(buffer << (5 - bits)) & 31]);
    }
    return out; // unpadded, as base64.b32encode(...).rstrip("=")
}

static std::string descriptor(const native_state_tensor & t) {
    // json.dumps({"dtype": ..., "shape": [...]}, separators=(",", ":")): insertion order, not sorted
    std::string out = "{\"dtype\":\"" + t.dtype + "\",\"shape\":[";
    for (size_t i = 0; i < t.shape.size(); ++i) {
        out += (i ? "," : "") + std::to_string(t.shape[i]);
    }
    return out + "]}";
}

std::string native_state_id(const native_state_block & block) {
    const std::string metadata = canonical(block.metadata);
    std::string head = std::string(NATIVE_HEADER) + '\0';
    put_u64(head, metadata.size());
    head += metadata;
    sha256_t ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, (const unsigned char *) head.data(), head.size());
    for (const auto & [name, t] : block.tensors) {
        std::string part;
        put_u32(part, (uint32_t) name.size());
        part += name;
        const std::string desc = descriptor(t);
        put_u32(part, (uint32_t) desc.size());
        part += desc;
        put_u64(part, t.data.size());
        sha256_update(&ctx, (const unsigned char *) part.data(), part.size());
        sha256_update(&ctx, (const unsigned char *) t.data.data(), t.data.size());
    }
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_final(&ctx, digest);
    return "nzs1_" + base32_lower(digest, sizeof(digest));
}

std::string native_state_encode(const native_state_block & block, const std::string & id) {
    ojson header = ojson::object();
    uint64_t offset = 0;
    for (const auto & [name, t] : block.tensors) {
        header[name] = { { "dtype", t.dtype }, { "shape", t.shape }, { "data_offsets", { offset, offset + t.data.size() } } };
        offset += t.data.size();
    }
    ojson metadata = block.metadata;
    metadata["id"] = id;
    header["__metadata__"] = { { "natlang.native_state", canonical(metadata) } };
    std::string text = canonical(header);
    text.append((8 - text.size() % 8) % 8, ' ');
    std::string out;
    put_u64(out, text.size());
    out += text;
    for (const auto & [name, t] : block.tensors) {
        out += t.data;
    }
    return out;
}

static size_t itemsize(const std::string & dtype) {
    if (dtype == "F16" || dtype == "BF16") return 2;
    if (dtype == "F32") return 4;
    if (dtype == "F64" || dtype == "I64") return 8;
    throw std::invalid_argument("unsupported native state dtype " + dtype);
}

static bool is_float(const std::string & dtype) { return dtype == "F16" || dtype == "BF16" || dtype == "F32" || dtype == "F64"; }

static bool is_sha256_hex(const ojson & v) {
    if (!v.is_string() || v.get<std::string>().size() != 64) return false;
    for (char c : v.get<std::string>()) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
}

static int64_t read_i64(const std::string & data, size_t i) {
    int64_t v;
    std::memcpy(&v, data.data() + 8*i, 8);
    return v;
}

// the block's layout against state_block.py NativeStateBlock.validate (for the layer kinds an LFM2/LFM3 server reads; sliding and full attention are both "attention")
static void validate(const native_state_block & block) {
    const ojson & m = block.metadata;
    static const std::vector<std::string> allowed = { "schema", "text", "architecture", "tokenizer_fingerprint",
        "checkpoint_sha256", "embedding_weight_sha256", "layer_kinds", "adapter_binding", "derivation" };
    for (const auto & item : m.items()) {
        if (std::find(allowed.begin(), allowed.end(), item.key()) == allowed.end()) {
            throw std::invalid_argument("unknown native state metadata field " + item.key());
        }
    }
    if (m.value("schema", std::string()) != NATIVE_SCHEMA) {
        throw std::invalid_argument("unsupported native state schema");
    }
    if (!m.contains("text") || !m["text"].is_string() || !m.contains("architecture") || !m["architecture"].is_string() ||
            m["architecture"].get<std::string>().empty()) {
        throw std::invalid_argument("native state needs text and an architecture identity");
    }
    for (const char * key : { "tokenizer_fingerprint", "checkpoint_sha256", "embedding_weight_sha256" }) {
        if (!m.contains(key) || !is_sha256_hex(m[key])) {
            throw std::invalid_argument(std::string("native state ") + key + " must be a SHA-256 digest");
        }
    }
    const ojson kinds = m.value("layer_kinds", ojson());
    if (!kinds.is_array() || kinds.empty()) {
        throw std::invalid_argument("native state needs a layer layout");
    }
    std::map<std::string, std::pair<int, size_t>> expected; // name -> dims, required
    expected["token_ids"] = { 1, 0 }; expected["positions"] = { 1, 0 }; expected["embeddings"] = { 2, 0 };
    for (size_t i = 0; i < kinds.size(); ++i) {
        const std::string kind = kinds[i].is_string() ? kinds[i].get<std::string>() : "";
        const std::string prefix = "layers." + std::to_string(i) + ".";
        if (kind == "attention") {
            expected[prefix + "raw_k"] = { 3, 0 }; expected[prefix + "v"] = { 3, 0 };
        } else if (kind == "finite_convolution") {
            expected[prefix + "gated_inputs"] = { 2, 0 };
        } else {
            throw std::invalid_argument("unsupported native shared layer state: " + kind);
        }
    }
    if (expected.size() != block.tensors.size()) {
        throw std::invalid_argument("native state tensor inventory differs from its layer layout");
    }
    const int64_t n = block.n_tokens();
    for (const auto & [name, want] : expected) {
        auto it = block.tensors.find(name);
        if (it == block.tensors.end()) {
            throw std::invalid_argument("native state tensor inventory differs from its layer layout");
        }
        const auto & t = it->second;
        if ((int) t.shape.size() != want.first || t.shape[0] != n) {
            throw std::invalid_argument("native state " + name + " must be token-first with " + std::to_string(want.first) + " dimensions");
        }
        for (size_t d = 1; d < t.shape.size(); ++d) {
            if (t.shape[d] <= 0) throw std::invalid_argument("native state " + name + " has an empty dimension");
        }
        const bool ints = name == "token_ids" || name == "positions";
        if (ints ? t.dtype != "I64" : !is_float(t.dtype)) {
            throw std::invalid_argument("native state " + name + (ints ? " must be int64" : " must be floating"));
        }
    }
    const auto & ids = block.tensors.at("token_ids").data;
    const auto & pos = block.tensors.at("positions").data;
    for (int64_t i = 0; i < n; ++i) {
        if (read_i64(ids, i) < 0) throw std::invalid_argument("native state token IDs must be nonnegative");
        if (read_i64(pos, i) < 0 || (i > 0 && read_i64(pos, i) != read_i64(pos, i - 1) + 1)) {
            throw std::invalid_argument("native state positions must form a contiguous nonnegative span");
        }
    }
    for (size_t i = 0; i < kinds.size(); ++i) {
        if (kinds[i] == "attention") {
            const auto & k = block.tensors.at("layers." + std::to_string(i) + ".raw_k");
            const auto & v = block.tensors.at("layers." + std::to_string(i) + ".v");
            if (k.shape != v.shape || k.dtype != v.dtype) {
                throw std::invalid_argument("attention raw keys and values must have identical shape and dtype");
            }
        }
    }
}

native_state_block native_state_decode(const std::string & body, std::string & id) {
    try {
        if (body.size() < 8) {
            throw std::invalid_argument("not a safetensors body");
        }
        uint64_t header_size = 0;
        std::memcpy(&header_size, body.data(), 8);
        if (header_size > body.size() - 8) {
            throw std::invalid_argument("header exceeds the body");
        }
        const ojson header = ojson::parse(body.substr(8, header_size));
        if (!header.is_object()) {
            throw std::invalid_argument("native state header must be an object");
        }
        native_state_block block;
        ojson metadata = ojson::parse(header.value("__metadata__", ojson::object()).value("natlang.native_state", std::string("{}")));
        if (!metadata.is_object()) {
            throw std::invalid_argument("native state metadata must be an object");
        }
        const std::string claimed = metadata.value("id", std::string());
        metadata.erase("id");
        block.metadata = metadata;
        const size_t payload = 8 + header_size, payload_size = body.size() - payload;
        std::vector<std::pair<uint64_t, uint64_t>> spans;
        for (const auto & item : header.items()) {
            if (item.key() == "__metadata__") {
                continue;
            }
            const ojson & e = item.value();
            native_state_tensor t;
            t.dtype = e.at("dtype").get<std::string>();
            const size_t size = itemsize(t.dtype);
            const ojson & shape = e.at("shape"), & offsets = e.at("data_offsets");
            if (!shape.is_array() || shape.empty() || !offsets.is_array() || offsets.size() != 2) {
                throw std::invalid_argument("invalid native tensor descriptor: " + item.key());
            }
            uint64_t count = 1;
            for (const auto & d : shape) {
                if (!d.is_number_integer() || d.get<int64_t>() < 0) {
                    throw std::invalid_argument("invalid native tensor descriptor: " + item.key());
                }
                t.shape.push_back(d.get<int64_t>());
                count *= (uint64_t) d.get<int64_t>();
            }
            const uint64_t start = offsets[0].get<uint64_t>(), end = offsets[1].get<uint64_t>();
            if (end < start || end > payload_size || end - start != count * size) {
                throw std::invalid_argument("invalid native tensor byte range: " + item.key());
            }
            t.data = body.substr(payload + start, end - start);
            spans.emplace_back(start, end);
            block.tensors[item.key()] = std::move(t);
        }
        std::sort(spans.begin(), spans.end());
        bool contiguous = !spans.empty() && spans.front().first == 0 && spans.back().second == payload_size;
        for (size_t i = 1; contiguous && i < spans.size(); ++i) {
            contiguous = spans[i - 1].second == spans[i].first;
        }
        if (!contiguous) {
            throw std::invalid_argument("native state tensor payload has gaps or overlaps");
        }
        validate(block);
        id = native_state_id(block);
        if (claimed != id) {
            throw std::invalid_argument("native state ID does not match its content (" + id + ")");
        }
        return block;
    } catch (const ojson::exception & e) {
        throw std::invalid_argument(std::string("malformed native state: ") + e.what());
    }
}

ojson native_state_meta(const std::string & id, const native_state_block & block) {
    ojson out = block.metadata;
    out["id"] = id;
    return out;
}

static void to_f32(const native_state_tensor & t, std::vector<float> & out) {
    const size_t n = t.data.size() / itemsize(t.dtype);
    out.resize(n);
    const char * src = t.data.data();
    for (size_t i = 0; i < n; ++i) {
        if (t.dtype == "F32") {
            std::memcpy(&out[i], src + 4*i, 4);
        } else if (t.dtype == "F64") {
            double d; std::memcpy(&d, src + 8*i, 8); out[i] = (float) d;
        } else {
            uint16_t h; std::memcpy(&h, src + 2*i, 2);
            out[i] = t.dtype == "BF16" ? ggml_bf16_to_fp32(ggml_bf16_t { h }) : ggml_fp16_to_fp32(h);
        }
    }
}

native_state_rows native_state_to_rows(const native_state_block & block, int32_t n_layer) {
    const ojson & kinds = block.metadata.at("layer_kinds");
    if ((int32_t) kinds.size() != n_layer) {
        throw std::invalid_argument("native state layer layout differs from the consuming backbone");
    }
    native_state_rows rows;
    rows.n_tokens = (int32_t) block.n_tokens();
    rows.k.assign(n_layer, {}); rows.v.assign(n_layer, {}); rows.conv.assign(n_layer, {});
    for (int32_t il = 0; il < n_layer; ++il) {
        const std::string prefix = "layers." + std::to_string(il) + ".";
        if (kinds[il] == "attention") {
            to_f32(block.tensors.at(prefix + "raw_k"), rows.k[il]);
            to_f32(block.tensors.at(prefix + "v"), rows.v[il]);
        } else {
            to_f32(block.tensors.at(prefix + "gated_inputs"), rows.conv[il]);
        }
    }
    return rows;
}

bool native_state_store::put(const std::string & id, const std::string & body, native_state_block block) {
    std::lock_guard<std::mutex> lock(m);
    if (entries.count(id)) {
        return false;
    }
    entries[id] = std::make_shared<const entry>(entry { body, std::move(block) });
    return true;
}

std::shared_ptr<const native_state_store::entry> native_state_store::get(const std::string & id) const {
    std::lock_guard<std::mutex> lock(m);
    auto it = entries.find(id);
    return it == entries.end() ? nullptr : it->second;
}

bool native_state_store::has(const std::string & id) const {
    std::lock_guard<std::mutex> lock(m);
    return entries.count(id) > 0;
}

void native_state_store::set_span(const std::string & id, llama_token first) {
    std::lock_guard<std::mutex> lock(m);
    spans[id] = first;
}

llama_token native_state_store::span(const std::string & id) const {
    std::lock_guard<std::mutex> lock(m);
    auto it = spans.find(id);
    return it == spans.end() ? LLAMA_TOKEN_NULL : it->second;
}

size_t native_state_store::size() const {
    std::lock_guard<std::mutex> lock(m);
    return entries.size();
}

void native_state_registry::add_static(const std::string & id, std::vector<llama_token> tokens) {
    std::lock_guard<std::mutex> lock(m);
    statics[id] = std::move(tokens);
}

bool native_state_registry::known(const std::string & id) const {
    {
        std::lock_guard<std::mutex> lock(m);
        if (statics.count(id)) {
            return true;
        }
    }
    return store.has(id);
}

native_state_use native_state_registry::resolve(const std::string & id) const {
    {
        std::lock_guard<std::mutex> lock(m);
        auto it = statics.find(id);
        if (it != statics.end()) {
            return { it->second, "exact-native-state" };
        }
    }
    const auto found = store.get(id);
    if (!found) {
        throw std::invalid_argument("neuralese-unknown-native-state: " + id);
    }
    native_state_use use;
    use.tokens.push_back(consume_open);
    const llama_token first = store.span(id);
    if (first != LLAMA_TOKEN_NULL) {
        for (int64_t i = 0; i < found->block.n_tokens(); ++i) {
            use.tokens.push_back(first + (llama_token) i);
        }
        use.mode = "exact-native-state";
    } else {
        const std::string text = found->block.text();
        std::vector<llama_token> ids(text.size() + 8);
        int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), ids.data(), (int32_t) ids.size(), false, false);
        if (n < 0) {
            ids.resize(-n);
            n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), ids.data(), (int32_t) ids.size(), false, false);
        }
        ids.resize(std::max<int32_t>(n, 0));
        use.tokens.insert(use.tokens.end(), ids.begin(), ids.end());
        use.mode = "best-effort-text-retokenization";
    }
    use.tokens.push_back(consume_close);
    return use;
}
