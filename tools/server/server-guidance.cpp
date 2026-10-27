#include "server-guidance.h"

#include <tree_sitter/api.h>

#include <algorithm>
#include <regex>
#include <stdexcept>

extern "C" const TSLanguage * tree_sitter_typescript(void);

static const size_t GUIDE_MIN_REPEAT_CHARS = 8;
static const size_t GUIDE_RUN_MIN          = 3;
static const size_t GUIDE_RUN_MAX          = 60;

server_guide_settings server_guide_settings::from_json(const json & value, const std::vector<std::string> & offered, bool require_call) {
    if (!value.is_boolean() && !value.is_object()) {
        throw std::invalid_argument("\"guidance\" must be true or an object");
    }
    const json body = value.is_object() ? value : json::object();
    server_guide_settings s;
    s.require_call = json_value(body, "require_call", require_call);
    s.tools        = json_value(body, "tools", std::vector<std::string>());
    if (s.tools.empty()) {
        s.tools = offered;
    }
    s.tools.erase(std::remove(s.tools.begin(), s.tools.end(), std::string()), s.tools.end());
    s.repeat  = json_value(body, "repeat", 3);
    s.run     = json_value(body, "run", 4);
    s.syntax  = json_value(body, "syntax", true);
    s.retries = json_value(body, "retries", 4);
    if (s.repeat < 0 || s.run < 0 || s.retries < 0 || s.retries > 64) {
        throw std::invalid_argument("\"guidance\": repeat and run must be >= 0, retries in [0, 64]");
    }
    return s;
}

json server_guide_settings::to_json() const {
    return {
        {"require_call", require_call},
        {"tools",        tools},
        {"repeat",       repeat},
        {"run",          run},
        {"syntax",       syntax},
        {"retries",      retries},
    };
}

// `line` ends with one chunk of GUIDE_RUN_MIN..GUIDE_RUN_MAX characters repeated `times` times in a row
static bool guide_ends_in_run(const std::string & line, int32_t times) {
    for (size_t size = GUIDE_RUN_MIN; size <= std::min(GUIDE_RUN_MAX, line.size() / (size_t) times); ++size) {
        const std::string unit = line.substr(line.size() - size);
        if (unit.find_first_not_of(" \t\r\n\f\v") == std::string::npos) {
            continue;
        }
        bool run = true;
        for (int32_t k = 1; k < times && run; ++k) {
            run = line.compare(line.size() - size * (k + 1), size, unit) == 0;
        }
        if (run) {
            return true;
        }
    }
    return false;
}

// a const/let/var declaration: its indentation and name
static bool guide_declaration(const std::string & line, std::string & indent, std::string & name) {
    static const std::regex pattern(R"(^(\s*)(?:const|let|var)\s+([A-Za-z_$][\w$]*))");
    std::smatch m;
    if (!std::regex_search(line, m, pattern)) {
        return false;
    }
    indent = m[1].str();
    name   = m[2].str();
    return true;
}

struct guide_code {
    std::string         text;      // the decoded code
    std::vector<size_t> raw_lines; // reply offset where each decoded line starts
    bool                closed;
};

static char guide_unescape(char c) {
    switch (c) {
        case 'n': return '\n';
        case 't': return '\t';
        case 'r': return '\r';
        case '0': return '\0';
        default:  return c; // \\ \' \" and anything else: the character itself
    }
}

// the code of every eval call in the reply, decoded from its quoted form
static std::vector<guide_code> guide_code_strings(const std::string & reply) {
    static const std::regex eval_code(R"(\beval\(\s*code\s*=\s*(['"]))");
    std::vector<guide_code> out;
    for (auto it = std::sregex_iterator(reply.begin(), reply.end(), eval_code); it != std::sregex_iterator(); ++it) {
        const char quote = (*it)[1].str()[0];
        size_t i = (size_t) (it->position(0) + it->length(0));
        guide_code code { "", { i }, false };
        while (i < reply.size()) {
            const char c = reply[i];
            if (c == '\\' && i + 1 < reply.size()) {
                const char decoded = guide_unescape(reply[i + 1]);
                code.text.push_back(decoded);
                i += 2;
                if (decoded == '\n') {
                    code.raw_lines.push_back(i);
                }
                continue;
            }
            if (c == '\\') {
                break; // an escape split across tokens: wait for the next one
            }
            if (c == quote) {
                code.closed = true;
                break;
            }
            code.text.push_back(c);
            ++i;
            if (c == '\n') {
                code.raw_lines.push_back(i);
            }
        }
        out.push_back(std::move(code));
    }
    return out;
}

static std::string guide_strip(const std::string & s) {
    const size_t a = s.find_first_not_of(" \t\r\n\f\v");
    if (a == std::string::npos) {
        return "";
    }
    const size_t b = s.find_last_not_of(" \t\r\n\f\v");
    return s.substr(a, b - a + 1);
}

static std::vector<std::string> guide_split_lines(const std::string & text) {
    std::vector<std::string> lines;
    size_t start = 0;
    while (true) {
        const size_t end = text.find('\n', start);
        if (end == std::string::npos) {
            lines.push_back(text.substr(start));
            return lines;
        }
        lines.push_back(text.substr(start, end - start));
        start = end + 1;
    }
}

std::vector<std::pair<size_t, size_t>> server_guide_syntax_errors(const std::string & code) {
    // one parser per thread: the check endpoint runs on the HTTP threads
    struct parser_holder {
        TSParser * parser;
        parser_holder() : parser(ts_parser_new()) { ts_parser_set_language(parser, tree_sitter_typescript()); }
        ~parser_holder() { ts_parser_delete(parser); }
    };
    static thread_local parser_holder holder;

    TSTree * tree = ts_parser_parse_string(holder.parser, nullptr, code.data(), (uint32_t) code.size());
    std::vector<std::pair<size_t, size_t>> found;
    std::vector<TSNode> stack { ts_tree_root_node(tree) };
    while (!stack.empty()) {
        const TSNode node = stack.back();
        stack.pop_back();
        if (ts_node_is_error(node) || ts_node_is_missing(node)) {
            found.emplace_back(ts_node_start_byte(node), ts_node_end_byte(node));
        } else if (ts_node_has_error(node)) {
            for (uint32_t i = 0; i < ts_node_child_count(node); ++i) {
                stack.push_back(ts_node_child(node, i));
            }
        }
    }
    ts_tree_delete(tree);
    return found;
}

void server_guide::rewind(size_t offset) {
    for (auto it = names_checked.begin(); it != names_checked.end();) {
        it = *it >= offset ? names_checked.erase(it) : std::next(it);
    }
    for (auto it = checked.begin(); it != checked.end();) {
        it = std::get<0>(*it) != "at" && std::get<1>(*it) >= offset ? checked.erase(it) : std::next(it);
    }
}

void server_guide::accept(size_t at) {
    checked.insert({ "at", at, false });
}

std::optional<server_guide_verdict> server_guide::check(const std::string & reply) {
    auto accepted = [&](size_t at) { return checked.count({ "at", at, false }) > 0; };

    if (!settings.tools.empty()) {
        // call names outside quoted arguments, at the same byte offsets
        static const std::regex name(R"((?:\[|,\s*)([A-Za-z_][\w.]*)\()");
        std::string visible = reply;
        char quote   = 0;
        bool escaped = false;
        for (size_t i = 0; i < reply.size(); ++i) {
            const char c = reply[i];
            if (quote) {
                visible[i] = ' ';
                if (escaped) {
                    escaped = false;
                } else if (c == '\\') {
                    escaped = true;
                } else if (c == quote) {
                    quote = 0;
                }
            } else if (c == '\'' || c == '"') {
                quote = c;
                visible[i] = ' ';
            }
        }
        for (auto it = std::sregex_iterator(visible.begin(), visible.end(), name); it != std::sregex_iterator(); ++it) {
            const size_t start = (size_t) it->position(1);
            if (names_checked.count(start)) {
                continue;
            }
            names_checked.insert(start);
            const std::string called = (*it)[1].str();
            if (std::find(settings.tools.begin(), settings.tools.end(), called) == settings.tools.end() && !accepted(start)) {
                return server_guide_verdict { "unknown-tool", start };
            }
        }
    }

    for (const auto & code : guide_code_strings(reply)) {
        const std::vector<std::string> lines = guide_split_lines(code.text);
        const size_t complete = code.closed ? lines.size() : lines.size() - 1;
        if (!code.closed && settings.run > 0 && !accepted(code.raw_lines.back()) && guide_ends_in_run(lines.back(), settings.run)) {
            return server_guide_verdict { "repetition", code.raw_lines.back() };
        }
        for (size_t index = 0; index < complete; ++index) {
            const bool final = code.closed && index == complete - 1;
            const auto key = std::make_tuple(std::string("line"), code.raw_lines[index], final);
            if (checked.count(key)) {
                continue;
            }
            checked.insert(key);
            const size_t at = code.raw_lines[index];
            if (accepted(at)) {
                continue;
            }
            const std::string line = guide_strip(lines[index]);
            if (settings.repeat > 0 && line.size() >= GUIDE_MIN_REPEAT_CHARS) {
                int32_t count = 0;
                for (size_t j = 0; j <= index; ++j) {
                    count += guide_strip(lines[j]) == line;
                }
                if (count >= settings.repeat) {
                    return server_guide_verdict { "repetition", at };
                }
            }
            std::string indent, name;
            if (guide_declaration(lines[index], indent, name)) {
                for (size_t j = index; j-- > 0;) {
                    const std::string & earlier = lines[j];
                    const size_t lead = earlier.find_first_not_of(" \t\r\n\f\v");
                    if (lead != std::string::npos && lead < indent.size()) {
                        break; // a block closed (or opened) at a smaller indentation
                    }
                    std::string other_indent, other_name;
                    if (guide_declaration(earlier, other_indent, other_name) && other_indent == indent && other_name == name) {
                        return server_guide_verdict { "redeclaration", at };
                    }
                }
            }
            if (settings.syntax) {
                std::string prefix, before;
                for (size_t j = 0; j <= index; ++j) {
                    if (j) {
                        prefix += "\n";
                    }
                    if (j == index) {
                        before = prefix;
                    }
                    prefix += lines[j];
                }
                if (!final) {
                    prefix += "\n";
                }
                const size_t line_start = before.size();
                const size_t last = prefix.find_last_not_of(" \t\r\n\f\v");
                const size_t end = last == std::string::npos ? 0 : last + 1;
                for (const auto & [start, stop] : server_guide_syntax_errors(prefix)) {
                    (void) start;
                    // errors that end before this line belong to accepted lines; before the code closes, an error at
                    // the end is an unfinished construct, not a wrong one
                    if (stop >= line_start && (final || stop < end)) {
                        return server_guide_verdict { "syntax", at };
                    }
                }
            }
        }
    }
    return std::nullopt;
}
