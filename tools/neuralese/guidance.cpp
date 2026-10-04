#include "guidance.h"

#include <tree_sitter/api.h>

#include <algorithm>
#include <regex>

extern "C" const TSLanguage * tree_sitter_typescript(void);

namespace {

const char * CALL_START = "<|tool_call_start|>";
const size_t MIN_REPEAT_CHARS = 8;
const size_t RUN_MIN = 3, RUN_MAX = 60;

// `line` ends with one chunk of RUN_MIN..RUN_MAX characters repeated `times` times in a row
bool ends_in_run(const std::string & line, int32_t times) {
    for (size_t size = RUN_MIN; size <= std::min(RUN_MAX, line.size() / (size_t) times); ++size) {
        const std::string unit = line.substr(line.size() - size);
        if (unit.find_first_not_of(" \t\r\n\f\v") == std::string::npos) continue;
        bool run = true;
        for (int32_t k = 1; k < times && run; ++k) run = line.compare(line.size() - size * (k + 1), size, unit) == 0;
        if (run) return true;
    }
    return false;
}

// const/let/var declaration: (indentation, name), or nothing
bool declaration(const std::string & line, std::string & indent, std::string & name) {
    static const std::regex pattern(R"(^(\s*)(?:const|let|var)\s+([A-Za-z_$][\w$]*))");
    std::smatch m;
    if (!std::regex_search(line, m, pattern)) return false;
    indent = m[1].str();
    name = m[2].str();
    return true;
}

struct code_string {
    size_t raw_start;
    std::string text;
    std::vector<size_t> raw_lines;
    bool closed;
};

char unescape(char c) {
    switch (c) {
        case 'n': return '\n';
        case 't': return '\t';
        case 'r': return '\r';
        case '0': return '\0';
        default:  return c; // \\ \' \" and anything else: the character itself
    }
}

std::vector<code_string> code_strings(const std::string & reply) {
    static const std::regex eval_code(R"(\beval\(\s*code\s*=\s*(['"]))");
    std::vector<code_string> out;
    for (auto it = std::sregex_iterator(reply.begin(), reply.end(), eval_code); it != std::sregex_iterator(); ++it) {
        const char quote = (*it)[1].str()[0];
        size_t i = (size_t) (it->position(0) + it->length(0));
        code_string code { i, "", { i }, false };
        while (i < reply.size()) {
            const char c = reply[i];
            if (c == '\\' && i + 1 < reply.size()) {
                const char decoded = unescape(reply[i + 1]);
                code.text.push_back(decoded);
                i += 2;
                if (decoded == '\n') code.raw_lines.push_back(i);
                continue;
            }
            if (c == '\\') break; // an escape split across tokens: wait for the next one
            if (c == quote) {
                code.closed = true;
                break;
            }
            code.text.push_back(c);
            ++i;
            if (c == '\n') code.raw_lines.push_back(i);
        }
        out.push_back(std::move(code));
    }
    return out;
}

std::string strip(const std::string & s) {
    const size_t a = s.find_first_not_of(" \t\r\n\f\v");
    if (a == std::string::npos) return "";
    const size_t b = s.find_last_not_of(" \t\r\n\f\v");
    return s.substr(a, b - a + 1);
}

std::vector<std::string> split_lines(const std::string & text) {
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

} // namespace

std::vector<std::pair<size_t, size_t>> nz_syntax_errors(const std::string & code) {
    static TSParser * parser = nullptr;
    if (!parser) {
        parser = ts_parser_new();
        ts_parser_set_language(parser, tree_sitter_typescript());
    }
    TSTree * tree = ts_parser_parse_string(parser, nullptr, code.data(), (uint32_t) code.size());
    std::vector<std::pair<size_t, size_t>> found;
    std::vector<TSNode> stack { ts_tree_root_node(tree) };
    while (!stack.empty()) {
        const TSNode node = stack.back();
        stack.pop_back();
        if (ts_node_is_error(node) || ts_node_is_missing(node)) {
            found.emplace_back(ts_node_start_byte(node), ts_node_end_byte(node));
        } else if (ts_node_has_error(node)) {
            for (uint32_t i = 0; i < ts_node_child_count(node); ++i) stack.push_back(ts_node_child(node, i));
        }
    }
    ts_tree_delete(tree);
    return found;
}

std::string nz_guide::forced_prefix() const {
    return settings.require_call ? std::string(CALL_START) + "[" : "";
}

void nz_guide::rewind(size_t offset) {
    for (auto it = names_checked.begin(); it != names_checked.end();) it = *it >= offset ? names_checked.erase(it) : std::next(it);
    for (auto it = checked.begin(); it != checked.end();) {
        it = std::get<0>(*it) != "at" && std::get<1>(*it) >= offset ? checked.erase(it) : std::next(it);
    }
}

void nz_guide::accept(size_t at) {
    checked.insert({ "at", at, false });
}

std::optional<nz_guide_verdict> nz_guide::check(const std::string & reply) {
    auto accepted = [&](size_t at) { return checked.count({ "at", at, false }) > 0; };
    if (!settings.tools.empty()) {
        static const std::regex name(R"((?:\[|,\s*)([A-Za-z_][\w.]*)\()");
        // Mask quoted arguments, keeping byte offsets for rollback.
        std::string visible = reply;
        char quote = 0;
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
            if (names_checked.count(start)) continue;
            names_checked.insert(start);
            const std::string called = (*it)[1].str();
            if (std::find(settings.tools.begin(), settings.tools.end(), called) == settings.tools.end() && !accepted(start)) {
                return nz_guide_verdict { "unknown-tool", start };
            }
        }
    }
    for (const auto & code : code_strings(reply)) {
        const std::vector<std::string> lines = split_lines(code.text);
        const size_t complete = code.closed ? lines.size() : lines.size() - 1;
        if (!code.closed && settings.run > 0 && !accepted(code.raw_lines.back()) && ends_in_run(lines.back(), settings.run)) {
            return nz_guide_verdict { "repetition", code.raw_lines.back() };
        }
        for (size_t index = 0; index < complete; ++index) {
            const bool final = code.closed && index == complete - 1;
            const auto key = std::make_tuple(std::string("line"), code.raw_lines[index], final);
            if (checked.count(key)) continue;
            checked.insert(key);
            const size_t at = code.raw_lines[index];
            if (accepted(at)) continue;
            const std::string line = strip(lines[index]);
            if (settings.repeat > 0 && line.size() >= MIN_REPEAT_CHARS) {
                int32_t count = 0;
                for (size_t j = 0; j <= index; ++j) count += strip(lines[j]) == line;
                if (count >= settings.repeat) return nz_guide_verdict { "repetition", at };
            }
            std::string indent, name;
            if (declaration(lines[index], indent, name)) {
                for (size_t j = index; j-- > 0;) {
                    const std::string & earlier = lines[j];
                    const size_t lead = earlier.find_first_not_of(" \t");
                    if (lead != std::string::npos && strip(earlier).size() && lead < indent.size()) break;
                    std::string other_indent, other_name;
                    if (declaration(earlier, other_indent, other_name) && other_indent == indent && other_name == name) {
                        return nz_guide_verdict { "redeclaration", at };
                    }
                }
            }
            if (settings.syntax) {
                std::string prefix, before;
                for (size_t j = 0; j <= index; ++j) {
                    if (j) prefix += "\n";
                    if (j == index) before = prefix;
                    prefix += lines[j];
                }
                if (!final) prefix += "\n";
                const size_t line_start = before.size();
                const size_t last = prefix.find_last_not_of(" \t\r\n\f\v");
                const size_t end = last == std::string::npos ? 0 : last + 1;
                for (const auto & [start, stop] : nz_syntax_errors(prefix)) {
                    (void) start;
                    // Errors ending before this line belong to accepted lines; before the code closes, an error
                    // reaching the end is an unfinished construct, not a wrong one.
                    if (stop >= line_start && (final || stop < end)) return nz_guide_verdict { "syntax", at };
                }
            }
        }
    }
    return std::nullopt;
}
