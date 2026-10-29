#include "server-guidance.h"

#include <tree_sitter/api.h>

#include <algorithm>
#include <regex>
#include <stdexcept>

extern "C" const TSLanguage * tree_sitter_typescript(void);

static const size_t GUIDE_MIN_REPEAT_CHARS = 8;
static const size_t GUIDE_RUN_MIN          = 3;
static const size_t GUIDE_RUN_MAX          = 60;

std::string server_guide_call_format(const std::string & template_source) {
    const bool xml = template_source.find("<tool_call>") != std::string::npos &&
                     template_source.find("<function=") != std::string::npos &&
                     template_source.find("<parameter=") != std::string::npos;
    return xml ? "xml" : "pythonic";
}

server_guide_settings server_guide_settings::from_json(const json & value, const std::vector<std::string> & offered, bool require_call,
                                                       const std::string & default_format) {
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
    s.call_format = json_value(body, "call_format", default_format);
    s.reasoning_open = json_value(body, "reasoning_open", false);
    s.reasoning_end  = json_value(body, "reasoning_end", std::vector<std::string>());
    if (s.call_format != "pythonic" && s.call_format != "xml") {
        throw std::invalid_argument("\"guidance\": call_format must be \"pythonic\" or \"xml\"");
    }
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
        {"call_format",  call_format},
        {"reasoning_open", reasoning_open},
        {"reasoning_end",  reasoning_end},
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

// the xml form: called names (with the offset where each starts, once its `>` arrived) and the `code` values of eval
// calls, raw, with the value boundary described in server-guidance.h
static const std::string GUIDE_XML_FUNCTION = "<function=";
static const std::string GUIDE_XML_PARAM    = "<parameter=";
static const std::string GUIDE_XML_VALUE_END = "\n</parameter>\n"; // upstream qwen3-coder.cpp until("\n</parameter>\n")
static const std::string GUIDE_XML_FUNCTION_END = "</function>";

// the length of the longest proper prefix of `marker` that `s` ends with
static size_t guide_held_prefix(const std::string & s, const std::string & marker) {
    for (size_t n = std::min(s.size(), marker.size() - 1); n > 0; --n) {
        if (s.compare(s.size() - n, n, marker, 0, n) == 0) {
            return n;
        }
    }
    return 0;
}

static void guide_xml_scan(const std::string & reply, std::vector<std::pair<std::string, size_t>> & names, std::vector<guide_code> & codes) {
    size_t i = 0;
    while (true) {
        const size_t open = reply.find(GUIDE_XML_FUNCTION, i);
        if (open == std::string::npos) {
            return;
        }
        const size_t name_start = open + GUIDE_XML_FUNCTION.size();
        const size_t name_end = reply.find('>', name_start);
        if (name_end == std::string::npos) {
            return; // the name is not complete yet
        }
        const std::string function = reply.substr(name_start, name_end - name_start);
        names.emplace_back(function, name_start);
        i = name_end + 1;
        // the parameters of this function
        while (true) {
            const size_t at = reply.find_first_not_of(" \t\r\n", i);
            if (at == std::string::npos) {
                return;
            }
            const std::string rest = reply.substr(at, std::max(GUIDE_XML_PARAM.size(), GUIDE_XML_FUNCTION_END.size()));
            if (reply.compare(at, GUIDE_XML_FUNCTION_END.size(), GUIDE_XML_FUNCTION_END) == 0) {
                i = at + GUIDE_XML_FUNCTION_END.size();
                break;
            }
            if (reply.compare(at, GUIDE_XML_PARAM.size(), GUIDE_XML_PARAM) != 0) {
                if (GUIDE_XML_PARAM.compare(0, rest.size(), rest) == 0 || GUIDE_XML_FUNCTION_END.compare(0, rest.size(), rest) == 0) {
                    return; // a marker that is still arriving
                }
                i = at; // not a parameter: look for the next function from here
                break;
            }
            const size_t key_start = at + GUIDE_XML_PARAM.size();
            const size_t key_end = reply.find('>', key_start);
            if (key_end == std::string::npos) {
                return;
            }
            const std::string key = reply.substr(key_start, key_end - key_start);
            size_t value_start = key_end + 1;
            const bool is_code = function == "eval" && key == "code";
            if (value_start == reply.size()) {
                return; // the newline that opens the value has not arrived
            }
            // the value starts after `<parameter=K>\n`; an empty value renders as `<parameter=K>\n\n</parameter>\n`
            if (reply[value_start] == '\n') {
                value_start++;
            }
            const size_t close = reply.find(GUIDE_XML_VALUE_END, value_start);
            const size_t value_end = close != std::string::npos ? close :
                reply.size() - guide_held_prefix(reply.substr(value_start), GUIDE_XML_VALUE_END);
            if (is_code) {
                guide_code code { reply.substr(value_start, value_end - value_start), { value_start }, close != std::string::npos };
                for (size_t k = value_start; k < value_end; ++k) {
                    if (reply[k] == '\n') {
                        code.raw_lines.push_back(k + 1);
                    }
                }
                codes.push_back(std::move(code));
            }
            if (close == std::string::npos) {
                return;
            }
            i = close + GUIDE_XML_VALUE_END.size();
        }
    }
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

size_t server_guide::action_start(const std::string & reply) const {
    if (!settings.reasoning_open || settings.reasoning_end.empty()) {
        return 0;
    }
    size_t end = std::string::npos;
    for (const auto & tag : settings.reasoning_end) {
        const size_t at = tag.empty() ? std::string::npos : reply.find(tag);
        if (at != std::string::npos && (end == std::string::npos || at + tag.size() < end)) {
            end = at + tag.size();
        }
    }
    return end;
}

std::optional<server_guide_verdict> server_guide::check(const std::string & whole) {
    // the reasoning is blanked rather than cut, so that every offset stays a reply offset
    const size_t from = action_start(whole);
    if (from == std::string::npos) {
        return std::nullopt;
    }
    std::string blanked;
    if (from > 0) {
        blanked = whole;
        for (size_t i = 0; i < from; ++i) {
            if (blanked[i] != '\n') {
                blanked[i] = ' ';
            }
        }
    }
    const std::string & reply = from > 0 ? blanked : whole;
    auto accepted = [&](size_t at) { return checked.count({ "at", at, false }) > 0; };

    const bool xml = settings.call_format == "xml";
    std::vector<std::pair<std::string, size_t>> xml_names;
    std::vector<guide_code> codes;
    if (xml) {
        guide_xml_scan(reply, xml_names, codes);
    } else {
        codes = guide_code_strings(reply);
    }

    if (!settings.tools.empty() && xml) {
        for (const auto & [called, start] : xml_names) {
            if (names_checked.count(start)) {
                continue;
            }
            names_checked.insert(start);
            if (std::find(settings.tools.begin(), settings.tools.end(), called) == settings.tools.end() && !accepted(start)) {
                return server_guide_verdict { "unknown-tool", start };
            }
        }
    } else if (!settings.tools.empty()) {
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

    for (const auto & code : codes) {
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
