#include "toml_loader.h"
#include <fstream>
#include <sstream>

namespace micronaut {

namespace {

static void trim(std::string& s) {
    const char* ws = " \t\r\n";
    s.erase(0, s.find_first_not_of(ws));
    auto last = s.find_last_not_of(ws);
    if (last != std::string::npos) s.erase(last + 1);
    else s.clear();
}

static std::string strip_quotes(const std::string& s) {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
        return s.substr(1, s.size() - 2);
    return s;
}

// Parse ["a", "b", "c"] or ["a","b"] into a vector.
static std::vector<std::string> parse_toml_array(const std::string& s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t q = s.find('"', i);
        if (q == std::string::npos) break;
        size_t e = s.find('"', q + 1);
        if (e == std::string::npos) break;
        out.push_back(s.substr(q + 1, e - q - 1));
        i = e + 1;
    }
    return out;
}

} // anon

AgentConfig load_agent_config(const std::string& toml_path) {
    AgentConfig cfg;
    std::ifstream f(toml_path);
    if (!f.is_open()) return cfg;

    bool in_agent = false;
    std::string line;
    while (std::getline(f, line)) {
        trim(line);
        if (line.empty() || line[0] == '#') continue;

        // Section header
        if (line[0] == '[') {
            in_agent = (line == "[agent]");
            continue;
        }
        if (!in_agent) continue;

        auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        trim(key); trim(val);

        if      (key == "id")         cfg.id         = strip_quotes(val);
        else if (key == "type")       cfg.type       = strip_quotes(val);
        else if (key == "fold")       cfg.fold       = strip_quotes(val);
        else if (key == "port")       cfg.port       = std::stoi(val);
        else if (key == "endpoint")   cfg.endpoint   = strip_quotes(val);
        else if (key == "swarm_role") cfg.swarm_role = strip_quotes(val);
        else if (key == "priority")   cfg.priority   = std::stoi(val);
        else if (key == "capabilities") cfg.capabilities = parse_toml_array(val);
        else if (key == "experts")      cfg.experts      = parse_toml_array(val);
    }

    cfg.valid = !cfg.id.empty();
    return cfg;
}

std::string agent_config_to_json(const AgentConfig& cfg) {
    std::ostringstream o;
    o << "{";
    o << "\"@kind\":\"micronaut.agent.v1\"";
    o << ",\"id\":\"" << json_escape(cfg.id) << "\"";
    o << ",\"type\":\"" << json_escape(cfg.type) << "\"";
    o << ",\"fold\":\"" << json_escape(cfg.fold) << "\"";
    o << ",\"port\":" << cfg.port;
    o << ",\"endpoint\":\"" << json_escape(cfg.endpoint) << "\"";
    o << ",\"role\":\"" << json_escape(cfg.swarm_role) << "\"";
    o << ",\"priority\":" << cfg.priority;

    o << ",\"capabilities\":[";
    for (size_t i = 0; i < cfg.capabilities.size(); ++i) {
        if (i) o << ",";
        o << "\"" << json_escape(cfg.capabilities[i]) << "\"";
    }
    o << "]";

    o << ",\"experts\":[";
    for (size_t i = 0; i < cfg.experts.size(); ++i) {
        if (i) o << ",";
        o << "\"" << json_escape(cfg.experts[i]) << "\"";
    }
    o << "]";

    o << "}";
    return o.str();
}

} // namespace micronaut
