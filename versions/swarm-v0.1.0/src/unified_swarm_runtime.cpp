#include "unified_swarm_runtime.h"
#include <iostream>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <filesystem>

namespace unified_swarm {

//=============================================================================
// PHASE UTILITIES
//=============================================================================

std::string phase_to_string(Phase p) {
    switch (p) {
        case Phase::POP: return "@pop";
        case Phase::WO: return "@wo";
        case Phase::SEK: return "@sek";
        case Phase::CH_EN: return "@ch'en";
        case Phase::YAX: return "@yax";
        case Phase::K_AYAB: return "@k'ayab";
        case Phase::KUMK_U: return "@kumk'u";
        case Phase::KAN: return "@kan";
        case Phase::COLLAPSE: return "@collapse";
        default: return "@unknown";
    }
}

Phase string_to_phase(const std::string& s) {
    if (s == "@pop") return Phase::POP;
    if (s == "@wo") return Phase::WO;
    if (s == "@sek") return Phase::SEK;
    if (s == "@ch'en") return Phase::CH_EN;
    if (s == "@yax") return Phase::YAX;
    if (s == "@k'ayab") return Phase::K_AYAB;
    if (s == "@kumk'u") return Phase::KUMK_U;
    if (s == "@kan") return Phase::KAN;
    if (s == "@collapse") return Phase::COLLAPSE;
    return Phase::POP;
}

//=============================================================================
// FOLD UTILITIES
//=============================================================================

std::string fold_to_string(Fold f) {
    switch (f) {
        case Fold::Pop:  return "Pop";
        case Fold::Wo:   return "Wo";
        case Fold::Yax:  return "Yax";
        case Fold::Sek:  return "Sek";
        case Fold::Chen: return "Chen";
        case Fold::Xul:  return "Xul";
        default:         return "UNASSIGNED";
    }
}

Fold string_to_fold(const std::string& s) {
    if (s == "Pop")  return Fold::Pop;
    if (s == "Wo")   return Fold::Wo;
    if (s == "Yax")  return Fold::Yax;
    if (s == "Sek")  return Fold::Sek;
    if (s == "Chen") return Fold::Chen;
    if (s == "Xul")  return Fold::Xul;
    return Fold::UNASSIGNED;
}

//=============================================================================
// VALUE IMPLEMENTATION
//=============================================================================

std::string Value::to_string() const {
    switch (type) {
        case NULL_TYPE: return "null";
        case INT: return std::to_string(data.int_val);
        case FLOAT: return std::to_string(data.float_val);
        case STRING: return str_val;
        case BYTES: return "[bytes:" + std::to_string(bytes_val.size()) + "]";
        case ARRAY: return "[array:" + std::to_string(array_val.size()) + "]";
        case OBJECT: return "[object]";
        case AGENT_REF: return agent_ref;
        case TENSOR: return "[tensor]";
        default: return "?";
    }
}

//=============================================================================
// CSS BINDING IMPLEMENTATION
//=============================================================================

std::string CSSBinding::generate_css() const {
    std::string css = "🤖[id=\"" + selector + "\"] {\n";
    for (const auto& [name, value] : css_variables) {
        css += "    " + name + ": \"" + value + "\";\n";
    }
    css += "}\n";
    return css;
}

//=============================================================================
// MICRONAUT IMPLEMENTATION
//=============================================================================

Micronaut::Micronaut(const std::string& agent_id, const json& config)
    : id(agent_id),
      type(config.value("type", "unknown")),
      fold(string_to_fold(config.value("fold", "UNASSIGNED"))),
      port(config.value("port", 0)),
      endpoint(config.value("endpoint", "")),
      swarm_role(config.value("role", "worker")),
      priority(config.value("priority", 0)),
      css_binding(agent_id)
{
    // Extract capabilities and experts from config
    if (config.contains("capabilities")) {
        for (const auto& cap : config["capabilities"]) {
            capabilities.push_back(cap.get<std::string>());
        }
    }
    
    if (config.contains("experts")) {
        for (const auto& exp : config["experts"]) {
            experts.push_back(exp.get<std::string>());
        }
    }
    
    // Initialize CSS variables
    css_binding.set_variable("--🤖-id", agent_id);
    css_binding.set_variable("--🤖-type", type);
    css_binding.set_variable("--🤖-fold", fold_to_string(fold));
    css_binding.set_variable("--🤖-port", std::to_string(port));
    css_binding.set_variable("--🤖-endpoint", endpoint);
    css_binding.set_variable("--🤖-state", "idle");
    css_binding.set_variable("--🤖-load", "0.0");
    css_binding.set_variable("--🤖-phase", phase_to_string(current_phase));
    css_binding.set_variable("--🤖-pc", "0");
}

Value Micronaut::pop_stack() {
    if (stack.empty()) return Value();
    Value v = stack.back();
    stack.pop_back();
    return v;
}

void Micronaut::update_css_state() {
    css_binding.set_variable("--🤖-state", running ? "running" : "idle");
    css_binding.set_variable("--🤖-load", std::to_string(load));
    css_binding.set_variable("--🤖-phase", phase_to_string(current_phase));
    css_binding.set_variable("--🤖-pc", std::to_string(pc));
}

void Micronaut::step() {
    // Simple state machine: POP -> WO -> SEK -> COLLAPSE
    if (current_phase == Phase::POP) {
        current_phase = Phase::WO;
    } else if (current_phase == Phase::WO) {
        current_phase = Phase::SEK;
    } else if (current_phase == Phase::SEK) {
        current_phase = Phase::COLLAPSE;
    } else if (current_phase == Phase::COLLAPSE) {
        running = false;
    }
    update_css_state();
}

Value Micronaut::execute_bytecode(const std::vector<uint8_t>& bytecode) {
    SCXQ2VM vm(bytecode, this);
    vm.run();
    return vm.get_result();
}

//=============================================================================
// SWARM CONSCIOUSNESS IMPLEMENTATION
//=============================================================================

SwarmConsciousness::SwarmConsciousness() {
    css_injector = [](const std::string&) { /* default no-op */ };
}

void SwarmConsciousness::register_agent(const json& config) {
    std::string id = config.value("id", "unknown");
    auto agent = std::make_shared<Micronaut>(id, config);
    
    agents[id] = agent;
    
    Fold fold = agent->get_fold();
    if (fold != Fold::UNASSIGNED) {
        fold_members[static_cast<uint8_t>(fold)].push_back(id);
    }
    
    // Inject CSS
    css_injector(agent->get_css_binding().generate_css());
    
    recalculate_coherence();
}

std::shared_ptr<Micronaut> SwarmConsciousness::get_agent(const std::string& id) {
    auto it = agents.find(id);
    return it != agents.end() ? it->second : nullptr;
}

std::vector<std::shared_ptr<Micronaut>> SwarmConsciousness::get_agents_by_fold(Fold fold) {
    std::vector<std::shared_ptr<Micronaut>> result;
    uint8_t fold_id = static_cast<uint8_t>(fold);
    
    auto it = fold_members.find(fold_id);
    if (it != fold_members.end()) {
        for (const auto& agent_id : it->second) {
            auto agent = get_agent(agent_id);
            if (agent) result.push_back(agent);
        }
    }
    
    return result;
}

void SwarmConsciousness::broadcast_to_fold(Fold fold, const json& message) {
    auto agents_in_fold = get_agents_by_fold(fold);
    for (auto& agent : agents_in_fold) {
        send_to_agent(agent->get_id(), message);
    }
}

void SwarmConsciousness::send_to_agent(const std::string& agent_id, const json& message) {
    auto agent = get_agent(agent_id);
    if (!agent) return;
    
    // For now, just update the agent's load based on message
    if (message.contains("load")) {
        agent->set_load(message["load"].get<double>());
    }
}

void SwarmConsciousness::optical_wave(const std::vector<float>& sh_coefficients, uint32_t frames) {
    // Convert SH coefficients to CSS and inject
    std::stringstream ss;
    ss << ":root {\n";
    ss << "    --optical-sh: \"";
    for (size_t i = 0; i < sh_coefficients.size(); i++) {
        if (i > 0) ss << ",";
        ss << sh_coefficients[i];
    }
    ss << "\";\n";
    ss << "    --optical-frames: \"" << frames << "\";\n";
    ss << "    animation: ⚡optical_broadcast " << (frames / 30.0) << "s infinite;\n";
    ss << "}\n";
    
    css_injector(ss.str());
    
    // Run for specified frames
    for (uint32_t i = 0; i < frames; i++) {
        tick();
    }
}

void SwarmConsciousness::tick() {
    tick_count++;
    
    // Execute one step for each agent in deterministic order
    for (auto& [id, agent] : agents) {
        agent->step();
    }
    
    recalculate_coherence();
    update_css_root();
}

void SwarmConsciousness::run_for_ticks(uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        tick();
    }
}

void SwarmConsciousness::recalculate_coherence() {
    std::unordered_map<uint8_t, size_t> phase_counts;
    
    for (const auto& [id, agent] : agents) {
        uint8_t phase = static_cast<uint8_t>(agent->get_phase());
        phase_counts[phase]++;
    }
    
    if (agents.empty()) {
        coherence = 0.94;
        entropy = 0.06;
        return;
    }
    
    size_t total = agents.size();
    double max_phase_ratio = 0.0;
    
    for (const auto& [phase, count] : phase_counts) {
        double ratio = static_cast<double>(count) / total;
        max_phase_ratio = std::max(max_phase_ratio, ratio);
    }
    
    coherence = 0.7 + (max_phase_ratio * 0.3);
    entropy = 1.0 - coherence;
}

void SwarmConsciousness::update_css_root() {
    std::stringstream ss;
    ss << ":root {\n";
    ss << "    --swarm-coherence: \"" << coherence << "\";\n";
    ss << "    --swarm-entropy: \"" << entropy << "\";\n";
    ss << "    --swarm-tick: \"" << tick_count << "\";\n";
    ss << "}\n";
    
    css_injector(ss.str());
}

json SwarmConsciousness::get_swarm_state() const {
    json state;
    state["coherence"] = coherence;
    state["entropy"] = entropy;
    state["tick_count"] = tick_count;
    state["agent_count"] = agents.size();
    state["agents"] = json::array();
    
    for (const auto& [id, agent] : agents) {
        json agent_state;
        agent_state["id"] = agent->get_id();
        agent_state["type"] = agent->get_type();
        agent_state["fold"] = fold_to_string(agent->get_fold());
        agent_state["phase"] = phase_to_string(agent->get_phase());
        agent_state["load"] = agent->get_load();
        state["agents"].push_back(agent_state);
    }
    
    return state;
}

//=============================================================================
// SCXQ2 VM IMPLEMENTATION
//=============================================================================

void SCXQ2VM::run() {
    while (pc < bytecode.size()) {
        step();
        if (phase == Phase::COLLAPSE) break;
    }
}

void SCXQ2VM::step() {
    if (pc >= bytecode.size()) {
        phase = Phase::COLLAPSE;
        return;
    }
    
    uint8_t opcode = bytecode[pc++];
    execute_opcode(opcode);
}

void SCXQ2VM::execute_opcode(uint8_t opcode) {
    // Simplified SCXQ2 opcode handling
    switch (opcode) {
        case 0x00: phase = Phase::WO; break;          // PHASE_WO
        case 0x01: phase = Phase::SEK; break;         // PHASE_SEK
        case 0x02: {                                   // PUSH_INT
            if (pc + 8 <= bytecode.size()) {
                int64_t val = 0;
                for (int i = 0; i < 8; i++) {
                    val |= (static_cast<int64_t>(bytecode[pc++]) << (i * 8));
                }
                stack.push_back(Value(val));
            }
            break;
        }
        case 0x03: {                                   // ADD
            if (stack.size() >= 2) {
                Value b = stack.back(); stack.pop_back();
                Value a = stack.back(); stack.pop_back();
                if (a.type == Value::INT && b.type == Value::INT) {
                    stack.push_back(Value(a.data.int_val + b.data.int_val));
                }
            }
            break;
        }
        case 0x04: {                                   // COLLAPSE
            phase = Phase::COLLAPSE;
            break;
        }
        default:
            // Unknown opcode, advance
            break;
    }
}

Value SCXQ2VM::get_result() {
    return stack.empty() ? Value() : stack.back();
}

//=============================================================================
// UNIFIED RUNTIME IMPLEMENTATION
//=============================================================================

UnifiedRuntime::UnifiedRuntime() {
    // Initialize swarm from dist/*/config.@.toml agent registration files
    auto configs = load_embedded_agent_configs();
    for (const auto& config : configs) {
        swarm.register_agent(config);
    }
}

void UnifiedRuntime::load_model(const std::string& model_dir) {
    // Load model configuration from directory
    // This would typically read from JSON or SCXQ2 files
    loaded_model_config["path"] = model_dir;
    loaded_model_config["status"] = "loaded";
}

void UnifiedRuntime::load_model_from_json(const json& model_spec) {
    loaded_model_config = model_spec;
}

std::string UnifiedRuntime::inference(const std::string& prompt, uint32_t max_tokens) {
    json request;
    request["prompt"] = prompt;
    request["max_tokens"] = max_tokens;
    
    auto response = inference_json(request);
    return response.value("response", "");
}

json UnifiedRuntime::inference_json(const json& request) {
    json response;
    response["prompt"] = request.value("prompt", "");
    response["response"] = "Unified swarm processed: " + request.value("prompt", "");
    response["swarm_state"] = swarm.get_swarm_state();
    return response;
}

void UnifiedRuntime::optical_compute(const std::vector<float>& sh_coefficients, uint32_t frames) {
    swarm.optical_wave(sh_coefficients, frames);
}

void UnifiedRuntime::execute_bytecode_on_agent(const std::string& agent_id,
                                               const std::vector<uint8_t>& bytecode) {
    auto agent = swarm.get_agent(agent_id);
    if (agent) {
        agent->execute_bytecode(bytecode);
    }
}

json UnifiedRuntime::get_system_status() const {
    json status;
    status["swarm"] = swarm.get_swarm_state();
    status["model"] = loaded_model_config;
    return status;
}

void UnifiedRuntime::print_swarm_status() const {
    auto state = swarm.get_swarm_state();
    
    std::cout << "\n⚛ UNIFIED MICRONAUT SWARM STATUS\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "Agents: " << state["agent_count"] << "\n";
    std::cout << "Coherence: " << state["coherence"] << "\n";
    std::cout << "Entropy: " << state["entropy"] << "\n";
    std::cout << "Tick: " << state["tick_count"] << "\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";
}

//=============================================================================
// AGENT CONFIG LOADER — discovers dist/*/config.@.toml at runtime
// Each skin directory under dist/ that contains a config.@.toml is registered.
// The [agent] section is parsed and converted to JSON for SwarmConsciousness.
//=============================================================================

namespace {

static void trim_str(std::string& s) {
    const char* ws = " \t\r\n";
    s.erase(0, s.find_first_not_of(ws));
    auto last = s.find_last_not_of(ws);
    if (last != std::string::npos) s.erase(last + 1);
    else s.clear();
}

// Parse the [agent] section of a config.@.toml file into a JSON object.
// Field mapping: swarm_role → "role"  (Micronaut constructor reads config["role"]).
static json parse_toml_agent_section(const std::filesystem::path& toml_path) {
    std::ifstream f(toml_path);
    json config;
    bool in_agent = false;

    std::string line;
    while (std::getline(f, line)) {
        trim_str(line);
        if (line.empty() || line[0] == '#') continue;

        if (line[0] == '[') {
            in_agent = (line == "[agent]");
            continue;
        }

        if (!in_agent) continue;

        auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        trim_str(key);
        trim_str(val);
        if (val.empty()) continue;

        // swarm_role in TOML → "role" in JSON (Micronaut reads config["role"])
        if (key == "swarm_role") key = "role";

        if (val[0] == '"') {
            // Quoted string
            auto q2 = val.rfind('"');
            config[key] = (q2 > 0) ? val.substr(1, q2 - 1) : std::string{};
        } else if (val[0] == '[') {
            // Array of quoted strings: ["a", "b", ...]
            json arr = json::array();
            size_t pos = 0;
            while ((pos = val.find('"', pos)) != std::string::npos) {
                auto q2 = val.find('"', pos + 1);
                if (q2 == std::string::npos) break;
                arr.push_back(val.substr(pos + 1, q2 - pos - 1));
                pos = q2 + 1;
            }
            config[key] = arr;
        } else {
            // Numeric or unquoted
            try { config[key] = std::stoi(val); }
            catch (...) { config[key] = val; }
        }
    }
    return config;
}

} // anonymous namespace

std::vector<json> load_embedded_agent_configs() {
    std::vector<json> configs;
    namespace fs = std::filesystem;

    fs::path dist_dir = "dist";
    if (!fs::exists(dist_dir) || !fs::is_directory(dist_dir)) {
        std::cerr << "[swarm] dist/ not found — no agent configs loaded\n";
        return configs;
    }

    for (const auto& entry : fs::directory_iterator(dist_dir)) {
        if (!entry.is_directory()) continue;
        fs::path toml_path = entry.path() / "config.@.toml";
        if (!fs::exists(toml_path)) continue;

        json config = parse_toml_agent_section(toml_path);
        if (!config.contains("id") || config["id"].get<std::string>().empty()) {
            std::cerr << "[swarm] skip " << toml_path << " — no id in [agent]\n";
            continue;
        }

        std::cout << "[swarm] registered " << config["id"].get<std::string>()
                  << " from " << toml_path.string() << "\n";
        configs.push_back(std::move(config));
    }

    if (configs.empty()) {
        std::cerr << "[swarm] warning: no config.@.toml files found under dist/\n";
    } else {
        std::cout << "[swarm] " << configs.size() << " agent(s) linked from dist/\n";
    }

    return configs;
}

} // namespace unified_swarm
