#pragma once
#include "common.h"
#include <vector>

namespace micronaut {

struct AgentConfig {
    std::string id;
    std::string type;
    std::string fold;
    int         port        = 0;
    std::string endpoint;
    std::string swarm_role;
    int         priority    = 5;
    std::vector<std::string> capabilities;
    std::vector<std::string> experts;
    bool valid = false;
};

// Parse [agent] section from config.@.toml.
// Returns AgentConfig with valid=true on success.
AgentConfig load_agent_config(const std::string& toml_path);

// Serialize AgentConfig as a flat JSON object (for swarm registration).
std::string agent_config_to_json(const AgentConfig& cfg);

} // namespace micronaut
