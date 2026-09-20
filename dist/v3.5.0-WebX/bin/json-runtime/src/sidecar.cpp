#include "sidecar.hpp"
#include "xcfe.hpp"
#include "file_system.hpp"

#include <iostream>
#include <stdexcept>

SidecarLoader::SidecarLoader()  = default;
SidecarLoader::~SidecarLoader() = default;

bool SidecarLoader::load(const std::string& path, const std::string& name,
                          const json* stdlib_ops) {
    // Sandbox check
    if (!path.empty() && (path[0] == '/' || path.find("..") != std::string::npos)) {
        std::cerr << "[sidecar] path rejected: " << path << "\n";
        return false;
    }

    json doc;
    try {
        doc = FileSystem::load_json(path);
    } catch (const std::exception& e) {
        std::cerr << "[sidecar] load failed (" << path << "): " << e.what() << "\n";
        return false;
    }

    Sidecar s;
    s.name = name;
    if (doc.contains("@sidecar") && doc["@sidecar"].is_object()) {
        s.capability = doc["@sidecar"];
    } else if (doc.contains("@sidecar") && doc["@sidecar"].is_string()) {
        s.capability = json{{"id", doc["@sidecar"]}};
    } else {
        s.capability = json::object();
    }

    const char* manifest_keys[] = {
        "@routes", "routes", "@ports", "ports", "@schema", "schema", "@ops", "ops",
        "@supernaut", "supernaut", "@contracts", "contracts", "@runstate", "runstate",
        "@mx2lm", "mx2lm", "@mx2lm_runtime", "mx2lm_runtime", "@llm", "llm",
        "@field_system", "field_system", "@ngrams", "ngrams", "@io", "io",
        "@coordinator", "coordinator", "@brain_js", "brain_js",
        "@compiler_files", "compiler_files",
        "@micronaut_brains", "micronaut_brains", "@profiles", "profiles",
        "@intents", "intents", "@ngram_routing", "ngram_routing",
        "@chat_hive", "chat_hive", "@kernel_processes", "kernel_processes",
        "@chat_config", "chat_config", "@data_sources", "data_sources",
        "@kuhul_chat_engine", "kuhul_chat_engine", "@python_interface", "python_interface",
        "@integration", "integration", "@boot_sequence", "boot_sequence",
        "@performance", "performance",
        "@primeos_agent_spawner", "primeos_agent_spawner", "@micronaut_factory", "micronaut_factory",
        "@model_types", "model_types", "@agent_specializations", "agent_specializations",
        "@static_tunnel_browser", "static_tunnel_browser", "@training_system", "training_system",
        "@message_channel", "message_channel",
        "@nnc", "nnc", "@xjson_runtime", "xjson_runtime", "@kuhul_engine", "kuhul_engine",
        "@scx_compressor", "scx_compressor", "@klh_backend", "klh_backend",
        "@asx_runtime", "asx_runtime",
        "@gravity_semantic_wells", "gravity_semantic_wells", "@wells", "wells",
        "@bigrams", "bigrams", "@semantic_codec", "semantic_codec",
        "@kuhul_xsd_grammar", "kuhul_xsd_grammar", "@hierarchy", "hierarchy",
        "@phase_machine", "phase_machine", "@knumatics", "knumatics",
        "@entropy_economy", "entropy_economy", "@xsd_schema", "xsd_schema",
        "@nnc_xsd_neural_grammar", "nnc_xsd_neural_grammar", "@integration", "integration",
        "@schema_refs", "schema_refs", "@neural_grammar", "neural_grammar",
        "@crown_matrix", "crown_matrix", "@actor_matrix", "actor_matrix",
        "@actors", "actors", "@actor_selection", "actor_selection",
        "@cheese_reward_policy", "cheese_reward_policy",
        "@personality_matrix", "personality_matrix", "@matrix_ops", "matrix_ops",
        "@matrix_refs", "matrix_refs",
        "@mathml_nnc_xsd", "mathml_nnc_xsd", "@mathml", "mathml",
        "@tensor_math", "tensor_math",
        "@xcfe_js_surfaces", "xcfe_js_surfaces", "@role_reversal", "role_reversal",
        "@projection", "projection", "@modules", "modules",
        "@kuhul_math_engine", "kuhul_math_engine", "@glyph_registry", "glyph_registry",
        "@executor_policy", "executor_policy",
        "@agl_xjson_contracts", "agl_xjson_contracts", "@reserved_keys", "reserved_keys",
        "@xjson_kuhul_merge", "xjson_kuhul_merge",
        "@core_merge_architecture", "core_merge_architecture",
        "@autonomous_ai_pipeline_xjson", "autonomous_ai_pipeline_xjson",
        "@geometric_execution_engine", "geometric_execution_engine",
        "@autonomous_agent_xjson", "autonomous_agent_xjson",
        "@direct_file_ops_no_chat", "direct_file_ops_no_chat",
        "@continuous_improvement_cycle", "continuous_improvement_cycle",
        "@performance_optimizations", "performance_optimizations",
        "@deployment_ready", "deployment_ready",
        "@kuhul_pi_virtual_cluster", "kuhul_pi_virtual_cluster",
        "@svg_model_distribution", "svg_model_distribution", "@cluster_nodes", "cluster_nodes",
        "@kuhul_cluster_analytics", "kuhul_cluster_analytics",
        "@archetypes", "archetypes", "@analytics", "analytics",
        "@kuhul_character_engine", "kuhul_character_engine",
        "@character_archetypes", "character_archetypes", "@virtual_pi_runtime", "virtual_pi_runtime",
        "@ecosystem", "ecosystem", "@asx_components", "asx_components",
        "@kuhul_learning_browser", "kuhul_learning_browser", "@learning_brain", "learning_brain",
        "@knowledge_graph", "knowledge_graph", "@attention_network", "attention_network",
        "@kuhul_auto_coder", "kuhul_auto_coder",
        "@ai_specialists_team", "ai_specialists_team", "@specialists", "specialists",
        "@workflow", "workflow", "@mx2lm_runtime_boot", "mx2lm_runtime_boot",
        "@boot_stack", "boot_stack", "@glyph_stream", "glyph_stream",
        "@mx2lm_api_brain", "mx2lm_api_brain", "@api_brain_layers", "api_brain_layers",
        "@api_brain_services", "api_brain_services", "@atomic_cluster_spec", "atomic_cluster_spec",
        "@entry", "entry", "@assets", "assets", "@actions", "actions",
        "@code_generation", "code_generation",
        "@rpc", "rpc", "@jsonrpc", "jsonrpc", "@methods", "methods",
        "@mcp", "mcp", "@resources", "resources", "@prompts", "prompts",
        "@network", "network", "@http", "http", "@api_client", "api_client",
        "@curl", "curl", "@requests", "requests", "@downloads", "downloads",
        "@headers", "headers", "@auth", "auth", "@endpoints", "endpoints",
        "@gpu", "gpu", "@d3d9", "d3d9", "@d3d10", "d3d10",
        "@d3d11", "d3d11", "@d3d11_1", "d3d11_1", "@d3d12", "d3d12",
        "@d3dconfig", "d3dconfig", "@direct3d", "direct3d",
        "@direct2d", "direct2d", "@d2d1", "d2d1",
        "@direct3d_core", "direct3d_core", "@direct3d_debug_layer", "direct3d_debug_layer",
        "@warp", "warp", "@d3d11on12", "d3d11on12", "@app_overrides", "app_overrides",
        "@d3dcompiler", "d3dcompiler", "@d3dx", "d3dx",
        "@dxc", "dxc", "@dxcompiler", "dxcompiler", "@dxil", "dxil",
        "@shader_model_6", "shader_model_6",
        "@shader_compilers", "shader_compilers", "@graphics_dlls", "graphics_dlls",
        "@webgpu", "webgpu", "@webgl2", "webgl2",
        "@threejs", "threejs", "@three_js", "three_js",
        "@orbit_controls", "orbit_controls", "@web_graphics_assets", "web_graphics_assets",
        "@opencl", "opencl",
        "@glsl", "glsl", "@opengl", "opengl", "@gl43", "gl43", "@gl_compute", "gl_compute",
        "@compute", "compute", "@kernels", "kernels",
        "@user", "user", "@users", "users", "@identity", "identity",
        "@memory", "memory", "@permissions", "permissions",
        "@dotnet", "dotnet", "@dotnet_cli", "dotnet_cli", "@sdk_manifests", "sdk_manifests",
        "@shared_frameworks", "shared_frameworks", "@windows_desktop", "windows_desktop",
        "@wpf", "wpf", "@winforms", "winforms", "@webview2", "webview2",
        "@wwahost", "wwahost", "@uwp", "uwp",
        "@windows_web_app_host", "windows_web_app_host",
        "@cmd", "cmd", "@command_prompt", "command_prompt", "@batch_files", "batch_files",
        "@py_launcher", "py_launcher", "@python", "python", "@python_scripts", "python_scripts",
        "@cscript", "cscript", "@wscript", "wscript",
        "@windows_script_host", "windows_script_host",
        "@vbscript", "vbscript", "@jscript", "jscript",
        "@script_files", "script_files",
        "@apphost", "apphost", "@workloads", "workloads",
        "@workload_manifests", "workload_manifests", "@workload_families", "workload_families",
        "@emscripten", "emscripten", "@mono_toolchain", "mono_toolchain",
        "@android", "android", "@ios", "ios", "@maccatalyst", "maccatalyst",
        "@macos", "macos", "@maui", "maui", "@tvos", "tvos", "@localize", "localize",
        "@roslyn", "roslyn", "@csc", "csc", "@vbc", "vbc",
        "@vbcscompiler", "vbcscompiler", "@dotnet_format", "dotnet_format",
        "@msbuild", "msbuild", "@analyzers", "analyzers",
        "@source_generators", "source_generators",
        "@cmake", "cmake", "@ctest", "ctest", "@cpack", "cpack",
        "@cmake_gui", "cmake_gui", "@cmcldeps", "cmcldeps",
        "@visual_studio_tools", "visual_studio_tools", "@vcvars", "vcvars",
        "@msvc", "msvc", "@native_tools", "native_tools",
        "@cross_tools", "cross_tools", "@toolchain", "toolchain",
        "@build_presets", "build_presets", "@test_presets", "test_presets",
        "@package_presets", "package_presets",
        "@session", "session", "@sessions", "sessions",
        "@conversation", "conversation", "@history", "history",
        "@database", "database", "@databases", "databases", "@tables", "tables",
        "@indexes", "indexes", "@queries", "queries",
        "@projects", "projects", "@project", "project", "@workspace", "workspace",
        "@workspaces", "workspaces",
        "@commands", "commands", "@tools", "tools", "@action_classes", "action_classes",
        "@bots", "bots",
        "@web_scrape_micronauts", "web_scrape_micronauts",
        "@bot", "bot", "@bot_helpers", "bot_helpers", "@native_bots", "native_bots",
        "@bots_cpp", "bots_cpp", "@micronaut", "micronaut", "@workers", "workers",
        "@worker", "worker", "@factory", "factory", "@beans", "beans",
        "@schedules", "schedules", "@events", "events", "@di", "di",
        "@experts", "experts", "@expert_pool", "expert_pool",
        "@evolution", "evolution", "@mutation", "mutation", "@reward", "reward",
        "@genome_contract", "genome_contract", "@opcode_surface", "opcode_surface",
        "@models", "models", "@model", "model", "@gguf", "gguf",
        "@mgguf", "mgguf", "@model_server", "model_server",
        "@adapters", "adapters", "@providers", "providers",
        "@glyphs", "glyphs", "@canonical_glyph_registry", "canonical_glyph_registry",
        "@fold_op_notation", "fold_op_notation", "@symbols", "symbols",
        "@angles", "angles", "@angle_contracts", "angle_contracts",
        "@lanes", "lanes", "@lane_contracts", "lane_contracts",
        "@opcodes", "opcodes", "@phases", "phases", "@folds", "folds",
        "@fold_algebra", "fold_algebra", "@xcfe", "xcfe",
        "@nodes", "nodes", "@phase", "phase", "@phase_id", "phase_id",
        "@bson", "bson", "@documents", "documents", "@collections", "collections",
        "@binary", "binary", "@binary_runtime_array", "binary_runtime_array",
        "@glyph_lane_keys", "glyph_lane_keys", "@codec", "codec", "@agents", "agents",
        "@agent", "agent", "@batches", "batches", "@batch", "batch",
        "@threads", "threads", "@thread", "thread", "@processes", "processes",
        "@process", "process", "@skills", "skills", "@skill", "skill",
        "@logs", "logs", "@log", "log", "@trace", "trace", "@audit", "audit",
        "@event", "event",
        "@xjson", "xjson", "@xcfe", "xcfe",
        "@ir", "ir", "@kimd", "kimd", "@linalg", "linalg",
        "@core_types", "core_types", "@scalar", "scalar",
        "@vector", "vector", "@matrix", "matrix", "@tensor", "tensor",
        "@geometry", "geometry", "@surface", "surface", "@cluster", "cluster",
        "@opcodes", "opcodes", "@ir_grammar", "ir_grammar",
        "@replay_log", "replay_log", "@svg_tessellation", "svg_tessellation",
        "@render_kernel", "render_kernel", "@d3d11_submission", "d3d11_submission",
        "@tiny_character_loop", "tiny_character_loop",
        "@animated_characters", "animated_characters",
        "@cartoon_character", "cartoon_character",
        "@character_rig", "character_rig", "@character_rigging", "character_rigging",
        "@animation_loop", "animation_loop", "@motion", "motion",
        "@procedural_animation", "procedural_animation",
        "@character_state_machine", "character_state_machine",
        "@screen", "screen", "@screen_vector_graphics", "screen_vector_graphics",
        "@vector_screen", "vector_screen",
        "@vector_surface", "vector_surface", "@visual_substrate", "visual_substrate",
        "@svg_nodes", "svg_nodes", "@tensor_surfaces", "tensor_surfaces",
        "@glyph_ops", "glyph_ops", "@json_columns", "json_columns",
        "@tensor_io", "tensor_io", "@native_projection_roots", "native_projection_roots",
        "@surface_syntax", "surface_syntax", "@lowering", "lowering",
        "@canonical_ast", "canonical_ast", "@canonical_json", "canonical_json",
        "@static_verifier", "static_verifier", "@runtime_walk", "runtime_walk",
        "@stdlib", "stdlib", "@pack_manifest", "pack_manifest",
        "@proof_envelope", "proof_envelope", "@signature", "signature",
        "@policy", "policy", "@crypto_pack", "crypto_pack",
        "@session_binding", "session_binding", "@scx_chain", "scx_chain",
        "@grammar", "grammar", "@grammars", "grammars",
        "@classes", "classes", "@class", "class",
        "@test_vectors", "test_vectors", "@expr_blocks", "expr_blocks",
        "@declarative_blocks", "declarative_blocks", "@flow_operator", "flow_operator",
        "@capabilities", "capabilities", "@determinism", "determinism",
        "@parse", "parse", "@parser", "parser", "@ast", "ast", "@kast", "kast",
        "@validate", "validate", "@validator", "validator", "@rules", "rules",
        "@compile", "compile", "@compiler", "compiler", "@targets", "targets",
        "@generate", "generate", "@generator", "generator", "@templates", "templates",
        "@program", "program", "@programs", "programs", "@scxq2", "scxq2",
        "@packer", "packer",
        "@sco", "sco"
    };
    for (const char* key : manifest_keys) {
        if (doc.contains(key) && !s.capability.contains(key)) s.capability[key] = doc[key];
    }

    s.op_defs = doc.value("@ops", json::object());

    // Create inner XCFE and load its ops
    s.xcfe = std::make_unique<XCFE>();
    if (stdlib_ops && stdlib_ops->is_object()) {
        s.xcfe->load_ops(*stdlib_ops);
    }
    if (!s.op_defs.is_null()) {
        s.xcfe->load_ops(s.op_defs);
    }

    // Run @init sequence
    if (doc.contains("@init")) {
        try {
            json init_prog = json{{"@control", doc["@init"]}};
            s.xcfe->execute(init_prog);
        } catch (const std::exception& e) {
            std::cerr << "[sidecar] init error (" << name << "): " << e.what() << "\n";
        }
    }

    s.initialized = true;
    sidecars_[name] = std::move(s);
    std::cout << "[sidecar] loaded: " << name
              << " ops=" << sidecars_[name].op_defs.size() << "\n";
    return true;
}

json SidecarLoader::call(const std::string& sidecar_name,
                          const std::string& op_name,
                          scope_t& scope) {
    auto it = sidecars_.find(sidecar_name);
    if (it == sidecars_.end()) {
        return json{{"error", "sidecar_not_found"}, {"name", sidecar_name}};
    }

    Sidecar& s = it->second;
    if (!s.initialized || !s.xcfe) {
        return json{{"error", "sidecar_not_initialized"}, {"name", sidecar_name}};
    }

    // Inject caller scope into sidecar's global_state
    for (auto& [k, v] : scope) s.xcfe->global_state[k] = v;

    // Build a call node
    json call_node = json{{"@op", op_name}};
    // Forward scope values as @state
    json step_prog = json{
        {"@state", json::object()},
        {"@control", json::array({call_node})}
    };
    for (auto& [k, v] : scope) step_prog["@state"][k] = v;

    try {
        s.xcfe->execute(step_prog);
        // Return result key if present
        json result = json::object();
        for (auto& [k, v] : s.xcfe->global_state) result[k] = v;
        return result;
    } catch (const std::exception& e) {
        return json{{"error", e.what()}, {"sidecar", sidecar_name}, {"op", op_name}};
    }
}

bool SidecarLoader::has(const std::string& name) const {
    return sidecars_.count(name) > 0;
}

json SidecarLoader::capability(const std::string& name) const {
    auto it = sidecars_.find(name);
    if (it == sidecars_.end()) return json(nullptr);
    return it->second.capability;
}

std::vector<std::string> SidecarLoader::list() const {
    std::vector<std::string> v;
    for (auto& [k, _] : sidecars_) v.push_back(k);
    return v;
}

const Sidecar* SidecarLoader::get(const std::string& name) const {
    auto it = sidecars_.find(name);
    if (it == sidecars_.end()) return nullptr;
    return &it->second;
}
