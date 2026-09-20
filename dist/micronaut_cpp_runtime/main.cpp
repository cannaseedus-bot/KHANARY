#include "input.h"
#include "output.h"
#include "scxq2_runtime.h"
#include "xcfe_router.h"
#include "toml_loader.h"
#include "xjson_loader.h"
#include "bots_worker.h"
#include "phase_router.h"
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

static std::string find_xjson(const fs::path& dir) {
    for (const auto& e : fs::directory_iterator(dir)) {
        if (e.path().extension() == ".xjson") return e.path().string();
    }
    return "";
}

int main(int argc, char** argv) {
    using namespace micronaut;

    // -----------------------------------------------------------------------
    // CLI flags
    // -----------------------------------------------------------------------
    std::string work_dir;
    bool pretty = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--pretty") { pretty = true; }
        else if ((arg == "--dir" || arg == "--xjson-dir") && i + 1 < argc) {
            work_dir = argv[++i];
        }
        else if (arg == "--chat" && i + 1 < argc) {
            // Quick one-shot chat (no config required)
            std::string prompt = argv[++i];
            MicronautOp op;
            op.kind = "micronaut.request.v1";
            op.id   = "cli-" + now_ms_string();
            op.op   = "chat.generate";
            op.payload.raw = prompt;
            op.payload.flat["prompt"] = prompt;
            register_builtin_handlers();
            output().result(scxq2_execute(op));
            return 0;
        }
    }

    if (pretty) output().set_mode(OutputMode::Pretty);

    // -----------------------------------------------------------------------
    // Load config.@.toml from work_dir (or current dir)
    // -----------------------------------------------------------------------
    fs::path dir = work_dir.empty() ? fs::current_path() : fs::path(work_dir);
    fs::path toml_path = dir / "config.@.toml";

    AgentConfig agent_cfg;
    if (fs::exists(toml_path)) {
        agent_cfg = load_agent_config(toml_path.string());
        if (agent_cfg.valid) {
            output().log("info", "[micronaut] loaded config: id=" + agent_cfg.id
                + " fold=" + agent_cfg.fold
                + " port=" + std::to_string(agent_cfg.port)
                + " role=" + agent_cfg.swarm_role);
        } else {
            output().log("warn", "[micronaut] config.@.toml found but [agent] section missing or invalid");
        }
    } else {
        output().log("warn", "[micronaut] no config.@.toml in " + dir.string() + " — running headless");
    }

    // -----------------------------------------------------------------------
    // Load *.xjson manifest
    // -----------------------------------------------------------------------
    std::string xjson_path = find_xjson(dir);
    XjsonManifest xjson;
    if (!xjson_path.empty()) {
        xjson = load_xjson(xjson_path);
        if (xjson.valid) {
            output().log("info", "[micronaut] loaded xjson: id=" + xjson.id
                + " fold=" + xjson.fold
                + (xjson.glyph.present
                    ? " glyph=" + xjson.glyph.token + " orientation=" + xjson.glyph.orientation
                    : ""));
        }
    }

    // -----------------------------------------------------------------------
    // Start bots.py worker
    // -----------------------------------------------------------------------
    bool bots_ok = bots_worker().start(dir.string());
    if (bots_ok) {
        std::string health = bots_worker().call("health");
        output().log("info", "[micronaut] bots.py health: " + health);
    } else {
        output().log("warn", "[micronaut] bots.py not started: " + bots_worker().last_error()
            + " — fold/task ops will fail");
    }

    // -----------------------------------------------------------------------
    // Register all XCFE + phase/fold handlers
    // -----------------------------------------------------------------------
    register_builtin_handlers();

    // Log lifecycle
    output().log("info", "[micronaut] phase lifecycle: " + PhaseRouter::lifecycle());

    // -----------------------------------------------------------------------
    // STDIN JSON-line loop
    // -----------------------------------------------------------------------
    return run_stdin_loop();
}
