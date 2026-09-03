#pragma once
#include "common.h"
#include "toml_loader.h"
#include "xjson_loader.h"

namespace micronaut {

// Opcode table entry — mirrors programs/opcode.table.json entries 0x01–0x0F.
struct OpcodeEntry {
    uint8_t     opcode;
    std::string glyph;       // e.g. "G_PHASE_POP"
    std::string fold;        // "Pop" | "Wo" | "Yax" | "Sek" | "Chen" | "Xul" | ""
    std::string semantic;    // "phase.enter.pop" / "fold.marker.pop" / etc.
    std::string token;       // fold token string if present
    std::string micronaut;   // bound micronaut ID if present
    int         phase_deg = 0;
    std::string phase_angle;
    std::string k_cube_face;
};

// Build-time opcode table (0x00–0x0F from programs/opcode.table.json).
const OpcodeEntry* opcode_table_begin();
const OpcodeEntry* opcode_table_end();
const OpcodeEntry* lookup_opcode(uint8_t opcode);
const OpcodeEntry* lookup_by_semantic(const std::string& semantic);
const OpcodeEntry* lookup_by_fold(const std::string& fold_name);  // phase enter

// Phase-aware routing.
// Maps op prefixes to opcode dispatch or bots.py forwarding.
class PhaseRouter {
public:
    // Route an op through the correct phase handler.
    // - "fold.*"  → G_FOLD_* opcode + bots.py call
    // - "phase.*" → G_PHASE_* opcode (state transition only, no payload)
    // - "glyph.*" → dual glyph dispatch
    // - "micronaut.cheese_eval" → G_CHEESE_EVAL → bots.py
    // - "micronaut.greymarch"   → G_GREYMARCH   → bots.py
    // - anything else           → bots.py passthrough (task = op suffix after last '.')
    MicronautResult route(const MicronautOp& op) const;

    // Human-readable phase lifecycle string.
    static std::string lifecycle();
};

PhaseRouter& phase_router();

} // namespace micronaut
