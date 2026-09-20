#include "phase_router.h"
#include "bots_worker.h"
#include "output.h"
#include <sstream>

namespace micronaut {

// ---------------------------------------------------------------------------
// Opcode table — mirrors programs/opcode.table.json 0x00–0x0F
// ---------------------------------------------------------------------------

static const OpcodeEntry k_table[] = {
    { 0x00, "G_NOP",            "",      "control.nop",              "",              "",        0,   "",      ""           },
    { 0x01, "G_PHASE_POP",      "Pop",   "phase.enter.pop",          "",              "memory",  0,   "0",     "Phi"        },
    { 0x02, "G_PHASE_WO",       "Wo",    "phase.enter.wo",           "",              "JYGG-1",  60,  "π/3",   "Fold"       },
    { 0x03, "G_PHASE_YAX",      "Yax",   "phase.enter.yax",          "",              "ALICE-1", 120, "2π/3",  "Gram"       },
    { 0x04, "G_PHASE_SEK",      "Sek",   "phase.enter.sek",          "",              "AM-1",    180, "π",     "Geodesic"   },
    { 0x05, "G_PHASE_CHEN",     "Chen",  "phase.enter.chen",         "",              "ELIZA-1", 240, "4π/3",  "Projection" },
    { 0x06, "G_PHASE_XUL",      "Xul",   "phase.enter.xul",          "",              "SHEOG-1", 300, "5π/3",  "Entropy"    },
    { 0x07, "G_FOLD_POP",       "Pop",   "fold.marker.pop",          "⟁POP_FOLD⟁",   "memory",  0,   "0",     "Phi"        },
    { 0x08, "G_FOLD_WO",        "Wo",    "fold.marker.wo",           "⟁WO_FOLD⟁",    "JYGG-1",  60,  "π/3",   "Fold"       },
    { 0x09, "G_FOLD_YAX",       "Yax",   "fold.marker.yax",          "⟁YAX_FOLD⟁",   "ALICE-1", 120, "2π/3",  "Gram"       },
    { 0x0A, "G_FOLD_SEK",       "Sek",   "fold.marker.sek",          "⟁SEK_FOLD⟁",   "AM-1",    180, "π",     "Geodesic"   },
    { 0x0B, "G_FOLD_CHEN",      "Chen",  "fold.marker.chen",         "⟁CHEN_FOLD⟁",  "ELIZA-1", 240, "4π/3",  "Projection" },
    { 0x0C, "G_FOLD_XUL",       "Xul",   "fold.marker.xul",          "⟁XUL_FOLD⟁",   "SHEOG-1", 300, "5π/3",  "Entropy"    },
    { 0x0D, "G_DUAL_SHEO_JYGG", "",      "glyph.dual.chaos_order",   "⟁DUAL_SHEO_JYGG⟁", "", 0, "",  ""           },
    { 0x0E, "G_CHEESE_EVAL",    "Chen",  "reinforcement.cheese_eval","",              "SHEOG-1", 240, "4π/3",  "Projection" },
    { 0x0F, "G_GREYMARCH",      "Xul",   "reinforcement.greymarch",  "",              "JYGG-1",  300, "5π/3",  "Entropy"    },
};

static constexpr size_t k_table_size = sizeof(k_table) / sizeof(k_table[0]);

const OpcodeEntry* opcode_table_begin() { return k_table; }
const OpcodeEntry* opcode_table_end()   { return k_table + k_table_size; }

const OpcodeEntry* lookup_opcode(uint8_t opcode) {
    for (const auto& e : k_table)
        if (e.opcode == opcode) return &e;
    return nullptr;
}

const OpcodeEntry* lookup_by_semantic(const std::string& semantic) {
    for (const auto& e : k_table)
        if (e.semantic == semantic) return &e;
    return nullptr;
}

const OpcodeEntry* lookup_by_fold(const std::string& fold_name) {
    // Returns the G_PHASE_* entry for a given fold name
    for (const auto& e : k_table)
        if (e.fold == fold_name && e.semantic.rfind("phase.enter.", 0) == 0) return &e;
    return nullptr;
}

// ---------------------------------------------------------------------------
// Phase routing helpers
// ---------------------------------------------------------------------------

namespace {

static std::string op_suffix(const std::string& op) {
    auto dot = op.rfind('.');
    return dot == std::string::npos ? op : op.substr(dot + 1);
}

static std::string emit_opcode_event(const std::string& req_id, const OpcodeEntry& e) {
    std::ostringstream o;
    o << "{\"opcode\":\"0x" << std::hex << (int)e.opcode << std::dec << "\""
      << ",\"glyph\":\"" << json_escape(e.glyph) << "\""
      << ",\"semantic\":\"" << json_escape(e.semantic) << "\"";
    if (!e.fold.empty())       o << ",\"fold\":\"" << json_escape(e.fold) << "\"";
    if (!e.phase_angle.empty())o << ",\"phase_angle\":\"" << json_escape(e.phase_angle) << "\"";
    if (e.phase_deg)           o << ",\"phase_deg\":" << e.phase_deg;
    if (!e.k_cube_face.empty())o << ",\"k_cube_face\":\"" << json_escape(e.k_cube_face) << "\"";
    if (!e.token.empty())      o << ",\"token\":\"" << json_escape(e.token) << "\"";
    if (!e.micronaut.empty())  o << ",\"micronaut\":\"" << json_escape(e.micronaut) << "\"";
    o << "}";
    return o.str();
}

} // anon

// ---------------------------------------------------------------------------
// PhaseRouter::route
// ---------------------------------------------------------------------------

MicronautResult PhaseRouter::route(const MicronautOp& op) const {
    auto start = Clock::now();
    MicronautResult r;
    r.id = op.id;
    r.op = op.op;

    const std::string& oper = op.op;

    // --- fold.* ---------------------------------------------------------
    // "fold.pop" / "fold.wo" / ... → G_FOLD_* opcode + bots.py passthrough
    if (oper.rfind("fold.", 0) == 0) {
        std::string name = op_suffix(oper);
        // Capitalize first letter for table lookup
        if (!name.empty()) name[0] = std::toupper((unsigned char)name[0]);

        const OpcodeEntry* e = nullptr;
        for (const auto& entry : k_table) {
            if (entry.fold == name && entry.semantic.rfind("fold.marker.", 0) == 0) {
                e = &entry; break;
            }
        }

        if (e) {
            output().event(op.id, "opcode", emit_opcode_event(op.id, *e));
            // Forward to bots.py with fold context in payload
            std::string payload = op.payload.has("payload_json")
                ? op.payload.get("payload_json", "{}")
                : op.payload.raw.empty() ? "{}" : op.payload.raw;
            std::string resp = bots_worker().call(op_suffix(oper), payload);
            r.ok = true;
            r.result_json = "{\"fold\":\"" + json_escape(name) + "\""
                          + ",\"opcode\":\"G_FOLD_" + name + "\""
                          + ",\"token\":\"" + json_escape(e->token) + "\""
                          + ",\"bots_response\":" + resp + "}";
        } else {
            r.ok = false;
            r.error_code = "FOLD_UNKNOWN";
            r.error_message = "No fold marker for: " + oper;
        }
        auto end = Clock::now();
        r.duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        return r;
    }

    // --- phase.* --------------------------------------------------------
    // "phase.enter.pop" etc. → state transition only, no bots.py call
    if (oper.rfind("phase.", 0) == 0) {
        const OpcodeEntry* e = lookup_by_semantic(oper);
        if (e) {
            output().event(op.id, "opcode", emit_opcode_event(op.id, *e));
            r.ok = true;
            r.result_json = "{\"phase\":\"" + json_escape(e->fold) + "\""
                          + ",\"opcode\":\"" + json_escape(e->glyph) + "\""
                          + ",\"deg\":" + std::to_string(e->phase_deg)
                          + ",\"face\":\"" + json_escape(e->k_cube_face) + "\"}";
        } else {
            r.ok = false;
            r.error_code = "PHASE_UNKNOWN";
            r.error_message = "No phase opcode for: " + oper;
        }
        auto end = Clock::now();
        r.duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        return r;
    }

    // --- glyph.* --------------------------------------------------------
    if (oper.rfind("glyph.", 0) == 0) {
        const OpcodeEntry* e = lookup_by_semantic(oper);
        if (!e) e = lookup_by_semantic("glyph.dual.chaos_order"); // fallback
        if (e) {
            output().event(op.id, "opcode", emit_opcode_event(op.id, *e));
            r.ok = true;
            r.result_json = "{\"token\":\"" + json_escape(e->token) + "\""
                          + ",\"opcode\":\"" + json_escape(e->glyph) + "\""
                          + ",\"shared_by\":[\"SHEOG-1\",\"JYGG-1\"]"
                          + ",\"boundary\":0.85}";
        } else {
            r.ok = false;
            r.error_code = "GLYPH_UNKNOWN";
            r.error_message = "No glyph opcode for: " + oper;
        }
        auto end = Clock::now();
        r.duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        return r;
    }

    // --- reinforcement ops ----------------------------------------------
    if (oper == "micronaut.cheese_eval" || oper == "reinforcement.cheese_eval") {
        const OpcodeEntry* e = lookup_opcode(0x0E);
        output().event(op.id, "opcode", emit_opcode_event(op.id, *e));
        std::string payload = op.payload.get("candidate_json", op.payload.raw.empty() ? "{}" : op.payload.raw);
        std::string resp = bots_worker().call("cheese", payload);
        r.ok = true;
        r.result_json = "{\"opcode\":\"G_CHEESE_EVAL\",\"bots_response\":" + resp + "}";
        auto end = Clock::now();
        r.duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        return r;
    }

    if (oper == "micronaut.greymarch" || oper == "reinforcement.greymarch") {
        const OpcodeEntry* e = lookup_opcode(0x0F);
        output().event(op.id, "opcode", emit_opcode_event(op.id, *e));
        std::string payload = op.payload.raw.empty() ? "{}" : op.payload.raw;
        std::string resp = bots_worker().call("greymarch", payload);
        r.ok = true;
        r.result_json = "{\"opcode\":\"G_GREYMARCH\",\"bots_response\":" + resp + "}";
        auto end = Clock::now();
        r.duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        return r;
    }

    // --- bots.py passthrough for all other ops --------------------------
    // task = last segment of op (e.g. "health", "expand", "plan")
    std::string task = op_suffix(oper);
    std::string payload = op.payload.raw.empty() ? "{}" : op.payload.raw;
    std::string resp = bots_worker().call(task, payload);
    r.ok = !resp.empty();
    r.result_json = resp.empty()
        ? "{\"error\":\"bots_worker no response\",\"task\":\"" + task + "\"}"
        : resp;

    auto end = Clock::now();
    r.duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return r;
}

std::string PhaseRouter::lifecycle() {
    return "Pop(0°) → Wo(60°) → Yax(120°) → Sek(180°) → Chen(240°) → Xul(300°)";
}

PhaseRouter& phase_router() {
    static PhaseRouter router;
    return router;
}

} // namespace micronaut
