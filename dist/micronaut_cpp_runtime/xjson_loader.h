#pragma once
#include "common.h"
#include <map>

namespace micronaut {

struct GlyphDef {
    std::string token;           // e.g. "⟁DUAL_SHEO_JYGG⟁"
    std::string shared_with;     // partner micronaut ID
    std::string orientation;     // "upright" | "inverted"
    int         rotation_deg = 0;
    std::string pole;            // "chaos" | "order"
    std::string phase_angle;     // "5π/3"
    int         phase_deg    = 0;
    double      boundary     = 0.85;
    bool        present      = false;
};

struct PhaseDef {
    std::string op;
    std::string desc;
};

struct XjsonManifest {
    std::string id;
    std::string name;
    std::string description;
    std::string version;
    std::string fold;            // fold token e.g. "⟁XUL_FOLD⟁"
    GlyphDef    glyph;
    std::map<std::string, PhaseDef> phases; // Pop/Wo/Yax/Sek/Chen/Xul
    bool valid = false;
};

// Load @meta, @glyph, and @phases from an xjson file.
// Does not require a full JSON parser — scans key-value pairs line by line.
XjsonManifest load_xjson(const std::string& xjson_path);

} // namespace micronaut
