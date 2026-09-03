#include "xjson_loader.h"
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

// Extract value of first "key": "value" match in a line.
static std::string extract_str(const std::string& line, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    auto pos = line.find(needle);
    if (pos == std::string::npos) return "";
    auto colon = line.find(':', pos + needle.size());
    if (colon == std::string::npos) return "";
    auto q1 = line.find('"', colon + 1);
    if (q1 == std::string::npos) return "";
    auto q2 = line.find('"', q1 + 1);
    if (q2 == std::string::npos) return "";
    return line.substr(q1 + 1, q2 - q1 - 1);
}

static int extract_int(const std::string& line, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    auto pos = line.find(needle);
    if (pos == std::string::npos) return 0;
    auto colon = line.find(':', pos + needle.size());
    if (colon == std::string::npos) return 0;
    // skip whitespace
    size_t i = colon + 1;
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
    if (i >= line.size() || !isdigit(line[i])) return 0;
    return std::stoi(line.substr(i));
}

} // anon

XjsonManifest load_xjson(const std::string& xjson_path) {
    XjsonManifest m;
    std::ifstream f(xjson_path);
    if (!f.is_open()) return m;

    // Section tracking — minimal finite state
    enum Section { NONE, META, GLYPH, PHASES, PHASE_ENTRY };
    Section sec = NONE;
    std::string cur_phase;

    std::string line;
    while (std::getline(f, line)) {
        trim(line);
        if (line.empty()) continue;

        // Section detection
        if (line.find("\"@meta\"") != std::string::npos)   { sec = META;   continue; }
        if (line.find("\"@glyph\"") != std::string::npos)  { sec = GLYPH;  m.glyph.present = true; continue; }
        if (line.find("\"@phases\"") != std::string::npos) { sec = PHASES; continue; }

        // Phase sub-entry detection (inside @phases)
        if (sec == PHASES) {
            for (const auto& ph : {"\"Pop\"","\"Wo\"","\"Yax\"","\"Sek\"","\"Ch'en\"","\"Chen\"","\"Xul\""}) {
                if (line.find(ph) != std::string::npos) {
                    cur_phase = std::string(ph + 1, strlen(ph) - 2);
                    sec = PHASE_ENTRY;
                    break;
                }
            }
        }

        // End of current block
        if (line == "}" || line == "},") {
            if (sec == PHASE_ENTRY) { sec = PHASES; cur_phase.clear(); }
            else if (sec == PHASES || sec == GLYPH || sec == META) { sec = NONE; }
            continue;
        }

        switch (sec) {
        case META:
            if (m.id.empty())          m.id          = extract_str(line, "id");
            if (m.name.empty())        m.name        = extract_str(line, "name");
            if (m.description.empty()) m.description = extract_str(line, "description");
            if (m.version.empty())     m.version     = extract_str(line, "version");
            break;

        case GLYPH:
            if (m.glyph.token.empty())       m.glyph.token       = extract_str(line, "token");
            if (m.glyph.shared_with.empty()) m.glyph.shared_with = extract_str(line, "shared_with");
            if (m.glyph.orientation.empty()) m.glyph.orientation = extract_str(line, "orientation");
            if (m.glyph.pole.empty())        m.glyph.pole        = extract_str(line, "pole");
            if (m.glyph.phase_angle.empty()) m.glyph.phase_angle = extract_str(line, "phase_angle");
            if (!m.glyph.rotation_deg)       m.glyph.rotation_deg = extract_int(line, "rotation_deg");
            if (!m.glyph.phase_deg)          m.glyph.phase_deg    = extract_int(line, "phase_deg");
            break;

        case PHASE_ENTRY:
            if (!cur_phase.empty()) {
                auto& pd = m.phases[cur_phase];
                if (pd.op.empty())   pd.op   = extract_str(line, "@op");
                if (pd.desc.empty()) pd.desc = extract_str(line, "@desc");
            }
            break;

        default: break;
        }

        // Pick up fold token from @agent.main or anywhere it appears
        if (m.fold.empty()) {
            auto v = extract_str(line, "fold");
            if (!v.empty() && v.find("FOLD") != std::string::npos) m.fold = v;
        }
    }

    m.valid = !m.id.empty();
    return m;
}

} // namespace micronaut
