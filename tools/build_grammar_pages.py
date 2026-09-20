# build_grammar_pages.py — turn the KHANARY grammar docs into W3Schools-style sandbox pages.
#
# The backend was built first, so these pages don't invent anything: each is a reference reader over
# an existing grammar (EBNF + schema + worked examples), with a "Try it" box. Self-contained static
# HTML (no npm build) so the PRIMEOS WebView2 shell — or any browser — can host sandbox/index.html.
# Add a grammar by dropping an entry in GRAMMARS below.
import os, html, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "sandbox")

def rd(rel):
    p = os.path.join(ROOT, rel)
    return open(p, encoding="utf-8").read() if os.path.exists(p) else f"(missing: {rel})"

# Each grammar: id, title, tagline, intro (HTML), code sections [(label, ('file',rel) | ('text',s))],
# a "Try it" seed (rel path to an example or None), and the validate command.
GRAMMARS = [
    {
        "id": "kuhul-3d", "title": "K'UHUL-3D", "tagline": "Recursive semantic + compute execution contract",
        "intro": "<p><b>The rule:</b> the AST preserves the prompt/context; K'UHUL traverses it (phase lanes); "
                 "XCFE routes legal graph moves; opcodes perform work; compute nodes lower to CPU / llama.cpp / "
                 "WebGPU-WGSL / D3D11-HLSL.</p>"
                 "<p><b>Laws (machine-enforced):</b> <code>P1</code> Phase &cap; Opcode = &empty; · "
                 "<code>R1</code> Node&rarr;PhaseTick&rarr;PhaseStep&rarr;Node (a node <i>is</i> a tick) · "
                 "<code>G1</code> the glyph <i>is</i> the token.</p>",
        "sections": [
            ("Syntax (EBNF)", ("file", "docs/kuhul-3d-vnext.ebnf")),
            ("AST schema (JSON Schema 2020-12)", ("file", "docs/kuhul.ast.v3.schema.json")),
            ("Example — attention + verify", ("file", "docs/examples/kuhul.ast.v3.example.json")),
            ("Example — recursion (a node's own tick)", ("file", "docs/examples/kuhul.ast.v3.recursion.example.json")),
            ("Example — glyph atom (nativity)", ("file", "docs/examples/kuhul.ast.v3.glyph_atom.example.json")),
        ],
        "tryit": "docs/examples/kuhul.ast.v3.example.json",
        "validate": "python tools/check_kuhul_ast_v3.py",
    },
    {
        "id": "birdsong", "title": "Birdsong Geometry", "tagline": "Bird-song as executable geometry",
        "intro": "<p>Formalizes <code>audio &rarr; spectrogram &rarr; ridges &rarr; mesh &rarr; graph &rarr; experts</code> "
                 "onto the Pop&rarr;Xul fold cycle. Grounded in the real <code>birdsong_mesh.stb</code>: "
                 "<b>30,628 nodes / 91,863 edges / 183,726 neighbours</b>.</p>",
        "sections": [
            ("Syntax (EBNF)", ("file", "docs/birdsong-geometry.ebnf")),
            ("Dataset schema", ("file", "docs/birdsong-brain.schema.json")),
            ("Example (real subgraph from the .stb)", ("file", "docs/examples/birdsong.example.json")),
        ],
        "tryit": "docs/examples/birdsong.example.json",
        "validate": "python tools/check_birdsong.py",
    },
    {
        "id": "xcfe", "title": "XCFE", "tagline": "The legal-move / routing layer",
        "intro": "<p>XCFE is the control layer over the phase graph — it decides which <b>moves are legal</b> and "
                 "which <b>backend</b> a compute node routes to (chess moves over phase topology). It <b>routes</b>; "
                 "it never redefines the phases.</p>"
                 "<h3>Legal phase transitions</h3>"
                 "<table class=ref><tr><th>from</th><th>to</th><th>guard</th></tr>"
                 "<tr><td>Pop</td><td>Wo</td><td>all inputs bound</td></tr>"
                 "<tr><td>Wo</td><td>Sek</td><td>intent declared</td></tr>"
                 "<tr><td>Wo</td><td>Yax</td><td>route selected</td></tr>"
                 "<tr><td>Yax</td><td>Sek</td><td>fit confirmed</td></tr>"
                 "<tr><td>Sek</td><td>Ch'en</td><td>execution complete</td></tr>"
                 "<tr><td>Ch'en</td><td>Xul</td><td>output emitted</td></tr>"
                 "<tr><td>Xul</td><td>Pop</td><td>reset (cycle)</td></tr></table>",
        "sections": [
            ("Routing productions (EBNF)", ("text",
                'xcfe            = "[XCFE]", { xcfe_rule }, "[/XCFE]" ;\n'
                'xcfe_rule       = condition_rule | route_rule | reward_rule | mutation_rule | capability_rule ;\n'
                'route_rule      = "@route", expression, "->", identifier ;\n'
                'reward_rule     = "@reward", expression ;\n'
                'capability_rule = "@requires", capability ;')),
            ("Route (best legal backend by cost)", ("text", json.dumps({
                "xcfe": {"route": {"candidates": [
                    {"backend": "cache", "requires": ["memory"], "cost": 0.01},
                    {"backend": "d3d11", "requires": ["d3d11"], "cost": 0.20},
                    {"backend": "webgpu", "requires": ["webgpu"], "cost": 0.25},
                    {"backend": "llama", "requires": ["llama"], "cost": 0.40}],
                    "policy": "semantic_best_legal_path"}}}, indent=2))),
            ("The 5 admission gates", ("text",
                "1. STRUCTURE     valid phases / nodes / edges\n"
                "2. SEMANTICS     references resolve; context preserved\n"
                "3. XCFE          the transition / route is legal      <-- this layer\n"
                "4. CAPABILITY    every requested backend is registered\n"
                "5. COMPUTE       buffer shapes / dtypes / kernel contract valid\n"
                "-> ADMIT")),
        ],
        "tryit": None,
        "validate": "python tools/check_kuhul_ast_v3.py   # XCFE routes are part of the v3 contract",
    },
]

# Palette adopted (as inspiration, not authority) from the ASX "Atomic" design kits:
# near-black navy bg, JetBrains Mono, teal-mint accent, soft cyan-white text, coral + amber accents.
# Three user-selectable themes (matches llama's theme selection) via [data-theme] on <html>.
CSS = """
:root{--bg:#020617;--panel:#050b16;--panel2:#050818;--fg:#e5f2ff;--accent:#16f2aa;
      --hot:#ff6b6b;--warn:#facc15;--muted:#94a3b8;--faint:#64748b;--line:#1f2937;--code:#050b16}
[data-theme=cyan]{--bg:#020307;--panel:#050814;--panel2:#0a1520;--fg:#e8f5ff;--accent:#00f5ff;--muted:#7a8699;--line:#0a1a2a;--code:#040a14}
[data-theme=matrix]{--bg:#0a0e27;--panel:#0f1629;--panel2:#12203a;--fg:#c9f7d8;--accent:#00ff99;--muted:#6a86a0;--line:#00ff0033;--code:#060a1c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-family:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:14px}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
header{background:linear-gradient(180deg,var(--panel),var(--bg));border-bottom:1px solid var(--accent);padding:12px 20px;display:flex;align-items:center;gap:16px}
header h1{margin:0;color:var(--accent);font-size:17px;letter-spacing:1px;flex:0 0 auto;text-shadow:0 0 12px var(--accent)}
header .tag{color:var(--muted);flex:1}
header label{color:var(--muted);font-size:12px}
header select{background:var(--panel);color:var(--fg);border:1px solid var(--accent);border-radius:5px;padding:4px 8px;font-family:inherit}
.wrap{display:flex;min-height:calc(100vh - 52px)}
nav{width:230px;background:var(--panel);border-right:1px solid var(--line);padding:16px}
nav h2{color:var(--accent);font-size:12px;text-transform:uppercase;letter-spacing:1px}
nav a{display:block;padding:6px 8px;border-left:2px solid transparent;color:var(--fg)}
nav a.active,nav a:hover{border-left:2px solid var(--accent);background:var(--panel2)}
main{flex:1;padding:24px 32px;max-width:1000px}
main>h1{color:var(--accent);text-shadow:0 0 14px var(--accent)}
h2.sec{color:var(--accent);border-bottom:1px solid var(--line);padding-bottom:4px;margin-top:32px}
h3{color:var(--accent)}
pre{background:var(--code);border:1px solid var(--line);border-radius:6px;padding:12px;overflow:auto;max-height:520px;color:var(--fg);line-height:1.45}
code{color:var(--accent)}
.ex-bar{background:var(--panel2);color:var(--accent);border:1px solid var(--accent);border-bottom:none;border-radius:6px 6px 0 0;padding:6px 12px;font-weight:bold;margin-top:16px}
.ex-bar + pre{border-radius:0 0 6px 6px;margin-top:0;border-color:var(--accent)}
table.ref{border-collapse:collapse;margin:8px 0}
table.ref th,table.ref td{border:1px solid var(--line);padding:4px 12px;text-align:left}
table.ref th{color:var(--accent)}
.tryit textarea{width:100%;height:220px;background:var(--code);color:var(--fg);border:1px solid var(--accent);border-radius:6px;padding:10px;font-family:inherit;font-size:13px}
.validate{background:var(--panel2);border-left:3px solid var(--accent);padding:8px 12px;margin:12px 0;color:var(--fg)}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px;margin-top:20px}
.card{background:var(--panel);border:1px solid var(--accent);border-radius:8px;padding:16px}
.card:hover{box-shadow:0 0 18px -4px var(--accent)}
.card h3{margin:0 0 6px}.card p{color:var(--muted)}
footer{color:var(--muted);padding:16px 32px;border-top:1px solid var(--line)}
"""

THEME_JS = ("<script>function setTheme(t){document.documentElement.dataset.theme=t;"
            "try{localStorage.setItem('khTheme',t)}catch(e){}"
            "var s=document.getElementById('themeSel');if(s)s.value=t;}"
            "setTheme((function(){try{return localStorage.getItem('khTheme')}catch(e){}})()||'atomic');</script>")
THEME_SELECT = ('<label>theme</label><select id="themeSel" onchange="setTheme(this.value)">'
                '<option value="atomic">Atomic</option><option value="cyan">Cyan</option>'
                '<option value="matrix">Matrix</option></select>')

def esc(s): return html.escape(s)

def nav_html(active):
    items = "".join(f'<a class="{"active" if g["id"]==active else ""}" href="{g["id"]}.html">{esc(g["title"])}</a>' for g in GRAMMARS)
    return f'<nav><h2>Grammars</h2><a class="{"active" if active=="index" else ""}" href="index.html">Overview</a>{items}</nav>'

def page(title, active, body):
    return (f'<!DOCTYPE html><html data-theme="atomic"><head><meta charset="utf-8"><title>KHANARY sandbox — {esc(title)}</title>'
            f'<link rel="stylesheet" href="style.css"></head><body>'
            f'<header><h1>KHΛNARY developer sandbox</h1><span class="tag">grammar reference &amp; try-it</span>{THEME_SELECT}</header>'
            f'<div class="wrap">{nav_html(active)}<main>{body}</main></div>'
            f'<footer>Generated from the grammar docs by <code>tools/build_grammar_pages.py</code> — the pages read the specs, they do not redefine them.</footer>'
            f'{THEME_JS}</body></html>')

def grammar_page(g):
    b = [f'<h1 style="color:var(--grn)">{esc(g["title"])}</h1><p class="tag">{esc(g["tagline"])}</p>', g["intro"]]
    for label, src in g["sections"]:
        kind, val = src
        text = rd(val) if kind == "file" else val
        b.append(f'<div class="ex-bar">{esc(label)}</div><pre>{esc(text)}</pre>')
    if g["tryit"]:
        seed = esc(rd(g["tryit"]))
        b.append('<h2 class="sec">Try it</h2><div class="tryit"><textarea spellcheck="false">' + seed +
                 '</textarea></div><p class="tag">Edit above, then validate locally:</p>')
    if g["validate"]:
        b.append(f'<div class="validate">$ {esc(g["validate"])}</div>')
    return page(g["title"], g["id"], "".join(b))

def index_page():
    cards = "".join(f'<a class="card" href="{g["id"]}.html"><h3>{esc(g["title"])}</h3><p>{esc(g["tagline"])}</p></a>' for g in GRAMMARS)
    body = ('<h1 style="color:var(--grn)">Grammar sandbox</h1>'
            '<p>W3Schools-style reference for the KHANARY grammars. The backend was built first — each page '
            'surfaces a spec that already exists (EBNF + schema + worked examples), each with machine-checked '
            'validators. Add a grammar by extending <code>tools/build_grammar_pages.py</code>.</p>'
            f'<div class="cards">{cards}</div>')
    return page("Overview", "index", body)

def main():
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "style.css"), "w", encoding="utf-8").write(CSS)
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(index_page())
    for g in GRAMMARS:
        open(os.path.join(OUT, f"{g['id']}.html"), "w", encoding="utf-8").write(grammar_page(g))
    print(f"[out] {OUT}  ->  index.html + {len(GRAMMARS)} grammar pages ({', '.join(g['id'] for g in GRAMMARS)}) + style.css")

if __name__ == "__main__":
    main()
