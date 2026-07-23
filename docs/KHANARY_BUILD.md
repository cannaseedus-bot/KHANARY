# KHANARY build — the branded app + the SvelteKit menu that exposes it

The backend was built first. This is the recipe for the last layer: a branded KHANARY app whose
**SvelteKit menu pages each surface a capability that already exists** — nothing new to invent under
the hood, just pages that apply it.

## Assembly

```
PRIMEOS (WebView2 shell, net8)         ← desktop/PRIMEOS  (Part 1, done: launches the server)
        │ hosts
        ▼
KHANARY SvelteKit menu app             ← the pages below  (Part 3)
        │ talks to
        ▼
khanary-server  (branded llama + ggml-xcfe)   +   KHANARY MCP server(s)
        │                                              (mcp_server_v2.1_…js — bundled alongside)
        ▼
ggml backend registry: ggml-cpu / ggml-opencl  (today)  +  ggml-xcfe  (Part 2)
```

The shell (Part 1) is done. Parts 2 (real `ggml-xcfe` + branded build) and 3 (these pages) are the
remaining work — and the pages are mostly *wiring existing tools to a UI*, because the backend exists.

## The menu pages → the backend each one exposes

| SvelteKit page | What it does | Backend it surfaces (already built) |
|---|---|---|
| **Chat** | talk to a model | llama-server's stock UI (baked in) / OpenAI-compat API |
| **Models** | list / mount / dual-quant tiers | `models/khanary-*` version folders; Q4-base + Q8-escalation (`khanary-qwen1_8b`) |
| **Quantize** | any safetensors → Q4/Q8 | `tools/quantize_safetensors.py` (+ `verify_quant.py`); sha256 gate (`docs/QUANT_BUILD.md`) |
| **Train** | dataset → LoRA/train run → export | vendored D3D11 trainer (`khanary-gpt2-v0.4.0/trainer`), `safetensors_to_stb.py` |
| **Grammar** | author/validate K'UHUL-3D + Birdsong | `check_kuhul_ast_v3.py` (laws P1/R1/G1), `check_birdsong.py`, the schemas |
| **Runtime / GPU** | residency budget, proof ladder | `proof/gpu_resident_ceiling_v1` (~1.75 GiB), the KGRC ladder, `kuhul_matmul_tick_v1` |
| **Tools / MCP** | the 12 KXML tool calls | `models/khanary-kxml-v0.5.0` + the KHANARY MCP server |
| **Geometry** | birdsong mesh → GPU geometry | `khanary-geometry-v0.3.0`, `brain_to_stb.py` |

Each row is a page whose "logic" is an existing script/model/proof — the page is a thin front-end
over a proven capability. That is why the ordering (backend first) works: the pages *apply* the map.

## Build recipe (branded KHANARY server)

Grounded in the vendored source (`.ASX.cpp/llama-b9968-.../llama.cpp/`, read-only — copy out, don't edit in place):

1. **Copy source** into a KHANARY build workspace (e.g. `build/khanary-llama/`).
2. **Make `ggml-xcfe` real** — today it's an orphan byte-copy of `ggml-webgpu` (`docs/llama-ggml-bridges.md`):
   - rename the target throughout: `ggml_add_backend_library(ggml-xcfe …)`, real `include/ggml-xcfe.h`, `ggml-xcfe.cpp`;
   - implement the vtable (`ggml_backend_xcfe_reg`, `supports_op`, `graph_compute`, buffer types) — start as a
     CPU-delegating pass-through so it *builds*, then lower ops → KHANARY glyph kernels (KLSL→WGSL/HLSL);
   - wire `ggml_add_backend(XCFE)` in `ggml/src/CMakeLists.txt` + a `GGML_XCFE` option.
3. **Build:** `cmake -B build -DGGML_XCFE=ON … && cmake --build build --config Release`.
4. **Bundle:** the branded `llama-server.exe` (+ `ggml*.dll` siblings) + KHANARY MCP server(s) + a default model,
   under `desktop/PRIMEOS/llama/` (gitignored — see `PRIMEOS/README.md`).
5. **Brand:** `khanary.svg`/`.png`, rename `llama-server` → `khanary-server`, window/title.
6. **UI:** ship the KHANARY pages one of two ways (below).

## UI: two ways to ship KHANARY pages

llama's web UI lives at **`tools/ui/`** in the llama.cpp source (SvelteKit + Svelte 5 runes +
shadcn-svelte + TailwindCSS 4 + Vite; IndexedDB/Dexie + LocalStorage; MODEL & ROUTER modes; it
already has a **dark/light theme**). Its build pipeline: `cd tools/ui && npm i && npm run build`
→ Vite + static adapter emit a **single inlined, gzipped `index.html`** into `build/tools/ui/dist/`
→ **llama-server compiles that into the binary** (single portable exe). Dev: `npm run dev` (Vite
:5173, proxies `/v1 /props /models /tools /slots` → llama-server :8080).

Upstream deliberately keeps **customizable themes and third-party plugins out of scope**
(`tools/server/README-dev.md`) — the sanctioned extension is **your own MCP server** (which KHANARY
already has). So KHANARY owns its branding/pages in a **fork**, not via an upstream plugin API.

| option | how | trade-off |
|---|---|---|
| **A. Fork `tools/ui`** | copy it out (read-only vendored), add routes (`/train`, `/grammar`, `/models`), make the ASX Atomic palette a Tailwind theme, brand the logo, `npm run build` → rebuild `khanary-server` | one integrated app in the branded binary; **needs Node + `npm build` + a server rebuild** |
| **B. Separate KHANARY web app** | ship KHANARY pages as their own static/SvelteKit app the PRIMEOS WebView2 shell also hosts (e.g. the grammar sandbox already does this) | no llama rebuild; **two surfaces** (llama chat UI + KHANARY pages) the shell switches between |

The **grammar/dev-sandbox pages are already Option B, done** (`tools/build_grammar_pages.py` →
`sandbox/`, static, no npm). The **Train/Models pages** are the interactive ones — either add them as
Option-A routes in the `tools/ui` fork, or as more Option-B pages. The ASX Atomic / Cyan / Matrix
themes the sandbox ships are the candidate KHANARY theme set (matches "user selects their theme").

## Honest scope

- Part 1 (WebView2 shell that launches the server) is **done and builds**.
- Part 2 (real `ggml-xcfe` + branded build) is a real C++ project — the source is present, but the folder is an
  **orphan today**; the vtable that lowers ggml ops → KHANARY kernels is the substantive work.
- Part 3 (these pages) is mostly UI wiring over existing tools, **plus** a UI build for the custom pages.
- MCP servers are **KHANARY's own** (`mcp_server_v2.1_…js`), not part of llama.cpp — bundled alongside.
