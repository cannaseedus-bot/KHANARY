# Branded KHANARY UI embedded + served by khanary-server (frozen)

**Part 3 — the UI is built, branded, sealed into the binary, and served.** The full loop works:
SvelteKit UI → KHANARY branding → `npm build` → embedded into `khanary-server` → served at runtime,
with the `ggml-xcfe` backend inside.

## The seal (three-step, as the source demands)

1. **`tools/build_khanary_ui.ps1`** — copies `tools/ui`, applies KHANARY branding (`APP_NAME=KHANARY`
   via `VITE_PUBLIC_APP_NAME`; ASX-Atomic teal recolor of `--primary/--ring/--sidebar-primary` in
   `app.css`), and `npm run build` with a larger Node heap (`--max-old-space-size=8192` — the stock
   CMake build OOM'd). Output: a branded `dist/index.html` (+ `_app/`).
2. **`tools/reseal_khanary_server.ps1`** — drops that `dist/` into the server workspace's
   `tools/ui/dist` (ui-assets.cmake **priority 1: pre-built assets → embed, no npm**) and rebuilds
   `llama-server` (incremental). Log: `-- UI: using pre-built assets from …/tools/ui/dist`.
3. The branded UI compiles into the binary (`ui.cpp`/`ui.h` C arrays → `llama-server-impl.dll`).

## Verified

```
embed:   llama-server-impl.dll contains "KHANARY" + 10 SvelteKit/HTML markers (UI baked in)
build:   "KHANARY" baked into the JS bundle (bundle.-ozWm5d2.js); app.css accent recolored to teal
runtime: khanary-server -m gpt2.Q8_0.gguf  ->  model loaded, listening
         GET /  ->  12638 bytes, serves the SvelteKit UI (app/svelte: true)
```

`khanary-server` now: (a) has the KHANARY `ggml-xcfe` backend (`--list-devices` lists XCFE,
`proof/khanary_server_v1`), and (b) serves the branded KHANARY Web UI.

## Honest scope

- **IS:** a branded `khanary-server` that serves the KHANARY-named, ASX-teal SvelteKit Web UI with the
  XCFE backend compiled in. The rebrand-lock (recompile embeds the new UI) works as designed.
- **IS NOT (yet):** rich KHANARY *feature* pages (interactive Train / Models / Grammar routes) inside
  the SvelteKit app — that is a larger frontend build. The **grammar/dev-sandbox pages already exist**
  as the separate static Option-B app (`sandbox/`, `tools/build_grammar_pages.py`), hostable by the
  PRIMEOS WebView2 shell; folding those in as SvelteKit routes is the next UI iteration.
- The embedded UI is uncompressed (gzip not on PATH at build; a warning, not an error).

## Reproduce
```
powershell -File tools/build_khanary_ui.ps1       # branded UI -> khanary-ui-build/dist
powershell -File tools/reseal_khanary_server.ps1  # embed it into khanary-server + re-bundle
dist/khanary-server/khanary-server.exe -m <model.gguf> --port 8913   # serves the branded UI at /
```
