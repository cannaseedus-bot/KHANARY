# PRIMEOS Shell (desktop app)

The WPF chat/terminal desktop front-end for the ASX/KHANARY stack — connects to a local LLAMA
inference server (`http://localhost:8888`) and shows a live connection/registry/command UI. This is
the desktop UI tier (parallel to the `www/` web UI and the `MICRONAUT.ps1` terminal).

## Why this copy exists

The original `.ASX.cpp` build (`bin/Debug/net6.0-windows/PRIMEOS.exe`) **did not launch**. Three
latent bugs were masked because the .NET 6 runtime isn't installed (only .NET 8/9/10 are):

1. **Runtime**: targeted `net6.0-windows`; no .NET 6 runtime present → "You must install .NET".
   Fixed by retargeting **`net8.0-windows`** (installed).
2. **Startup**: `App.xaml` had `StartupUri="MainWindow.xaml"`, but the window file is
   `PRIMEOS-Shell.xaml` (`x:Class="PRIMEOS.MainWindow"`) → `IOException: Cannot locate resource
   'mainwindow.xaml'`. Fixed the `StartupUri`.
3. **XAML**: `PRIMEOS-Shell.xaml` line 190 had `Margin="auto,0,4,0"` — `auto` is not a valid
   `Thickness` → `XamlParseException`. Fixed to `Margin="0,0,4,0"`.

The shell is self-contained (only `System.*` + `HttpClient` + `System.Text.Json`; it does **not**
use MCP, so the `ModelContextProtocol` package ref was dropped).

## Build & run

```
dotnet build -c Release
bin/Release/net8.0-windows/PRIMEOS.exe
```

Launches the chat shell; connect it to a running LLAMA server at `localhost:8888` for inference.

## WebView2 shell (Part 1 of the KHANARY desktop architecture)

PRIMEOS now **launches `llama-server.exe` itself** and hosts its web UI in a **WebView2** control
(`Microsoft.Web.WebView2`, `1.0.*`) — replacing the legacy IE `WebBrowser`. On startup
`LaunchLlamaServerAsync()`:

1. resolves `llama-server.exe` — bundled under `.\llama\` next to `PRIMEOS.exe`, else the
   `.ASX.cpp\llama-b9968-bin-win-cpu-x64\` dev build (that folder is **the whole runtime**:
   `llama-server.exe` + `llama-server-impl.dll` + the `ggml*.dll` siblings, ~465 MB, the free
   non-enterprise build);
2. picks a **free loopback port** and starts the server (`--host 127.0.0.1 --port <free>`,
   `-m <model>` if one is configured), working dir set so the sibling DLLs load;
3. waits on the server's `/health`, then points `CanvasDisplay.Source` at `http://127.0.0.1:<port>`.

**No npm / SvelteKit build is needed** — llama-server's web UI is **baked into the binary**, so it
serves the UI at its root the moment it's up. The server process is killed when the window closes.
Model is configurable (`_modelPath`); with none set the UI still loads (no inference). Builds clean
on net8 (`dotnet build -c Release`, 0 warnings). Runtime needs the **WebView2 Runtime** (ships with
Edge on Win10/11); bundle the `llama\` binaries at package time (gitignored, not committed).

This is **Part 1** of the two-part plan: a real KHANARY desktop app **today** on stock
`ggml-cpu`/`ggml-opencl`. **Part 2** (later) swaps in a real `ggml-xcfe` backend *beneath* llama —
the UI never changes, KHANARY plugs in at the ggml layer. See the main README roadmap and
`docs/llama-ggml-bridges.md`.
