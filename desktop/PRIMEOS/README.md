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

The canvas is now a **WebView2** control (`Microsoft.Web.WebView2`, `Version="1.0.*"`) instead of the
legacy IE-based `WebBrowser` — so it can host a **modern (SvelteKit) web UI**. On startup
`InitCanvasAsync()` calls `EnsureCoreWebView2Async()` and points `CanvasDisplay.Source` at the
llama-server web UI (`http://localhost:8888`); model HTML/SVG output still renders via
`NavigateToString`. Builds clean on net8 (`dotnet build -c Release`, 0 warnings). Needs the
**WebView2 Runtime** (ships with Edge on Win10/11) and a running `llama-server` at `:8888` to display.

This is **Part 1** of the two-part plan: PRIMEOS is a thin WebView2 shell over llama's own UI,
running today on stock `ggml-cpu`/`ggml-opencl`. **Part 2** (later) swaps in a real `ggml-xcfe`
backend *beneath* llama — the UI never changes, KHANARY plugs in at the ggml layer. See the main
README roadmap and `docs/llama-ggml-bridges.md`.
