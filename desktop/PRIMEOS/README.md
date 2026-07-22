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
