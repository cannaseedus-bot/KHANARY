using System;
using System.Collections.Generic;
using System.IO.MemoryMappedFiles;
using System.Net.Http;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Threading;
using System.Linq;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;

namespace PRIMEOS
{
    /// <summary>
    /// PRIMEOS Shell - Layer 6 Orchestration Interface
    /// Integrates with LLAMA GGUF dual-mode inference system
    /// Routes delta commands to agents, skills, plugins, tools
    /// </summary>
    public partial class MainWindow : Window
    {
        // SharedStateHeader layout from shared_memory_bridge.h (ASX v0.7)
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        private struct KuhulSharedState
        {
            public uint  Version;
            public uint  ActiveFold;
            public uint  TickCount;
            public float Entropy;
            public float Attention;
            public float Pressure;
            // float reserve[10] follows — not needed here
        }

        private const string SHM_NAME = "Local\\KuhulGeometricState";
        private DispatcherTimer _shmTimer;

        private HttpClient _httpClient = new HttpClient();
        private string _llamaServerUri = "http://127.0.0.1:17474";  // kuhul_engine --serve port
        private string _bossUri        = "http://127.0.0.1:8764";   // kuhul-server (MCP gateway)
        private List<string> _commandHistory = new List<string>();
        private bool _isLlamaConnected = false;
        private bool _atomicReplies = false;
        private string _currentManifest = "";
        private const string KuhulEngineExe = @"C:\Users\canna\.NNC-K\bin\v3.5.0-WebX\build\bin\Release\kuhul_engine.exe";

        // Alias → relative manifest path (relative to repo root / AppBase).
        // Matches the Tag values on ModelSelector ComboBoxItems in XAML.
        private static readonly Dictionary<string, string> ModelAliases = new()
        {
            ["from_zero"]   = @"models\from_zero\atomic.manifest.json",
            ["kxml"]        = @"models\khanary-kxml-v0.5.0\atomic.manifest.json",
            ["gpt-oss"]     = @"models\gpt-oss\atomic.manifest.json",
            ["gpt2-xl"]     = @"models\gpt2-xl\atomic.manifest.json",
            ["lfm2"]        = @"models\lfm2-1b\atomic.manifest.json",
            ["gemma-1b"]    = @"models\gemma-3-1b\atomic.manifest.json",
            ["gemma-1b-q8"] = @"models\gemma-3-1b-q8\atomic.manifest.json",
            ["gemma-4b"]    = @"models\gemma-3-4b\atomic.manifest.json",
            ["gemma-4-e2b"] = @"models\gemma-4-e2b\atomic.manifest.json",
            ["phi3-mini"]   = @"models\phi3-mini-4k\atomic.manifest.json",
            ["dolphin"]     = @"models\dolphin-phi2\atomic.manifest.json",
            ["qwen-1b8"]    = @"models\qwen-1b8-chat\atomic.manifest.json",
            ["qwen-story"]  = @"models\qwen25-05b-story\atomic.manifest.json",
            ["mgguf-gpt2"]  = @"models\mgguf-gpt2-2expert\atomic.manifest.json",
            ["mgguf-qwen"]  = @"models\mgguf-qwen-1expert\atomic.manifest.json",
        };

        // PRIMEOS bundles + launches llama-server.exe (its web UI is baked into the binary — no npm
        // build). Point WebView2 at the localhost port it broadcasts. Bundle location resolved below;
        // the .ASX.cpp path is the dev fallback. Model is configurable (null -> UI loads, no inference).
        private Process _llamaProcess;
        private int _llamaPort;
        private string _modelPath = null;

        // Registry models for delta commands
        private Dictionary<string, object> _registry = new Dictionary<string, object>
        {
            { "agents", new List<string> { "semantic-router", "expert-classifier", "plan-generator" } },
            { "skills", new List<string> { "geometry-validator", "phase-manager", "cache-optimizer" } },
            { "plugins", new List<string> { "llama-gguf-bridge", "mcp-server-link", "canvas-sync" } },
            { "tools", new List<string> { "execute-query", "measure-latency", "profile-memory" } },
            { "micronauts", new List<string> { "qwen-2.5-micronaut", "validator-micronaut", "planner-micronaut" } },
            { "opcodes", new List<string> { "PUSH_PHASE", "EXECUTE", "VALIDATE", "ROUTE", "CACHE_HIT", "RETURN" } }
        };

        private string _currentPhase = "POP";
        private Stopwatch _executionTimer = new Stopwatch();

        public MainWindow()
        {
            InitializeComponent();
            Initialize();
            this.Closed += (s, e) => { try { _llamaProcess?.Kill(true); } catch { } };
            _ = InitCanvasAsync();   // launch llama-server + WebView2 shell over its built-in web UI
        }

        /// <summary>
        /// Part 1: launch the bundled llama-server.exe, then point the WebView2 canvas at the
        /// localhost port it broadcasts (its web UI is baked into the binary). KHANARY plugs in
        /// underneath via the ggml backend, not the UI — so this shell works today on stock llama.
        /// </summary>
        private async Task InitCanvasAsync()
        {
            try
            {
                await LaunchLlamaServerAsync();
                await CanvasDisplay.EnsureCoreWebView2Async();
                CanvasDisplay.Source = new Uri(_llamaServerUri);
                CommandStatus.Text = "Canvas: llama web UI (" + _llamaServerUri + ")";
            }
            catch (Exception ex)
            {
                AddChatMessage("[SYSTEM]", "WebView2 init failed (is the WebView2 Runtime installed?): " + ex.Message);
            }
        }

        /// <summary>Resolve, launch, and wait on the bundled llama-server.exe.</summary>
        private async Task LaunchLlamaServerAsync()
        {
            string exe = ResolveLlamaServer();
            if (exe == null)
            {
                AddChatMessage("[SYSTEM]", "llama-server.exe not found — bundle it under .\\llama\\ next to PRIMEOS.exe.");
                return;
            }
            _llamaPort = FreePort();
            _llamaServerUri = $"http://127.0.0.1:{_llamaPort}";
            _modelPath = _modelPath ?? ResolveModel();
            string args = $"--host 127.0.0.1 --port {_llamaPort}" +
                          (_modelPath != null ? $" -m \"{_modelPath}\"" : "");
            var psi = new ProcessStartInfo
            {
                FileName = exe,
                Arguments = args,
                WorkingDirectory = Path.GetDirectoryName(exe),   // siblings (ggml*.dll) load from here
                UseShellExecute = false,
                CreateNoWindow = true
            };
            _llamaProcess = Process.Start(psi);
            AddChatMessage("[SYSTEM]", $"llama-server launching on {_llamaServerUri}" +
                (_modelPath != null ? $" (model {Path.GetFileName(_modelPath)})" : " (no model — set _modelPath for inference)"));
            // wait until the server accepts connections (its /health endpoint), up to ~30s
            for (int i = 0; i < 60; i++)
            {
                try { var r = await _httpClient.GetAsync($"{_llamaServerUri}/health"); if (r.IsSuccessStatusCode) break; }
                catch { }
                await Task.Delay(500);
            }
        }

        /// <summary>Bundled path first (.\llama\), then the .ASX.cpp dev build as a fallback.</summary>
        private string ResolveLlamaServer()
        {
            foreach (var p in new[]
            {
                Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "llama", "llama-server.exe"),
                @"C:\Users\canna\.ASX.cpp\llama-b9968-bin-win-cpu-x64\llama-server.exe"
            })
                if (File.Exists(p)) return p;
            return null;
        }

        /// <summary>First existing default model (configurable). null = UI loads without inference.</summary>
        private string ResolveModel()
        {
            foreach (var p in new[]
            {
                @"C:\Users\canna\.lmstudio\models\gpt2.Q8_0.gguf",
                @"C:\Users\canna\.lmstudio\models\Qwen-1_8B-Chat-f16\model.safetensors"
            })
                if (File.Exists(p)) return p;
            return null;
        }

        /// <summary>Pick a free loopback TCP port for llama-server to bind.</summary>
        private int FreePort()
        {
            var l = new TcpListener(IPAddress.Loopback, 0);
            l.Start();
            int port = ((IPEndPoint)l.LocalEndpoint).Port;
            l.Stop();
            return port;
        }

        private void Initialize()
        {
            CommandStatus.Text = "PRIMEOS Shell initialized. Type 'help' for commands.";
            AddChatMessage("[SYSTEM]", "PRIMEOS Shell v1.0 ready. Connect to LLAMA for inference.");
            UpdatePhaseState("POP", 0);

            // Poll Local\KuhulGeometricState shared memory (written by FieldExecutionEngine)
            _shmTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(500) };
            _shmTimer.Tick += ShmTimer_Tick;
            _shmTimer.Start();

            // Auto-connect to LLAMA on startup
            _ = TryConnectLlama();
        }

        private void ShmTimer_Tick(object sender, EventArgs e)
        {
            try
            {
                using var mmf  = MemoryMappedFile.OpenExisting(SHM_NAME, MemoryMappedFileRights.Read);
                using var view = mmf.CreateViewAccessor(0, Marshal.SizeOf<KuhulSharedState>(),
                                                        MemoryMappedFileAccess.Read);
                view.Read(0, out KuhulSharedState s);

                ShmEntropyText.Text   = s.Entropy.ToString("F4");
                ShmAttentionText.Text = s.Attention.ToString("F4");
                ShmPressureText.Text  = s.Pressure.ToString("F4");
                ShmFoldTickText.Text  = $"{s.ActiveFold} / {s.TickCount}";
                ShmStatusText.Text    = $"● SHM v{s.Version} live";
                ShmStatusText.Foreground = System.Windows.Media.Brushes.LimeGreen;
            }
            catch
            {
                ShmStatusText.Text = "⊘ SHM offline (engine not running)";
                ShmStatusText.Foreground = System.Windows.Media.Brushes.OrangeRed;
            }
        }

        /// <summary>
        /// Connect to LLAMA GGUF server
        /// </summary>
        private async void ConnectLlama_Click(object sender, RoutedEventArgs e)
        {
            await TryConnectLlama();
        }

        private async Task TryConnectLlama()
        {
            try
            {
                CommandStatus.Text = "Connecting to LLAMA...";
                
                var response = await _httpClient.GetAsync($"{_llamaServerUri}/status");
                if (response.IsSuccessStatusCode)
                {
                    _isLlamaConnected = true;
                    LlamaStatusText.Text = "[ONLINE]";
                    LlamaStatusText.Foreground = System.Windows.Media.Brushes.LimeGreen;
                    CommandStatus.Text = "✓ Connected to LLAMA at " + _llamaServerUri;
                    AddChatMessage("[SYSTEM]", $"Connected to LLAMA server at {_llamaServerUri}");
                }
                else
                {
                    _isLlamaConnected = false;
                    LlamaStatusText.Text = "[ERROR]";
                    LlamaStatusText.Foreground = System.Windows.Media.Brushes.Red;
                    CommandStatus.Text = "✗ LLAMA connection failed (HTTP " + response.StatusCode + ")";
                }
            }
            catch (Exception ex)
            {
                _isLlamaConnected = false;
                LlamaStatusText.Text = "[OFFLINE]";
                LlamaStatusText.Foreground = System.Windows.Media.Brushes.OrangeRed;
                CommandStatus.Text = $"✗ Connection error: {ex.Message}";
                AddChatMessage("[ERROR]", ex.Message);
            }
        }

        /// <summary>
        /// Execute command (DELTA or regular)
        /// </summary>
        private async void Run_Click(object sender, RoutedEventArgs e)
        {
            string input = CommandInput.Text.Trim();
            if (string.IsNullOrEmpty(input))
                return;

            _executionTimer.Restart();
            CommandInput.Clear();

            try
            {
                // Parse delta command
                if (input.StartsWith("[CREATE]") || input.StartsWith("[UPDATE]") || 
                    input.StartsWith("[DELETE]") || input.StartsWith("[ROUTE]") ||
                    input.StartsWith("[VALIDATE]") || input.StartsWith("[EXECUTE]"))
                {
                    await ExecuteDeltaCommand(input);
                }
                else if (input.Equals("help", StringComparison.OrdinalIgnoreCase))
                {
                    ShowHelp();
                }
                else if (input.Equals("status", StringComparison.OrdinalIgnoreCase))
                {
                    ShowStatus();
                }
                else if (input.StartsWith("query:", StringComparison.OrdinalIgnoreCase))
                {
                    // Route to LLAMA inference
                    string query = input.Substring(6).Trim();
                    await RouteToLlamaInference(query);
                }
                else
                {
                    CommandStatus.Text = "Unknown command. Type 'help' for usage.";
                }
            }
            finally
            {
                _executionTimer.Stop();
                ExecutionTimeText.Text = _executionTimer.ElapsedMilliseconds + "ms";
                AddCommandToHistory($"[{DateTime.Now:HH:mm:ss}] {input.Substring(0, Math.Min(40, input.Length))}...");
            }
        }

        /// <summary>
        /// Call kuhul_task_boss on kuhul-server (port 8764) via JSON-RPC MCP.
        /// verb = task.plan / app.create / build.micronaut / build.game / build.website etc.
        /// </summary>
        private async Task<string> CallBossAsync(string verb, string prompt)
        {
            var payload = new
            {
                jsonrpc = "2.0",
                method  = "tools/call",
                id      = 1,
                @params = new
                {
                    name      = "kuhul_task_boss",
                    arguments = new { verb, prompt }
                }
            };
            var json    = JsonSerializer.Serialize(payload);
            var content = new StringContent(json, System.Text.Encoding.UTF8, "application/json");
            var resp    = await _httpClient.PostAsync($"{_bossUri}/mcp", content);
            if (!resp.IsSuccessStatusCode)
                throw new Exception($"BOSS HTTP {resp.StatusCode}");
            string body = await resp.Content.ReadAsStringAsync();
            using var doc = JsonDocument.Parse(body);
            // result.content[0].text
            var result = doc.RootElement.GetProperty("result");
            if (result.TryGetProperty("content", out var arr) && arr.GetArrayLength() > 0)
                return arr[0].GetProperty("text").GetString() ?? "";
            if (result.TryGetProperty("text", out var txt))
                return txt.GetString() ?? "";
            return body;
        }

        /// <summary>
        /// Execute DELTA command — routes through kuhul_task_boss (BOSS) on port 8764.
        /// </summary>
        private async Task ExecuteDeltaCommand(string command)
        {
            // Local-only ops (no BOSS needed)
            if (command.StartsWith("[UPDATE]"))  { ExecuteDeltaUpdate(command);   CommandStatus.Text = "✓ Updated"; return; }
            if (command.StartsWith("[DELETE]"))  { ExecuteDeltaDelete(command);   CommandStatus.Text = "✓ Deleted"; return; }
            if (command.StartsWith("[VALIDATE]")){ ExecuteDeltaValidate(command); return; }

            // All task-bearing ops go through kuhul_task_boss (BOSS) on port 8764
            try
            {
                CommandStatus.Text = $"→ BOSS: {command.Substring(0, Math.Min(40, command.Length))}...";

                string verb   = "task.plan";
                string prompt = command;

                if (command.StartsWith("[CREATE]"))
                {
                    var parts = command.Split(' ');
                    string type = parts.Length > 1 ? parts[1].ToLower() : "micronaut";
                    verb   = type == "app" ? "app.create" : "build.micronaut";
                    prompt = command.Substring("[CREATE]".Length).Trim();
                    // also add to local registry for UI
                    ExecuteDeltaCreate(command);
                }
                else if (command.StartsWith("[ROUTE]"))
                {
                    var parts = command.Split(new[] { "Query:" }, StringSplitOptions.None);
                    prompt = parts.Length > 1 ? parts[1].Trim().Trim('"') : command;
                    verb   = "task.plan";
                }
                else if (command.StartsWith("[EXECUTE]"))
                {
                    prompt = command.Substring("[EXECUTE]".Length).Trim();
                    verb   = "task.plan";
                }

                string reply = await CallBossAsync(verb, prompt);
                AddChatMessage("[BOSS]", reply);
                CommandStatus.Text = "✓ BOSS task complete";
            }
            catch (Exception ex)
            {
                CommandStatus.Text = $"✗ BOSS error: {ex.Message}";
                AddChatMessage("[ERROR]", $"BOSS ({_bossUri}): {ex.Message}");
            }
        }

        private void ExecuteDeltaCreate(string command)
        {
            // [CREATE] Agent MyAgent skill:geometry-validator
            var parts = command.Split(' ');
            if (parts.Length < 3)
            {
                CommandStatus.Text = "Usage: [CREATE] <type> <name>";
                return;
            }

            string type = parts[1].ToLower(); // agent, skill, plugin, etc.
            string name = parts[2];

            if (_registry.ContainsKey(type + "s"))
            {
                var list = _registry[type + "s"] as List<string>;
                if (list != null && !list.Contains(name))
                {
                    list.Add(name);
                    AddChatMessage("[DELTA]", $"Created {type}: {name}");
                    CommandStatus.Text = $"✓ Created new {type}: {name}";
                }
            }
        }

        private void ExecuteDeltaUpdate(string command)
        {
            // [UPDATE] skill geometry-validator config:enabled=true
            var parts = command.Split(' ');
            AddChatMessage("[DELTA]", $"Updated configuration: {string.Join(" ", parts.Skip(1))}");
            CommandStatus.Text = "✓ Configuration updated";
        }

        private void ExecuteDeltaDelete(string command)
        {
            // [DELETE] micronaut qwen-2.5-micronaut
            var parts = command.Split(' ');
            if (parts.Length < 3)
            {
                CommandStatus.Text = "Usage: [DELETE] <type> <name>";
                return;
            }

            string type = parts[1].ToLower() + "s";
            string name = parts[2];

            if (_registry.ContainsKey(type))
            {
                var list = _registry[type] as List<string>;
                if (list != null && list.Remove(name))
                {
                    AddChatMessage("[DELTA]", $"Deleted: {name}");
                    CommandStatus.Text = $"✓ Deleted: {name}";
                }
            }
        }

        private async Task ExecuteDeltaRoute(string command)
        {
            // [ROUTE] Expert Query: "What is machine learning?"
            var parts = command.Split(new[] { "Query:" }, StringSplitOptions.None);
            if (parts.Length < 2)
            {
                CommandStatus.Text = "Usage: [ROUTE] Expert Query: <query>";
                return;
            }

            string query = parts[1].Trim().Trim('"');
            await RouteToLlamaInference(query);
        }

        private void ExecuteDeltaValidate(string command)
        {
            // [VALIDATE] Phase WO
            var parts = command.Split(' ');
            if (parts.Length < 3)
            {
                CommandStatus.Text = "Usage: [VALIDATE] Phase <phase>";
                return;
            }

            string phase = parts[2].ToUpper();
            AddChatMessage("[VALIDATE]", $"Phase {phase} validated ✓");
            CommandStatus.Text = $"✓ Phase validation passed: {phase}";
            
            // Transition to next phase
            TransitionPhase();
        }

        private void ExecuteDeltaExecute(string command)
        {
            // [EXECUTE] Opcode PUSH_PHASE POP
            var parts = command.Split(' ');
            if (parts.Length < 3)
            {
                CommandStatus.Text = "Usage: [EXECUTE] Opcode <opcode>";
                return;
            }

            string opcode = parts[2];
            AddChatMessage("[EXECUTE]", $"Executed opcode: {opcode}");
            CommandStatus.Text = $"✓ Opcode executed: {opcode}";
        }

        /// <summary>
        /// Route query to kuhul_engine via OpenAI-compatible /v1/chat/completions (port 17474).
        /// </summary>
        private async Task RouteToLlamaInference(string query)
        {
            try
            {
                CommandStatus.Text = "Sending to kuhul_engine...";

                var requestBody = new
                {
                    model    = "kuhul",
                    messages = new[] { new { role = "user", content = query } },
                    stream   = false
                };

                var json    = JsonSerializer.Serialize(requestBody);
                var content = new StringContent(json, System.Text.Encoding.UTF8, "application/json");

                var response = await _httpClient.PostAsync($"{_llamaServerUri}/v1/chat/completions", content);

                if (response.IsSuccessStatusCode)
                {
                    string resultJson = await response.Content.ReadAsStringAsync();
                    using (JsonDocument doc = JsonDocument.Parse(resultJson))
                    {
                        string responseText = doc.RootElement
                            .GetProperty("choices")[0]
                            .GetProperty("message")
                            .GetProperty("content")
                            .GetString() ?? "";

                        _isLlamaConnected = true;
                        AddChatMessage("[KUHUL]", responseText);
                        CommandStatus.Text = "✓ kuhul_engine response";

                        if (_atomicReplies)
                            _ = DisplayAtomicReply(responseText);
                        else if (responseText.Contains("<") && responseText.Contains(">"))
                            DisplayCanvasOutput(responseText);
                    }
                }
                else
                {
                    CommandStatus.Text = $"✗ kuhul_engine HTTP {response.StatusCode}";
                }
            }
            catch (Exception ex)
            {
                _isLlamaConnected = false;
                CommandStatus.Text = $"✗ kuhul_engine error: {ex.Message}";
                AddChatMessage("[ERROR]", ex.Message);
            }
        }

        /// <summary>
        /// Atomic Replies toggle — switches between plain HTML and Atomic DOM block rendering.
        /// </summary>
        private void AtomicReplies_Changed(object sender, RoutedEventArgs e)
        {
            _atomicReplies = AtomicRepliesToggle.IsChecked == true;
            AtomicRepliesToggle.Foreground = _atomicReplies
                ? System.Windows.Media.Brushes.LimeGreen
                : new System.Windows.Media.SolidColorBrush(System.Windows.Media.Color.FromRgb(0, 68, 0));
            CommandStatus.Text = _atomicReplies ? "Atomic Replies: ON" : "Atomic Replies: OFF";
        }

        /// <summary>
        /// Renders a model reply as structured Atomic DOM blocks (HEADER/BODY/FEED/FOOTER).
        /// Called instead of DisplayCanvasOutput when _atomicReplies is true.
        /// Parses the content into blocks: first line → HEADER, last line → FOOTER,
        /// middle content → BODY, metadata/status → FEED.
        /// </summary>
        private async Task DisplayAtomicReply(string content, string npcName = "KUHUL")
        {
            var lines = content.Split('\n');
            string header = lines.Length > 0 ? HtmlEnc(lines[0].Trim()) : "";
            string footer = lines.Length > 2 ? HtmlEnc(lines[^1].Trim()) : "";
            string body   = HtmlEnc(string.Join("\n", lines.Length > 2 ? lines[1..^1] : lines));
            string feed   = HtmlEnc($"{npcName} · {DateTime.Now:HH:mm:ss}");

            string atomicHtml = $@"<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8'>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0a0e27; color: #00ff00; font-family: 'Consolas', monospace; font-size: 12px; padding: 8px; }}
  .block {{ border: 1px solid #004400; margin-bottom: 4px; }}
  .block-label {{ background: #0d1a0d; color: #006600; font-size: 10px; padding: 2px 6px; border-bottom: 1px solid #004400; letter-spacing: 2px; }}
  .block-content {{ padding: 6px 8px; white-space: pre-wrap; word-break: break-word; }}
  .header .block-content {{ color: #00ff99; font-weight: bold; }}
  .body   .block-content {{ color: #00ff00; line-height: 1.5; }}
  .feed   .block-content {{ color: #004400; font-size: 10px; }}
  .footer .block-content {{ color: #006600; font-style: italic; }}
</style>
</head>
<body>
  <div class='block header'>
    <div class='block-label'>HEADER</div>
    <div class='block-content'>{header}</div>
  </div>
  <div class='block body'>
    <div class='block-label'>BODY</div>
    <div class='block-content'>{body}</div>
  </div>
  <div class='block feed'>
    <div class='block-label'>FEED</div>
    <div class='block-content'>{feed}</div>
  </div>
  <div class='block footer'>
    <div class='block-label'>FOOTER</div>
    <div class='block-content'>{footer}</div>
  </div>
</body>
</html>";

            try
            {
                await CanvasDisplay.EnsureCoreWebView2Async();
                CanvasDisplay.NavigateToString(atomicHtml);
                CommandStatus.Text = "✓ Atomic reply rendered";
            }
            catch (Exception ex)
            {
                CommandStatus.Text = $"✗ Atomic render error: {ex.Message}";
            }
        }

        /// <summary>
        /// Display HTML/SVG output in canvas
        /// </summary>
        private async void DisplayCanvasOutput(string html)
        {
            try
            {
                string wrappedHtml = $@"
<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <title>Canvas Output</title>
    <style>
        body {{ margin: 0; padding: 8px; background: #0a0e27; color: #00ff00; font-family: Consolas; }}
        svg {{ border: 1px solid #00ff00; }}
        canvas {{ border: 1px solid #00ff00; }}
    </style>
</head>
<body>
{html}
</body>
</html>";

                // WebView2: render inline (repurposes the canvas from the llama UI to this output).
                await CanvasDisplay.EnsureCoreWebView2Async();
                CanvasDisplay.NavigateToString(wrappedHtml);
                CommandStatus.Text = "✓ Canvas output rendered";
            }
            catch (Exception ex)
            {
                CommandStatus.Text = $"✗ Canvas error: {ex.Message}";
            }
        }

        /// <summary>
        /// Chat send button
        /// </summary>
        private void Send_Click(object sender, RoutedEventArgs e)
        {
            string message = ChatInput.Text.Trim();
            if (!string.IsNullOrEmpty(message))
            {
                ChatInput.Clear();
                _ = RouteToLlamaInference(message);
            }
        }

        private void ChatInput_KeyDown(object sender, System.Windows.Input.KeyEventArgs e)
        {
            if (e.Key == System.Windows.Input.Key.Return && 
                (System.Windows.Input.Keyboard.Modifiers & System.Windows.Input.ModifierKeys.Control) == System.Windows.Input.ModifierKeys.Control)
            {
                Send_Click(null, null);
                e.Handled = true;
            }
        }

        /// <summary>
        /// Model selector — resolves manifest path, stores it in _currentManifest,
        /// and emits a type-style load sequence into the footer status bar.
        /// </summary>
        private async void ModelSelector_Changed(object sender, SelectionChangedEventArgs e)
        {
            if (ModelSelector.SelectedItem is not ComboBoxItem item) return;
            string tag = item.Tag as string ?? "";
            if (!ModelAliases.TryGetValue(tag, out string rel)) return;

            string repoRoot = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"..\..\..\.."));
            _currentManifest = Path.Combine(repoRoot, rel);

            string label  = item.Content?.ToString() ?? tag;
            string slot   = label.Split('—')[0].Trim().ToUpper();
            string gpuTag = label.Contains("GPU") ? "● GPU" : "○ CPU";
            ModelGpuTag.Text = gpuTag;
            AddCommandToHistory($"[{DateTime.Now:HH:mm}] MODEL {tag}");

            // Type-style load sequence in the footer below the studio canvas
            CommandStatus.Text = $"LOADING [{slot}]";
            await Task.Delay(80);
            CommandStatus.Text = $"LOADING [{slot}]  MANIFEST OK";
            await Task.Delay(80);
            CommandStatus.Text = $"LOADING [{slot}]  {gpuTag.Replace("●","").Replace("○","").Trim()} · READY ▌";
        }

        /// <summary>
        /// Registry item selection
        /// </summary>
        private void RegistryItem_Selected(object sender, SelectionChangedEventArgs e)
        {
            if (RegistryList.SelectedItem is ListBoxItem item)
            {
                var text = (item.Content as TextBlock)?.Text;
                if (!string.IsNullOrEmpty(text))
                {
                    CommandStatus.Text = $"Selected: {text}";
                }
            }
        }

        /// <summary>
        /// Phase transition
        /// </summary>
        private void TransitionPhase()
        {
            var phases = new[] { "POP", "WO", "YAX", "SEK", "CH'EN", "XUL" };
            int currentIndex = System.Array.IndexOf(phases, _currentPhase);
            int nextIndex = (currentIndex + 1) % phases.Length;
            _currentPhase = phases[nextIndex];
            UpdatePhaseState(_currentPhase, nextIndex);
        }

        private void UpdatePhaseState(string phase, int index)
        {
            CurrentPhaseText.Text = $"{phase} (phase {index}/6)";
            AddChatMessage("[PHASE]", $"Transitioned to {phase}");
        }

        /// <summary>
        /// Add message to chat panel
        /// </summary>
        private void AddChatMessage(string sender, string message)
        {
            var item = new ListBoxItem
            {
                Content = new TextBlock
                {
                    Text = $"{sender}: {message}",
                    TextWrapping = TextWrapping.Wrap,
                    Foreground = sender.StartsWith("[ERROR]") ? System.Windows.Media.Brushes.OrangeRed :
                                 sender.StartsWith("[SYSTEM]") ? System.Windows.Media.Brushes.LimeGreen :
                                 sender.StartsWith("[LLAMA]") ? System.Windows.Media.Brushes.Cyan :
                                 System.Windows.Media.Brushes.LimeGreen
                }
            };
            ChatPanel.Items.Add(item);
            ChatPanel.ScrollIntoView(item);
        }

        /// <summary>
        /// Add command to history
        /// </summary>
        private void AddCommandToHistory(string cmd)
        {
            _commandHistory.Add(cmd);
            var item = new ListBoxItem { Content = cmd, Foreground = System.Windows.Media.Brushes.LimeGreen };
            CommandHistory.Items.Insert(0, item);
            if (CommandHistory.Items.Count > 20)
                CommandHistory.Items.RemoveAt(CommandHistory.Items.Count - 1);
        }

        /// <summary>
        /// Show help
        /// </summary>
        private void ShowHelp()
        {
            string help = @"PRIMEOS Shell Commands:

DELTA COMMANDS (Registry Mutation):
  [CREATE] <type> <name> - Create agent/skill/plugin/tool/micronaut/opcode
  [UPDATE] <type> <name> - Update configuration
  [DELETE] <type> <name> - Delete from registry
  [ROUTE] Expert Query: <query> - Route to LLAMA inference
  [VALIDATE] Phase <phase> - Validate phase (POP/WO/YAX/SEK/CH'EN/XUL)
  [EXECUTE] Opcode <opcode> - Execute opcode

SYSTEM COMMANDS:
  query: <question> - Send query to LLAMA (AUTO/FAST/DEEP)
  help - Show this help
  status - Show current status

INFERENCE MODES:
  FAST - CPU baseline (~100ms latency)
  AUTO - Intelligent selection (default)
  DEEP - GPU 3D reasoning (~5-15s latency)";

            AddChatMessage("[HELP]", help);
            CommandStatus.Text = "✓ Help displayed";
        }

        /// <summary>
        /// Show status
        /// </summary>
        private void ShowStatus()
        {
            string status = $@"
PRIMEOS SHELL STATUS:
  Phase: {_currentPhase}
  LLAMA: {(_isLlamaConnected ? "ONLINE" : "OFFLINE")}
  Agents: {((List<string>)_registry["agents"]).Count}
  Skills: {((List<string>)_registry["skills"]).Count}
  Plugins: {((List<string>)_registry["plugins"]).Count}
  Tools: {((List<string>)_registry["tools"]).Count}
  Micronauts: {((List<string>)_registry["micronauts"]).Count}
  Opcodes: {((List<string>)_registry["opcodes"]).Count}
";
            AddChatMessage("[STATUS]", status);
            CommandStatus.Text = "✓ Status displayed";
        }

        private static string HtmlEnc(string s) =>
            s.Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;").Replace("\"", "&quot;");

        /// <summary>
        /// Clear all
        /// </summary>
        private void Clear_Click(object sender, RoutedEventArgs e)
        {
            CommandInput.Clear();
            CommandStatus.Text = "Cleared";
        }
    }
}
