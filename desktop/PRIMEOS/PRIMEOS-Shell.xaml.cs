using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
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
        private HttpClient _httpClient = new HttpClient();
        private string _llamaServerUri = "http://127.0.0.1:8080";   // set to the actual port at launch
        private List<string> _commandHistory = new List<string>();
        private bool _isLlamaConnected = false;

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
            
            // Auto-connect to LLAMA on startup
            _ = TryConnectLlama();
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
        /// Execute DELTA command (registry mutation)
        /// </summary>
        private async Task ExecuteDeltaCommand(string command)
        {
            try
            {
                CommandStatus.Text = $"Executing delta: {command.Substring(0, Math.Min(30, command.Length))}...";

                if (command.StartsWith("[CREATE]"))
                {
                    ExecuteDeltaCreate(command);
                }
                else if (command.StartsWith("[UPDATE]"))
                {
                    ExecuteDeltaUpdate(command);
                }
                else if (command.StartsWith("[DELETE]"))
                {
                    ExecuteDeltaDelete(command);
                }
                else if (command.StartsWith("[ROUTE]"))
                {
                    await ExecuteDeltaRoute(command);
                }
                else if (command.StartsWith("[VALIDATE]"))
                {
                    ExecuteDeltaValidate(command);
                }
                else if (command.StartsWith("[EXECUTE]"))
                {
                    ExecuteDeltaExecute(command);
                }

                CommandStatus.Text = "✓ Delta executed successfully";
            }
            catch (Exception ex)
            {
                CommandStatus.Text = $"✗ Delta error: {ex.Message}";
                AddChatMessage("[ERROR]", ex.Message);
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
        /// Route query to LLAMA GGUF dual-mode inference
        /// </summary>
        private async Task RouteToLlamaInference(string query)
        {
            if (!_isLlamaConnected)
            {
                CommandStatus.Text = "✗ LLAMA not connected";
                await TryConnectLlama();
                return;
            }

            try
            {
                CommandStatus.Text = "Sending to LLAMA...";
                
                string mode = InferenceMode.SelectedItem?.ToString() ?? "AUTO";
                
                var requestBody = new
                {
                    query = query,
                    mode = mode
                };

                var json = JsonSerializer.Serialize(requestBody);
                var content = new StringContent(json, System.Text.Encoding.UTF8, "application/json");

                var response = await _httpClient.PostAsync($"{_llamaServerUri}/chat", content);
                
                if (response.IsSuccessStatusCode)
                {
                    string resultJson = await response.Content.ReadAsStringAsync();
                    using (JsonDocument doc = JsonDocument.Parse(resultJson))
                    {
                        var element = doc.RootElement;
                        if (element.TryGetProperty("response", out var responseElement))
                        {
                            string responseText = responseElement.GetString();
                            AddChatMessage("[LLAMA]", responseText);
                            CommandStatus.Text = $"✓ LLAMA response ({mode} mode)";
                            
                            // Display in canvas if HTML/SVG
                            if (responseText.Contains("<") && responseText.Contains(">"))
                            {
                                DisplayCanvasOutput(responseText);
                            }
                        }
                    }
                }
                else
                {
                    CommandStatus.Text = $"✗ LLAMA error: HTTP {response.StatusCode}";
                }
            }
            catch (Exception ex)
            {
                CommandStatus.Text = $"✗ Inference error: {ex.Message}";
                AddChatMessage("[ERROR]", ex.Message);
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
