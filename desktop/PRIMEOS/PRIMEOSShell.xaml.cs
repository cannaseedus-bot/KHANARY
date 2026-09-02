using System;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Threading;
using System.Diagnostics;
using System.Linq;

namespace PRIMEOS
{
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
        }

        private const string SHM_NAME = "Local\\KuhulGeometricState";
        private DispatcherTimer _shmTimer;
        private DispatcherTimer _statusTimer;

        private readonly HttpClient _httpClient = new HttpClient { Timeout = System.Threading.Timeout.InfiniteTimeSpan };
        // Chat endpoint — read from active-model.json written by START-SERVERS.bat.
        // Default: port 9000 (main llama-server lane). kuhul_engine runs on :17480.
        private string _chatEndpoint   = "http://127.0.0.1:8764/v1/chat/completions";  // MCP gateway — always up
        private string _engineEndpoint = "http://127.0.0.1:17480/v1/chat/completions";
        private string _bossUri        = "http://127.0.0.1:8764";
        private bool   _isKuhulOnline  = false;
        private bool   _isBossOnline   = false;
        private bool   _atomicReplies  = false;
        private string _currentAlias   = "gemma-1b";
        private string _currentPhase   = "POP";
        private int    _phaseIndex     = 0;

        private Process _llamaProcess = null;  // set if PRIMEOS manages its own server process

        // ── Auth ─────────────────────────────────────────────────────────────
        private const string GH_CLIENT_ID = "Ov23liVSfnLVNpTnMr2Y";
        private const string GH_SCOPE     = "user:email repo workflow";
        private string _ghToken = null;
        private string _ghLogin = null;
        private CancellationTokenSource _deviceFlowCts = null;

        private static readonly string _keyDir  = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".khanary");
        private static readonly string _keyFile = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".khanary", "primeos-access.key");

        private record GHUser(string Login, string Name, string Email, string AvatarUrl);

        private static readonly string RepoRoot = Path.GetFullPath(
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"..\..\..\..\..")
        );

        // alias → relative manifest path (from repo root)
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

        // known built-in system ops — ps1 or bat (id → repo-relative path)
        // User apps live in apps/{id}/app.ps1 and are discovered by ScanApps().
        private static readonly Dictionary<string, string> BuiltinApps = new()
        {
            ["start-servers"] = @"START-SERVERS.bat",
        };

        public MainWindow()
        {
            InitializeComponent();
            this.Closed += (_, _) => { try { _llamaProcess?.Kill(true); } catch { } };
            _ = InitShellAsync();
        }

        // ── Startup ───────────────────────────────────────────────────────────

        private async Task InitShellAsync()
        {
            try
            {
                // Create WebView2 environment with explicit user data folder
                var userDataPath = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                    "PRIMEOS", "WebView2");
                Directory.CreateDirectory(userDataPath);
                
                var env = await Microsoft.Web.WebView2.Core.CoreWebView2Environment.CreateAsync(null, userDataPath).ConfigureAwait(false);
                
                // Use event-based initialization to avoid potential deadlocks
                await Dispatcher.InvokeAsync(() => {
                    MainWebView.CoreWebView2InitializationCompleted += (s, e) => {
                        if (!e.IsSuccess) {
                            PostToUI(new { type = "error", message = $"WebView2 init failed: {e.InitializationException?.Message}" });
                            return;
                        }
                        
                        Debug.WriteLine($"[PRIMEOS] CoreWebView2 initialized successfully");
                        MainWebView.CoreWebView2.WebMessageReceived += OnWebMessage;
                        
                        // After first navigation: discover apps + silently restore auth from key file
                        bool _firstNav = true;
                        MainWebView.CoreWebView2.NavigationCompleted += (s2, e2) => {
                            Debug.WriteLine($"[PRIMEOS] NavigationCompleted: {e2.IsSuccess}");
                            ScanApps();
                            if (_firstNav) { _firstNav = false; _ = Task.Run(InitAuthAsync); }
                        };

                        string htmlPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "primeos-app.html");
                        Debug.WriteLine($"[PRIMEOS] Looking for HTML at: {htmlPath}, exists: {File.Exists(htmlPath)}");
                        if (!File.Exists(htmlPath)) {
                            htmlPath = Path.Combine(RepoRoot, @"desktop\PRIMEOS\primeos-app.html");
                            Debug.WriteLine($"[PRIMEOS] Fallback path: {htmlPath}, exists: {File.Exists(htmlPath)}");
                        }

                        if (File.Exists(htmlPath)) {
                            Debug.WriteLine($"[PRIMEOS] Navigating to: {htmlPath}");
                            MainWebView.Source = new Uri(htmlPath);
                        } else {
                            Debug.WriteLine("[PRIMEOS] HTML not found, using inline HTML");
                            MainWebView.NavigateToString("<body style='background:#010a0f;color:#0d9b7a;font-family:monospace;padding:20px'>PRIMEOS: primeos-app.html not found</body>");
                        }
                    };
                    
                    _ = MainWebView.EnsureCoreWebView2Async(env);
                });
            }
            catch (Exception ex)
            {
                PostToUI(new { type = "error", message = $"WebView2 init failed: {ex.Message}" });
                return;
            }

            // Read active-model.json (written by START-SERVERS.bat) for actual endpoints
            LoadActiveModel();

            // SHM polling (FieldExecutionEngine geodesic state)
            _shmTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(500) };
            _shmTimer.Tick += ShmTimer_Tick;
            _shmTimer.Start();

            // Periodic status + active-model.json refresh (fully off UI thread)
            _statusTimer = new DispatcherTimer { Interval = TimeSpan.FromSeconds(8) };
            _statusTimer.Tick += (_, _) => _ = Task.Run(async () => { LoadActiveModel(); await ProbeStatus(); });
            _statusTimer.Start();

            await ProbeStatus();
        }

        // Reads active-model.json (written by START-SERVERS.bat) and updates endpoints.
        private void LoadActiveModel()
        {
            // 1. chat.manifest.json — canonical source (written by START-SERVERS / kuhul-server)
            string chatManifest = Path.Combine(RepoRoot, "chat.manifest.json");
            if (File.Exists(chatManifest))
            {
                try
                {
                    using var cm = JsonDocument.Parse(File.ReadAllText(chatManifest));
                    var r = cm.RootElement;
                    if (r.TryGetProperty("gateway", out var gw))
                    {
                        if (gw.TryGetProperty("url", out var gu) && !string.IsNullOrEmpty(gu.GetString()))
                            _bossUri = gu.GetString();
                        if (gw.TryGetProperty("chat", out var gc) && !string.IsNullOrEmpty(gc.GetString()))
                            _chatEndpoint = gc.GetString();
                    }
                    if (r.TryGetProperty("engine", out var eng2) && eng2.TryGetProperty("chat", out var ec2) && !string.IsNullOrEmpty(ec2.GetString()))
                        _engineEndpoint = ec2.GetString();
                }
                catch { }
            }

            // 2. active-model.json — overrides active_model port if present
            string path = Path.Combine(RepoRoot, "active-model.json");
            if (!File.Exists(path)) return;
            try
            {
                using var doc = JsonDocument.Parse(File.ReadAllText(path));
                var root = doc.RootElement;
                // Prefer MCP gateway for chat (always up, proxies to active model)
                if (root.TryGetProperty("mcp_gateway", out var mcp) && !string.IsNullOrEmpty(mcp.GetString()))
                {
                    _bossUri      = mcp.GetString();
                    _chatEndpoint = mcp.GetString().TrimEnd('/') + "/v1/chat/completions";
                }
                else if (root.TryGetProperty("endpoint", out var ep) && !string.IsNullOrEmpty(ep.GetString()))
                    _chatEndpoint = ep.GetString();
                if (root.TryGetProperty("engine_endpoint", out var eng) && !string.IsNullOrEmpty(eng.GetString()))
                    _engineEndpoint = eng.GetString();
            }
            catch { }
        }

        // ── SHM ──────────────────────────────────────────────────────────────

        private void ShmTimer_Tick(object sender, EventArgs e)
        {
            try
            {
                using var mmf  = MemoryMappedFile.OpenExisting(SHM_NAME, MemoryMappedFileRights.Read);
                using var view = mmf.CreateViewAccessor(0, Marshal.SizeOf<KuhulSharedState>(),
                                                         MemoryMappedFileAccess.Read);
                view.Read(0, out KuhulSharedState s);
                PostToUI(new {
                    type      = "shm",
                    version   = s.Version,
                    entropy   = s.Entropy,
                    attention = s.Attention,
                    pressure  = s.Pressure,
                    fold      = s.ActiveFold,
                    tick      = s.TickCount
                });
            }
            catch
            {
                PostToUI(new { type = "shm_offline" });
            }
        }

        // ── Status probe ─────────────────────────────────────────────────────

        private async Task ProbeStatus()
        {
            // chat server (9000 or port from active-model.json)
            string chatBase = _chatEndpoint.Replace("/v1/chat/completions", "");
            bool kuhul = await PingAsync($"{chatBase}/health");
            bool boss  = await PingAsync($"{_bossUri}/health");
            bool llama = _llamaProcess is { HasExited: false };
            _isKuhulOnline = kuhul;
            _isBossOnline  = boss;

            // extract port from chatEndpoint for display
            int chatPort = 9000;
            try { chatPort = new Uri(_chatEndpoint).Port; } catch { }

            PostToUI(new { type = "status", kuhul, boss, llama, chatPort });
        }

        private async Task<bool> PingAsync(string url)
        {
            try
            {
                using var cts  = new System.Threading.CancellationTokenSource(2000);
                var resp = await _httpClient.GetAsync(url, cts.Token);
                return resp.IsSuccessStatusCode;
            }
            catch { return false; }
        }

        private static int FreePort()
        {
            var l = new TcpListener(IPAddress.Loopback, 0);
            l.Start();
            int port = ((IPEndPoint)l.LocalEndpoint).Port;
            l.Stop();
            return port;
        }

        // ── WebMessage handler ────────────────────────────────────────────────

        private void OnWebMessage(object sender, Microsoft.Web.WebView2.Core.CoreWebView2WebMessageReceivedEventArgs e)
        {
            string raw;
            try   { raw = e.TryGetWebMessageAsString(); }
            catch { return; }

            JsonDocument doc;
            try   { doc = JsonDocument.Parse(raw); }
            catch { return; }

            using (doc)
            {
                string type = doc.RootElement.TryGetProperty("type", out var t) ? t.GetString() : "";
                switch (type)
                {
                    case "get_status":
                        _ = Task.Run(ProbeStatus);
                        break;

                    case "set_model":
                        if (doc.RootElement.TryGetProperty("alias", out var a))
                        {
                            _currentAlias = a.GetString() ?? _currentAlias;
                            PostToUI(new { type = "system", message = $"Model → {_currentAlias}" });
                            PostToUI(new { type = "model_switched", alias = _currentAlias });
                        }
                        break;

                    case "chat":
                    {
                        string msg   = doc.RootElement.TryGetProperty("message", out var m) ? m.GetString() : "";
                        string model = doc.RootElement.TryGetProperty("model",   out var mo) ? mo.GetString() : _currentAlias;
                        _ = Task.Run(() => RouteInference(msg, model));
                        break;
                    }

                    case "call_boss":
                    {
                        string verb   = doc.RootElement.TryGetProperty("verb",   out var v)  ? v.GetString()  : "task.plan";
                        string prompt = doc.RootElement.TryGetProperty("prompt", out var pr) ? pr.GetString() : "";
                        _ = Task.Run(async () => {
                            try {
                                string reply = await CallBossAsync(verb, prompt);
                                PostToUI(new { type = "boss_reply", text = reply });
                            } catch (Exception ex) {
                                PostToUI(new { type = "chat_error", message = $"BOSS: {ex.Message}" });
                            }
                        });
                        break;
                    }

                    case "launch_app":
                        string appId = doc.RootElement.TryGetProperty("id", out var id) ? id.GetString() : "";
                        LaunchApp(appId);
                        break;

                    case "set_atomic":
                        _atomicReplies = doc.RootElement.TryGetProperty("enabled", out var en) && en.GetBoolean();
                        break;

                    case "get_models":
                    {
                        System.Diagnostics.Debug.WriteLine($"[PRIMEOS] get_models called, RepoRoot={RepoRoot}");
                        var models = new List<object>();
                        foreach (var kvp in ModelAliases)
                        {
                            string manifestPath = Path.Combine(RepoRoot, kvp.Value);
                            System.Diagnostics.Debug.WriteLine($"[PRIMEOS] Checking {kvp.Key}: {manifestPath}, exists={File.Exists(manifestPath)}");
                            bool hasToolCalls = false;
                            bool isChatModel = false;
                            string format = "";
                            
                            if (File.Exists(manifestPath))
                            {
                                try
                                {
                                    using var manDoc = JsonDocument.Parse(File.ReadAllText(manifestPath));
                                    var root = manDoc.RootElement;
                                    
                                    // Check if it's a chat model
                                    if (root.TryGetProperty("app", out var app) && 
                                        app.TryGetProperty("kind", out var kind))
                                    {
                                        isChatModel = kind.GetString() == "chat";
                                        System.Diagnostics.Debug.WriteLine($"[PRIMEOS] {kvp.Key} is_chat={isChatModel}");
                                    }
                                    
                                    // Check for tool_call support in chat_template
                                    if (root.TryGetProperty("chat_template", out var ct) &&
                                        ct.TryGetProperty("tool_call", out var tc))
                                    {
                                        hasToolCalls = true;
                                        if (tc.TryGetProperty("format", out var fmt))
                                            format = fmt.GetString() ?? "";
                                        System.Diagnostics.Debug.WriteLine($"[PRIMEOS] {kvp.Key} has_tool_calls=true, format={format}");
                                    }
                                    
                                    // Also check chat_template.format for chatml, gemma3, etc.
                                    if (ct.TryGetProperty("format", out var fmt2))
                                        format = fmt2.GetString() ?? format;
                                }
                                catch (Exception ex)
                                {
                                    System.Diagnostics.Debug.WriteLine($"[PRIMEOS] Error reading {kvp.Key}: {ex.Message}");
                                }
                            }
                            else
                            {
                                System.Diagnostics.Debug.WriteLine($"[PRIMEOS] Manifest not found: {manifestPath}");
                            }
                            
                            models.Add(new {
                                alias = kvp.Key,
                                has_tool_calls = hasToolCalls,
                                is_chat = isChatModel,
                                format = format
                            });
                        }
                        System.Diagnostics.Debug.WriteLine($"[PRIMEOS] Sending {models.Count} models to UI");
                        PostToUI(new { type = "models", items = models });
                        break;
                    }

                    case "download_model":
                    {
                        string dlId     = doc.RootElement.TryGetProperty("id",          out var dlIdP)     ? dlIdP.GetString()     : "";
                        string hfRepo   = doc.RootElement.TryGetProperty("hf_repo",     out var hfP)       ? hfP.GetString()       : "";
                        string filename = doc.RootElement.TryGetProperty("filename",    out var fnP)       ? fnP.GetString()       : "";
                        string destFold = doc.RootElement.TryGetProperty("dest_folder", out var dfP)       ? dfP.GetString()       : "";
                        if (!string.IsNullOrWhiteSpace(dlId) && !string.IsNullOrWhiteSpace(filename))
                            _ = Task.Run(() => DownloadModelAsync(dlId, hfRepo, filename, destFold));
                        break;
                    }

                    case "open_url":
                    {
                        string url = doc.RootElement.TryGetProperty("url", out var urlP) ? urlP.GetString() : "";
                        if (!string.IsNullOrWhiteSpace(url))
                            Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
                        break;
                    }

                    case "gh_login":
                    {
                        _deviceFlowCts?.Cancel();
                        _deviceFlowCts = new CancellationTokenSource();
                        _ = Task.Run(() => StartGitHubDeviceFlowAsync(_deviceFlowCts.Token));
                        break;
                    }

                    case "gh_cancel":
                        _deviceFlowCts?.Cancel();
                        break;

                    case "download_key":
                        _ = Task.Run(GenerateKeyFileAsync);
                        break;

                    case "gh_logout":
                    {
                        _ghToken = null;
                        _ghLogin = null;
                        if (File.Exists(_keyFile)) try { File.Delete(_keyFile); } catch { }
                        break;
                    }

                    case "gh_push":
                        _ = Task.Run(GhPushAsync);
                        break;

                    case "gh_dispatch":
                        _ = Task.Run(GhDispatchAsync);
                        break;

                    case "gh_clone":
                    {
                        string repo = doc.RootElement.TryGetProperty("repo", out var rv) ? rv.GetString() ?? "" : "";
                        string dest = doc.RootElement.TryGetProperty("dest", out var dv) ? dv.GetString() ?? "" : "";
                        _ = Task.Run(() => GhCloneAsync(repo, dest));
                        break;
                    }

                    // ── WebGPU WGSL bridge ─────────────────────────────────
                    // JS sends: { type:"wgsl_dispatch", kernel:"FusedAttention" }
                    //        or: { type:"wgsl_dispatch", kernel:"...", wgsl:"<source>" }
                    // C# loads the .wgsl from the compiled outputs dir (if not inline),
                    // then calls window.__kuhulWgslDispatch(wgsl, kernelName) via ExecuteScriptAsync.
                    case "wgsl_dispatch":
                    {
                        string kernelName = doc.RootElement.TryGetProperty("kernel", out var kn) ? kn.GetString() ?? "" : "";
                        string inlineWgsl = doc.RootElement.TryGetProperty("wgsl",   out var ws) ? ws.GetString() ?? "" : "";
                        if (string.IsNullOrWhiteSpace(kernelName)) break;
                        string wgslSource = inlineWgsl;
                        if (string.IsNullOrWhiteSpace(wgslSource))
                        {
                            string wgslPath = Path.Combine(RepoRoot,
                                @"versions\kxc-v1.0.0\examples\outputs",
                                kernelName, kernelName + ".wgsl");
                            if (!File.Exists(wgslPath))
                            {
                                PostToUI(new { type = "wgsl_error", kernel = kernelName,
                                               error = $".wgsl not found: {wgslPath}" });
                                break;
                            }
                            wgslSource = File.ReadAllText(wgslPath);
                        }
                        InvokeWgslKernel(kernelName, wgslSource);
                        break;
                    }
                }
            }
        }

        private async Task DownloadModelAsync(string id, string hfRepo, string filename, string destFolder)
        {
            try
            {
                string lmDir  = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".lmstudio", "models");
                string outDir = Path.Combine(lmDir, destFolder.Replace('/', Path.DirectorySeparatorChar));
                string dest   = Path.Combine(outDir, filename);
                Directory.CreateDirectory(outDir);

                string url = $"https://huggingface.co/{hfRepo}/resolve/main/{filename}";
                using var dlClient = new HttpClient { Timeout = System.Threading.Timeout.InfiniteTimeSpan };
                using var resp = await dlClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
                resp.EnsureSuccessStatusCode();
                long total = resp.Content.Headers.ContentLength ?? -1;

                await using var stream = await resp.Content.ReadAsStreamAsync();
                await using var file   = File.OpenWrite(dest);
                var buf  = new byte[81920];
                long done = 0; int read; int lastPct = -1;
                while ((read = await stream.ReadAsync(buf)) > 0)
                {
                    await file.WriteAsync(buf.AsMemory(0, read));
                    done += read;
                    if (total > 0)
                    {
                        int pct = (int)(done * 100 / total);
                        if (pct != lastPct) { lastPct = pct; PostToUI(new { type = "model_download_progress", id, pct, mb = done / 1048576 }); }
                    }
                }
                PostToUI(new { type = "model_download_done", id });
            }
            catch (Exception ex)
            {
                PostToUI(new { type = "model_download_error", id, error = ex.Message });
            }
        }

        // ── Inference ────────────────────────────────────────────────────────

        private async Task RouteInference(string message, string alias)
        {
            if (string.IsNullOrWhiteSpace(message)) return;

            // Prefer the main chat endpoint from active-model.json (MCP gateway).
            // Fall back to manifest provider.url if present (e.g. phi3-mini on :17480).
            string endpoint = _chatEndpoint;
            if (ModelAliases.TryGetValue(alias, out string rel))
            {
                string manifestPath = Path.Combine(RepoRoot, rel);
                endpoint = ResolveEndpoint(manifestPath) ?? endpoint;
            }

            // Immediate feedback so the user sees something within milliseconds
            PostToUI(new { type = "chat_routing", endpoint = endpoint });

            // Quick reachability check (2s) before full inference request
            bool reachable = await PingAsync(endpoint.Replace("/v1/chat/completions", "/health"));
            if (!reachable)
            {
                PostToUI(new { type = "chat_error", message = $"Model offline — start servers first (Quick Launch → Start Servers). Tried: {endpoint}" });
                return;
            }

            try
            {
                var body    = new { model = alias, messages = new[] { new { role = "user", content = message } }, stream = false };
                var json    = JsonSerializer.Serialize(body);
                var content = new StringContent(json, System.Text.Encoding.UTF8, "application/json");
                // Per-request timeout — long enough for CPU inference on slow hardware
                using var reqCts = new System.Threading.CancellationTokenSource(TimeSpan.FromSeconds(120));
                var resp    = await _httpClient.PostAsync(endpoint, content, reqCts.Token);

                if (resp.IsSuccessStatusCode)
                {
                    string resultJson = await resp.Content.ReadAsStringAsync();
                    using var doc = JsonDocument.Parse(resultJson);
                    string text = doc.RootElement
                        .GetProperty("choices")[0]
                        .GetProperty("message")
                        .GetProperty("content")
                        .GetString() ?? "";

                    _isKuhulOnline = true;
                    PostToUI(new { type = "chat_reply", model = alias, text });
                }
                else
                {
                    PostToUI(new { type = "chat_error", message = $"HTTP {(int)resp.StatusCode} from {endpoint}" });
                }
            }
            catch (Exception ex)
            {
                _isKuhulOnline = false;
                PostToUI(new { type = "chat_error", message = ex.Message });
            }
        }

        private string ResolveEndpoint(string manifestPath)
        {
            if (!File.Exists(manifestPath)) return null;
            try
            {
                using var doc = JsonDocument.Parse(File.ReadAllText(manifestPath));
                if (doc.RootElement.TryGetProperty("provider", out var prov) &&
                    prov.TryGetProperty("url", out var url))
                    return url.GetString();
            }
            catch { }
            return null;
        }

        // ── BOSS/MCP ──────────────────────────────────────────────────────────

        private async Task<string> CallBossAsync(string verb, string prompt)
        {
            var payload = new
            {
                jsonrpc = "2.0",
                method  = "tools/call",
                id      = 1,
                @params = new { name = "kuhul_task_boss", arguments = new { verb, prompt } }
            };
            var json    = JsonSerializer.Serialize(payload);
            var content = new StringContent(json, System.Text.Encoding.UTF8, "application/json");
            var resp    = await _httpClient.PostAsync($"{_bossUri}/mcp", content);
            if (!resp.IsSuccessStatusCode) throw new Exception($"BOSS HTTP {resp.StatusCode}");

            string body = await resp.Content.ReadAsStringAsync();
            using var doc = JsonDocument.Parse(body);
            var result = doc.RootElement.GetProperty("result");
            if (result.TryGetProperty("content", out var arr) && arr.GetArrayLength() > 0)
                return arr[0].GetProperty("text").GetString() ?? "";
            if (result.TryGetProperty("text", out var txt))
                return txt.GetString() ?? "";
            return body;
        }

        // ── App launch ────────────────────────────────────────────────────────

        // Read app.ops.kuhul for launch runtime/entry — returns (runtime, entryRelPath)
        private (string runtime, string entry) ReadOpsKuhul(string appDir)
        {
            string opsPath = Path.Combine(appDir, "app.ops.kuhul");
            if (!File.Exists(opsPath)) return ("powershell", "app.ps1");
            try
            {
                using var doc = JsonDocument.Parse(File.ReadAllText(opsPath));
                var root = doc.RootElement;
                if (root.TryGetProperty("ops", out var ops) && ops.TryGetProperty("launch", out var launch))
                {
                    string rt    = launch.TryGetProperty("runtime", out var rtP) ? rtP.GetString() ?? "powershell" : "powershell";
                    string entry = launch.TryGetProperty("entry",   out var en)  ? en.GetString()  ?? "app.ps1"   : "app.ps1";
                    return (rt, entry);
                }
            }
            catch { }
            return ("powershell", "app.ps1");
        }

        private void LaunchApp(string appId)
        {
            if (string.IsNullOrEmpty(appId)) return;

            string appDir    = null;
            string scriptPath = null;

            if (BuiltinApps.TryGetValue(appId, out string rel))
            {
                string p = Path.Combine(RepoRoot, rel);
                if (File.Exists(p)) { scriptPath = p; appDir = Path.GetDirectoryName(p); }
            }

            if (scriptPath == null)
            {
                string dir = Path.Combine(RepoRoot, "apps", appId);
                // Read ops.kuhul for the declared entry (may not be app.ps1)
                var (rt2, entry2) = ReadOpsKuhul(dir);
                string candidate = Path.Combine(dir, entry2);
                if (File.Exists(candidate)) { scriptPath = candidate; appDir = dir; }
                else
                {
                    // fallback: app.ps1
                    string ps1 = Path.Combine(dir, "app.ps1");
                    if (File.Exists(ps1)) { scriptPath = ps1; appDir = dir; }
                }
            }

            if (scriptPath == null)
            {
                PostToUI(new { type = "app_error", id = appId, message = "entry script not found" });
                return;
            }

            // Determine runtime from ops.kuhul, then from file extension
            var (runtime, _) = ReadOpsKuhul(appDir ?? Path.GetDirectoryName(scriptPath));

            try
            {
                ProcessStartInfo psi;
                string ext = Path.GetExtension(scriptPath).ToLowerInvariant();

                if (ext == ".bat" || ext == ".cmd" || runtime == "cmd")
                {
                    psi = new ProcessStartInfo
                    {
                        FileName         = "cmd.exe",
                        Arguments        = $"/c \"{scriptPath}\"",
                        UseShellExecute  = true,
                        WorkingDirectory = appDir ?? RepoRoot,
                    };
                }
                else if (runtime == "node" || ext == ".mjs" || ext == ".cjs" || ext == ".js")
                {
                    psi = new ProcessStartInfo
                    {
                        FileName         = "node.exe",
                        Arguments        = $"\"{scriptPath}\"",
                        UseShellExecute  = true,
                        WorkingDirectory = appDir ?? RepoRoot,
                    };
                }
                else if (runtime == "pwsh")
                {
                    psi = new ProcessStartInfo
                    {
                        FileName         = "pwsh.exe",
                        Arguments        = $"-NoProfile -ExecutionPolicy Bypass -File \"{scriptPath}\"",
                        UseShellExecute  = true,
                        WorkingDirectory = appDir ?? RepoRoot,
                    };
                }
                else if (runtime == "exe" || ext == ".exe")
                {
                    psi = new ProcessStartInfo
                    {
                        FileName         = scriptPath,
                        UseShellExecute  = true,
                        WorkingDirectory = appDir ?? RepoRoot,
                    };
                }
                else
                {
                    // Default: powershell (most app.ps1 files)
                    psi = new ProcessStartInfo
                    {
                        FileName         = "powershell.exe",
                        Arguments        = $"-NoProfile -ExecutionPolicy Bypass -STA -File \"{scriptPath}\"",
                        UseShellExecute  = true,
                        WorkingDirectory = appDir ?? RepoRoot,
                    };
                }
                Process.Start(psi);
                PostToUI(new { type = "app_launched", id = appId });
            }
            catch (Exception ex)
            {
                PostToUI(new { type = "app_error", id = appId, message = ex.Message });
            }
        }

        // ── App discovery ─────────────────────────────────────────────────────

        // Scans apps/{id}/app.program.json and posts apps_loaded so the HTML
        // can mark discovered apps as installed in the store grid.
        private void ScanApps()
        {
            try
            {
                string appsDir = Path.Combine(RepoRoot, "apps");
                if (!Directory.Exists(appsDir)) return;

                var manifests = new System.Text.StringBuilder();
                manifests.Append('[');
                bool first = true;
                foreach (var dir in Directory.GetDirectories(appsDir))
                {
                    string jsonPath   = Path.Combine(dir, "app.program.json");
                    string ps1Path    = Path.Combine(dir, "app.ps1");
                    string shaderPath = Path.Combine(dir, "app.shader.klsl");
                    if (!File.Exists(jsonPath) || !File.Exists(ps1Path)) continue;

                    string rawJson = File.ReadAllText(jsonPath).TrimEnd();
                    // Inject has_shader flag by appending to JSON object before closing brace
                    bool hasShader = File.Exists(shaderPath);
                    if (rawJson.EndsWith("}"))
                        rawJson = rawJson[..^1] + $",\"has_shader\":{(hasShader ? "true" : "false")}}}";

                    if (!first) manifests.Append(',');
                    manifests.Append(rawJson);
                    first = false;
                }
                manifests.Append(']');

                if (!first) // at least one app found
                {
                    string payload = $"{{\"type\":\"apps_loaded\",\"apps\":{manifests}}}";
                    Dispatcher.InvokeAsync(() =>
                    {
                        try { MainWebView.CoreWebView2?.PostWebMessageAsString(payload); }
                        catch { }
                    });
                }
            }
            catch { }
        }

        // ── Phase cycle ───────────────────────────────────────────────────────

        private void AdvancePhase()
        {
            var phases = new[] { "POP", "WO", "YAX", "SEK", "CH'EN", "XUL" };
            _phaseIndex = (_phaseIndex + 1) % phases.Length;
            _currentPhase = phases[_phaseIndex];
            PostToUI(new { type = "phase", phase = _currentPhase, index = _phaseIndex });
        }

        // ── Bridge helpers ────────────────────────────────────────────────────

        private void PostToUI(object payload)
        {
            string json = JsonSerializer.Serialize(payload);
            Dispatcher.InvokeAsync(() =>
            {
                try { MainWebView.CoreWebView2?.PostWebMessageAsString(json); }
                catch { }
            });
        }

        // Passes a compiled WGSL kernel into the page via ExecuteScriptAsync so the
        // JS side can call navigator.gpu.createComputePipeline() with it.
        // The page must expose window.__kuhulWgslDispatch(wgsl, kernelName).
        private void InvokeWgslKernel(string kernelName, string wgslSource)
        {
            string jsWgsl   = JsonSerializer.Serialize(wgslSource);   // safe JSON string literal
            string jsKernel = JsonSerializer.Serialize(kernelName);
            string script   = $"window.__kuhulWgslDispatch({jsWgsl},{jsKernel})";
            Dispatcher.InvokeAsync(async () =>
            {
                try
                {
                    if (MainWebView.CoreWebView2 != null)
                        await MainWebView.CoreWebView2.ExecuteScriptAsync(script);
                }
                catch (Exception ex)
                {
                    PostToUI(new { type = "wgsl_error", kernel = kernelName, error = ex.Message });
                }
            });
        }

        // ── Auth ──────────────────────────────────────────────────────────────

        private async Task InitAuthAsync()
        {
            if (!File.Exists(_keyFile)) return;
            try
            {
                string raw = await File.ReadAllTextAsync(_keyFile);
                using var doc = JsonDocument.Parse(raw);
                var root = doc.RootElement;
                if (!root.TryGetProperty("token_enc", out var te)) return;

                byte[] encBytes  = Convert.FromBase64String(te.GetString() ?? "");
                byte[] rawBytes  = ProtectedData.Unprotect(encBytes, null, DataProtectionScope.CurrentUser);
                string token     = Encoding.UTF8.GetString(rawBytes);
                string fp        = root.TryGetProperty("fingerprint", out var fpP) ? fpP.GetString() : "";

                var user = await GetGitHubUserAsync(token);
                if (user == null) { try { File.Delete(_keyFile); } catch { } return; }

                _ghToken = token;
                _ghLogin = user.Login;
                PostToUI(new {
                    type = "auth_restore",
                    login = user.Login, name = user.Name, email = user.Email,
                    avatar_url = user.AvatarUrl, key_exists = true, key_fingerprint = fp
                });
            }
            catch { }
        }

        private async Task StartGitHubDeviceFlowAsync(CancellationToken ct)
        {
            try
            {
                using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(30) };
                client.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));

                // Step 1 — request device + user codes
                var form1 = new FormUrlEncodedContent(new[] {
                    new KeyValuePair<string,string>("client_id", GH_CLIENT_ID),
                    new KeyValuePair<string,string>("scope",     GH_SCOPE)
                });
                var r1 = await client.PostAsync("https://github.com/login/device/code", form1, ct);
                r1.EnsureSuccessStatusCode();
                using var d1  = JsonDocument.Parse(await r1.Content.ReadAsStringAsync(ct));
                string devCode  = d1.RootElement.GetProperty("device_code").GetString();
                string userCode = d1.RootElement.GetProperty("user_code").GetString();
                int    interval = d1.RootElement.TryGetProperty("interval", out var iv) ? iv.GetInt32() : 5;

                PostToUI(new { type = "gh_device_code", user_code = userCode });

                // Step 2 — poll until granted, denied, or cancelled
                while (!ct.IsCancellationRequested)
                {
                    await Task.Delay(interval * 1000, ct);

                    var form2 = new FormUrlEncodedContent(new[] {
                        new KeyValuePair<string,string>("client_id",   GH_CLIENT_ID),
                        new KeyValuePair<string,string>("device_code", devCode),
                        new KeyValuePair<string,string>("grant_type",  "urn:ietf:params:oauth:grant-type:device_code")
                    });
                    var r2 = await client.PostAsync("https://github.com/login/oauth/access_token", form2, ct);
                    using var d2 = JsonDocument.Parse(await r2.Content.ReadAsStringAsync(ct));

                    if (d2.RootElement.TryGetProperty("access_token", out var tok))
                    {
                        string token = tok.GetString();
                        var user = await GetGitHubUserAsync(token);
                        if (user == null) { PostToUI(new { type = "auth_error", message = "Could not retrieve GitHub profile" }); return; }
                        _ghToken = token;
                        _ghLogin = user.Login;
                        PostToUI(new { type = "auth_success", login = user.Login, name = user.Name, email = user.Email, avatar_url = user.AvatarUrl });
                        return;
                    }

                    if (d2.RootElement.TryGetProperty("error", out var err))
                    {
                        string e = err.GetString();
                        if (e == "authorization_pending") continue;
                        if (e == "slow_down") { interval += 5; continue; }
                        if (e == "expired_token") { PostToUI(new { type = "auth_error", message = "Device code expired — please try again" }); return; }
                        PostToUI(new { type = "auth_error", message = e });
                        return;
                    }
                }
            }
            catch (TaskCanceledException) { }
            catch (Exception ex) { PostToUI(new { type = "auth_error", message = ex.Message }); }
        }

        private async Task<GHUser> GetGitHubUserAsync(string token)
        {
            try
            {
                using var c = new HttpClient { Timeout = TimeSpan.FromSeconds(10) };
                c.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", token);
                c.DefaultRequestHeaders.UserAgent.ParseAdd("PRIMEOS/1.0");
                c.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/vnd.github+json"));
                var resp = await c.GetAsync("https://api.github.com/user");
                if (!resp.IsSuccessStatusCode) return null;
                using var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync());
                var r = doc.RootElement;
                return new GHUser(
                    Login:     r.TryGetProperty("login",      out var lo) ? lo.GetString() ?? "" : "",
                    Name:      r.TryGetProperty("name",       out var na) ? na.GetString() ?? "" : "",
                    Email:     r.TryGetProperty("email",      out var em) ? em.GetString() ?? "" : "",
                    AvatarUrl: r.TryGetProperty("avatar_url", out var av) ? av.GetString() ?? "" : ""
                );
            }
            catch { return null; }
        }

        private async Task GenerateKeyFileAsync()
        {
            if (string.IsNullOrEmpty(_ghToken))
            { PostToUI(new { type = "key_error", message = "Not authenticated" }); return; }
            try
            {
                Directory.CreateDirectory(_keyDir);

                // Encrypt token with DPAPI (CurrentUser scope = machine-bound)
                byte[] raw = Encoding.UTF8.GetBytes(_ghToken);
                byte[] enc = ProtectedData.Protect(raw, null, DataProtectionScope.CurrentUser);
                string tokenEnc = Convert.ToBase64String(enc);

                // Build stable fingerprint
                string created  = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ");
                string fpSrc    = $"{_ghLogin}:{created}:{Environment.MachineName}";
                byte[] fpBytes  = SHA256.HashData(Encoding.UTF8.GetBytes(fpSrc));
                string fp       = "sha256:" + BitConverter.ToString(fpBytes, 0, 8).Replace("-","").ToLower();

                var obj = new {
                    version     = 1,
                    provider    = "github",
                    login       = _ghLogin,
                    machine     = Environment.MachineName,
                    created     = created,
                    fingerprint = fp,
                    token_enc   = tokenEnc
                };
                string json = JsonSerializer.Serialize(obj, new JsonSerializerOptions { WriteIndented = true });

                await File.WriteAllTextAsync(_keyFile, json);

                // Also save copy to Desktop so the user can pick it up
                string desktop = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
                string copy    = Path.Combine(desktop, "primeos-access.key");
                await File.WriteAllTextAsync(copy, json);

                PostToUI(new { type = "key_ready", fingerprint = fp, path = copy });
            }
            catch (Exception ex) { PostToUI(new { type = "key_error", message = ex.Message }); }
        }

        private async Task GhPushAsync()
        {
            if (string.IsNullOrEmpty(_ghToken))
            { PostToUI(new { type = "gh_action_error", message = "Not authenticated" }); return; }
            try
            {
                // Get current remote URL, inject token, push, restore
                string origUrl = await RunGitAsync("remote get-url origin");
                if (string.IsNullOrWhiteSpace(origUrl))
                { PostToUI(new { type = "gh_action_error", message = "No git remote 'origin' found" }); return; }

                string tokenUrl = origUrl.Replace("https://", $"https://{_ghLogin}:{_ghToken}@");
                await RunGitAsync($"remote set-url origin {tokenUrl}");
                try
                {
                    string result = await RunGitAsync("push");
                    PostToUI(new { type = "gh_action_ok", message = string.IsNullOrWhiteSpace(result) ? "Push succeeded" : result.Trim() });
                }
                finally
                {
                    await RunGitAsync($"remote set-url origin {origUrl}");
                }
            }
            catch (Exception ex) { PostToUI(new { type = "gh_action_error", message = ex.Message }); }
        }

        private async Task GhDispatchAsync()
        {
            if (string.IsNullOrEmpty(_ghToken))
            { PostToUI(new { type = "gh_action_error", message = "Not authenticated" }); return; }
            try
            {
                // Derive owner/repo from remote URL
                string remoteUrl = await RunGitAsync("remote get-url origin");
                if (string.IsNullOrWhiteSpace(remoteUrl))
                { PostToUI(new { type = "gh_action_error", message = "No git remote 'origin' found" }); return; }

                // Parse https://github.com/owner/repo.git  or  git@github.com:owner/repo.git
                string ownerRepo = remoteUrl.Trim()
                    .Replace("https://github.com/", "")
                    .Replace("git@github.com:", "")
                    .TrimEnd('/').TrimEnd(".git".ToCharArray());

                using var c = new HttpClient { Timeout = TimeSpan.FromSeconds(15) };
                c.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", _ghToken);
                c.DefaultRequestHeaders.UserAgent.ParseAdd("PRIMEOS/1.0");
                c.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/vnd.github+json"));

                // Trigger the first workflow_dispatch-capable workflow found
                var wfResp = await c.GetAsync($"https://api.github.com/repos/{ownerRepo}/actions/workflows");
                if (!wfResp.IsSuccessStatusCode)
                { PostToUI(new { type = "gh_action_error", message = $"Could not list workflows: {wfResp.StatusCode}" }); return; }

                using var wfDoc = JsonDocument.Parse(await wfResp.Content.ReadAsStringAsync());
                var workflows = wfDoc.RootElement.GetProperty("workflows");
                if (workflows.GetArrayLength() == 0)
                { PostToUI(new { type = "gh_action_error", message = "No workflows found in repo" }); return; }

                string wfId = workflows[0].GetProperty("id").GetRawText();
                var dispatch = new StringContent(
                    JsonSerializer.Serialize(new { @ref = "main" }),
                    Encoding.UTF8, "application/json");
                var dResp = await c.PostAsync(
                    $"https://api.github.com/repos/{ownerRepo}/actions/workflows/{wfId}/dispatches", dispatch);

                if (dResp.StatusCode == System.Net.HttpStatusCode.NoContent)
                    PostToUI(new { type = "gh_action_ok", message = "CI workflow dispatched on main" });
                else
                    PostToUI(new { type = "gh_action_error", message = $"Dispatch failed: {(int)dResp.StatusCode}" });
            }
            catch (Exception ex) { PostToUI(new { type = "gh_action_error", message = ex.Message }); }
        }

        private async Task GhCloneAsync(string repo, string destBase)
        {
            if (string.IsNullOrWhiteSpace(repo))
            { PostToUI(new { type = "gh_clone_error", message = "No repo specified" }); return; }

            // Normalise: accept "owner/repo", full HTTPS URL, or ssh URL
            string cloneUrl = repo;
            if (!repo.StartsWith("http") && !repo.StartsWith("git@"))
                cloneUrl = $"https://github.com/{repo}.git";

            // If authenticated, embed token for private repos
            if (!string.IsNullOrEmpty(_ghToken) && cloneUrl.StartsWith("https://github.com/"))
            {
                string repoPath = cloneUrl.Replace("https://github.com/", "");
                cloneUrl = $"https://{_ghLogin}:{_ghToken}@github.com/{repoPath}";
            }

            // Destination: use provided dest, or a projects/ subfolder next to the repo
            string repoName = repo.Split('/').Last().Replace(".git", "");
            if (string.IsNullOrWhiteSpace(destBase))
                destBase = Path.Combine(RepoRoot, "projects");
            string dest = Path.Combine(destBase, repoName);

            try
            {
                PostToUI(new { type = "gh_clone_progress", message = $"Cloning {repoName}…" });
                Directory.CreateDirectory(destBase);
                string result = await RunGitInDirAsync(destBase, $"clone \"{cloneUrl}\" \"{repoName}\"");
                PostToUI(new { type = "gh_clone_ok", message = $"Cloned to {dest}", path = dest, name = repoName });
            }
            catch (Exception ex)
            {
                PostToUI(new { type = "gh_clone_error", message = ex.Message });
            }
        }

        private async Task<string> RunGitAsync(string args)
        {
            var psi = new ProcessStartInfo("git", args)
            {
                UseShellExecute        = false,
                RedirectStandardOutput = true,
                RedirectStandardError  = true,
                WorkingDirectory       = RepoRoot,
                CreateNoWindow         = true,
            };
            var proc = Process.Start(psi);
            string stdout = await proc.StandardOutput.ReadToEndAsync();
            await proc.WaitForExitAsync();
            return stdout.Trim();
        }

        private async Task<string> RunGitInDirAsync(string workDir, string args)
        {
            var psi = new ProcessStartInfo("git", args)
            {
                UseShellExecute        = false,
                RedirectStandardOutput = true,
                RedirectStandardError  = true,
                WorkingDirectory       = workDir,
                CreateNoWindow         = true,
            };
            var proc = Process.Start(psi);
            string stdout = await proc.StandardOutput.ReadToEndAsync();
            string stderr = await proc.StandardError.ReadToEndAsync();
            await proc.WaitForExitAsync();
            if (proc.ExitCode != 0 && !string.IsNullOrWhiteSpace(stderr))
                throw new Exception(stderr.Trim());
            return stdout.Trim();
        }
    }
}
