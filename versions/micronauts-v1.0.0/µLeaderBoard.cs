using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Khanary.Micronauts
{
    // ── µ entry (live, computed state) ───────────────────────────────────────

    public sealed class MuEntry
    {
        public string Name       { get; set; } = "";
        public string Fold       { get; set; } = "";    // Pop/Wo/Yax/Sek/Ch'en/Xul or null
        public string Category   { get; set; } = "";    // system/specialist/fold/meta/native/orchestrator/stack
        public string QuantTier  { get; set; } = "";    // fast/standard/quality/none
        public double Confidence { get; set; }           // registry baseline (0–1)

        // Accumulated across all project submissions
        public long              Invocations { get; set; }
        public long              Successes   { get; set; }
        public HashSet<string>   Projects    { get; set; } = new(StringComparer.Ordinal);

        // Computed — call Recalculate() after loading saved state
        [JsonIgnore] public double SuccessRate    { get; private set; }
        [JsonIgnore] public double AdoptionWeight { get; private set; }
        [JsonIgnore] public double Score          { get; private set; }
        [JsonIgnore] public int    Rank           { get; internal set; }

        // score = invocations × successRate × (1 + log-adoption) × confidence
        internal void Recalculate(int maxProjects)
        {
            SuccessRate    = Invocations == 0 ? 0 : (double)Successes / Invocations;
            double logMax  = Math.Log(1 + Math.Max(maxProjects, 1));
            AdoptionWeight = logMax == 0 ? 0 : Math.Log(1 + Projects.Count) / logMax;
            Score          = Invocations * SuccessRate * (1 + AdoptionWeight) * Confidence;
        }

        public override string ToString() =>
            $"#{Rank:D2} {Name,-22} score={Score:F1,10} inv={Invocations,6} ok={SuccessRate:P0} proj={Projects.Count,2} fold={Fold ?? "—",-6} cat={Category}";
    }

    // ── event written to .µleaderboard.jsonl ────────────────────────────────

    internal sealed class BoardEvent
    {
        public string Op        { get; set; } = "";  // "submit" | "seed"
        public string Timestamp { get; set; } = "";
        public string MuId      { get; set; } = "";
        public string ProjectId { get; set; } = "";
        public string EventType { get; set; } = "";  // "invoke" | "error" | "promote" | "seed"
        public bool   Success   { get; set; }
        public string Hash      { get; set; } = "";  // SHA-256 of prev_hash + payload
    }

    // ── µLeaderBoard ─────────────────────────────────────────────────────────

    public sealed class MuLeaderBoard
    {
        private readonly Dictionary<string, MuEntry> _entries =
            new(StringComparer.OrdinalIgnoreCase);
        private string _logPath  = ".µleaderboard.jsonl";
        private string _prevHash = "";

        // ── Factory ───────────────────────────────────────────────────────────

        // Load registry.json to pre-seed all known µ's, then replay event log if present.
        public static MuLeaderBoard Load(string registryPath, string logPath = null)
        {
            var board = new MuLeaderBoard();
            if (logPath != null) board._logPath = logPath;

            if (File.Exists(registryPath))
            {
                using var fs  = File.OpenRead(registryPath);
                using var doc = JsonDocument.Parse(fs);
                if (doc.RootElement.TryGetProperty("micronauts", out var arr))
                {
                    foreach (var m in arr.EnumerateArray())
                    {
                        string name = m.TryGetProperty("name", out var n) ? n.GetString() ?? "" : "";
                        if (string.IsNullOrEmpty(name)) continue;
                        board._entries[name] = new MuEntry
                        {
                            Name       = name,
                            Fold       = m.TryGetProperty("fold",      out var f)  ? f.GetString() ?? "" : "",
                            Category   = m.TryGetProperty("category",  out var c)  ? c.GetString() ?? "" : "",
                            QuantTier  = m.TryGetProperty("quant_tier",out var qt) ? qt.GetString() ?? "" : "",
                            Confidence = m.TryGetProperty("confidence",out var co) ? co.GetDouble()      : 0.5,
                        };
                    }
                }
            }

            // Replay existing event log to rebuild live state
            if (File.Exists(board._logPath))
            {
                foreach (var line in File.ReadLines(board._logPath))
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    try
                    {
                        using var ev = JsonDocument.Parse(line);
                        var r = ev.RootElement;
                        string op  = r.TryGetProperty("Op",  out var o) ? o.GetString() ?? "" : "";
                        string mu  = r.TryGetProperty("MuId",out var m) ? m.GetString() ?? "" : "";
                        string prj = r.TryGetProperty("ProjectId", out var p) ? p.GetString() ?? "" : "";
                        bool ok    = r.TryGetProperty("Success",   out var s) && s.GetBoolean();
                        board._prevHash = r.TryGetProperty("Hash", out var h) ? h.GetString() ?? "" : "";

                        if (op == "submit" && !string.IsNullOrEmpty(mu))
                        {
                            if (!board._entries.TryGetValue(mu, out var entry))
                            {
                                entry = new MuEntry { Name = mu, Confidence = 0.5 };
                                board._entries[mu] = entry;
                            }
                            entry.Invocations++;
                            if (ok) entry.Successes++;
                            if (!string.IsNullOrEmpty(prj)) entry.Projects.Add(prj);
                        }
                    }
                    catch { }
                }
            }

            board.Recalculate();
            return board;
        }

        // ── Submission ────────────────────────────────────────────────────────

        // Any project calls this to record a µ invocation result.
        // muId      — name from registry (e.g. "eliza", "coder", "think")
        // projectId — caller's project identifier (e.g. "PRIMEOS", "kxc-v1.0.0", "my-game")
        // success   — whether the µ completed its fold phase successfully
        // eventType — "invoke" | "error" | "promote" (default "invoke")
        public void Submit(string muId, string projectId, bool success, string eventType = "invoke")
        {
            if (string.IsNullOrEmpty(muId)) return;

            if (!_entries.TryGetValue(muId, out var entry))
            {
                entry = new MuEntry { Name = muId, Confidence = 0.5 };
                _entries[muId] = entry;
            }
            entry.Invocations++;
            if (success) entry.Successes++;
            if (!string.IsNullOrEmpty(projectId)) entry.Projects.Add(projectId);

            AppendEvent(new BoardEvent
            {
                Op        = "submit",
                Timestamp = DateTimeOffset.UtcNow.ToString("o"),
                MuId      = muId,
                ProjectId = projectId ?? "",
                EventType = eventType,
                Success   = success,
            });

            Recalculate();
        }

        // ── Queries ───────────────────────────────────────────────────────────

        public IReadOnlyList<MuEntry> GetLeaderboard() =>
            _entries.Values.OrderByDescending(e => e.Score).ToList();

        public IReadOnlyList<MuEntry> GetTopN(int n) =>
            GetLeaderboard().Take(n).ToList();

        public MuEntry GetTop() => GetLeaderboard().FirstOrDefault();

        // All µ's assigned to a K-CUBE fold lane.
        // fold: "Pop" | "Wo" | "Yax" | "Sek" | "Ch'en" | "Xul"
        public IReadOnlyList<MuEntry> GetFoldBoard(string fold) =>
            _entries.Values
                .Where(e => string.Equals(e.Fold, fold, StringComparison.OrdinalIgnoreCase))
                .OrderByDescending(e => e.Score)
                .ToList();

        public IReadOnlyList<MuEntry> GetCategoryBoard(string category) =>
            _entries.Values
                .Where(e => string.Equals(e.Category, category, StringComparison.OrdinalIgnoreCase))
                .OrderByDescending(e => e.Score)
                .ToList();

        // µ's whose score exceeds the fold median across 3+ contributing projects —
        // candidates for promotion to the next micronauts version snapshot.
        public IReadOnlyList<MuEntry> GetPromotionCandidates(int minProjects = 3)
        {
            var byFold = _entries.Values.GroupBy(e => e.Fold ?? "").ToDictionary(
                g => g.Key, g => g.Select(e => e.Score).OrderBy(s => s).ToList());

            return _entries.Values
                .Where(e =>
                {
                    if (e.Projects.Count < minProjects) return false;
                    string f = e.Fold ?? "";
                    if (!byFold.TryGetValue(f, out var scores) || scores.Count == 0) return false;
                    double median = scores.Count % 2 == 0
                        ? (scores[scores.Count / 2 - 1] + scores[scores.Count / 2]) / 2.0
                        : scores[scores.Count / 2];
                    return e.Score > median;
                })
                .OrderByDescending(e => e.Score)
                .ToList();
        }

        // ── Display ───────────────────────────────────────────────────────────

        public void PrintLeaderboard(int topN = 10)
        {
            var board = GetLeaderboard().Take(topN).ToList();
            Console.WriteLine("\n⚡ µ LEADERBOARD");
            Console.WriteLine(new string('─', 80));
            foreach (var e in board) Console.WriteLine(e);
            Console.WriteLine(new string('─', 80));
            Console.WriteLine($"  Total µ's tracked: {_entries.Count}  |  Events: {_eventCount}");
        }

        public void PrintFoldSummary()
        {
            var folds = new[] { "Pop", "Wo", "Yax", "Sek", "Ch'en", "Xul" };
            Console.WriteLine("\n📐 µ FOLD SUMMARY");
            Console.WriteLine(new string('─', 60));
            foreach (var f in folds)
            {
                var top = GetFoldBoard(f).FirstOrDefault();
                string leader = top != null ? $"{top.Name} (score={top.Score:F1})" : "—";
                Console.WriteLine($"  {f,-8} → {leader}");
            }
        }

        // ── Serialisation helpers ─────────────────────────────────────────────

        // Save a JSON snapshot of the current leaderboard state (not the event log).
        public void SaveSnapshot(string path)
        {
            var snapshot = new
            {
                generated  = DateTimeOffset.UtcNow.ToString("o"),
                eventCount = _eventCount,
                leaderboard = GetLeaderboard().Select(e => new
                {
                    rank       = e.Rank,
                    name       = e.Name,
                    fold       = e.Fold,
                    category   = e.Category,
                    score      = Math.Round(e.Score, 3),
                    invocations = e.Invocations,
                    successRate = Math.Round(e.SuccessRate, 3),
                    projects   = e.Projects.OrderBy(p => p).ToList(),
                    confidence = e.Confidence,
                })
            };
            File.WriteAllText(path,
                JsonSerializer.Serialize(snapshot, new JsonSerializerOptions { WriteIndented = true }),
                Encoding.UTF8);
        }

        // ── Internal ──────────────────────────────────────────────────────────

        private long _eventCount;

        private void Recalculate()
        {
            int maxProjects = _entries.Values.Select(e => e.Projects.Count).DefaultIfEmpty(0).Max();
            var ranked = _entries.Values.OrderByDescending(e => { e.Recalculate(maxProjects); return e.Score; }).ToList();
            for (int i = 0; i < ranked.Count; i++) ranked[i].Rank = i + 1;
        }

        // Append a hash-chained event to .µleaderboard.jsonl (same pattern as JROM/IDB).
        private void AppendEvent(BoardEvent ev)
        {
            string payload = JsonSerializer.Serialize(ev, new JsonSerializerOptions { WriteIndented = false });
            byte[] hashBytes = SHA256.HashData(
                Encoding.UTF8.GetBytes(_prevHash + "|" + payload));
            ev.Hash = Convert.ToHexString(hashBytes).ToLowerInvariant();
            _prevHash = ev.Hash;
            _eventCount++;

            File.AppendAllText(_logPath,
                JsonSerializer.Serialize(ev, new JsonSerializerOptions { WriteIndented = false }) + "\n",
                Encoding.UTF8);
        }
    }

    // ── Quick smoke test (run as: dotnet script µLeaderBoard.cs) ─────────────

    internal static class MuLeaderBoardSmoke
    {
        internal static void Run(string registryPath = "micronauts/registry.json",
                                  string logPath     = ".µleaderboard-test.jsonl")
        {
            if (File.Exists(logPath)) File.Delete(logPath);
            var board = MuLeaderBoard.Load(registryPath, logPath);

            // Simulate submissions from three projects
            foreach (var (mu, ok, proj) in new (string, bool, string)[] {
                ("eliza",   true,  "PRIMEOS"),  ("eliza",   true,  "PRIMEOS"),
                ("coder",   true,  "PRIMEOS"),  ("coder",   false, "PRIMEOS"),
                ("eliza",   true,  "kxc-v1.0.0"),
                ("think",   true,  "PRIMEOS"),  ("think",   true,  "kxc-v1.0.0"),
                ("ast",     true,  "PRIMEOS"),  ("ast",     true,  "kxc-v1.0.0"), ("ast", true, "my-game"),
                ("tool",    true,  "PRIMEOS"),  ("tool",    false, "my-game"),
                ("sek",     true,  "kxc-v1.0.0"),
                ("factory", true,  "PRIMEOS"),
                ("memory",  true,  "my-game"),  ("memory",  true,  "my-game"),
            })
                board.Submit(mu, proj, ok);

            board.PrintLeaderboard(10);
            board.PrintFoldSummary();
            board.SaveSnapshot(".µleaderboard-snapshot.json");
            Console.WriteLine($"\nPromotion candidates (≥3 projects): {board.GetPromotionCandidates(3).Count}");

            if (File.Exists(logPath)) File.Delete(logPath);
            if (File.Exists(".µleaderboard-snapshot.json")) File.Delete(".µleaderboard-snapshot.json");
            Console.WriteLine("smoke OK");
        }
    }
}
