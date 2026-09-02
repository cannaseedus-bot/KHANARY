// ============================================================
// K'UHUL Fold-Geometry System
// C# implementation: π-phase attention bounds
// A_ij = G(Δθ_ij) · S(Q_i, K_j)
// ============================================================

using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;

namespace KUHUL.FoldGeometry
{
    // ============================================================
    // 1. CORE TYPES
    // ============================================================

    /// <summary>
    /// Six-fold phase angle (0, π/3, 2π/3, π, 4π/3, 5π/3)
    /// </summary>
    public enum PhaseAngle
    {
        Pop   = 0,   // 0      — observe / Q-read
        Wo    = 1,   // π/3    — weight / mask
        Yax   = 2,   // 2π/3  — enumerate / K-read
        Sek   = 3,   // π      — compute / QKᵀ
        Chen  = 4,   // 4π/3  — collect / V-gather
        Xul   = 5    // 5π/3  — entropy / output-project
    }

    public enum FoldType { Pop, Wo, Yax, Sek, Chen, Xul }

    public struct FoldPosition
    {
        public double Theta { get; set; }
        public double R     { get; set; }
        public double Z     { get; set; }

        public static FoldPosition FromPhase(PhaseAngle phase, double radius = 1.0, double depth = 0.0)
        {
            double theta = (int)phase * (Math.PI / 3.0);
            return new FoldPosition { Theta = theta, R = radius, Z = depth };
        }

        public Vector3 ToVector3() => new Vector3(
            (float)(R * Math.Cos(Theta)),
            (float)(R * Math.Sin(Theta)),
            (float)Z
        );
    }

    public class FoldGeometry
    {
        public PhaseAngle Phase    { get; set; }
        public FoldType FoldType   => (FoldType)(int)Phase;
        public FoldPosition Position { get; set; }
        public double Radius       { get; set; } = 1.0;

        public FoldGeometry(PhaseAngle phase, double radius = 1.0, double depth = 0.0)
        {
            Phase    = phase;
            Radius   = radius;
            Position = FoldPosition.FromPhase(phase, radius, depth);
        }

        public double Theta => Position.Theta;
        public double R     => Position.R;
        public double Z     => Position.Z;
    }

    public class Token
    {
        public string         Id       { get; set; }
        public float[]        Vector   { get; set; }
        public FoldGeometry   Geometry { get; set; }
        public TokenMetadata  Metadata { get; set; }

        public Token(string id, float[] vector, PhaseAngle phase, double radius = 1.0)
        {
            Id       = id;
            Vector   = vector;
            Geometry = new FoldGeometry(phase, radius);
            Metadata = new TokenMetadata { FoldPhase = phase };
        }
    }

    public class TokenMetadata
    {
        public PhaseAngle                 FoldPhase   { get; set; }
        public string                     TokenType   { get; set; } = "input";
        public float                      Confidence  { get; set; } = 1.0f;
        public Dictionary<string, object> Attributes  { get; set; } = new Dictionary<string, object>();
    }

    // ============================================================
    // 2. ATTENTION RESULT
    // ============================================================

    public class AttentionResult
    {
        public Token                    Output           { get; set; }
        public List<AttentionWeight>    AttentionWeights { get; set; }
        public List<Token>              AdmittedTokens   { get; set; }
        public List<double>             GeometricWeights { get; set; }
        public List<double>             SemanticWeights  { get; set; }
        public double                   Entropy          { get; set; }
        public double                   Sparsity         { get; set; }
    }

    public class AttentionWeight
    {
        public Token  Token           { get; set; }
        public double Weight          { get; set; }
        public double GeometricWeight { get; set; }
        public double SemanticWeight  { get; set; }
    }

    // ============================================================
    // 3. FOLD ATTENTION SYSTEM
    // ============================================================

    public class FoldAttentionSystem
    {
        private readonly Dictionary<string, FoldGeometry>         _phaseMap    = new Dictionary<string, FoldGeometry>();
        private readonly Dictionary<PhaseAngle, List<string>>     _foldBuckets = new Dictionary<PhaseAngle, List<string>>();

        public FoldAttentionSystem()
        {
            foreach (PhaseAngle phase in Enum.GetValues(typeof(PhaseAngle)))
                _foldBuckets[phase] = new List<string>();
        }

        public PhaseAngle GetPhase(string tokenId)
            => _phaseMap.TryGetValue(tokenId, out var g) ? g.Phase : PhaseAngle.Pop;

        public FoldGeometry GetGeometry(string tokenId)
            => _phaseMap.TryGetValue(tokenId, out var g) ? g : null;

        public double AngularSeparation(PhaseAngle theta1, PhaseAngle theta2)
            => WrapAngle(((int)theta1 - (int)theta2) * (Math.PI / 3.0));

        private static double WrapAngle(double angle)
        {
            const double twoPi = 2.0 * Math.PI;
            double w = angle % twoPi;
            if (w >  Math.PI) w -= twoPi;
            if (w < -Math.PI) w += twoPi;
            return w;
        }

        /// <summary>G(Δθ) = cos(Δθ)</summary>
        public double GeometricRelevance(PhaseAngle theta1, PhaseAngle theta2)
            => Math.Cos(AngularSeparation(theta1, theta2));

        public bool IsGeometricallyRelated(PhaseAngle theta1, PhaseAngle theta2, double threshold = 0.5)
            => GeometricRelevance(theta1, theta2) > threshold;

        public List<string> GetFoldNeighborhood(string tokenId, double threshold = 0.5)
        {
            var theta     = GetPhase(tokenId);
            var neighbors = new List<string>();
            foreach (var kvp in _phaseMap)
            {
                if (kvp.Key == tokenId) continue;
                if (GeometricRelevance(theta, kvp.Value.Phase) > threshold)
                    neighbors.Add(kvp.Key);
            }
            return neighbors;
        }

        public void RegisterToken(string tokenId, FoldGeometry geometry)
        {
            _phaseMap[tokenId] = geometry;
            _foldBuckets[geometry.Phase].Add(tokenId);
        }

        public void RegisterTokens(IEnumerable<Token> tokens)
        {
            foreach (var t in tokens) RegisterToken(t.Id, t.Geometry);
        }

        public void RemoveToken(string tokenId)
        {
            if (_phaseMap.TryGetValue(tokenId, out var geometry))
            {
                _phaseMap.Remove(tokenId);
                _foldBuckets[geometry.Phase].Remove(tokenId);
            }
        }

        public List<string> GetTokensInFold(PhaseAngle phase)
            => _foldBuckets.TryGetValue(phase, out var t) ? new List<string>(t) : new List<string>();

        public FoldStatistics GetStatistics()
        {
            var stats = new FoldStatistics();
            foreach (var kvp in _foldBuckets)
            {
                stats.Counts[kvp.Key] = kvp.Value.Count;
                stats.Total += kvp.Value.Count;
            }
            stats.Entropy = ComputeEntropy(stats.Counts.Values);
            return stats;
        }

        private static double ComputeEntropy(IEnumerable<int> counts)
        {
            int total = counts.Sum();
            if (total == 0) return 0;
            double entropy = 0;
            foreach (int c in counts)
            {
                if (c > 0) { double p = (double)c / total; entropy -= p * Math.Log(p, 2); }
            }
            return entropy;
        }
    }

    public class FoldStatistics
    {
        public Dictionary<PhaseAngle, int> Counts  { get; set; } = new Dictionary<PhaseAngle, int>();
        public int    Total   { get; set; }
        public double Entropy { get; set; }
        public double Balance => Entropy / Math.Log(6, 2);
    }

    // ============================================================
    // 4. SEMANTIC ATTENTION (QKV)
    // ============================================================

    public class SemanticAttention
    {
        private readonly float[][] _Wq;
        private readonly float[][] _Wk;
        private readonly float[][] _Wv;
        private readonly Random    _random = new Random();

        public SemanticAttention(int dModel = 512, int dK = 64)
        {
            _Wq = InitMatrix(dModel, dK);
            _Wk = InitMatrix(dModel, dK);
            _Wv = InitMatrix(dModel, dK);
        }

        private float[][] InitMatrix(int rows, int cols)
        {
            var m = new float[rows][];
            for (int i = 0; i < rows; i++)
            {
                m[i] = new float[cols];
                for (int j = 0; j < cols; j++)
                    m[i][j] = (float)((_random.NextDouble() - 0.5) * 0.02);
            }
            return m;
        }

        public float[] Compute(Token query, Token[] keys)
        {
            float[]   Q      = Project(query.Vector, _Wq);
            float[][] Ks     = keys.Select(k => Project(k.Vector, _Wk)).ToArray();
            float[]   scores = new float[Ks.Length];
            for (int i = 0; i < Ks.Length; i++)
                scores[i] = Dot(Q, Ks[i]);
            return scores;
        }

        private static float[] Project(float[] v, float[][] W)
        {
            float[] r = new float[W[0].Length];
            for (int i = 0; i < W.Length; i++)
                for (int j = 0; j < W[i].Length; j++)
                    r[j] += v[i] * W[i][j];
            return r;
        }

        private static float Dot(float[] a, float[] b)
        {
            float s = 0;
            for (int i = 0; i < a.Length; i++) s += a[i] * b[i];
            return s;
        }
    }

    // ============================================================
    // 5. FOLD-BOUNDED ATTENTION
    // ============================================================

    public class FoldBoundedAttention
    {
        private readonly FoldAttentionSystem _foldSystem;
        private readonly SemanticAttention   _semanticAttention;

        public FoldAttentionSystem FoldSystem => _foldSystem;

        public FoldBoundedAttention(FoldAttentionSystem foldSystem = null, SemanticAttention semanticAttention = null)
        {
            _foldSystem        = foldSystem        ?? new FoldAttentionSystem();
            _semanticAttention = semanticAttention ?? new SemanticAttention();
        }

        /// <summary>A_ij = G(Δθ_ij) · S(Q_i, K_j)</summary>
        public AttentionResult Attend(Token query, Token[] keys, Token[] values, double threshold = 0.5)
        {
            PhaseAngle queryTheta = _foldSystem.GetPhase(query.Id);

            var admittedTokens   = new List<Token>();
            var geometricWeights = new List<double>();

            foreach (var key in keys)
            {
                double g = _foldSystem.GeometricRelevance(queryTheta, _foldSystem.GetPhase(key.Id));
                if (g > threshold) { admittedTokens.Add(key); geometricWeights.Add(g); }
            }

            if (admittedTokens.Count == 0)
            {
                admittedTokens   = new List<Token>(keys);
                geometricWeights = admittedTokens
                    .Select(k => _foldSystem.GeometricRelevance(queryTheta, _foldSystem.GetPhase(k.Id)))
                    .ToList();
            }

            float[] rawSemantic  = _semanticAttention.Compute(query, admittedTokens.ToArray());
            var semanticScores   = rawSemantic.Select(s => (double)s).ToList();

            var finalWeights = new List<AttentionWeight>();
            for (int i = 0; i < admittedTokens.Count; i++)
            {
                finalWeights.Add(new AttentionWeight
                {
                    Token           = admittedTokens[i],
                    Weight          = semanticScores[i] * geometricWeights[i],
                    GeometricWeight = geometricWeights[i],
                    SemanticWeight  = semanticScores[i]
                });
            }

            Softmax(finalWeights);
            Token  output  = WeightedSum(finalWeights, values);
            double entropy = Entropy(finalWeights);

            return new AttentionResult
            {
                Output           = output,
                AttentionWeights = finalWeights,
                AdmittedTokens   = admittedTokens,
                GeometricWeights = geometricWeights,
                SemanticWeights  = semanticScores,
                Entropy          = entropy,
                Sparsity         = Sparsity(finalWeights)
            };
        }

        public List<AttentionResult> AttendBatch(Token[] queries, Token[] keys, Token[] values, double threshold = 0.5)
            => queries.Select(q => Attend(q, keys, values, threshold)).ToList();

        private static void Softmax(List<AttentionWeight> weights)
        {
            double maxW  = weights.Max(w => w.Weight);
            double sumExp = weights.Sum(w => Math.Exp(w.Weight - maxW));
            foreach (var w in weights)
                w.Weight = Math.Exp(w.Weight - maxW) / sumExp;
        }

        // internal so XCFEFoldAttention in the same assembly can call it
        internal static Token WeightedSum(List<AttentionWeight> weights, Token[] values)
        {
            int     dim    = values[0].Vector.Length;
            float[] result = new float[dim];
            for (int i = 0; i < weights.Count; i++)
            {
                float[] vec = values[i % values.Length].Vector;
                double  w   = weights[i].Weight;
                for (int j = 0; j < dim; j++)
                    result[j] += (float)(w * vec[j]);
            }
            return new Token("attended", result, PhaseAngle.Pop)
            {
                Metadata = new TokenMetadata
                {
                    FoldPhase  = PhaseAngle.Pop,
                    TokenType  = "attended",
                    Confidence = (float)(1.0 - Entropy(weights))
                }
            };
        }

        internal static double Entropy(List<AttentionWeight> weights)
        {
            double e = 0;
            foreach (var w in weights)
                if (w.Weight > 0) e -= w.Weight * Math.Log(w.Weight, 2);
            return e;
        }

        internal static double Sparsity(List<AttentionWeight> weights)
        {
            if (weights.Count == 0) return 0;
            double mean = weights.Sum(w => w.Weight) / weights.Count;
            double var_ = weights.Sum(w => Math.Pow(w.Weight - mean, 2)) / weights.Count;
            return 1.0 - Math.Sqrt(var_) / (mean + 1e-8);
        }
    }

    // ============================================================
    // 6. FOLD OPTIMIZER
    // ============================================================

    public class FoldOptimizer
    {
        private readonly FoldAttentionSystem _foldSystem;

        public FoldOptimizer(FoldAttentionSystem foldSystem = null)
            => _foldSystem = foldSystem ?? new FoldAttentionSystem();

        public PhaseAngle AssignFold(Token token, List<FoldGeometry> existingFolds)
        {
            double    bestScore = double.NegativeInfinity;
            PhaseAngle bestPhase = PhaseAngle.Pop;
            for (int phase = 0; phase < 6; phase++)
            {
                double score = existingFolds.Sum(f =>
                    Math.Cos((phase - (int)f.Phase) * (Math.PI / 3.0)));
                if (score > bestScore) { bestScore = score; bestPhase = (PhaseAngle)phase; }
            }
            return bestPhase;
        }

        public Dictionary<PhaseAngle, List<Token>> BalanceFolds(Token[] tokens)
        {
            var map = new Dictionary<PhaseAngle, List<Token>>();
            for (int i = 0; i < 6; i++) map[(PhaseAngle)i] = new List<Token>();
            foreach (var t in tokens) map[t.Geometry.Phase].Add(t);
            return map;
        }

        public bool Rebalance(Token[] tokens)
        {
            var map           = BalanceFolds(tokens);
            int targetPerFold = (int)Math.Ceiling(tokens.Length / 6.0);
            bool rebalanced   = false;

            var overfull  = map.Where(kvp => kvp.Value.Count > targetPerFold + 1)
                              .OrderByDescending(kvp => kvp.Value.Count).ToList();
            var underfull = map.Where(kvp => kvp.Value.Count < targetPerFold - 1)
                              .OrderBy(kvp => kvp.Value.Count).ToList();

            foreach (var over in overfull)
            {
                foreach (var under in underfull)
                {
                    if (over.Value.Count > targetPerFold + 1 && under.Value.Count < targetPerFold - 1)
                    {
                        var token = over.Value[over.Value.Count - 1];
                        over.Value.RemoveAt(over.Value.Count - 1);
                        under.Value.Add(token);
                        token.Geometry = new FoldGeometry(under.Key);
                        rebalanced = true;
                    }
                }
            }
            return rebalanced;
        }

        public double FoldEntropy(Token[] tokens)
        {
            var counts = new int[6];
            foreach (var t in tokens) counts[(int)t.Geometry.Phase]++;
            int total = tokens.Length;
            double entropy = 0;
            foreach (int c in counts)
                if (c > 0) { double p = (double)c / total; entropy -= p * Math.Log(p, 2); }
            return entropy;
        }
    }

    // ============================================================
    // 7. XCFE CONTROL
    // ============================================================

    public class XCFEControl
    {
        private readonly Dictionary<string, object>                  _state    = new Dictionary<string, object>();
        private readonly Dictionary<string, Func<object, object>>    _handlers = new Dictionary<string, Func<object, object>>();

        public async Task<T> Evaluate<T>(string expression)
        {
            await Task.Delay(1);
            switch (expression)
            {
                case "hasFoldGeometry && active":
                    return (T)(object)(bool)(_state.ContainsKey("foldGeometry") && GetState<bool>("active"));
                case "getFoldThreshold":
                    return (T)(object)(double)0.5;
                default:
                    return default(T);
            }
        }

        public async Task Branch(string condition, string target)
        {
            await Task.Delay(1);
            _state["branch"] = new { condition, target };
        }

        public async Task<T> Schedule<T>(string operation, object data)
        {
            await Task.Delay(1);
            return (T)data;
        }

        public void SetState<T>(string key, T value)  => _state[key] = value;
        public T    GetState<T>(string key)
            => _state.TryGetValue(key, out var v) ? (T)v : default(T);

        public void RegisterHandler(string name, Func<object, object> handler)
            => _handlers[name] = handler;
        public object CallHandler(string name, object input)
            => _handlers.TryGetValue(name, out var h) ? h(input) : null;
    }

    // ============================================================
    // 8. XCFE FOLD ATTENTION INTEGRATION
    // ============================================================

    public class XCFEFoldAttention
    {
        private readonly FoldBoundedAttention _attention;
        private readonly XCFEControl          _xcfe;

        public FoldBoundedAttention Attention => _attention;

        public XCFEFoldAttention(XCFEControl xcfe = null)
        {
            _attention = new FoldBoundedAttention();
            _xcfe      = xcfe ?? new XCFEControl();
        }

        public async Task<AttentionResult> ExecuteAttention(
            Token query, Token[] keys, Token[] values, double threshold = 0.5)
        {
            // Explicit type args required — C# can't infer T from return-type assignment alone
            bool useFoldGeometry = await _xcfe.Evaluate<bool>("hasFoldGeometry && active");

            if (!useFoldGeometry)
                return StandardAttention(query, keys, values);

            double admissionThreshold = await _xcfe.Evaluate<double>("getFoldThreshold");

            var result = _attention.Attend(query, keys, values, admissionThreshold);

            if (result.AdmittedTokens.Count == 0)
                return StandardAttention(query, keys, values);

            if (result.AdmittedTokens.Count < 3)
                await _xcfe.Branch("sparseAttention", "useDenseFallback");

            return await _xcfe.Schedule<AttentionResult>("executeAttention", result);
        }

        private AttentionResult StandardAttention(Token query, Token[] keys, Token[] values)
        {
            var     semantic = new SemanticAttention();
            float[] scores   = semantic.Compute(query, keys);
            float   maxScore = scores.Max();
            float   sumExp   = scores.Sum(s => (float)Math.Exp(s - maxScore));

            var weights = new List<AttentionWeight>();
            for (int i = 0; i < keys.Length; i++)
            {
                float w = (float)Math.Exp(scores[i] - maxScore) / sumExp;
                weights.Add(new AttentionWeight
                {
                    Token           = keys[i],
                    Weight          = w,
                    GeometricWeight = 1.0,
                    SemanticWeight  = scores[i]
                });
            }

            // internal static — accessible within the assembly
            Token output = FoldBoundedAttention.WeightedSum(weights, values);

            return new AttentionResult
            {
                Output           = output,
                AttentionWeights = weights,
                AdmittedTokens   = new List<Token>(keys),
                GeometricWeights = weights.Select(_ => 1.0).ToList(),
                SemanticWeights  = weights.Select(w => w.SemanticWeight).ToList(),
                Entropy          = FoldBoundedAttention.Entropy(weights),
                Sparsity         = FoldBoundedAttention.Sparsity(weights)
            };
        }
    }

    // ============================================================
    // 9. DEMO ENTRY POINT
    // ============================================================

    public static class Program
    {
        public static async Task Main()
        {
            Console.WriteLine("K'UHUL Fold-Geometry Attention System");
            Console.WriteLine("======================================\n");

            var foldSystem = new FoldAttentionSystem();
            var random     = new Random();

            var tokens = new[]
            {
                new Token("token-A", new float[512], PhaseAngle.Pop),
                new Token("token-B", new float[512], PhaseAngle.Wo),
                new Token("token-C", new float[512], PhaseAngle.Chen),
                new Token("token-D", new float[512], PhaseAngle.Yax),
                new Token("token-E", new float[512], PhaseAngle.Xul),
                new Token("token-F", new float[512], PhaseAngle.Sek)
            };

            foreach (var t in tokens)
                for (int i = 0; i < t.Vector.Length; i++)
                    t.Vector[i] = (float)(random.NextDouble() * 2 - 1);

            foldSystem.RegisterTokens(tokens);

            Console.WriteLine("Fold Statistics:");
            var stats = foldSystem.GetStatistics();
            foreach (var kvp in stats.Counts)
                Console.WriteLine($"  {kvp.Key,6}: {kvp.Value} tokens");
            Console.WriteLine($"  Entropy: {stats.Entropy:F3}   Balance: {stats.Balance:F3}\n");

            var attention = new FoldBoundedAttention(foldSystem);
            var query     = new Token("query", new float[512], PhaseAngle.Pop);
            for (int i = 0; i < query.Vector.Length; i++)
                query.Vector[i] = (float)(random.NextDouble() * 2 - 1);

            foldSystem.RegisterToken(query.Id, query.Geometry);

            Console.WriteLine("Computing Fold-Bounded Attention...");
            var result = attention.Attend(query, tokens, tokens, threshold: 0.5);

            Console.WriteLine($"Admitted: {string.Join(", ", result.AdmittedTokens.Select(t => t.Id))}");
            foreach (var w in result.AttentionWeights)
                Console.WriteLine($"  {w.Token.Id}: {w.Weight:F4} (G:{w.GeometricWeight:F4} S:{w.SemanticWeight:F4})");
            Console.WriteLine($"Entropy: {result.Entropy:F4}   Sparsity: {result.Sparsity:F4}\n");

            Console.WriteLine("Geometric Relationships:");
            foreach (var (id1, id2) in new[] { ("token-A","token-B"), ("token-A","token-D"), ("token-A","token-E") })
            {
                double g  = foldSystem.GeometricRelevance(foldSystem.GetPhase(id1), foldSystem.GetPhase(id2));
                double dt = foldSystem.AngularSeparation(foldSystem.GetPhase(id1), foldSystem.GetPhase(id2));
                Console.WriteLine($"  {id1} <-> {id2}: G={g:F4} dt={dt:F4} rad");
            }

            Console.WriteLine("\nFold Neighborhoods:");
            foreach (var t in tokens)
            {
                var nb = foldSystem.GetFoldNeighborhood(t.Id);
                Console.WriteLine($"  {t.Id} ({t.Geometry.Phase}): {string.Join(", ", nb)}");
            }

            Console.WriteLine("\nXCFE Integration:");
            var xcfe = new XCFEControl();
            xcfe.SetState("foldGeometry", true);
            xcfe.SetState("active", true);
            var xcfeAttention = new XCFEFoldAttention(xcfe);
            var xcfeResult    = await xcfeAttention.ExecuteAttention(query, tokens, tokens);
            Console.WriteLine($"  Admitted: {string.Join(", ", xcfeResult.AdmittedTokens.Select(t => t.Id))}");
            Console.WriteLine($"  Entropy:  {xcfeResult.Entropy:F4}");

            Console.WriteLine("\nFold Optimization:");
            var optimizer = new FoldOptimizer(foldSystem);
            Console.WriteLine($"  Initial entropy: {optimizer.FoldEntropy(tokens):F4}");
            bool rebalanced = optimizer.Rebalance(tokens);
            Console.WriteLine($"  Rebalanced: {rebalanced}");
            Console.WriteLine($"  Final entropy:   {optimizer.FoldEntropy(tokens):F4}");

            Console.WriteLine("\nDone.");
        }
    }
}
