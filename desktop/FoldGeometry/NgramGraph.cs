using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace KUHUL.FoldGeometry
{
    public class NgramNode
    {
        public string     Id        { get; set; }
        public string[]   Parts     { get; set; }
        public int        Frequency { get; set; }
        public PhaseAngle Phase     { get; set; }
        public double     Theta     => (int)Phase * (Math.PI / 3.0);
    }

    public class NgramEdge
    {
        public string From            { get; set; }
        public string To              { get; set; }
        public int    TransitionCount { get; set; }
        public double GeometricWeight { get; set; }  // G(Δθ) = cos(θ_i − θ_j)
        public double CombinedWeight  { get; set; }  // TransitionCount × max(0, G)
    }

    public class NgramGraphStats
    {
        public int    NodeCount        { get; set; }
        public int    EdgeCount        { get; set; }
        public int    TokenCount       { get; set; }
        public int    NgramCount       { get; set; }
        public double Entropy          { get; set; }
        public double AvgDegree        { get; set; }
        public double GeometricDensity { get; set; }
        public Dictionary<PhaseAngle, int> PhaseDistribution { get; set; }
    }

    public class NgramGraphData
    {
        public int            N         { get; set; }
        public bool           CharLevel { get; set; }
        public List<NgramNode> Nodes    { get; set; }
        public List<NgramEdge> Edges    { get; set; }
        public NgramGraphStats Stats    { get; set; }
        public Dictionary<string, List<NgramEdge>> Adjacency { get; set; }

        public List<NgramEdge>      OutEdges(string nodeId)
            => Adjacency.TryGetValue(nodeId, out var e) ? e : new List<NgramEdge>();

        public IEnumerable<NgramNode> GetPhaseGroup(PhaseAngle phase)
            => Nodes.Where(n => n.Phase == phase);

        // Walk the graph from a start node up to maxDepth hops, returning
        // nodes in descending combined-weight order at each step.
        public List<NgramNode> Walk(string startId, int maxDepth = 3)
        {
            var visited = new HashSet<string>();
            var result  = new List<NgramNode>();
            var queue   = new Queue<(string id, int depth)>();
            queue.Enqueue((startId, 0));
            while (queue.Count > 0)
            {
                var (id, depth) = queue.Dequeue();
                if (!visited.Add(id)) continue;
                var node = Nodes.FirstOrDefault(n => n.Id == id);
                if (node != null) result.Add(node);
                if (depth >= maxDepth) continue;
                foreach (var edge in OutEdges(id).OrderByDescending(e => e.CombinedWeight))
                    queue.Enqueue((edge.To, depth + 1));
            }
            return result;
        }
    }

    public static class NgramGraphBuilder
    {
        // Build an n-gram graph from plain text.
        // n          — n-gram width (default 2)
        // charLevel  — true → character n-grams, false → word n-grams
        // threshold  — minimum CombinedWeight for an edge to be retained
        public static NgramGraphData Build(
            string text,
            int    n          = 2,
            bool   charLevel  = false,
            double threshold  = 0.0)
        {
            string[] tokens = charLevel
                ? text.Select(c => c.ToString()).ToArray()
                : Regex.Matches(text.ToLowerInvariant(), @"\w+")
                       .Cast<Match>()
                       .Select(m => m.Value)
                       .ToArray();

            var ngrams = new List<string>(tokens.Length);
            string sep = charLevel ? "" : " ";
            for (int i = 0; i <= tokens.Length - n; i++)
                ngrams.Add(string.Join(sep, tokens.Skip(i).Take(n)));

            // node frequency counts
            var nodeCounts = new Dictionary<string, int>(ngrams.Count);
            foreach (var ng in ngrams)
                nodeCounts[ng] = nodeCounts.TryGetValue(ng, out int c) ? c + 1 : 1;

            // directed transition counts
            var edgeCounts = new Dictionary<(string, string), int>();
            for (int i = 0; i < ngrams.Count - 1; i++)
            {
                var key = (ngrams[i], ngrams[i + 1]);
                edgeCounts[key] = edgeCounts.TryGetValue(key, out int c) ? c + 1 : 1;
            }

            var nodes = nodeCounts.Select(kvp => new NgramNode
            {
                Id        = kvp.Key,
                Parts     = charLevel ? new[] { kvp.Key } : kvp.Key.Split(' '),
                Frequency = kvp.Value,
                Phase     = DeterministicPhase(kvp.Key)
            }).ToList();

            var edges = new List<NgramEdge>(edgeCounts.Count);
            foreach (var kvp in edgeCounts)
            {
                double gw       = PhaseGeometricWeight(DeterministicPhase(kvp.Key.Item1),
                                                       DeterministicPhase(kvp.Key.Item2));
                double combined = kvp.Value * Math.Max(0.0, gw);
                if (combined >= threshold)
                {
                    edges.Add(new NgramEdge
                    {
                        From            = kvp.Key.Item1,
                        To              = kvp.Key.Item2,
                        TransitionCount = kvp.Value,
                        GeometricWeight = gw,
                        CombinedWeight  = combined
                    });
                }
            }
            edges = edges.OrderByDescending(e => e.CombinedWeight).ToList();

            var adjacency = new Dictionary<string, List<NgramEdge>>();
            foreach (var e in edges)
            {
                if (!adjacency.TryGetValue(e.From, out var list))
                    adjacency[e.From] = list = new List<NgramEdge>();
                list.Add(e);
            }

            return new NgramGraphData
            {
                N         = n,
                CharLevel = charLevel,
                Nodes     = nodes,
                Edges     = edges,
                Stats     = ComputeStats(nodes, edges, tokens.Length, ngrams.Count),
                Adjacency = adjacency
            };
        }

        // Assign a fold phase deterministically from the n-gram string.
        // Polynomial rolling hash mod 6 → same n-gram always maps to the same phase.
        private static PhaseAngle DeterministicPhase(string ngram)
        {
            int hash = ngram.Aggregate(0, (acc, c) => unchecked(acc * 31 + c));
            return (PhaseAngle)(Math.Abs(hash) % 6);
        }

        // G(Δθ) = cos(Δθ) with Δθ wrapped to [−π, π]
        private static double PhaseGeometricWeight(PhaseAngle p1, PhaseAngle p2)
        {
            double dt = ((int)p1 - (int)p2) * (Math.PI / 3.0);
            while (dt >  Math.PI) dt -= 2.0 * Math.PI;
            while (dt < -Math.PI) dt += 2.0 * Math.PI;
            return Math.Cos(dt);
        }

        private static NgramGraphStats ComputeStats(
            List<NgramNode> nodes, List<NgramEdge> edges,
            int tokenCount, int ngramCount)
        {
            var phaseDist = new Dictionary<PhaseAngle, int>();
            foreach (PhaseAngle p in Enum.GetValues(typeof(PhaseAngle)))
                phaseDist[p] = 0;
            foreach (var node in nodes)
                phaseDist[node.Phase]++;

            double totalFreq = nodes.Sum(n => (double)n.Frequency);
            double entropy   = 0;
            if (totalFreq > 0)
                foreach (var node in nodes)
                {
                    double p = node.Frequency / totalFreq;
                    if (p > 0) entropy -= p * Math.Log(p, 2);
                }

            double avgDegree = nodes.Count > 0 ? edges.Count * 2.0 / nodes.Count : 0;
            double avgGW     = edges.Count > 0 ? edges.Average(e => e.GeometricWeight) : 0;

            return new NgramGraphStats
            {
                NodeCount         = nodes.Count,
                EdgeCount         = edges.Count,
                TokenCount        = tokenCount,
                NgramCount        = ngramCount,
                Entropy           = entropy,
                AvgDegree         = avgDegree,
                GeometricDensity  = avgGW,
                PhaseDistribution = phaseDist
            };
        }
    }
}
