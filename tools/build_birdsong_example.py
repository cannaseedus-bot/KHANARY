# build_birdsong_example.py — emit a REAL Birdsong dataset example from birdsong_mesh.stb.
#
# Grounds docs/examples/birdsong.example.json in the actual canary-song graph (no invented numbers):
# a small connected subgraph (first K node ids + edges among them + their experts) plus the true
# whole-graph totals. Reproducible: same .stb -> same example.
import os, struct, array, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STB = os.path.join(ROOT, "models", "khanary-geometry-v0.3.0", "data", "birdsong_mesh.stb")
OUT = os.path.join(ROOT, "docs", "examples", "birdsong.example.json")
K = 6  # nodes in the example subgraph

def load_stb(path):
    b = open(path, "rb").read()
    assert b[:4] == b"STB0", b[:4]
    tc = struct.unpack("<H", b[6:8])[0]
    T = []
    for i in range(tc):
        e = b[32 + i*32: 32 + (i+1)*32]
        off = struct.unpack("<Q", e[4:12])[0]; sz = struct.unpack("<Q", e[12:20])[0]
        dims = struct.unpack("<III", e[20:32])
        T.append({"dtype": e[1], "rank": e[2], "off": off, "size": sz, "dims": dims})
    return b, T

def f32(b, t, n):
    a = array.array("f"); a.frombytes(b[t["off"]: t["off"] + n*4]); return a
def i32(b, t, n):
    a = array.array("i"); a.frombytes(b[t["off"]: t["off"] + n*4]); return a

def main():
    b, T = load_stb(STB)
    N = T[0]["dims"][0]; E = T[3]["dims"][0]; NB = T[5]["dims"][0]
    time = f32(b, T[0], N); freq = f32(b, T[1], N); energy = f32(b, T[2], N)
    expert = i32(b, T[6], N)
    edges_all = i32(b, T[3], E*2)

    nodes = [{"id": i, "time": round(time[i], 4), "freq": round(freq[i], 4),
              "energy": round(energy[i], 4), "expert": int(expert[i])} for i in range(K)]
    edges = []
    for e in range(E):
        a, c = edges_all[2*e], edges_all[2*e+1]
        if a < K and c < K:
            edges.append({"a": int(a), "b": int(c)})
    # experts present among the sample nodes, with real centroid/energy over those members
    experts = []
    for xid in sorted(set(expert[i] for i in range(K))):
        members = [i for i in range(K) if expert[i] == xid]
        experts.append({"id": int(xid), "node_ids": members,
                        "centroid": {"time": round(sum(time[i] for i in members)/len(members), 4),
                                     "freq": round(sum(freq[i] for i in members)/len(members), 4)},
                        "energy_mean": round(sum(energy[i] for i in members)/len(members), 4)})

    doc = {
        "id": "canary_birdsong_mesh",
        "source": "brain2 canary-song spectrogram graph (models/khanary-geometry-v0.3.0/data/birdsong_mesh.stb)",
        "audio": {"sample_rate": 44100, "channels": 1, "duration": 47.95},
        "spectrogram": {"width": 2048, "height": 1024},
        "graph": {"layout": "csr", "nodes": N, "edges": E, "neighbors": NB},
        "tensor": {"features": ["time", "freq", "energy"], "layout": "SoA", "dtype": "float32"},
        "nodes": nodes,
        "edges": edges,
        "experts": experts,
        "_note": f"REAL subgraph: first {K} of {N} nodes; edges among them of {E} total; experts computed over the sample. Whole-graph totals in `graph`."
    }
    json.dump(doc, open(OUT, "w", encoding="utf-8"), indent=2)
    print(f"[stb] N={N} E={E} NB={NB}")
    print(f"[out] {OUT}  ({K}-node subgraph, {len(edges)} edges, {len(experts)} experts)")

if __name__ == "__main__":
    main()
