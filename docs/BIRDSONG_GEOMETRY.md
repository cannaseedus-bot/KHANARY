# Birdsong Geometry Grammar

Bird-song formalized as **geometry + tensors**, not audio playback. It is the structural language
of the pipeline the repo already runs:

```
audio → spectrogram → ridges → mesh → graph → experts
```

## Grounded in real data

This is not illustrative — it describes
`models/khanary-geometry-v0.3.0/data/birdsong_mesh.stb` (the brain2 canary-song graph), verified by
`tools/check_birdsong.py`:

| whole-graph total | value (from the `.stb`) |
|---|---|
| nodes | **30,628** |
| Delaunay edges | **91,863** |
| CSR neighbours | **183,726** |

(The `10168 / 30491` figures in the original sketch were a different/illustrative graph — the
grammar and example use the **real** counts.) `docs/examples/birdsong.example.json` is a real
6-node subgraph extracted straight from the `.stb` by `tools/build_birdsong_example.py`.

## Three equivalent representations

| form | file | role |
|---|---|---|
| **EBNF** | `docs/birdsong-geometry.ebnf` | syntax — the language |
| **JSON dataset** | `docs/birdsong-brain.schema.json` (+ example) | human-readable twin, trainable |
| **KXML** | fold-shaped `<kuhul.brain>` (below) | execution tree |

```
EBNF  = syntax        JSON = dataset        KXML = execution tree
```

All three describe one thing: a **Geometric MoE neural substrate**.

## Maps onto the fold cycle

The graph *is* the K'UHUL tick — the same Pop→Xul lanes as the v3 grammar:

| fold | birdsong stage | KXML element |
|---|---|---|
| **Pop** | load spectrogram/audio | `<input>` |
| **Wo** | define geometry space | `<spectrum>` |
| **Yax** | extract ridges (harmonics/chirps/trills) | `<ridges>` |
| **Sek** | build the tensor graph (nodes + edges) | `<graph><nodes><edges>` |
| **Ch'en** | activate geometric-MoE experts | `<experts>` |
| **Xul** | run inference (propagate/route) | `<inference>` |

```xml
<kuhul.brain id="canary_birdsong_mesh">
  <Pop><input type="audio" sampleRate="44100" channels="1" duration="47.95"/></Pop>
  <Wo><spectrum width="2048" height="1024"/></Wo>
  <Yax><ridges>…</ridges></Yax>
  <Sek><graph layout="csr"><nodes>…</nodes><edges>…</edges></graph></Sek>
  <Chen><experts>…</experts></Chen>
  <Xul><inference mode="propagate"><rule type="diffusion" radius="2"/></inference></Xul>
</kuhul.brain>
```

## `.stb` tensor mapping (the runtime twin)

`birdsong_mesh.stb` (STB0, 7 tensors) is the binary `.brain`; the JSON is its readable twin:

| tensor | field | dtype / shape |
|---|---|---|
| t0 | node time (SVG x) | f32 [30628] |
| t1 | node freq (SVG y) | f32 [30628] |
| t2 | node energy | f32 [30628] |
| t3 | Delaunay edges | i32 [91863, 2] |
| t4 | CSR index | i32 [30629] |
| t5 | CSR neighbours | i32 [183726] |
| t6 | geometric-MoE expert id | i32 [30628] |

## Runtime semantics

```
node.energy → propagate via edges → update expert → repeat     ≡  graph neural field + MoE routing
```

The optional `Inference` production (`ACTIVATE → PROPAGATE → ROUTE → UPDATE`) turns the spec from a
data description into an execution language.

## Why this matters

Bird-song becomes a **formal language**, not just data — parseable like code, compilable into
brains, safely mergeable, and trainable **directly on geometric structure instead of text tokens**.
It plugs into the π-cluster runtime as a tensor source: `nodes → energy → propagation → experts →
weights`.

## Self-checks
```
python tools/build_birdsong_example.py   # re-extract the real example from the .stb
python tools/check_birdsong.py           # JSON validates + EBNF well-formed + totals match the .stb
```
