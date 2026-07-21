# DirectML GEMM benchmark (HD 4600)

Measures DirectML's `DML_OPERATOR_GEMM` against KHΛNARY's tiled `cs_5_0` kernel on the exact
gpt2 QKV shape (`C[64,2304] = A[64,768] @ B[768,2304]`), reusing the same `../gemm_A/B/Cref.bin`
and 100-iter one-submit timing model as `../matmul_run.cpp` so the numbers compare directly.

## Result (Intel HD 4600, FL 11_1)

| Path | ms/iter | vs tiled |
|---|---|---|
| **DirectML `DML_OPERATOR_GEMM`** (D3D12) | **~1.9** | **~4.9× faster** |
| Tiled `cs_5_0` 16×16 groupshared (D3D11) | ~9.3 | 1× |
| Naive `cs_5_0` (D3D11) | ~32 | 0.3× |

Correct to scale-normalized `1.01e-06` vs numpy f64. DirectML runs on this FL 11_1 tier-1 iGPU
(both the 1.15.4 NuGet redist and the 2020 inbox `directml.dll`). **Caveat:** the tiled kernel
is timed on D3D11 and DML on D3D12 — different dispatch paths, so the ratio is indicative, but
the ~4.9× gap dwarfs any dispatch-overhead difference.

**Gotcha:** DML GEMM has 3 input slots `[A, B, C]`. With `CTensor = nullptr` you must still
`BindInputs(3, …)` with the C slot as `DML_BINDING_TYPE_NONE`, or the dispatch silently no-ops
(zero output, ~0 ms).

## Build & run (the DirectML redist is NOT committed — fetch it first)

The header/lib/dll come from the `Microsoft.AI.DirectML` NuGet package (gitignored here):

```bash
# from scratch/dml/, with python + curl available:
curl -s "https://api.nuget.org/v3-flatcontainer/microsoft.ai.directml/1.15.4/microsoft.ai.directml.1.15.4.nupkg" -o dml.nupkg
python -c "import zipfile,shutil,os; z=zipfile.ZipFile('dml.nupkg'); os.makedirs('include',exist_ok=True); os.makedirs('lib',exist_ok=True); [shutil.copyfileobj(z.open(s),open(d,'wb')) for s,d in [('include/DirectML.h','include/DirectML.h'),('include/DirectMLConfig.h','include/DirectMLConfig.h'),('bin/x64-win/DirectML.lib','lib/DirectML.lib'),('bin/x64-win/DirectML.dll','lib/DirectML.dll')]]"
```

Then, from an MSVC x64 dev shell:

```
build.bat          # compiles dml_gemm_bench.exe and copies lib\DirectML.dll next to it
dml_gemm_bench.exe # run from scratch/dml/ (reads ../gemm_*.bin)
```

## DirectML matmul path wired into the inference driver (prototype)

`dml_gemm_dll.cpp` builds `dml_gemm.dll` — DirectML GEMM behind a C ABI
(`dml_gemm_f32(A,B,C,M,N,K)`, persistent D3D12+DML device, compiled operators cached by shape).
`tools/dml_gemm.py` loads it via ctypes, and `tools/kxml_inference_driver.op_matmul` routes
through it when **`KXML_DML=1`** — so `G_MATMUL` runs on DirectML across a full forward.

Build the DLL and verify it threads the whole 12-layer model:

```
cl /nologo /std:c++17 /EHsc /O2 /LD /I include dml_gemm_dll.cpp /link /LIBPATH:lib /OUT:dml_gemm.dll
python compare_driver_dml.py    # runs the driver numpy vs DirectML, compares logits + argmax
```

Result: DirectML matmul vs numpy over the full 12-layer GPT-2 driver → logits **scale-norm
2.2e-06**, next-token argmax **matches**.

### Amortization (persistent buffers + GPU-resident weights)

The DLL was progressively amortized (8-token, 12-layer forward, repeated):

| DLL version | ms/pass | vs baseline |
|---|---|---|
| per-matmul buffer alloc + 3 GPU syncs | 1300 | 1× |
| persistent per-shape buffers, single fused flush | 939 | 1.4× |
| **+ GPU-resident weights** (upload each weight once, keyed by pointer) | **447** | **2.9×** |

The dominant lever is keeping **weights resident** on the GPU across calls (they don't change
across tokens) — the driver passes the model's own `W` (stable pointer) and uses the
`dml_gemm_bt_f32` transpose-B kernel for `lm_head` so its 154 MB weight caches instead of being
re-copied every pass. Two entry points, both verified vs numpy by `test_dml_gemm.py`:

- `dml_gemm_f32(A,B,C,M,N,K)`     → `C = A @ B`
- `dml_gemm_bt_f32(A,B,C,M,N,K)`  → `C = A @ B^T` (B is `[N,K]`) = **ggml MUL_MAT** shape

`dml_gemm_bt_f32` is the same call the `ggml-xcfe` backend loads via `LoadLibrary` — one
DirectML implementation serves both the driver and llama.cpp.

## Fused MLP block on-device (activations resident)

The next lever beyond per-matmul calls: keep activations **resident on the GPU** across a whole
sub-block so they don't round-trip through the CPU between ops. `dml_mlp_run.cpp` runs a full
gpt2 MLP block — `ln → fc(+bias) → gelu → proj(+bias) → +residual` — as **five DML ops chained
into one command list** with intermediate GPU buffers and a **single flush**; only the final
output is read back.

De-risked first (`dml_ops_test.cpp`): every non-GEMM op executes correctly on the HD 4600 (FL
11_1) — GELU `3.97e-17`, LayerNorm (MVN1) `1.16e-07`, Add1 `0.0`. Two gotchas found:
`DML_OPERATOR_ELEMENT_WISE_ADD` (and therefore DirectMLX's `dml::Add`/`operator+`) fails
`E_FAIL` to compile here — use `ELEMENT_WISE_ADD1`; and newer ops need
`#define DML_TARGET_VERSION_USE_LATEST 1`.

```
python mlp_prep.py      # dumps x/weights/biases + a numpy erf-gelu reference
cl ... dml_mlp_run.cpp  # (same MSVC line as the others)
dml_mlp_run.exe
```

Result: fused MLP vs numpy → **scale-norm 1.74e-06**. Honest caveat: DirectML's GELU is the
exact/**erf** form; the KHANARY driver uses the **tanh** approximation, so the fused block
differs from the driver's MLP by **~1.3e-04** (a modeling-choice gap, not a bug).

## Fused attention block on-device

The other half of a transformer layer. `dml_attn_run.cpp` runs a full gpt2 attention block —
`ln → Q/K/V(gemm+bias) → causal MHA → proj(gemm+bias) → +residual` — as **seven DML ops in one
command list** with resident intermediates and a single flush.

De-risked first (`dml_mha_test.cpp`): `DML_OPERATOR_MULTIHEAD_ATTENTION` runs correct causal
self-attention on the HD 4600 (scale-norm 7.48e-08). Hard-won findings: device DML feature level
is 0x6200 (MHA needs 0x6100); the **stacked-QKV path fails `E_INVALIDARG`** here — use **separate
Q/K/V `{1,S,E}`**; and binding is strictly positional — MHA has **11 input slots** (Q/K/V at 0-2,
`RelativePositionBias` at 8, `NONE` elsewhere) and **3 output slots** (Output + 2 `NONE`), or it
silently returns zeros (same no-op trap as the GEMM C-slot). Causal mask via `RelativePositionBias`
`[1,Hn,S,S]` (0 / -1e9), `Scale = 1/√Hd`.

```
python attn_prep.py     # dumps x/weights/biases + numpy causal-attention reference
cl ... dml_attn_run.cpp
dml_attn_run.exe
```

Result: fused attention vs numpy → **scale-norm 9.89e-07** (S=8, E=768, 12 heads). With both
sub-blocks fused, a full transformer layer runs with activations resident across each block —
~2 GPU syncs per layer instead of ~10.

