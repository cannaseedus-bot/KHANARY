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
2.2e-06**, next-token argmax **matches**. (This proves the path/integration; it is not yet an
optimized runtime — the DLL syncs the GPU per matmul.)

