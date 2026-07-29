# from_zero_v0.1 (GPT-2) -> GGUF, running in khanary-server

A KHANARY-trained GPT-2 small (124M, `trainer/from_zero_v0.1.safetensors`) now LOADS and
GENERATES in `khanary-server` on the XCFE (DirectML) backend.

## Path (why a custom converter was needed)
- The runtime supports `gpt2` (`LLM_ARCH_GPT2` in llama-arch.cpp) BUT this llama.cpp (b9968)
  dropped gpt2 from `convert_hf_to_gguf.py`.
- The trainer wrote the safetensors WITHOUT shapes (`'shape': []`) -> reconstructed from
  GPT-2 dims (n_layer=12, n_embd=768, n_head=12, n_ctx=1024, n_ff=3072, vocab=50257).
- `tools/gpt2_safetensors_to_gguf.py`: reshape flat F32, map HF names -> llama gpt2 GGUF
  names, TRANSPOSE the four Conv1D weight matrices (attn.c_attn/c_proj, mlp.c_fc/c_proj),
  copy the vetted GPT-2 vocab from `models/ggml-vocab-gpt-2.gguf`, tie output<-wte.

## Evidence
- convert: 148 in -> 149 out tensors, vocab=50257 merges=50000, 653.9 MB f32 gguf
- load: `llama_server: model loaded` with NO shape errors (llama validates every tensor
  shape against the gpt2 arch on load -> the mapping + transpose are correct)
- backend: `[ggml-xcfe] MUL_MAT path: DirectML (GPU)`
- generate: /completion returned 16 tokens; content dominated by code/shell tokens
  (`function`, `start`, `$`, backticks, `ums`/`sums`) -> reflects the model's CODE-heavy
  training corpus. v0.1 is under-trained (repetition), so this proves the
  train->convert->serve PIPELINE is faithful, not language fluency.

## What it proves (and doesn't)
Proves: a KHANARY-native trained model runs in the branded server on this rig's iGPU path.
Does NOT prove: fluency/quality -- that is a function of the corpus + training steps, which
for from_zero_v0.1 is minimal.
