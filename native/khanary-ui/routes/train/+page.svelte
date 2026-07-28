<script lang="ts">
	// KHANARY Train route — the training/quantization pipeline. Runs on the host (there is no
	// browser training endpoint on llama-server yet), so this surfaces the steps + commands.
	const steps = [
		{ n: 1, title: 'Dataset', body: 'Prepare a corpus or safetensors weights (HuggingFace, local, or a KHANARY .stb).' },
		{ n: 2, title: 'Quantize', body: 'python tools/quantize_safetensors.py --src model.safetensors --out dist  (Q4 + Q8, tensor-aligned, deterministic sha256)' },
		{ n: 3, title: 'Train / adapt', body: 'D3D11 GPT-2 trainer (khanary-gpt2) or LoRA-DDS deltas on a frozen base; micronaut meta on top.' },
		{ n: 4, title: 'Serve', body: 'Mount the result on this khanary-server — the XCFE backend runs MUL_MAT on the HD 4600 via DirectML (CPU fallback).' }
	];
</script>

<div class="mx-auto w-full max-w-3xl p-4 md:p-8">
	<h1 class="text-primary text-2xl font-bold">Train</h1>
	<p class="text-muted-foreground mt-1">
		The KHANARY training pipeline. Steps run on the host; the trained model is served here.
	</p>

	<ol class="mt-6 space-y-4">
		{#each steps as s}
			<li class="border-border rounded-md border p-4">
				<div class="text-primary font-semibold">{s.n}. {s.title}</div>
				<div class="text-muted-foreground mt-1 font-mono text-sm break-words">{s.body}</div>
			</li>
		{/each}
	</ol>

	<p class="text-muted-foreground mt-6 text-sm">
		Reference: <span class="font-mono">docs/QUANT_BUILD.md</span>, <span class="font-mono">docs/KHANARY_BUILD.md</span>.
		A browser-driven training endpoint on khanary-server is future work.
	</p>
</div>
