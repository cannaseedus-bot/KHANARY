<script lang="ts">
	// KHANARY Models route — lists the model(s) this khanary-server reports (/v1/models) plus the
	// KHANARY model packages. Self-contained (no app-component deps) to keep the embed build robust.
	import { onMount } from 'svelte';

	let models = $state<Array<{ id?: string; name?: string; model?: string }>>([]);
	let error = $state('');
	let loading = $state(true);

	const packages = [
		'khanary-qwen1_8b — Q4 base (0.91 GiB) + Q8 escalation tier (1.71 GiB), tensor-aligned',
		'khanary-gpt2 — 5 GPT-2 forward-op glyphs (verified on HD 4600)',
		'khanary-geometry — birdsong mesh geometry ops',
		'khanary-kxml — 12 tool calls + node ops registry',
		'khanary-gpu-resident — KGRC resident proof ladder'
	];

	onMount(async () => {
		try {
			const r = await fetch('/v1/models');
			const j = await r.json();
			models = j?.data ?? j?.models ?? [];
		} catch (e) {
			error = String(e);
		} finally {
			loading = false;
		}
	});
</script>

<div class="mx-auto w-full max-w-3xl p-4 md:p-8">
	<h1 class="text-primary text-2xl font-bold">Models</h1>
	<p class="text-muted-foreground mt-1">Model(s) loaded on this khanary-server, plus the KHANARY model packages.</p>

	<h2 class="text-primary mt-6 text-lg font-semibold">Loaded</h2>
	{#if loading}
		<p class="text-muted-foreground mt-2">Loading /v1/models…</p>
	{:else if error}
		<p class="mt-2 text-red-400">Could not read /v1/models: {error}</p>
	{:else}
		<ul class="mt-2 space-y-2">
			{#each models as m}
				<li class="border-border rounded-md border p-3 font-mono text-sm">{m.id ?? m.name ?? m.model}</li>
			{:else}
				<li class="text-muted-foreground">No models reported (single-model mode shows one; none if unloaded).</li>
			{/each}
		</ul>
	{/if}

	<h2 class="text-primary mt-8 text-lg font-semibold">KHANARY packages</h2>
	<ul class="text-muted-foreground mt-2 space-y-1 text-sm">
		{#each packages as p}
			<li>• {p}</li>
		{/each}
	</ul>
</div>
