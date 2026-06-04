<script lang="ts">
	import Button from './ui/Button.svelte';
	import { studio } from '$lib/studio.svelte';
	import type { Decision } from '$lib/types';

	const actions: { decision: Decision; label: string; key: string; variant: 'success' | 'default' | 'destructive' | 'secondary' }[] = [
		{ decision: 'accept', label: 'Accept', key: 'a', variant: 'success' },
		{ decision: 'edit', label: 'Save Edit', key: 'e', variant: 'default' },
		{ decision: 'reject', label: 'Reject', key: 'r', variant: 'destructive' },
		{ decision: 'non_speech', label: 'Non-speech', key: 'n', variant: 'secondary' }
	];

	const shortcuts: [string, string][] = [
		['a', 'accept'],
		['e', 'save edit'],
		['r', 'reject'],
		['n', 'non-speech'],
		['u', 'undo'],
		['space', 'play / pause'],
		['j / ↓', 'next'],
		['k / ↑', 'previous']
	];

	let disabled = $derived(!studio.context || studio.saving);
</script>

<aside class="flex h-full flex-col gap-5 border-l border-border bg-card/40 p-4">
	<div class="space-y-2">
		<h2 class="text-xs font-semibold tracking-wide text-muted-foreground uppercase">Decision</h2>
		<div class="grid grid-cols-2 gap-2">
			{#each actions as action (action.decision)}
				<Button
					variant={action.variant}
					{disabled}
					onclick={() => studio.decide(action.decision)}
				>
					{action.label}
					<kbd class="rounded bg-black/20 px-1 text-[10px] opacity-70">{action.key}</kbd>
				</Button>
			{/each}
		</div>
		<Button variant="outline" class="w-full" disabled={!studio.selectedId || studio.saving} onclick={() => studio.undo()}>
			Undo
			<kbd class="rounded bg-black/20 px-1 text-[10px] opacity-70">u</kbd>
		</Button>
	</div>

	<div class="space-y-1.5">
		<h2 class="text-xs font-semibold tracking-wide text-muted-foreground uppercase">State</h2>
		{#if studio.context}
			{#if studio.context.reviewed}
				<p class="text-sm text-emerald-400">
					Reviewed · {studio.context.review_decision ?? studio.context.source}
				</p>
			{:else}
				<p class="text-sm text-muted-foreground">Not yet reviewed</p>
			{/if}
		{:else}
			<p class="text-sm text-muted-foreground">No clip selected</p>
		{/if}
		{#if studio.error}
			<p class="rounded border border-destructive/40 bg-destructive/10 p-2 text-xs text-destructive-foreground">
				{studio.error}
			</p>
		{/if}
	</div>

	<div class="mt-auto space-y-1.5">
		<h2 class="text-xs font-semibold tracking-wide text-muted-foreground uppercase">Shortcuts</h2>
		<dl class="space-y-1 text-xs text-muted-foreground">
			{#each shortcuts as [key, desc] (key)}
				<div class="flex justify-between">
					<dt>{desc}</dt>
					<dd><kbd class="rounded border border-border bg-secondary px-1.5 py-0.5 font-mono">{key}</kbd></dd>
				</div>
			{/each}
		</dl>
	</div>
</aside>
