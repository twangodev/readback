<script lang="ts">
	import Button from './ui/Button.svelte';
	import Progress from './ui/Progress.svelte';
	import TierBadge from './TierBadge.svelte';
	import ReasonBadge from './ReasonBadge.svelte';
	import { studio, type QueueMode } from '$lib/studio.svelte';

	const modes: { id: QueueMode; label: string }[] = [
		{ id: 'plan', label: 'Plan' },
		{ id: 'all', label: 'All' }
	];
	const ROW = 68;
	const BUFFER = 6;

	let scrollTop = $state(0);
	let viewport = $state(600);

	let count = $derived(studio.items.length);
	let start = $derived(Math.max(0, Math.floor(scrollTop / ROW) - BUFFER));
	let end = $derived(Math.min(count, Math.ceil((scrollTop + viewport) / ROW) + BUFFER));
	let visible = $derived(studio.items.slice(start, end));

	function fmt(value: number): string {
		return value.toFixed(2);
	}

	async function pick(mode: QueueMode) {
		scrollTop = 0;
		await studio.setMode(mode);
	}
</script>

<aside class="flex h-full flex-col border-r border-border bg-card/40">
	<div class="space-y-3 border-b border-border p-4">
		<div class="flex items-center justify-between">
			<h1 class="text-sm font-semibold tracking-tight">Data Studio</h1>
			<div class="inline-flex rounded-md border border-input p-0.5">
				{#each modes as m (m.id)}
					<button
						class="rounded px-2.5 py-1 text-xs font-medium transition-colors {studio.mode === m.id
							? 'bg-primary text-primary-foreground'
							: 'text-muted-foreground hover:text-foreground'}"
						onclick={() => pick(m.id)}
					>
						{m.label}
					</button>
				{/each}
			</div>
		</div>

		<div class="space-y-1.5">
			<div class="flex justify-between text-xs text-muted-foreground">
				<span>{studio.reviewed} / {studio.total} reviewed</span>
				{#if studio.stats}
					<span class="font-mono">{studio.stats.remaining} left</span>
				{/if}
			</div>
			<Progress value={studio.reviewed} max={studio.total} />
		</div>

		{#if studio.stats}
			<div class="flex gap-4 text-xs text-muted-foreground">
				<span>silver <span class="font-mono text-foreground">{studio.stats.by_tier.silver}</span></span>
				<span>tail <span class="font-mono text-foreground">{studio.stats.by_tier.tail}</span></span>
			</div>
		{/if}
	</div>

	<div
		class="min-h-0 flex-1 overflow-y-auto"
		bind:clientHeight={viewport}
		onscroll={(event) => (scrollTop = event.currentTarget.scrollTop)}
	>
		{#if studio.queueLoading && count === 0}
			<p class="p-4 text-sm text-muted-foreground">Loading queue…</p>
		{:else if count === 0}
			<p class="p-4 text-sm text-muted-foreground">Queue is empty.</p>
		{:else}
			<div style="height: {count * ROW}px; position: relative;">
				{#each visible as item, i (item.utterance_id)}
					{@const index = start + i}
					{@const selected = item.utterance_id === studio.selectedId}
					<button
						class="absolute right-0 left-0 flex flex-col justify-center gap-1.5 border-b border-border/60 px-3 text-left transition-colors {selected
							? 'bg-accent'
							: 'hover:bg-accent/50'}"
						style="top: {index * ROW}px; height: {ROW}px;"
						onclick={() => studio.select(item.utterance_id)}
					>
						<div class="flex items-center justify-between gap-2">
							<span class="truncate font-mono text-xs" title={item.utterance_id}>
								{item.utterance_id}
							</span>
							{#if item.reviewed}
								<span class="shrink-0 text-emerald-400" aria-label="reviewed">✓</span>
							{/if}
						</div>
						<div class="flex flex-wrap items-center gap-1.5">
							<TierBadge tier={item.tier} />
							{#if item.reason}
								<ReasonBadge reason={item.reason} />
							{/if}
							<span class="ml-auto font-mono text-[11px] text-muted-foreground">
								agr {fmt(item.agreement_score)}
							</span>
							{#if item.advisory_disagree !== null}
								<span class="font-mono text-[11px] text-amber-400/80">
									adv {item.advisory_disagree.toFixed(1)}
								</span>
							{/if}
						</div>
					</button>
				{/each}
			</div>
		{/if}
	</div>

	<div class="border-t border-border p-2">
		<Button variant="ghost" size="sm" class="w-full" onclick={() => studio.loadQueue()}>
			Refresh
		</Button>
	</div>
</aside>
