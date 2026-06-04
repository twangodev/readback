<script lang="ts">
	import AudioPlayer from './AudioPlayer.svelte';
	import AlignedLine from './AlignedLine.svelte';
	import TierBadge from './TierBadge.svelte';
	import Textarea from './ui/Textarea.svelte';
	import TrackMap from './TrackMap.svelte';
	import Conversation from './Conversation.svelte';
	import { studio } from '$lib/studio.svelte';

	interface Props {
		toggle?: () => void;
	}

	let { toggle = $bindable() }: Props = $props();

	function fmt(value: number | null | undefined): string {
		return value === null || value === undefined ? '—' : value.toFixed(2);
	}
</script>

<section class="flex h-full min-h-0 flex-col overflow-y-auto">
	{#if studio.clipLoading && !studio.context}
		<p class="p-6 text-sm text-muted-foreground">Loading clip…</p>
	{:else if !studio.context}
		<p class="p-6 text-sm text-muted-foreground">Select a clip from the queue.</p>
	{:else}
		{@const c = studio.context}
		<div class="space-y-5 p-5">
			<div class="flex items-center gap-2">
				<TierBadge tier={c.tier} />
				<span class="font-mono text-xs text-muted-foreground" title={c.utterance_id}>
					{c.utterance_id}
				</span>
				{#if c.reviewed}
					<span class="ml-auto text-xs text-emerald-400">
						reviewed · {c.review_decision ?? c.source}
					</span>
				{/if}
			</div>

			{#if studio.audioSrc}
				<AudioPlayer src={studio.audioSrc} bind:toggle />
			{/if}

			{#if (c.conversation ?? []).length > 1}
				<Conversation clips={c.conversation ?? []} />
			{/if}

			<TrackMap tracks={c.tracks ?? []} />

			<div class="space-y-2 rounded-lg border border-border bg-card/50 p-4">
				<h2 class="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
					Voter hypotheses
				</h2>
				<div class="space-y-2">
					{#each c.aligned_hypotheses as hyp, i (i)}
						<AlignedLine ops={hyp.ops} />
					{/each}
				</div>
				<div class="border-t border-border/60 pt-2">
					<span class="text-[11px] tracking-wide text-muted-foreground uppercase">base hyp</span>
					<p class="font-mono text-sm break-words">{c.base_hyp}</p>
				</div>
			</div>

			<div class="space-y-1.5">
				<label
					for="transcript"
					class="text-xs font-semibold tracking-wide text-muted-foreground uppercase"
				>
					Transcript
				</label>
				<Textarea id="transcript" bind:value={studio.draft} rows={3} class="font-mono" />
			</div>

			<dl class="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-4">
				<div><dt class="text-xs text-muted-foreground">airport</dt><dd class="font-mono uppercase">{c.airport}</dd></div>
				<div><dt class="text-xs text-muted-foreground">aircraft</dt><dd class="font-mono">{c.n_aircraft ?? '—'}</dd></div>
				<div><dt class="text-xs text-muted-foreground">duration</dt><dd class="font-mono">{fmt(c.duration_s)}s</dd></div>
				<div><dt class="text-xs text-muted-foreground">agreement</dt><dd class="font-mono">{fmt(c.agreement_score)}</dd></div>
				<div><dt class="text-xs text-muted-foreground">rover conf</dt><dd class="font-mono">{fmt(c.rover_confidence)}</dd></div>
				<div><dt class="text-xs text-muted-foreground">models agree</dt><dd class="font-mono">{c.n_models_agree}</dd></div>
				<div><dt class="text-xs text-muted-foreground">advisory dis</dt><dd class="font-mono">{fmt(c.advisory_disagree)}</dd></div>
				<div>
					<dt class="text-xs text-muted-foreground">callsign</dt>
					<dd class="font-mono">
						{c.callsign_tail ?? '—'}
						<span class={c.callsign_matched ? 'text-emerald-400' : 'text-rose-400'}>
							{c.callsign_matched ? '✓' : '✗'}
						</span>
					</dd>
				</div>
			</dl>

			{#if c.tails.length > 0}
				<div class="flex flex-wrap items-center gap-1.5">
					<span class="text-xs text-muted-foreground">tails:</span>
					{#each c.tails as tail (tail)}
						<span class="rounded border border-border bg-secondary px-1.5 py-0.5 font-mono text-xs">
							{tail}
						</span>
					{/each}
				</div>
			{/if}
		</div>
	{/if}
</section>
