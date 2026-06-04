<script lang="ts">
	import { audioUrl } from '$lib/api';
	import type { ConversationClip } from '$lib/types';

	interface Props {
		clips: ConversationClip[];
	}

	let { clips }: Props = $props();

	let player: HTMLAudioElement | undefined;
	let playingId = $state<string | null>(null);
	let sequence: string[] = [];
	let cursor = -1;

	function element(): HTMLAudioElement {
		if (!player) {
			player = new Audio();
			player.addEventListener('ended', onEnded);
		}
		return player;
	}

	function start(id: string) {
		const audio = element();
		audio.src = audioUrl(id);
		playingId = id;
		void audio.play().catch(() => undefined);
	}

	function onEnded() {
		if (cursor >= 0 && cursor < sequence.length - 1) {
			cursor += 1;
			start(sequence[cursor]);
		} else {
			playingId = null;
			cursor = -1;
			sequence = [];
		}
	}

	function playOne(id: string) {
		if (playingId === id) {
			element().pause();
			playingId = null;
			cursor = -1;
			sequence = [];
			return;
		}
		sequence = [];
		cursor = -1;
		start(id);
	}

	function playExchange() {
		sequence = clips.map((clip) => clip.utterance_id);
		cursor = 0;
		start(sequence[0]);
	}

	$effect(() => {
		clips;
		if (player) player.pause();
		playingId = null;
		cursor = -1;
		sequence = [];
	});

	$effect(() => {
		return () => {
			if (player) {
				player.pause();
				player.src = '';
			}
		};
	});

	function clock(start: number | null): string {
		if (start == null) return '';
		return new Date(start * 1000).toISOString().slice(11, 19);
	}
</script>

<div class="space-y-2 rounded-lg border border-border bg-card p-4">
	<div class="flex items-center justify-between">
		<h2 class="text-xs font-semibold tracking-wide text-muted-foreground uppercase">Conversation</h2>
		<button
			class="rounded-[3px] border border-input px-2 py-0.5 text-[11px] font-medium tracking-wide uppercase transition-colors hover:bg-accent"
			onclick={playExchange}
		>
			Play exchange ▸
		</button>
	</div>
	<ul class="space-y-1">
		{#each clips as clip (clip.utterance_id)}
			<li
				class="flex items-start gap-2 rounded-[3px] px-2 py-1 {clip.current
					? 'bg-primary/10 ring-1 ring-primary/40'
					: 'hover:bg-accent/60'}"
			>
				<button
					class="mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full bg-foreground text-[8px] text-background transition hover:opacity-80"
					onclick={() => playOne(clip.utterance_id)}
					aria-label="play clip"
				>
					{playingId === clip.utterance_id ? '❚❚' : '▶'}
				</button>
				<span class="mt-0.5 shrink-0 font-mono text-[11px] tabular-nums text-muted-foreground">
					{clock(clip.start)}
				</span>
				<span
					class="font-mono text-xs leading-relaxed break-words {clip.transcript
						? ''
						: 'text-muted-foreground italic'}"
				>
					{clip.transcript || '(non-speech)'}
				</span>
			</li>
		{/each}
	</ul>
</div>
