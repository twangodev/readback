<script lang="ts">
	import WaveSurfer from 'wavesurfer.js';

	interface Props {
		src: string;
		toggle?: () => void;
	}

	let { src, toggle = $bindable() }: Props = $props();

	let container = $state<HTMLDivElement>();
	let ws = $state<WaveSurfer>();
	let playing = $state(false);
	let current = $state(0);
	let total = $state(0);

	$effect(() => {
		const node = container;
		if (!node) return;
		const instance = WaveSurfer.create({
			container: node,
			height: 56,
			waveColor: '#d4d4d8',
			progressColor: '#ff4d00',
			cursorColor: '#52525b',
			barWidth: 2,
			barGap: 1,
			barRadius: 2,
			normalize: true
		});
		toggle = () => void instance.playPause();
		instance.on('play', () => (playing = true));
		instance.on('pause', () => (playing = false));
		instance.on('finish', () => (playing = false));
		instance.on('timeupdate', (time: number) => (current = time));
		instance.on('ready', (duration: number) => {
			total = duration;
			void instance.play().catch(() => undefined);
		});
		ws = instance;
		return () => {
			toggle = undefined;
			instance.destroy();
			ws = undefined;
		};
	});

	$effect(() => {
		const url = src;
		const instance = ws;
		if (!instance || !url) return;
		current = 0;
		total = 0;
		void instance.load(url).catch(() => undefined);
	});

	function fmt(seconds: number): string {
		const whole = Math.max(0, Math.floor(seconds));
		const s = (whole % 60).toString().padStart(2, '0');
		return `${Math.floor(whole / 60)}:${s}`;
	}
</script>

<div class="flex items-center gap-3 rounded-lg border border-border bg-card/50 p-3">
	<button
		class="flex size-9 shrink-0 items-center justify-center rounded-full bg-primary text-sm text-primary-foreground transition hover:opacity-90"
		onclick={() => toggle?.()}
		aria-label={playing ? 'Pause' : 'Play'}
	>
		{playing ? '❚❚' : '▶'}
	</button>
	<div bind:this={container} class="min-w-0 flex-1"></div>
	<span class="shrink-0 font-mono text-xs text-muted-foreground">{fmt(current)} / {fmt(total)}</span>
</div>
