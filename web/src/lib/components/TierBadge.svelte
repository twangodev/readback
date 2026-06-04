<script lang="ts">
	import Badge from './ui/Badge.svelte';
	import type { Tier } from '$lib/types';

	interface Props {
		tier: Tier;
	}

	let { tier }: Props = $props();

	const styles: Record<Tier, string> = {
		gold: 'border-amber-500 bg-amber-50 text-amber-700',
		silver: 'border-zinc-400 bg-zinc-100 text-zinc-700',
		tail: 'border-sky-600 bg-sky-50 text-sky-700'
	};
</script>

<Badge class={styles[tier]}>{tier}</Badge>
