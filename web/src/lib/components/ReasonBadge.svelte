<script lang="ts">
	import Badge from './ui/Badge.svelte';
	import type { Reason } from '$lib/types';

	interface Props {
		reason: Reason;
	}

	let { reason }: Props = $props();

	const styles: Record<Reason, string> = {
		calibration: 'border-violet-500 bg-violet-50 text-violet-700',
		boundary: 'border-orange-500 bg-orange-50 text-orange-700'
	};
</script>

<Badge class={styles[reason]}>{reason}</Badge>
