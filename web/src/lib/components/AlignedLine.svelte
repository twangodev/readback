<script lang="ts">
	import type { AlignOp, AlignedToken } from '$lib/types';

	interface Props {
		ops: AlignedToken[];
	}

	let { ops }: Props = $props();

	const tokenClass: Record<AlignOp, string> = {
		equal: 'text-foreground',
		ins: 'text-emerald-700',
		del: 'text-rose-600 line-through',
		sub: 'text-orange-600'
	};
</script>

<p class="font-mono text-sm leading-relaxed break-words">
	{#each ops as token, i (i)}
		<span class={tokenClass[token.op]}>{token.token}</span>{' '}
	{/each}
</p>
