<script lang="ts">
	import type { Snippet } from 'svelte';
	import { cn } from '$lib/utils';

	interface Props {
		class?: string;
		children: Snippet;
	}

	let { class: className, children }: Props = $props();
</script>

<span
	class={cn(
		'inline-flex items-center rounded-[2px] border px-1.5 py-0.5 font-mono text-[10px] font-medium tracking-wide uppercase leading-none',
		className
	)}
>
	{@render children()}
</span>
