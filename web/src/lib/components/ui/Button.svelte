<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { HTMLButtonAttributes } from 'svelte/elements';
	import { cn } from '$lib/utils';

	type Variant = 'default' | 'secondary' | 'outline' | 'ghost' | 'destructive' | 'success';
	type Size = 'sm' | 'md' | 'lg';

	interface Props extends HTMLButtonAttributes {
		variant?: Variant;
		size?: Size;
		children: Snippet;
	}

	let {
		variant = 'default',
		size = 'md',
		class: className,
		children,
		...rest
	}: Props = $props();

	const variants: Record<Variant, string> = {
		default: 'bg-primary text-primary-foreground hover:bg-primary/90',
		secondary: 'bg-secondary text-secondary-foreground hover:bg-secondary/80',
		outline: 'border border-input bg-transparent hover:bg-accent hover:text-accent-foreground',
		ghost: 'hover:bg-accent hover:text-accent-foreground',
		destructive: 'bg-destructive text-destructive-foreground hover:bg-destructive/90',
		success: 'bg-emerald-600 text-white hover:bg-emerald-600/90'
	};

	const sizes: Record<Size, string> = {
		sm: 'h-8 px-3 text-xs',
		md: 'h-9 px-4 text-sm',
		lg: 'h-11 px-6 text-base'
	};
</script>

<button
	class={cn(
		'inline-flex items-center justify-center gap-2 rounded-md font-medium whitespace-nowrap transition-colors focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none disabled:pointer-events-none disabled:opacity-50',
		variants[variant],
		sizes[size],
		className
	)}
	{...rest}
>
	{@render children()}
</button>
