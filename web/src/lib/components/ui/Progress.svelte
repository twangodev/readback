<script lang="ts">
	interface Props {
		value: number;
		max: number;
	}

	let { value, max }: Props = $props();
	let pct = $derived(max > 0 ? Math.min(100, (value / max) * 100) : 0);
</script>

<div class="h-2 w-full overflow-hidden rounded-full bg-secondary">
	<div class="h-full bg-primary transition-all" style="width: {pct}%"></div>
</div>
