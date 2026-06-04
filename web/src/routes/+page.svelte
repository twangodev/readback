<script lang="ts">
	import { onMount } from 'svelte';
	import QueuePane from '$lib/components/QueuePane.svelte';
	import ClipPane from '$lib/components/ClipPane.svelte';
	import DecisionPane from '$lib/components/DecisionPane.svelte';
	import { studio } from '$lib/studio.svelte';

	let toggle = $state<(() => void) | undefined>();

	onMount(() => {
		void studio.loadQueue();
	});

	function togglePlay() {
		toggle?.();
	}

	function isEditing(target: EventTarget | null): boolean {
		return target instanceof HTMLTextAreaElement || target instanceof HTMLInputElement;
	}

	function onKeydown(event: KeyboardEvent) {
		if (event.metaKey || event.ctrlKey || event.altKey) return;
		const editing = isEditing(event.target);

		if (event.key === ' ' && !editing) {
			event.preventDefault();
			togglePlay();
			return;
		}
		if (editing) return;

		switch (event.key) {
			case 'a':
				event.preventDefault();
				void studio.decide('accept');
				break;
			case 'e':
				event.preventDefault();
				void studio.decide('edit');
				break;
			case 'r':
				event.preventDefault();
				void studio.decide('reject');
				break;
			case 'n':
				event.preventDefault();
				void studio.decide('non_speech');
				break;
			case 'u':
				event.preventDefault();
				void studio.undo();
				break;
			case 'j':
			case 'ArrowDown':
				event.preventDefault();
				studio.move(1);
				break;
			case 'k':
			case 'ArrowUp':
				event.preventDefault();
				studio.move(-1);
				break;
		}
	}
</script>

<svelte:window onkeydown={onKeydown} />

<div class="grid h-screen grid-cols-[320px_minmax(0,1fr)_300px] grid-rows-1 overflow-hidden">
	<QueuePane />
	<ClipPane bind:toggle />
	<DecisionPane />
</div>
