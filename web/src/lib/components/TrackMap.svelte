<script lang="ts">
	import L from 'leaflet';
	import 'leaflet/dist/leaflet.css';
	import type { Aircraft } from '$lib/types';

	interface Props {
		tracks: Aircraft[];
	}

	let { tracks }: Props = $props();
	let container = $state<HTMLDivElement>();
	let map = $state<L.Map>();
	let layer: L.LayerGroup | undefined;

	const palette = [
		'#ff4d00',
		'#0096ff',
		'#16a34a',
		'#7c3aed',
		'#ea384c',
		'#d946ef',
		'#0891b2',
		'#ca8a04'
	];

	$effect(() => {
		const node = container;
		if (!node) return;
		const instance = L.map(node, { zoomControl: true, attributionControl: false });
		L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
			subdomains: 'abcd',
			maxZoom: 19
		}).addTo(instance);
		layer = L.layerGroup().addTo(instance);
		instance.setView([39.5, -98.35], 4);
		const resize = setTimeout(() => instance.invalidateSize(), 60);
		map = instance;
		return () => {
			clearTimeout(resize);
			instance.remove();
			layer = undefined;
			map = undefined;
		};
	});

	$effect(() => {
		const list = tracks;
		const instance = map;
		if (!instance || !layer) return;
		layer.clearLayers();
		const bounds = L.latLngBounds([]);
		list.forEach((aircraft, index) => {
			const latlngs = aircraft.points
				.filter((p) => p.lat != null && p.lon != null)
				.map((p) => [p.lat, p.lon] as [number, number]);
			if (latlngs.length === 0) return;
			const color = palette[index % palette.length];
			L.polyline(latlngs, { color, weight: 2, opacity: 0.85 }).addTo(layer!);
			L.circleMarker(latlngs[latlngs.length - 1], {
				radius: 4,
				color,
				weight: 2,
				fillColor: color,
				fillOpacity: 1
			})
				.bindTooltip(aircraft.tail ?? aircraft.aircraft_id ?? '?', { direction: 'top' })
				.addTo(layer!);
			latlngs.forEach((point) => bounds.extend(point));
		});
		if (bounds.isValid()) instance.fitBounds(bounds, { padding: [24, 24] });
		instance.invalidateSize();
	});
</script>

<div class="relative">
	<div
		bind:this={container}
		class="h-72 w-full overflow-hidden rounded-lg border border-border"
	></div>
	{#if tracks.length === 0}
		<div
			class="pointer-events-none absolute inset-0 flex items-center justify-center rounded-lg bg-background/70 text-xs text-muted-foreground"
		>
			no ADS-B tracks for this clip
		</div>
	{/if}
</div>
