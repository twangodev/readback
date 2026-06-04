import type { Context, Queue, QueueStats, ReviewBody } from './types';

async function getJson<T>(url: string): Promise<T> {
	const response = await fetch(url, { headers: { accept: 'application/json' } });
	if (!response.ok) throw new Error(`${response.status} ${response.statusText} (${url})`);
	return response.json() as Promise<T>;
}

export function fetchPlan(): Promise<Queue> {
	return getJson<Queue>('/api/plan');
}

export function fetchQueue(): Promise<Queue> {
	return getJson<Queue>('/api/queue');
}

export function fetchStats(): Promise<QueueStats> {
	return getJson<QueueStats>('/api/queue/stats');
}

export function fetchContext(utteranceId: string): Promise<Context> {
	return getJson<Context>(`/api/clips/${utteranceId}`);
}

export function audioUrl(utteranceId: string): string {
	return `/api/clips/${utteranceId}/audio`;
}

export async function postReview(utteranceId: string, body: ReviewBody): Promise<void> {
	const response = await fetch(`/api/review/${utteranceId}`, {
		method: 'POST',
		headers: { 'content-type': 'application/json' },
		body: JSON.stringify(body)
	});
	if (!response.ok) throw new Error(`review failed: ${response.status} ${response.statusText}`);
}

export async function undoReview(utteranceId: string): Promise<void> {
	const response = await fetch(`/api/review/${utteranceId}`, { method: 'DELETE' });
	if (!response.ok) throw new Error(`undo failed: ${response.status} ${response.statusText}`);
}
