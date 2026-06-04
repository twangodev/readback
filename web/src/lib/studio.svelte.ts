import {
	audioUrl,
	fetchContext,
	fetchPlan,
	fetchQueue,
	fetchStats,
	postReview,
	undoReview
} from './api';
import type { Context, Decision, PlanItem, QueueStats } from './types';

export type QueueMode = 'plan' | 'all';

class Studio {
	mode = $state<QueueMode>('plan');
	items = $state<PlanItem[]>([]);
	total = $state(0);
	reviewed = $state(0);
	stats = $state<QueueStats | null>(null);

	selectedId = $state<string | null>(null);
	context = $state<Context | null>(null);
	draft = $state('');

	queueLoading = $state(false);
	clipLoading = $state(false);
	saving = $state(false);
	error = $state<string | null>(null);

	selectedIndex = $derived(this.items.findIndex((item) => item.utterance_id === this.selectedId));

	get audioSrc() {
		return this.selectedId ? audioUrl(this.selectedId) : '';
	}

	async loadQueue() {
		this.queueLoading = true;
		this.error = null;
		try {
			const [queue, stats] = await Promise.all([
				this.mode === 'plan' ? fetchPlan() : fetchQueue(),
				fetchStats()
			]);
			this.items = queue.items;
			this.total = queue.total;
			this.reviewed = queue.reviewed;
			this.stats = stats;
			if (this.selectedId && !this.items.some((item) => item.utterance_id === this.selectedId)) {
				this.selectedId = null;
				this.context = null;
			}
			if (!this.selectedId && this.items.length > 0) {
				await this.select(this.items[0].utterance_id);
			}
		} catch (error) {
			this.error = error instanceof Error ? error.message : String(error);
		} finally {
			this.queueLoading = false;
		}
	}

	async setMode(mode: QueueMode) {
		if (mode === this.mode) return;
		this.mode = mode;
		await this.loadQueue();
	}

	async select(utteranceId: string) {
		this.selectedId = utteranceId;
		this.clipLoading = true;
		this.error = null;
		try {
			const context = await fetchContext(utteranceId);
			this.context = context;
			this.draft = context.effective_transcript;
		} catch (error) {
			this.context = null;
			this.error = error instanceof Error ? error.message : String(error);
		} finally {
			this.clipLoading = false;
		}
	}

	selectIndex(index: number) {
		if (index < 0 || index >= this.items.length) return;
		void this.select(this.items[index].utterance_id);
	}

	move(delta: number) {
		const index = this.selectedIndex;
		if (index < 0) {
			if (this.items.length > 0) this.selectIndex(0);
			return;
		}
		this.selectIndex(index + delta);
	}

	private nextUnreviewedIndex(from: number): number {
		for (let i = from; i < this.items.length; i++) {
			if (!this.items[i].reviewed) return i;
		}
		for (let i = 0; i < from; i++) {
			if (!this.items[i].reviewed) return i;
		}
		return -1;
	}

	private transcriptFor(decision: Decision): string {
		if (!this.context) return '';
		if (decision === 'accept') return this.context.base_hyp;
		if (decision === 'edit') return this.draft;
		return '';
	}

	async decide(decision: Decision) {
		if (!this.context || !this.selectedId || this.saving) return;
		this.saving = true;
		this.error = null;
		const utteranceId = this.selectedId;
		try {
			await postReview(utteranceId, {
				decision,
				transcript: this.transcriptFor(decision),
				base_hyp: this.context.transcript
			});
			this.markReviewed(utteranceId, decision);
			await this.advance(utteranceId);
		} catch (error) {
			this.error = error instanceof Error ? error.message : String(error);
		} finally {
			this.saving = false;
		}
	}

	async undo() {
		if (!this.selectedId || this.saving) return;
		this.saving = true;
		this.error = null;
		const utteranceId = this.selectedId;
		try {
			await undoReview(utteranceId);
			await Promise.all([this.refreshStats(), this.select(utteranceId)]);
			this.syncReviewedFlag(utteranceId);
		} catch (error) {
			this.error = error instanceof Error ? error.message : String(error);
		} finally {
			this.saving = false;
		}
	}

	private markReviewed(utteranceId: string, decision: Decision) {
		const index = this.items.findIndex((item) => item.utterance_id === utteranceId);
		if (index >= 0 && !this.items[index].reviewed) {
			this.items[index] = { ...this.items[index], reviewed: true };
			this.reviewed += 1;
		}
		if (this.context && this.context.utterance_id === utteranceId) {
			this.context = { ...this.context, reviewed: true, review_decision: decision };
		}
		void this.refreshStats();
	}

	private syncReviewedFlag(utteranceId: string) {
		if (!this.context) return;
		const index = this.items.findIndex((item) => item.utterance_id === utteranceId);
		if (index < 0) return;
		const wasReviewed = this.items[index].reviewed;
		if (wasReviewed !== this.context.reviewed) {
			this.items[index] = { ...this.items[index], reviewed: this.context.reviewed };
			this.reviewed += this.context.reviewed ? 1 : -1;
		}
	}

	private async refreshStats() {
		try {
			this.stats = await fetchStats();
		} catch {
			void 0;
		}
	}

	private async advance(fromId: string) {
		const current = this.items.findIndex((item) => item.utterance_id === fromId);
		const next = this.nextUnreviewedIndex(current < 0 ? 0 : current + 1);
		if (next >= 0) await this.select(this.items[next].utterance_id);
	}
}

export const studio = new Studio();
