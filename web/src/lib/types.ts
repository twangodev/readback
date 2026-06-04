export type Tier = 'gold' | 'silver' | 'tail';
export type Reason = 'calibration' | 'boundary';
export type Decision = 'accept' | 'edit' | 'reject' | 'non_speech';
export type AlignOp = 'equal' | 'ins' | 'del' | 'sub';

export interface PlanItem {
	utterance_id: string;
	tier: Tier;
	rover_confidence: number;
	agreement_score: number;
	n_models_agree: number;
	callsign_matched: boolean;
	advisory_disagree: number | null;
	shard: number;
	reviewed: boolean;
	reason?: Reason;
}

export interface Queue {
	items: PlanItem[];
	total: number;
	reviewed: number;
}

export interface QueueStats {
	total: number;
	reviewed: number;
	remaining: number;
	by_tier: { silver: number; tail: number };
}

export interface AlignedToken {
	op: AlignOp;
	token: string;
}

export interface AlignedHypothesis {
	text: string;
	ops: AlignedToken[];
}

export interface TrackPoint {
	t: number | null;
	lat: number;
	lon: number;
	alt: number;
	speed: number;
	heading: number;
}

export interface Aircraft {
	tail: string | null;
	aircraft_id: string | null;
	points: TrackPoint[];
}

export interface ConversationClip {
	utterance_id: string;
	transcript: string;
	tier: Tier | 'non_speech';
	current: boolean;
	start: number | null;
}

export interface Context {
	utterance_id: string;
	transcript: string;
	tier: Tier;
	rover_confidence: number;
	agreement_score: number;
	n_models_agree: number;
	n_tails: number;
	callsign_matched: boolean;
	callsign_tail: string | null;
	callsign_score: number;
	snapped: boolean;
	advisory_disagree: number | null;
	base_hyp: string;
	voting_text: string[];
	aligned_hypotheses: AlignedHypothesis[];
	airport: string;
	tails: string[];
	reviewed: boolean;
	review_decision: string | null;
	effective_transcript: string;
	source: string;
	start?: number | null;
	end?: number | null;
	duration_s?: number | null;
	n_aircraft?: number | null;
	clip_offset_s?: number | null;
	tracks?: Aircraft[];
	conversation?: ConversationClip[];
}

export interface ReviewBody {
	decision: Decision;
	transcript: string;
	base_hyp: string;
}
