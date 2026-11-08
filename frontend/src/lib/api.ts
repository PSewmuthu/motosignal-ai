export type VideoRecord = {
	id: string;
	filename: string;
	role: string;
	size_bytes: number;
	duration_seconds?: number | null;
	fps?: number | null;
	width?: number | null;
	height?: number | null;
};

export type Job = {
	id: string;
	status: string;
	progress: number;
	message: string;
	diagnostics: Record<string, unknown>;
};

export type TrafficEvent = {
	id: string;
	timestamp_seconds: number;
	track_id: string;
	rear_track_id?: string | null;
	direction: string;
	classification: string;
	classification_reason: string;
	signal: {
		color: string;
		status: string;
		reason: string;
		phase_start_seconds?: number | null;
		phase_end_seconds?: number | null;
		direction?: string | null;
	};
	association: { status: string; confidence: number; reason: string };
	plate: {
		text?: string | null;
		status: string;
		confidence: number;
		observations: number;
		conflicts: number;
	};
	traffic_frame_url?: string;
	plate_evidence: string[];
	rear_frame_urls?: string[];
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
	const response = await fetch(path, init);
	if (!response.ok) {
		let message = `Request failed (${response.status})`;
		try {
			const body = await response.json();
			message = body.detail ?? message;
		} catch {
			/* keep fallback */
		}
		throw new Error(message);
	}
	return response.json() as Promise<T>;
}

export async function uploadVideo(
	file: File,
	role: "traffic" | "rear" | "combined",
): Promise<VideoRecord> {
	const form = new FormData();
	form.append("file", file);
	form.append("role", role);
	return request<VideoRecord>("/api/videos", { method: "POST", body: form });
}

export async function createJob(payload: unknown): Promise<Job> {
	return request<Job>("/api/jobs", {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(payload),
	});
}

export const getJob = (id: string) => request<Job>(`/api/jobs/${id}`);

export const getEvents = (id: string) =>
	request<TrafficEvent[]>(`/api/jobs/${id}/events`);

export const getHealth = () =>
	request<{
		status: string;
		motorcycle_model_configured: boolean;
		plate_model_configured: boolean;
		ocr_enabled: boolean;
	}>("/api/health");
