import { useEffect, useMemo, useState } from "react";
import {
	Activity,
	AlertTriangle,
	ArrowDownToLine,
	Camera,
	CheckCircle2,
	CircleDot,
	Clock3,
	FileVideo2,
	Gauge,
	Play,
	RadioTower,
	ShieldCheck,
	UploadCloud,
} from "lucide-react";
import {
	createJob,
	getEvents,
	getHealth,
	getJob,
	TrafficEvent,
	uploadVideo,
	VideoRecord,
	Job,
} from "./lib/api";

const sampleTimeline = JSON.stringify([], null, 2);

function formatTime(seconds: number) {
	const safe = Math.max(0, seconds || 0);
	const mm = Math.floor(safe / 60);
	const ss = (safe % 60).toFixed(2).padStart(5, "0");
	return `${String(mm).padStart(2, "0")}:${ss}`;
}
function StatusPill({ value }: { value: string }) {
	const styles: Record<string, string> = {
		SUSPECTED_VIOLATION: "danger",
		NOT_A_VIOLATION: "success",
		INSUFFICIENT_EVIDENCE: "warning",
		completed: "success",
		processing: "info",
		queued: "muted",
		failed: "danger",
	};
	return (
		<span className={`pill ${styles[value] ?? "muted"}`}>
			{value.replaceAll("_", " ").toLowerCase()}
		</span>
	);
}

export default function App() {
	const [trafficFile, setTrafficFile] = useState<File | null>(null);
	const [rearFile, setRearFile] = useState<File | null>(null);
	const [sameCamera, setSameCamera] = useState(true);
	const [busy, setBusy] = useState(false);
	const [message, setMessage] = useState("");
	const [job, setJob] = useState<Job | null>(null);
	const [events, setEvents] = useState<TrafficEvent[]>([]);
	const [selected, setSelected] = useState<string | null>(null);
	const [health, setHealth] = useState<{
		motorcycle_model_configured: boolean;
		plate_model_configured: boolean;
		ocr_enabled: boolean;
	} | null>(null);
	const [line, setLine] = useState({
		x1: "0.08",
		y1: "0.52",
		x2: "0.92",
		y2: "0.52",
	});
	const [approach, setApproach] = useState("either");
	const [direction, setDirection] = useState("all");
	const [timeline, setTimeline] = useState(sampleTimeline);
	const [modeOffset, setModeOffset] = useState("0");
	const [travelMin, setTravelMin] = useState("0");
	const [travelMax, setTravelMax] = useState("15");
	const [stride, setStride] = useState("1");
	const [activeTab, setActiveTab] = useState<"events" | "diagnostics">(
		"events",
	);
	const [trafficPreviewUrl, setTrafficPreviewUrl] = useState<string | null>(
		null,
	);
	const [rearPreviewUrl, setRearPreviewUrl] = useState<string | null>(null);

	useEffect(() => {
		if (!trafficFile) {
			setTrafficPreviewUrl(null);
			return;
		}
		const url = URL.createObjectURL(trafficFile);
		setTrafficPreviewUrl(url);
		return () => URL.revokeObjectURL(url);
	}, [trafficFile]);
	useEffect(() => {
		if (!rearFile) {
			setRearPreviewUrl(null);
			return;
		}
		const url = URL.createObjectURL(rearFile);
		setRearPreviewUrl(url);
		return () => URL.revokeObjectURL(url);
	}, [rearFile]);
	useEffect(() => {
		getHealth()
			.then(setHealth)
			.catch(() => setHealth(null));
	}, []);
	useEffect(() => {
		if (!job || !["queued", "processing"].includes(job.status)) return;
		const timer = window.setInterval(async () => {
			try {
				const updated = await getJob(job.id);
				setJob(updated);
				if (updated.status === "completed" || updated.status === "failed") {
					const rows = await getEvents(job.id);
					setEvents(rows);
					if (rows.length && !selected) setSelected(rows[0].id);
				}
			} catch (error) {
				setMessage(
					error instanceof Error ? error.message : "Could not refresh job",
				);
			}
		}, 1500);
		return () => window.clearInterval(timer);
	}, [job?.id, job?.status]);

	const chosenEvent = useMemo(
		() => events.find((e) => e.id === selected) ?? events[0] ?? null,
		[events, selected],
	);
	const summary = useMemo(
		() => ({
			suspected: events.filter(
				(e) => e.classification === "SUSPECTED_VIOLATION",
			).length,
			clear: events.filter((e) => e.classification === "NOT_A_VIOLATION")
				.length,
			uncertain: events.filter(
				(e) => e.classification === "INSUFFICIENT_EVIDENCE",
			).length,
		}),
		[events],
	);

	async function startAnalysis() {
		if (!trafficFile) {
			setMessage("Choose a traffic video before starting.");
			return;
		}
		if (!sameCamera && !rearFile) {
			setMessage("Choose a rear-view video for separate-camera mode.");
			return;
		}
		let phaseRows: unknown[];
		try {
			phaseRows = JSON.parse(timeline);
			if (!Array.isArray(phaseRows))
				throw new Error("Signal timeline must be a JSON array.");
		} catch (error) {
			setMessage(
				error instanceof Error ? error.message : "Invalid signal timeline JSON",
			);
			return;
		}
		setBusy(true);
		setMessage("Uploading source video…");
		setEvents([]);
		setSelected(null);
		try {
			const traffic = await uploadVideo(
				trafficFile,
				sameCamera ? "combined" : "traffic",
			);
			let rear: VideoRecord | null = null;
			if (!sameCamera) {
				if (!rearFile)
					throw new Error("Choose a rear-view video for separate-camera mode.");
				setMessage("Uploading rear-view video…");
				rear = await uploadVideo(rearFile, "rear");
			}
			setMessage("Creating analysis job…");
			const payload = {
				traffic_video_id: traffic.id,
				rear_video_id: rear?.id ?? null,
				config: {
					mode: sameCamera ? "same_camera" : "separate_cameras",
					stop_line: [
						{ x: Number(line.x1), y: Number(line.y1) },
						{ x: Number(line.x2), y: Number(line.y2) },
					],
					approach_side: approach,
					direction,
					signal_timeline: phaseRows,
					rear_time_offset_seconds: Number(modeOffset),
					min_travel_seconds: Number(travelMin),
					max_travel_seconds: Number(travelMax),
					frame_stride: Number(stride),
					min_track_observations: 3,
				},
			};
			const created = await createJob(payload);
			setJob(created);
			setActiveTab("events");
			setMessage("Analysis queued. Processing will begin in the background.");
		} catch (error) {
			setMessage(
				error instanceof Error ? error.message : "Unable to start analysis",
			);
		} finally {
			setBusy(false);
		}
	}

	return (
		<div className="app-shell">
			<aside className="sidebar">
				<div className="brand">
					<div className="brand-mark">
						<RadioTower size={21} />
					</div>
					<div>
						<strong>MotoSignal</strong>
						<span>AI ANALYTICS</span>
					</div>
				</div>
				<div className="side-label">WORKSPACE</div>
				<div className="nav-item active">
					<Activity size={17} /> Analysis workspace
				</div>
				<div className="nav-item">
					<FileVideo2 size={17} /> Video sources
				</div>
				<div className="nav-item">
					<ShieldCheck size={17} /> Event review
				</div>
				<div className="side-bottom">
					<div className="privacy-icon">
						<ShieldCheck size={17} />
					</div>
					<strong>Local-first processing</strong>
					<p>
						Videos and evidence stay in your configured storage volume. Do not
						expose this demo server publicly.
					</p>
				</div>
			</aside>

			<main className="main-content">
				<header className="topbar">
					<div>
						<div className="eyebrow">TRAFFIC INTELLIGENCE / WORKSPACE</div>
						<h1>Red-light event analysis</h1>
						<p>
							Motorcycle tracking, signal verification and rear-plate evidence
							review.
						</p>
					</div>
					<div className="system-status">
						<span className="live-dot" /> API{" "}
						{health ? "connected" : "not checked"}
					</div>
				</header>

				<section className="metric-grid">
					<div className="metric-card">
						<div className="metric-icon">
							<FileVideo2 size={18} />
						</div>
						<span>TRAFFIC EVENTS</span>
						<strong>{events.length}</strong>
						<small>From selected processing job</small>
					</div>
					<div className="metric-card">
						<div className="metric-icon red">
							<AlertTriangle size={18} />
						</div>
						<span>SUSPECTED</span>
						<strong>{summary.suspected}</strong>
						<small>Red phase + supported association</small>
					</div>
					<div className="metric-card">
						<div className="metric-icon green">
							<CheckCircle2 size={18} />
						</div>
						<span>NOT A VIOLATION</span>
						<strong>{summary.clear}</strong>
						<small>Green phase at crossing</small>
					</div>
					<div className="metric-card">
						<div className="metric-icon amber">
							<CircleDot size={18} />
						</div>
						<span>INSUFFICIENT</span>
						<strong>{summary.uncertain}</strong>
						<small>Missing or ambiguous evidence</small>
					</div>
				</section>

				<div className="workspace-grid">
					<section className="panel setup-panel">
						<div className="panel-heading">
							<div>
								<div className="eyebrow">01 / CONFIGURATION</div>
								<h2>New analysis</h2>
							</div>
							<span className="step-tag">VIDEO INPUT</span>
						</div>
						<div className="field-block">
							<label>
								Traffic camera video <span className="required">*</span>
							</label>
							<label className="upload-box">
								<UploadCloud size={22} />
								<strong>{trafficFile?.name ?? "Choose a video file"}</strong>
								<span>MP4, MOV, AVI, MKV · up to configured upload limit</span>
								<input
									type="file"
									accept="video/*,.mkv"
									onChange={(e) => setTrafficFile(e.target.files?.[0] ?? null)}
								/>
							</label>
						</div>
						{trafficPreviewUrl && (
							<div className="source-preview">
								<small>TRAFFIC SOURCE PREVIEW</small>
								<video src={trafficPreviewUrl} controls preload="metadata" />
							</div>
						)}
						<div className="toggle-row">
							<div>
								<strong>Same camera stream</strong>
								<small>Use one video for traffic and rear-plate analysis</small>
							</div>
							<button
								className={`switch ${sameCamera ? "on" : ""}`}
								onClick={() => setSameCamera((v) => !v)}
								aria-label="Toggle same camera"
							>
								<span />
							</button>
						</div>
						{!sameCamera && (
							<>
								<div className="field-block">
									<label>
										Rear-view camera video <span className="required">*</span>
									</label>
									<label className="upload-box compact">
										<Camera size={20} />
										<strong>
											{rearFile?.name ?? "Choose rear-view video"}
										</strong>
										<span>Separate stream; set clock offset below</span>
										<input
											type="file"
											accept="video/*,.mkv"
											onChange={(e) => setRearFile(e.target.files?.[0] ?? null)}
										/>
									</label>
								</div>
								{rearPreviewUrl && (
									<div className="source-preview">
										<small>REAR-VIEW SOURCE PREVIEW</small>
										<video src={rearPreviewUrl} controls preload="metadata" />
									</div>
								)}
							</>
						)}

						<div className="section-divider">
							<span>Intersection geometry</span>
						</div>
						<div className="field-block">
							<label>
								Stop-line endpoints{" "}
								<span className="hint">Normalized 0–1 coordinates</span>
							</label>
							<div className="four-inputs">
								{(["x1", "y1", "x2", "y2"] as const).map((k) => (
									<div key={k}>
										<small>{k.toUpperCase()}</small>
										<input
											value={line[k]}
											onChange={(e) =>
												setLine((s) => ({ ...s, [k]: e.target.value }))
											}
											type="number"
											min="0"
											max="1"
											step="0.01"
										/>
									</div>
								))}
							</div>
							<p className="field-help">
								Coordinates are relative to the video frame. Example: y=0.52
								places a horizontal line just below the frame midpoint.
							</p>
						</div>
						<div className="two-inputs">
							<div className="field-block">
								<label>Approach side</label>
								<select
									value={approach}
									onChange={(e) => setApproach(e.target.value)}
								>
									<option value="either">Either side</option>
									<option value="positive">Positive side</option>
									<option value="negative">Negative side</option>
								</select>
							</div>
							<div className="field-block">
								<label>Signal direction key</label>
								<input
									value={direction}
									onChange={(e) => setDirection(e.target.value)}
									placeholder="all / northbound"
								/>
							</div>
						</div>

						<div className="section-divider">
							<span>Signal phase timeline</span>
							<span className="hint">Video-relative seconds</span>
						</div>
						<div className="field-block">
							<label>Phase schedule (JSON)</label>
							<textarea
								className="json-input"
								value={timeline}
								onChange={(e) => setTimeline(e.target.value)}
								spellCheck={false}
							/>
							<p className="field-help">
								Replace the example intervals with verified signal times for
								this camera direction. End times are exclusive. Gaps and
								overlaps become insufficient evidence.
							</p>
						</div>

						{!sameCamera && (
							<>
								<div className="section-divider">
									<span>Cross-camera synchronization</span>
								</div>
								<div className="two-inputs">
									<div className="field-block">
										<label>Rear time offset (seconds)</label>
										<input
											value={modeOffset}
											onChange={(e) => setModeOffset(e.target.value)}
											type="number"
											step="0.1"
										/>
									</div>
									<div className="field-block">
										<label>Travel window (seconds)</label>
										<div className="inline-inputs">
											<input
												value={travelMin}
												onChange={(e) => setTravelMin(e.target.value)}
												type="number"
											/>
											<input
												value={travelMax}
												onChange={(e) => setTravelMax(e.target.value)}
												type="number"
											/>
										</div>
									</div>
								</div>
								<p className="field-help">
									Cross-camera matching is conservative and requires a usable
									visual descriptor as well as a plausible time window.
								</p>
							</>
						)}
						<div className="field-block">
							<label>Frame sampling stride</label>
							<select
								value={stride}
								onChange={(e) => setStride(e.target.value)}
							>
								<option value="1">Every frame (highest compute)</option>
								<option value="2">Every 2 frames</option>
								<option value="3">Every 3 frames</option>
								<option value="5">Every 5 frames</option>
								<option value="10">Every 10 frames</option>
							</select>
						</div>
						<button
							className="primary-button"
							onClick={startAnalysis}
							disabled={busy || !trafficFile}
						>
							<Play size={17} fill="currentColor" />
							{busy ? "Preparing upload…" : "Start analysis"}
						</button>
						{message && (
							<div className="message-box">
								<InfoMark />
								{message}
							</div>
						)}
						<div className="model-readiness">
							<div className="readiness-title">
								<Gauge size={15} /> MODEL READINESS
							</div>
							<div className="readiness-row">
								<span>Motorcycle detector</span>
								<b
									className={
										health?.motorcycle_model_configured ? "ready" : "missing"
									}
								>
									{health?.motorcycle_model_configured
										? "Configured"
										: "Not configured"}
								</b>
							</div>
							<div className="readiness-row">
								<span>Rear plate detector</span>
								<b
									className={
										health?.plate_model_configured ? "ready" : "missing"
									}
								>
									{health?.plate_model_configured
										? "Configured"
										: "Not configured"}
								</b>
							</div>
							<div className="readiness-row">
								<span>OCR engine</span>
								<b className={health?.ocr_enabled ? "ready" : "missing"}>
									{health?.ocr_enabled ? "Enabled" : "Disabled"}
								</b>
							</div>
							<p>
								Without model weights, the service only validates video
								ingestion and returns diagnostics; it does not invent
								detections.
							</p>
						</div>
					</section>

					<section className="panel results-panel">
						<div className="panel-heading">
							<div>
								<div className="eyebrow">02 / INVESTIGATION</div>
								<h2>Analysis results</h2>
							</div>
							{job && <StatusPill value={job.status} />}
						</div>
						{job && (
							<div className="job-progress">
								<div className="job-line">
									<span>Job {job.id.slice(0, 8)}</span>
									<b>{Math.round(job.progress * 100)}%</b>
								</div>
								<div className="progress-track">
									<span
										style={{ width: `${Math.round(job.progress * 100)}%` }}
									/>
								</div>
								<small>{job.message}</small>
							</div>
						)}
						<div className="tabs">
							<button
								className={activeTab === "events" ? "selected" : ""}
								onClick={() => setActiveTab("events")}
							>
								Event timeline <span>{events.length}</span>
							</button>
							<button
								className={activeTab === "diagnostics" ? "selected" : ""}
								onClick={() => setActiveTab("diagnostics")}
							>
								Diagnostics
							</button>
						</div>
						{activeTab === "events" ? (
							<>
								{events.length === 0 ? (
									<div className="empty-state">
										<div className="empty-icon">
											<Clock3 size={25} />
										</div>
										<strong>No event records yet</strong>
										<p>
											Upload a video and start an analysis. Crossing events
											appear here with their signal state, association result
											and plate evidence.
										</p>
										{job?.status === "completed" && (
											<span>
												Processing completed. Check diagnostics for missing
												models or missing stop-line crossings.
											</span>
										)}
									</div>
								) : (
									<div className="event-layout">
										<div className="event-list">
											{events.map((event, index) => (
												<button
													key={event.id}
													onClick={() => setSelected(event.id)}
													className={`event-row ${selected === event.id ? "active" : ""}`}
												>
													<div className="event-row-top">
														<span className="event-index">
															{String(index + 1).padStart(2, "0")}
														</span>
														<b>{formatTime(event.timestamp_seconds)}</b>
														<StatusPill value={event.classification} />
													</div>
													<div className="event-row-bottom">
														<span>Track #{event.track_id}</span>
														<span>
															{event.plate.text ?? "Plate unreadable"}
														</span>
													</div>
												</button>
											))}
										</div>
										{chosenEvent && (
											<div className="event-detail">
												<div className="detail-heading">
													<div>
														<small>SELECTED EVENT</small>
														<h3>{formatTime(chosenEvent.timestamp_seconds)}</h3>
													</div>
													<StatusPill value={chosenEvent.classification} />
												</div>
												{chosenEvent.traffic_frame_url ? (
													<img
														className="evidence-image"
														src={chosenEvent.traffic_frame_url}
														alt="Traffic frame at crossing"
													/>
												) : (
													<div className="evidence-placeholder">
														No traffic frame stored
													</div>
												)}
												<div className="detail-grid">
													<div>
														<small>Signal state</small>
														<b>{chosenEvent.signal.color}</b>
													</div>
													<div>
														<small>Signal direction</small>
														<b>{chosenEvent.signal.direction ?? "unknown"}</b>
													</div>
													<div>
														<small>Signal interval</small>
														<b>
															{chosenEvent.signal.phase_start_seconds != null
																? `${chosenEvent.signal.phase_start_seconds}–${chosenEvent.signal.phase_end_seconds}s`
																: "unknown"}
														</b>
													</div>
													<div>
														<small>Traffic track</small>
														<b>#{chosenEvent.track_id}</b>
													</div>
													<div>
														<small>Association</small>
														<b>{chosenEvent.association.status}</b>
													</div>
													<div>
														<small>Association score</small>
														<b>
															{(
																chosenEvent.association.confidence * 100
															).toFixed(0)}
															%
														</b>
													</div>
												</div>
												<div className="plate-result">
													<div>
														<small>RECOGNIZED PLATE</small>
														<strong>
															{chosenEvent.plate.text ?? "UNREADABLE"}
														</strong>
													</div>
													<span>
														{chosenEvent.plate.status} ·{" "}
														{chosenEvent.plate.observations} frames ·{" "}
														{(chosenEvent.plate.confidence * 100).toFixed(0)}%
													</span>
												</div>
												{chosenEvent.rear_frame_urls?.length ? (
													<div className="plate-crops">
														<small>ORIGINAL REAR-VIEW FRAMES</small>
														<div>
															{chosenEvent.rear_frame_urls.map((url, i) => (
																<img
																	className="rear-frame-thumb"
																	key={url}
																	src={url}
																	alt={`Rear frame ${i + 1}`}
																/>
															))}
														</div>
													</div>
												) : null}
												{chosenEvent.plate_evidence?.length > 0 && (
													<div className="plate-crops">
														<small>AUTOMATICALLY LOCALIZED PLATE CROPS</small>
														<div>
															{chosenEvent.plate_evidence.map((url, i) => (
																<img
																	key={url}
																	src={url}
																	alt={`Plate crop ${i + 1}`}
																/>
															))}
														</div>
													</div>
												)}
												<div className="reason-box">
													<b>Evidence interpretation</b>
													<p>{chosenEvent.classification_reason}</p>
													<small>{chosenEvent.signal.reason}</small>
												</div>
											</div>
										)}
									</div>
								)}
								{job?.status === "completed" && (
									<div className="export-row">
										<span>
											<ArrowDownToLine size={15} /> Export event data
										</span>
										<a href={`/api/jobs/${job.id}/export.json`} target="_blank">
											JSON
										</a>
										<a href={`/api/jobs/${job.id}/export.csv`}>CSV</a>
									</div>
								)}
							</>
						) : (
							<div className="diagnostics">
								<pre>
									{JSON.stringify(
										job?.diagnostics ?? {
											message:
												"Run a job to view model and processing diagnostics.",
										},
										null,
										2,
									)}
								</pre>
							</div>
						)}
					</section>
				</div>
				<footer>
					<span>MOTOSIGNAL AI · LOCAL ANALYSIS BUILD</span>
					<span>
						<ShieldCheck size={13} /> Evidence-led results · Human review
						required
					</span>
				</footer>
			</main>
		</div>
	);
}
function InfoMark() {
	return <AlertTriangle size={15} />;
}
