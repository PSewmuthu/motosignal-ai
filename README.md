# MotoSignal AI

**Evidence-driven motorcycle red-light event analysis and rear-view license-plate recognition.**

MotoSignal AI is a local-first, modular computer-vision application for a technical assignment. It provides a FastAPI backend, a React dashboard, background video processing, configurable stop-line and signal-phase logic, automatic plate localization hooks, multi-frame OCR consensus, conservative cross-camera association, evidence storage, JSON/CSV exports, Docker Compose, and automated tests.

## Quick start (Docker)

Prerequisites: Docker Engine and Docker Compose v2.

```bash
git clone https://github.com/PSewmuthu/motosignal-ai.git
cd motosignal-ai
cp .env.example .env
docker compose up --build
```

Open:

- Dashboard: <http://localhost:8080>
- API documentation: <http://localhost:8000/docs>
- Health: <http://localhost:8000/api/health>

The default image is CPU-compatible and does not install the optional YOLO/PaddleOCR/EasyOCR packages. Uploading and job management work without model files; model readiness and missing configuration are shown in the dashboard diagnostics.

Stop the services with `Ctrl+C`, then:

```bash
docker compose down
```

Uploaded videos, SQLite data, and event evidence are stored under `data/` through a persistent volume. This directory is git-ignored.

## Enable model inference

1. Obtain or train a motorcycle detector and a rear license-plate detector. Check dataset, checkpoint, and dependency licenses before using or distributing them.
2. Place weights in `models/` (the directory is mounted read-only into the API container). Keep model weights out of Git unless their license and size permit it.
3. Set model paths in `.env`:

```dotenv
MOTOSIGNAL_MOTORCYCLE_MODEL=/models/motorcycle.pt
MOTOSIGNAL_MOTORCYCLE_CLASS_ID=3
MOTOSIGNAL_PLATE_MODEL=/models/license_plate.pt
MOTOSIGNAL_ENABLE_OCR=true
INSTALL_ML_DEPS=true
```

`MOTOSIGNAL_MOTORCYCLE_CLASS_ID=3` is the COCO motorcycle class index. Change it to the motorcycle class index used by your fine-tuned checkpoint. The plate detector uses all model classes; ideally export a plate-only detector.

4. Rebuild:

```bash
docker compose build --no-cache api
docker compose up
```

The optional ML dependency set (Ultralytics, PaddleOCR, PaddlePaddle and EasyOCR) is version-pinned as a starting point, not a guarantee that every model checkpoint or hardware combination is compatible. Test the exact PaddleOCR/PaddlePaddle, EasyOCR and Ultralytics versions you deploy. Review the Ultralytics license and any model-specific terms; the repository's own license does not override third-party licenses.

## Local development

Backend (Python 3.11 or 3.12 recommended):

```bash
cd backend
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# Optional, when you have model weights:
# pip install -r requirements-ml.txt
MOTOSIGNAL_DATA_DIR=../data uvicorn app.main:app --reload --port 8000
```

Frontend (Node 22 recommended):

```bash
cd frontend
npm install
npm run dev
```

The Vite development server listens on port 5173. For local development, either configure a proxy to `http://localhost:8000` or use the Docker/Nginx setup, which proxies `/api` and `/evidence` automatically. The production-style Compose deployment is the recommended demonstration path.

Run backend tests:

```bash
cd backend
pip install -e ".[test]"
pytest -q
```

## First analysis workflow

1. Start the services and check **Model readiness** in the dashboard.
2. Upload your traffic clip; the browser provides a local video preview for checking the source before processing. For the supplied single-view clip, select **Same camera stream**.
3. Set the stop-line endpoints in normalized coordinates (`x` and `y` from 0 to 1). Place the line on the actual road stop line and select the approach side. The system uses the bottom-centre of each motorcycle box and a time-ordered transition across the finite line segment; bounding-box overlap alone is not used.
4. Add a verified signal timeline as JSON. Example format:

```json
[
	{
		"start_seconds": 0.0,
		"end_seconds": 12.0,
		"color": "green",
		"direction": "northbound"
	},
	{
		"start_seconds": 12.0,
		"end_seconds": 20.0,
		"color": "red",
		"direction": "northbound"
	},
	{
		"start_seconds": 20.0,
		"end_seconds": 35.0,
		"color": "green",
		"direction": "northbound"
	}
]
```

These values are illustrative only. Replace them with timestamps actually verified from your footage or a reliable signal-phase source. Time intervals are half-open (`start_seconds <= t < end_seconds`). A gap, overlap, `amber`, or `unknown` phase is treated as insufficient evidence.

5. Select a sampling stride and start analysis.
6. Inspect each result's crossing timestamp, traffic frame, signal phase, association status, plate consensus, and rear-frame crops. Export JSON or CSV.

### Single-camera and two-camera operation

- **Same camera:** traffic and rear-plate detections share one decoded frame stream and one tracker. This is the preferred first demonstration when only one clip is available.
- **Separate cameras:** upload both streams, provide the rear-camera clock offset and a plausible travel-time interval. The matcher requires both a valid temporal window and a visual appearance descriptor, and it abstains on low-confidence or ambiguous candidates. The current colour-histogram descriptor is a baseline, not a validated re-identification model. Do not claim robust cross-camera re-identification until evaluated on paired, labelled camera footage.

## Classification semantics

| Output                  | Meaning                                                                                                                                                                                  |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SUSPECTED_VIOLATION`   | A tracked motorcycle crossed the configured line during a configured red phase, with a supported same-camera or cross-camera association. This is a review flag, not a legal conclusion. |
| `NOT_A_VIOLATION`       | The configured signal timeline is green at the confirmed crossing time.                                                                                                                  |
| `INSUFFICIENT_EVIDENCE` | The crossing, signal state, or required association is uncertain or unavailable.                                                                                                         |

OCR uses pretrained PaddleOCR and, when installed, EasyOCR. It compares a small number of conservative image variants (upscaled/CLAHE, grayscale and adaptive threshold), prefers agreement between engines, and combines results across distinct video frames. No OCR model is trained or fine-tuned. Disagreement or weak evidence should remain unreadable rather than being guessed.

Plate readability is independent of event classification. A suspected event may have an unreadable plate. OCR consensus is based on multiple detections from video frames and abstains when evidence is weak or conflicting; it does not fill missing characters from a presumed registration format.

## Experiment first, implement second

The `notebooks/` directory contains independent experiments. Notebooks do not import project scripts or application modules. Use them first to inspect data, test models, compare settings, and understand results through annotated images, plots, tables, and timelines. Once an approach is understood and worth keeping, implement it separately in the application and add tests. The application does not execute notebooks. See [`notebooks/README.md`](notebooks/README.md).

## Architecture

```text
React dashboard :8080
        │ /api and /evidence
        ▼
FastAPI :8000 ── SQLite (jobs, videos, events)
        │
        └── Background worker
              │
              ├── Decode frame stream once per camera
              ├── YOLO motorcycle detector + persistent BoT-SORT tracker (standard configuration)
              ├── Anchor trajectory → finite stop-line crossing
              ├── Crossing timestamp → direction-specific signal verifier
              ├── Rear plate detector → plate crops → OCR predictions
              ├── Multi-frame consensus / abstention
              ├── Cross-camera time + appearance association (separate-camera mode)
              └── Explainable event classification + evidence exports
```

The code is kept as a straightforward modular application rather than a collection of networked microservices. The motorcycle path uses the Ultralytics built-in tracking API with a custom BoT-SORT profile; the old IoU tracker is retained only for deterministic tests/fallback. Frame stride defaults to 1 for crossing-sensitive tracking. Each camera stream gets its own persistent tracker instance. The single background worker keeps model loading and long video work out of HTTP request handlers. For larger workloads, replace the in-process executor with a durable task queue and add retention/access controls before exposing the application to multiple users.

See [`docs/architecture.md`](docs/architecture.md), [`docs/api.md`](docs/api.md), and [`docs/evaluation.md`](docs/evaluation.md).

## Motorcycle tracking with YOLO + BoT-SORT

The project uses `model.track(frame, persist=True, tracker=...)`, not a separate `model.predict()` followed by a hand-written IoU association. The default configuration is `backend/app/configs/trackers/botsort_standard.yaml`: BoT-SORT with a straightforward fixed-camera configuration and no optional appearance ReID. Controlled baselines (`botsort_standard.yaml` and `bytetrack_baseline.yaml`) are included for evaluation. See [tracking configuration and evaluation](docs/architecture.md#tracking-configuration-and-accuracy-notes).

Configure these in `.env`:

```dotenv
MOTOSIGNAL_MOTORCYCLE_MODEL=/models/motorcycle.pt
MOTOSIGNAL_MOTORCYCLE_CLASS_ID=3
MOTOSIGNAL_MOTORCYCLE_CONFIDENCE=0.08
MOTOSIGNAL_MOTORCYCLE_NMS_IOU=0.65
MOTOSIGNAL_MOTORCYCLE_IMAGE_SIZE=1280
MOTOSIGNAL_YOLO_DEVICE=0
MOTOSIGNAL_FRAME_STRIDE=1
INSTALL_ML_DEPS=true
```

For a custom motorcycle-only checkpoint, set its motorcycle class index (often `0`, but verify the training `data.yaml`). For a COCO checkpoint, motorcycle is class `3`. Keep `MOTOSIGNAL_FRAME_STRIDE=1` for accurate stop-line crossing times. More detail in [model documentation](docs/model_documentation.md).

You can test the tracker independently before running the full application:

```bash
pip install -r backend/requirements-ml.txt
python scripts/track_video.py --source video.mp4 --weights models/motorcycle.pt \
  --class-id 3 --stride 1 --device 0
```

The script writes `runs/tracking/tracked.mp4` and `runs/tracking/tracks.csv` for visual inspection and evaluation. It does not imply that tracking is accurate until ID switches, fragmentation and track recall have been measured on labelled video.

## Validation and evaluation helpers

Validate a signal schedule before submitting it:

```bash
python scripts/validate_signal_timeline.py configs/signal_example.json
```

Evaluate labelled event classifications (`y_true`, `y_pred`) and plate OCR (`ground_truth`, `prediction`) with:

```bash
python scripts/evaluate_events.py event_labels.csv
python scripts/evaluate_ocr.py plate_labels.csv
```

The OCR helper reports readable coverage, exact match, and character error rate on readable predictions. Keep unreadable ground-truth cases in a separately labelled set to measure false-readable and abstention behaviour. Metric values are meaningful only when the CSV labels are independently reviewed.

## Repository layout

```text
motosignal-ai/
├── backend/             # FastAPI API, SQLAlchemy, worker, CV pipeline, tests
├── frontend/            # React + TypeScript dashboard
├── backend/app/configs/trackers/  # Custom Ultralytics BoT-SORT YAML profile
├── notebooks/           # Dataset exploration and experiments; never runtime dependencies
├── configs/             # Example camera and signal configuration
├── data/                # Local uploads
├── models/              # Local model weights
├── docs/                # Architecture, and model documentation
└── docker-compose.yaml  # CPU-first deployment
```
