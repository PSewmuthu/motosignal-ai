from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import logging

from app.db.session import SessionLocal
from app.models.entities import ProcessingJob, TrafficEvent, Video
from app.services.pipeline import run_video_analysis

logger = logging.getLogger(__name__)
executor = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="motosignal-worker")


def submit(job_id: str):
    executor.submit(process_job, job_id)


def process_job(job_id: str):
    db = SessionLocal()

    try:
        job = db.get(ProcessingJob, job_id)

        if not job:
            return

        traffic = db.get(Video, job.traffic_video_id)
        rear = db.get(Video, job.rear_video_id) if job.rear_video_id else None
        job.status, job.progress, job.message = "processing", 0.05, "Opening video streams"
        job.started_at = datetime.now(timezone.utc)
        db.commit()
        last_bucket = [-1]

        def report_progress(value: float, message: str):
            bucket = int(max(0.0, min(0.99, value)) * 20)

            if bucket != last_bucket[0]:
                job.progress = max(job.progress, min(0.99, value))
                job.message = message
                db.commit()
                last_bucket[0] = bucket

        result = run_video_analysis(traffic.stored_path, rear.stored_path if rear else None,
                                    job.id, job.config, progress_callback=report_progress)

        for payload in result["events"]:
            event = TrafficEvent(id=payload["id"], job_id=job.id, timestamp_seconds=payload["timestamp_seconds"],
                                 track_id=payload["track_id"], classification=payload["classification"],
                                 plate_text=payload["plate"].get("text"), association_confidence=payload["association_confidence"], payload=payload)
            db.add(event)

        job.status, job.progress, job.message = "completed", 1.0, "Video analysis completed"
        job.diagnostics = result["diagnostics"]
        job.finished_at = datetime.now(timezone.utc)
        db.commit()
    except Exception as exc:
        logger.exception("Processing job %s failed", job_id)
        job = db.get(ProcessingJob, job_id)

        if job:
            job.status, job.message = "failed", str(exc)
            job.finished_at = datetime.now(timezone.utc)
            job.diagnostics = {"error": str(exc)}
            db.commit()
    finally:
        db.close()
