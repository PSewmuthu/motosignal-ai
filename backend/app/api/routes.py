from __future__ import annotations
import csv
import io
import uuid
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.models.entities import ProcessingJob, TrafficEvent, Video
from app.schemas.api import CreateJobRequest
from app.workers.video_worker import submit

router = APIRouter(prefix="/api")


def _video_dict(video: Video):
    return {"id": video.id, "filename": video.filename, "role": video.role, "size_bytes": video.size_bytes,
            "duration_seconds": video.duration_seconds, "fps": video.fps, "width": video.width,
            "height": video.height, "created_at": video.created_at}


def _job_dict(job: ProcessingJob):
    return {"id": job.id, "traffic_video_id": job.traffic_video_id, "rear_video_id": job.rear_video_id,
            "status": job.status, "progress": job.progress, "message": job.message, "config": job.config,
            "diagnostics": job.diagnostics, "created_at": job.created_at, "started_at": job.started_at,
            "finished_at": job.finished_at}


@router.get("/health")
def health():
    return {"status": "ok", "service": "motosignal-api",
            "motorcycle_model_configured": bool(settings.motorcycle_model and Path(settings.motorcycle_model).is_file()),
            "plate_model_configured": bool(settings.plate_model and Path(settings.plate_model).is_file()),
            "ocr_enabled": settings.enable_ocr}


@router.get("/videos")
def list_videos(db: Session = Depends(get_db)):
    return [_video_dict(v) for v in db.query(Video).order_by(Video.created_at.desc()).all()]


@router.post("/videos", status_code=201)
async def upload_video(file: UploadFile = File(...), role: str = Form("traffic"), db: Session = Depends(get_db)):
    if role not in ("traffic", "rear", "combined"):
        raise HTTPException(422, "role must be traffic, rear, or combined")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise HTTPException(
            415, f"Unsupported video extension: {suffix or '(missing)'}")

    video_id = str(uuid.uuid4())
    destination = settings.data_dir / "uploads" / f"{video_id}{suffix}"
    size = 0
    max_bytes = settings.max_upload_mb * 1024 * 1024

    try:
        with destination.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    destination.unlink(missing_ok=True)
                    raise HTTPException(
                        413, f"Video exceeds configured {settings.max_upload_mb} MB limit")

                output.write(chunk)
    finally:
        await file.close()

    cap = cv2.VideoCapture(str(destination))
    if not cap.isOpened():
        destination.unlink(missing_ok=True)
        raise HTTPException(400, "Uploaded file is not a readable video")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    video = Video(id=video_id, filename=Path(file.filename or "video").name, stored_path=str(destination), role=role,
                  size_bytes=size, fps=fps or None, duration_seconds=(frames / fps if fps > 0 and frames > 0 else None),
                  width=width or None, height=height or None)
    db.add(video)
    db.commit()
    db.refresh(video)

    return _video_dict(video)


@router.post("/jobs", status_code=202)
def create_job(request: CreateJobRequest, db: Session = Depends(get_db)):
    traffic = db.get(Video, request.traffic_video_id)
    if not traffic:
        raise HTTPException(404, "traffic_video_id was not found")

    rear = db.get(
        Video, request.rear_video_id) if request.rear_video_id else None
    if request.rear_video_id and not rear:
        raise HTTPException(404, "rear_video_id was not found")

    if request.config.mode == "separate_cameras" and traffic.id == rear.id:
        raise HTTPException(
            422, "Use same_camera mode when both roles refer to the same uploaded video")

    job = ProcessingJob(id=str(uuid.uuid4()), traffic_video_id=traffic.id, rear_video_id=rear.id if rear else None,
                        config=request.config.model_dump(), status="queued", progress=0.0, message="Queued")
    db.add(job)
    db.commit()
    db.refresh(job)
    submit(job.id)

    return _job_dict(job)


@router.get("/jobs")
def list_jobs(db: Session = Depends(get_db)):
    return [_job_dict(job) for job in db.query(ProcessingJob).order_by(ProcessingJob.created_at.desc()).limit(100).all()]


@router.get("/jobs/{job_id}")
def get_job(job_id: str, db: Session = Depends(get_db)):
    job = db.get(ProcessingJob, job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    return _job_dict(job)


@router.get("/jobs/{job_id}/events")
def get_events(job_id: str, db: Session = Depends(get_db)):
    if not db.get(ProcessingJob, job_id):
        raise HTTPException(404, "Job not found")

    events = db.query(TrafficEvent).filter(TrafficEvent.job_id == job_id).order_by(
        TrafficEvent.timestamp_seconds).all()

    return [event.payload for event in events]


@router.get("/jobs/{job_id}/export.json")
def export_json(job_id: str, db: Session = Depends(get_db)):
    job = db.get(ProcessingJob, job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    events = db.query(TrafficEvent).filter(TrafficEvent.job_id == job_id).order_by(
        TrafficEvent.timestamp_seconds).all()

    return {"job_id": job.id, "status": job.status, "config": job.config,
            "diagnostics": job.diagnostics, "events": [event.payload for event in events]}


@router.get("/jobs/{job_id}/export.csv")
def export_csv(job_id: str, db: Session = Depends(get_db)):
    if not db.get(ProcessingJob, job_id):
        raise HTTPException(404, "Job not found")

    events = db.query(TrafficEvent).filter(TrafficEvent.job_id == job_id).order_by(
        TrafficEvent.timestamp_seconds).all()
    output = io.StringIO()
    fields = ["event_id", "timestamp_seconds", "traffic_track_id", "rear_track_id", "signal_color", "classification",
              "signal_direction", "signal_phase_start_seconds", "signal_phase_end_seconds",
              "plate_text", "plate_status", "plate_confidence", "association_status", "association_confidence",
              "traffic_frame_url", "rear_frame_urls", "plate_evidence_urls", "classification_reason"]

    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for event in events:
        p = event.payload
        writer.writerow({"event_id": event.id, "timestamp_seconds": event.timestamp_seconds,
                         "traffic_track_id": event.track_id, "rear_track_id": p.get("rear_track_id"),
                         "signal_color": p.get("signal", {}).get("color"),
                         "signal_direction": p.get("signal", {}).get("direction"),
                         "signal_phase_start_seconds": p.get("signal", {}).get("phase_start_seconds"),
                         "signal_phase_end_seconds": p.get("signal", {}).get("phase_end_seconds"),
                         "classification": event.classification,
                         "plate_text": p.get("plate", {}).get("text") or "UNREADABLE", "plate_status": p.get("plate", {}).get("status"),
                         "plate_confidence": p.get("plate", {}).get("confidence"),
                         "association_status": p.get("association", {}).get("status"),
                         "association_confidence": event.association_confidence,
                         "traffic_frame_url": p.get("traffic_frame_url"),
                         "rear_frame_urls": "|".join(p.get("rear_frame_urls", [])),
                         "plate_evidence_urls": "|".join(p.get("plate_evidence", [])),
                         "classification_reason": p.get("classification_reason")})
    output.seek(0)

    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f"attachment; filename=motosignal-{job_id}.csv"})
