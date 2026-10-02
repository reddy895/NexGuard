"""
NexGuard FastAPI REST API Router
"""

import os
import base64
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, Response, JSONResponse
import torch

from backend.app.config import settings, UPLOADS_DIR, INCIDENTS_DIR
from backend.app.detection.detector import NexGuardDetector
from backend.app.tracking.tracker import NexGuardTracker
from backend.app.accident.analyzer import AccidentDetector
from backend.app.severity.classifier import SeverityClassifier
from backend.app.notifications.whatsapp import whatsapp_notifier
from backend.app.incidents.storage import incident_store

router = APIRouter(prefix="/api")

# Singletons for system
detector = NexGuardDetector()
tracker = NexGuardTracker()
accident_detector = AccidentDetector()

# Live CCTV state simulation
class CCTVState:
    active: bool = False
    source: str = "0"
    cap: Optional[cv2.VideoCapture] = None

cctv_state = CCTVState()


class SeverityTestRequest(BaseModel):
    test_id: int  # 1 to 5
    collision_detected: bool
    collision_score: float
    vehicles_involved: int
    people_involved: int
    has_vulnerable_user: bool = False
    post_stop: bool = False


@router.get("/status")
def get_system_status() -> Dict[str, Any]:
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "READY",
        "yolo": "READY" if detector.ready else "ERROR",
        "device": device,
        "tracker": "READY",
        "accident_ai": "READY",
        "whatsapp": "CONFIGURED" if settings.whatsapp_configured else "NOT CONFIGURED",
        "whatsapp_details": whatsapp_notifier.get_status()
    }


@router.post("/upload")
async def upload_media(file: UploadFile = File(...)) -> Dict[str, Any]:
    allowed_exts = {".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = Path(file.filename).suffix.lower()
    
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(allowed_exts)}"
        )

    file_path = UPLOADS_DIR / f"{int(time.time())}_{file.filename}"
    contents = await file.read()
    
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    with open(file_path, "wb") as f:
        f.write(contents)

    media_type = "video" if ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"} else "image"

    return {
        "status": "SUCCESS",
        "filename": file.filename,
        "saved_path": str(file_path),
        "media_type": media_type,
        "file_size_bytes": len(contents)
    }


@router.post("/detect/image")
async def detect_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image file.")

    # Reset tracker for single image analysis
    tracker.reset()
    accident_detector.reset()

    detections = detector.detect(frame)
    tracked_objects = tracker.update(detections)
    analysis = accident_detector.analyze_frame(tracked_objects)

    # Count objects by class
    counts: Dict[str, int] = {}
    for d in detections:
        counts[d.class_name] = counts.get(d.class_name, 0) + 1

    # Annotate frame
    annotated = detector.draw_annotations(frame, detections, show_ids=True)

    # Draw accident overlay banner if incident
    if analysis.is_accident:
        h, w = annotated.shape[:2]
        banner_h = 60
        cv2.rectangle(annotated, (0, 0), (w, banner_h), (0, 0, 180), -1)
        text = f"ACCIDENT DETECTED | SEVERITY: {analysis.severity.level} ({int(analysis.severity.confidence * 100)}%)"
        cv2.putText(annotated, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Save evidence
        rec = incident_store.record_incident(
            frame=annotated,
            severity_level=analysis.severity.level,
            confidence=analysis.severity.confidence,
            detected_objects=counts,
            reasons=analysis.severity.reasons
        )
        analysis.incident_id = rec.incident_id

    # Encode annotated image to JPEG base64
    _, buffer = cv2.imencode(".jpg", annotated)
    b64_image = base64.b64encode(buffer).decode("utf-8")

    return {
        "status": "COMPLETED",
        "filename": file.filename,
        "media_type": "image",
        "detection_count": len(detections),
        "detected_classes": counts,
        "is_accident": analysis.is_accident,
        "severity": analysis.severity.level,
        "severity_score": analysis.severity.score,
        "reasons": analysis.severity.reasons,
        "incident_id": analysis.incident_id,
        "annotated_image_b64": f"data:image/jpeg;base64,{b64_image}"
    }


@router.post("/detect/video")
async def detect_video(filepath: str) -> Dict[str, Any]:
    path = Path(filepath)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found at path: {filepath}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Failed to open video file.")

    tracker.reset()
    accident_detector.reset()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    frame_index = 0
    max_counts: Dict[str, int] = {}
    incident_triggered = False
    highest_severity = "NORMAL"
    highest_score = 0.0
    recorded_incident_id = None

    sampled_b64_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        # Process every Nth frame for performance optimization if needed
        detections = detector.detect(frame)
        tracked_objects = tracker.update(detections)
        analysis = accident_detector.analyze_frame(tracked_objects)

        # Update max detected counts
        curr_counts: Dict[str, int] = {}
        for d in detections:
            curr_counts[d.class_name] = curr_counts.get(d.class_name, 0) + 1
        for cls, cnt in curr_counts.items():
            max_counts[cls] = max(max_counts.get(cls, 0), cnt)

        if analysis.is_accident:
            incident_triggered = True
            if analysis.severity.score > highest_score:
                highest_score = analysis.severity.score
                highest_severity = analysis.severity.level

            if not recorded_incident_id:
                annotated_ev = detector.draw_annotations(frame, detections, show_ids=True)
                rec = incident_store.record_incident(
                    frame=annotated_ev,
                    severity_level=analysis.severity.level,
                    confidence=analysis.severity.confidence,
                    detected_objects=max_counts,
                    reasons=analysis.severity.reasons
                )
                recorded_incident_id = rec.incident_id

        # Sample preview frames (first, middle, incident frame, last)
        if frame_index == 1 or frame_index == (total_frames // 2) or (analysis.is_accident and len(sampled_b64_frames) < 4):
            annotated = detector.draw_annotations(frame, detections, show_ids=True)
            _, buf = cv2.imencode(".jpg", annotated)
            b64 = base64.b64encode(buf).decode("utf-8")
            sampled_b64_frames.append(f"data:image/jpeg;base64,{b64}")

    cap.release()

    return {
        "status": "COMPLETED",
        "filepath": str(path),
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "detected_classes": max_counts,
        "is_accident": incident_triggered,
        "severity": highest_severity,
        "highest_score": round(highest_score, 4),
        "incident_id": recorded_incident_id,
        "preview_frames": sampled_b64_frames
    }


@router.post("/cctv/start")
def start_cctv(source: str = "0") -> Dict[str, Any]:
    if cctv_state.active:
        return {"status": "ALREADY_RUNNING", "source": cctv_state.source}

    src_val: Any = 0 if source in ["0", "webcam"] else source
    cap = cv2.VideoCapture(src_val)
    if not cap.isOpened():
        raise HTTPException(
            status_code=400,
            detail=f"Could not connect to camera/RTSP stream source: '{source}'"
        )

    cctv_state.active = True
    cctv_state.source = source
    cctv_state.cap = cap
    tracker.reset()
    accident_detector.reset()

    return {"status": "STARTED", "source": source}


@router.post("/cctv/stop")
def stop_cctv() -> Dict[str, Any]:
    if cctv_state.cap:
        cctv_state.cap.release()
    cctv_state.active = False
    cctv_state.cap = None
    return {"status": "STOPPED"}


@router.get("/cctv/frame")
def get_cctv_frame() -> Dict[str, Any]:
    if not cctv_state.active or not cctv_state.cap:
        raise HTTPException(status_code=400, detail="CCTV stream is not active. Call /api/cctv/start first.")

    ret, frame = cctv_state.cap.read()
    if not ret:
        cctv_state.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video simulation
        ret, frame = cctv_state.cap.read()
        if not ret:
            raise HTTPException(status_code=500, detail="Failed to read frame from CCTV source.")

    detections = detector.detect(frame)
    tracked_objects = tracker.update(detections)
    analysis = accident_detector.analyze_frame(tracked_objects)

    annotated = detector.draw_annotations(frame, detections, show_ids=True)

    counts: Dict[str, int] = {}
    for d in detections:
        counts[d.class_name] = counts.get(d.class_name, 0) + 1

    if analysis.is_accident:
        h, w = annotated.shape[:2]
        cv2.rectangle(annotated, (0, 0), (w, 50), (0, 0, 180), -1)
        cv2.putText(
            annotated,
            f"ACCIDENT DETECTED | SEVERITY: {analysis.severity.level} ({int(analysis.severity.confidence * 100)}%)",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    _, buf = cv2.imencode(".jpg", annotated)
    b64 = base64.b64encode(buf).decode("utf-8")

    return {
        "active": True,
        "detection_count": len(detections),
        "detected_classes": counts,
        "is_accident": analysis.is_accident,
        "severity": analysis.severity.level,
        "confidence": analysis.severity.confidence,
        "reasons": analysis.severity.reasons,
        "frame_b64": f"data:image/jpeg;base64,{b64}"
    }


@router.get("/incidents")
def list_incidents() -> List[Dict[str, Any]]:
    return incident_store.list_incidents()


@router.get("/incidents/{incident_id}")
def get_incident_details(incident_id: str) -> Dict[str, Any]:
    rec = incident_store.get_incident(incident_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Incident '{incident_id}' not found.")
    return rec.to_dict()


@router.get("/incidents/{incident_id}/evidence")
def get_incident_evidence(incident_id: str):
    img_path = INCIDENTS_DIR / f"{incident_id}.jpg"
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Evidence image for '{incident_id}' not found.")
    return FileResponse(str(img_path), media_type="image/jpeg")


@router.post("/test/severity")
def run_severity_test(req: SeverityTestRequest) -> Dict[str, Any]:
    """Automated severity evaluation test system according to section 5 specifications."""
    eval_result = SeverityClassifier.classify(
        collision_detected=req.collision_detected,
        collision_score=req.collision_score,
        vehicles_involved=req.vehicles_involved,
        people_involved=req.people_involved,
        has_vulnerable_user=req.has_vulnerable_user,
        post_stop=req.post_stop
    )

    expected_levels = {
        1: "NORMAL",
        2: "LOW",
        3: "MEDIUM",
        4: "HIGH",
        5: "CRITICAL"
    }
    expected = expected_levels.get(req.test_id, "UNKNOWN")
    passed = (eval_result.level == expected)

    return {
        "test_id": req.test_id,
        "expected_severity": expected,
        "evaluated_severity": eval_result.level,
        "score": eval_result.score,
        "passed": passed,
        "reasons": eval_result.reasons
    }
