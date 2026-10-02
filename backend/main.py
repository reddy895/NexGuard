"""
NexGuard — Edge AI CCTV Surveillance System Backend Server
"""

import sys
import os
from pathlib import Path
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch

from backend.app.config import settings, UPLOADS_DIR, INCIDENTS_DIR
from backend.app.api import router as api_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="NexGuard Edge AI CCTV Surveillance & Accident Detection System"
)

# Enable CORS for React frontend dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router)


def print_startup_banner():
    device_str = "CUDA" if torch.cuda.is_available() else "CPU"
    whatsapp_str = "CONFIGURED" if settings.whatsapp_configured else "NOT CONFIGURED"

    print("\n" + "=" * 60)
    print("                    NEXGUARD")
    print("             AI CCTV SAFETY SYSTEM")
    print("=" * 60)
    print(f"YOLO:        READY ({settings.YOLO_MODEL})")
    print(f"DEVICE:      {device_str}")
    print("TRACKER:     READY")
    print("ACCIDENT AI: READY")
    print(f"WHATSAPP:    {whatsapp_str}")
    print(f"API:         http://localhost:{settings.PORT}")
    print("=" * 60)
    print("SYSTEM STATUS: READY")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print_startup_banner()
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_level="info"
    )
