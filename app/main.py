from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uvicorn
import logging
from typing import Optional, List, Dict, Any
import os
import time
import json

from .routers import diarization
from .routers import gpu_test
from .config import settings
from .services.diarization_service import DiarizationService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instance - will be loaded at startup
diarization_service = None

# Create necessary directories
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.DIARIZATION_OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.TRANSCRIPT_OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.GPU_REPORTS_DIR, exist_ok=True)

app = FastAPI(
    title="Audio Diarization API",
    description="API for audio diarization using Whisper and Pyannote models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    """Load models at application startup"""
    global diarization_service
    logger.info("Loading models at application startup...")
    try:
        diarization_service = DiarizationService()
        logger.info("Models loaded successfully at startup")
    except Exception as e:
        logger.error(f"Failed to load models at startup: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    global diarization_service
    logger.info("Application shutting down...")
    diarization_service = None

# Include routers
app.include_router(diarization.router, prefix="/api/diarization", tags=["diarization"])
app.include_router(gpu_test.router, prefix="/api/gpu-test", tags=["GPU Testing"])

@app.get("/")
async def root():
    return {"message": "Audio Diarization API is running"}

@app.get("/health")
async def health_check():
    global diarization_service
    return {
        "status": "healthy",
        "models_loaded": diarization_service is not None,
        "gpu_monitoring_enabled": settings.GPU_MONITORING_ENABLED
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=5000, reload=True)
