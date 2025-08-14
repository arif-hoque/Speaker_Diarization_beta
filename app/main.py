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
from .config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.DIARIZATION_OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.TRANSCRIPT_OUTPUT_DIR, exist_ok=True)

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

# Include routers
app.include_router(diarization.router, prefix="/api/diarization", tags=["diarization"])

@app.get("/")
async def root():
    return {"message": "Audio Diarization API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=5000, reload=True)
