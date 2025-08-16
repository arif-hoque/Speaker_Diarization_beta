from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from typing import Optional
import os
import logging
from pathlib import Path
import shutil
import uuid

from ..services.diarization_service import DiarizationService
from ..core.dependencies import get_diarization_service
from ..config import settings
from ..schemas.diarization import DiarizationResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/process", response_model=DiarizationResponse)
async def process_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    num_speakers: Optional[int] = None,
    language: Optional[str] = None,
    translate: bool = False,
    prompt: str = "",
    service: DiarizationService = Depends(get_diarization_service)
):
    """
    Process an audio file for diarization and transcription.
    
    Args:
        file: Audio file to process
        num_speakers: Number of speakers (optional, auto-detect)
        language: Language code (e.g., 'en', 'es')
        translate: Whether to translate to English
        prompt: Optional prompt for better transcription
        
    Returns:
        JSON response with diarization and transcription results
    """
    # Validate that filename exists
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="File must have a filename"
        )
    
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower().lstrip('.')
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        allowed_types = ', '.join(settings.ALLOWED_EXTENSIONS)
        raise HTTPException(
            status_code=400,
            detail=f"File type '{file_ext}' not allowed. "
                   f"Allowed types: {allowed_types}"
        )
    
    try:
        # Ensure upload directory exists
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        
        # Create a unique filename
        unique_id = str(uuid.uuid4())
        filename = f"{unique_id}_{file.filename}"
        file_path = os.path.join(settings.UPLOAD_DIR, filename)
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the file
        result = service.process_audio(
            file_path=file_path,
            num_speakers=num_speakers,
            language=language,
            translate=translate,
            prompt=prompt
        )
        
        # Add cleanup task
        background_tasks.add_task(cleanup_file, file_path)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio: {str(e)}"
        )


@router.get("/status/{task_id}")
async def get_status(task_id: str):
    """Check the status of a diarization task"""
    # TODO: Implement task status tracking
    return {"status": "completed", "task_id": task_id}


def cleanup_file(file_path: str):
    """Clean up temporary files"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {str(e)}")
