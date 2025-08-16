"""
Dependency injection for shared services
"""
from fastapi import HTTPException

from ..services.diarization_service import DiarizationService


def get_diarization_service() -> DiarizationService:
    """Get the global diarization service instance"""
    from ..main import diarization_service
    if diarization_service is None:
        raise HTTPException(
            status_code=503, 
            detail="Models not loaded yet, please wait for startup to complete"
        )
    return diarization_service
