"""
GPU monitoring and performance testing endpoints for audio diarization.
"""
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from fastapi.responses import JSONResponse, FileResponse

from ..libs.gpu_reporter import GPUReporter
from ..config import settings

router = APIRouter(tags=["GPU Testing"])

# Initialize GPU reporter
gpu_reporter = GPUReporter(reports_dir=settings.GPU_REPORTS_DIR)

def get_diarization_service():
    """Get the global diarization service instance"""
    from ..main import diarization_service
    if diarization_service is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet, please wait for startup to complete")
    return diarization_service


@router.get("/baseline/{duration}", response_model=Dict[str, Any])
async def test_baseline_gpu_usage(duration: int = 30) -> Dict[str, Any]:
    """
    Test baseline GPU usage without any audio processing
    
    Args:
        duration: Duration in seconds to monitor baseline usage (default: 30)
    
    Returns:
        JSON report of baseline GPU usage
    """
    if not settings.GPU_MONITORING_ENABLED:
        raise HTTPException(status_code=400, detail="GPU monitoring is disabled")
    
    if duration < 5 or duration > 300:
        raise HTTPException(status_code=400, detail="Duration must be between 5 and 300 seconds")
    
    try:
        # Generate baseline report
        report_data = gpu_reporter.generate_baseline_report(duration=duration)
        
        # Save the report in multiple formats
        saved_files = gpu_reporter.save_report(
            report_data, 
            f"baseline_{duration}s",
            formats=['json', 'txt']
        )
        
        # Add file paths to response
        report_data['saved_reports'] = saved_files
        
        return JSONResponse(content=report_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate baseline report: {str(e)}")


@router.post("/single-audio", response_model=Dict[str, Any])
async def test_single_audio_gpu_usage(
    file: UploadFile = File(...),
    service = Depends(get_diarization_service)
) -> Dict[str, Any]:
    """
    Test GPU usage for processing a single audio file using parallel processing
    
    Args:
        file: Audio file to process
    
    Returns:
        JSON report including processing result and GPU usage statistics
    """
    if not settings.GPU_MONITORING_ENABLED:
        raise HTTPException(status_code=400, detail="GPU monitoring is disabled")
    
    # Validate file type
    if not file.filename or not any(file.filename.lower().endswith(ext) for ext in ['.wav', '.mp3', '.mp4', '.flac', '.m4a']):
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    # Save uploaded file temporarily
    temp_file_path = Path(settings.UPLOAD_DIR) / f"temp_{int(time.time())}_{file.filename}"
    temp_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Generate report using parallel processing
        def process_parallel(file_path):
            return service.process_audio(str(file_path))
            
        report_data = gpu_reporter.generate_single_audio_report(
            audio_filename=file.filename,
            process_func=process_parallel,
            file_path=temp_file_path
        )
        
        # Save the report in multiple formats
        saved_files = gpu_reporter.save_report(
            report_data, 
            f"single_{file.filename.replace('.', '_')}",
            formats=['json', 'txt', 'csv']
        )
        
        # Add file paths to response
        report_data['saved_reports'] = saved_files
        
        return JSONResponse(content=report_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process audio file: {str(e)}")
    
    finally:
        # Clean up temp file
        if temp_file_path.exists():
            temp_file_path.unlink()


@router.post("/multiple-audio", response_model=Dict[str, Any])
async def test_multiple_audio_gpu_usage(
    files: List[UploadFile] = File(...),
    service = Depends(get_diarization_service)
) -> Dict[str, Any]:
    """
    Test GPU usage for processing multiple audio files sequentially
    
    Args:
        files: List of audio files to process
    
    Returns:
        JSON report with individual results and aggregate statistics
    """
    if not settings.GPU_MONITORING_ENABLED:
        raise HTTPException(status_code=400, detail="GPU monitoring is disabled")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per request")
    
    temp_files = []
    
    try:
        # Save all uploaded files temporarily
        for file in files:
            if not file.filename or not any(file.filename.lower().endswith(ext) for ext in ['.wav', '.mp3', '.mp4', '.flac', '.m4a']):
                raise HTTPException(status_code=400, detail=f"Unsupported file format: {file.filename}")
            
            temp_file_path = Path(settings.UPLOAD_DIR) / f"temp_{int(time.time())}_{file.filename}"
            temp_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(temp_file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            temp_files.append((file.filename, temp_file_path))
        
        # Generate report for multiple files
        audio_filenames = [filename for filename, _ in temp_files]
        
        def process_multiple_with_service(temp_files):
            return _process_multiple_audio_sync(temp_files, service)
        
        report_data = gpu_reporter.generate_multiple_audio_report(
            audio_files=audio_filenames,
            process_func=process_multiple_with_service,
            temp_files=temp_files
        )
        
        # Save the report in multiple formats
        saved_files = gpu_reporter.save_report(
            report_data, 
            f"multiple_{len(files)}_files",
            formats=['json', 'txt', 'csv']
        )
        
        # Add file paths to response
        report_data['saved_reports'] = saved_files
        
        return JSONResponse(content=report_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process audio files: {str(e)}")
    
    finally:
        # Clean up temp files
        for _, temp_file_path in temp_files:
            if temp_file_path.exists():
                temp_file_path.unlink()


@router.get("/reports/{report_filename}")
async def download_report(report_filename: str):
    """
    Download a generated report file
    
    Args:
        report_filename: Name of the report file to download
    
    Returns:
        File download response
    """
    report_path = Path(settings.GPU_REPORTS_DIR) / report_filename
    
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report file not found")
    
    # Determine media type based on file extension
    if report_filename.endswith('.json'):
        media_type = 'application/json'
    elif report_filename.endswith('.csv'):
        media_type = 'text/csv'
    else:
        media_type = 'text/plain'
    
    return FileResponse(
        path=str(report_path),
        filename=report_filename,
        media_type=media_type
    )


@router.get("/reports")
async def list_reports() -> Dict[str, List[str]]:
    """
    List all available report files
    
    Returns:
        Dictionary with lists of report files by type
    """
    reports_dir = Path(settings.GPU_REPORTS_DIR)
    
    if not reports_dir.exists():
        return {"json": [], "txt": [], "csv": []}
    
    json_files = [f.name for f in reports_dir.glob("*.json")]
    txt_files = [f.name for f in reports_dir.glob("*.txt")]
    csv_files = [f.name for f in reports_dir.glob("*.csv")]
    
    return {
        "json": sorted(json_files),
        "txt": sorted(txt_files),
        "csv": sorted(csv_files)
    }


@router.get("/status")
async def gpu_monitoring_status(service = Depends(get_diarization_service)) -> Dict[str, Any]:
    """
    Get current GPU monitoring status and configuration
    
    Returns:
        Status information about GPU monitoring setup
    """
    gpu_usage = gpu_reporter.gpu_monitor.get_gpu_usage()
    
    return {
        "gpu_monitoring_enabled": settings.GPU_MONITORING_ENABLED,
        "monitoring_interval": settings.GPU_MONITORING_INTERVAL,
        "reports_directory": str(settings.GPU_REPORTS_DIR),
        "current_gpu_usage": gpu_usage,
        "gpu_available": gpu_usage is not None
    }


# Helper functions for processing audio
def _process_multiple_audio_sync(temp_files: List[tuple], service) -> List[Dict[str, Any]]:
    """Synchronously process multiple audio files"""
    results = []
    for filename, file_path in temp_files:
        try:
            result = service.process_audio(str(file_path))
            results.append(result)
        except Exception as e:
            # Add error result
            results.append({
                "success": False,
                "message": f"Error processing {filename}: {str(e)}",
                "language": "unknown",
                "num_speakers": 0,
                "duration": 0.0,
                "segments": []
            })
    
    return results
