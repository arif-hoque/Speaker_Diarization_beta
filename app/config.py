from pydantic_settings import BaseSettings
from pathlib import Path
import os

class Settings(BaseSettings):
    # Application settings
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    PORT: int = int(os.getenv("PORT", "5000"))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    
    # File storage paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    UPLOAD_DIR: str = os.path.join(BASE_DIR, "static/uploads")
    DIARIZATION_OUTPUT_DIR: str = os.path.join(BASE_DIR, "diarization_output")
    TRANSCRIPT_OUTPUT_DIR: str = os.path.join(BASE_DIR, "transcript_output")
    
    # Model settings
    WHISPER_MODEL: str = "large"
    WHISPER_DEVICE: str = "cuda"
    WHISPER_COMPUTE_TYPE: str = "float16"
    DIARIZATION_MODEL: str = "pyannote/speaker-diarization-3.1"
    
    # Authentication token for Pyannote (should be set as environment variable in production)
    HUGGINGFACE_TOKEN: str = os.getenv("HUGGINGFACE_TOKEN", "")
    
    # Maximum file size (100MB)
    MAX_CONTENT_LENGTH: int = 100 * 1024 * 1024
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS: set = {"wav", "mp3", "ogg", "flac", "m4a"}

    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()
