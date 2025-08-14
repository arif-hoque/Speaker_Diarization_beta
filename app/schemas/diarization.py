from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class WordSegment(BaseModel):
    word: str
    start: float
    end: float
    probability: Optional[float] = None

class DiarizationSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    words: List[WordSegment] = []

class DiarizationRequest(BaseModel):
    num_speakers: Optional[int] = Field(
        None,
        ge=1,
        le=50,
        description="Number of speakers, leave empty to autodetect."
    )
    language: Optional[str] = Field(
        None,
        description="Language of the spoken words as a language code like 'en'. Leave empty to auto detect language."
    )
    translate: bool = Field(
        False,
        description="Translate the speech into English."
    )
    prompt: str = Field(
        "",
        description="Vocabulary: provide names, acronyms and loanwords in a list. Use punctuation for best accuracy."
    )

class DiarizationResponse(BaseModel):
    language: str
    num_speakers: int
    duration: float
    segments: List[DiarizationSegment]
