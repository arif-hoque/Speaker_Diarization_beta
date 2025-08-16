import os
import time
import logging
import torch
import torchaudio
import subprocess
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
from pyannote.audio.pipelines.utils.hook import ProgressHook

from ..config import settings

logger = logging.getLogger(__name__)

class DiarizationService:
    def __init__(self):
        self.whisper_model = None
        self.diarization_model = None
        self._load_models()
    
    def _load_models(self):
        """Load Whisper and Pyannote models"""
        logger.info("Loading Whisper model...")
        self.whisper_model = WhisperModel(
            model_size_or_path=settings.WHISPER_MODEL,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE
        )
        
        logger.info("Loading Diarization model...")
        self.diarization_model = Pipeline.from_pretrained(
            settings.DIARIZATION_MODEL,
            use_auth_token=settings.HUGGINGFACE_TOKEN or None
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    def process_audio(
        self,
        file_path: str,
        num_speakers: Optional[int] = None,
        language: Optional[str] = None,
        translate: bool = False,
        prompt: str = ""
    ) -> Dict[str, Any]:
        """
        Process an audio file for diarization and transcription with parallel processing.
        
        Args:
            file_path: Path to the audio file
            num_speakers: Number of speakers (optional)
            language: Language code (optional)
            translate: Whether to translate to English
            prompt: Optional prompt for better transcription
            
        Returns:
            Dictionary containing diarization and transcription results
        """
        start_time = time.time()
        
        try:
            # Convert audio to WAV format if needed
            conversion_start = time.time()
            wav_path = self._convert_to_wav(file_path)
            conversion_time = time.time() - conversion_start
            logger.info(f"Audio conversion completed in {conversion_time:.2f} seconds")
            
            # Process diarization and transcription in parallel
            parallel_start = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both tasks simultaneously
                logger.info("Starting parallel diarization and transcription...")
                
                diarization_future = executor.submit(
                    self._diarize_audio, wav_path, num_speakers
                )
                transcription_future = executor.submit(
                    self._transcribe_audio, wav_path, language, translate, prompt
                )
                
                # Wait for both to complete
                diarization_result = diarization_future.result()
                transcription_result = transcription_future.result()
            
            parallel_time = time.time() - parallel_start
            logger.info(f"Parallel processing completed in {parallel_time:.2f} seconds")
            
            # Combine results
            combine_start = time.time()
            result = self._combine_results(
                diarization_result,
                transcription_result,
                language
            )
            combine_time = time.time() - combine_start
            logger.info(f"Result combination completed in {combine_time:.2f} seconds")
            
            # Clean up temporary WAV file if it was created
            if wav_path != file_path:
                os.remove(wav_path)
            
            total_processing_time = time.time() - start_time
            logger.info(f"Total processing completed in {total_processing_time:.2f} seconds")
            logger.info(f"Performance breakdown - Conversion: {conversion_time:.1f}s, "
                       f"Parallel: {parallel_time:.1f}s, Combine: {combine_time:.1f}s")
            
            # Add timing information to result
            result['processing_times'] = {
                'total': total_processing_time,
                'conversion': conversion_time,
                'parallel_processing': parallel_time,
                'combination': combine_time
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}", exc_info=True)
            raise
    
    def process_audio_sequential(
        self,
        file_path: str,
        num_speakers: Optional[int] = None,
        language: Optional[str] = None,
        translate: bool = False,
        prompt: str = ""
    ) -> Dict[str, Any]:
        """
        Process an audio file for diarization and transcription sequentially (original method).
        Kept for comparison purposes.
        """
        start_time = time.time()
        
        try:
            # Convert audio to WAV format if needed
            wav_path = self._convert_to_wav(file_path)
            
            # Perform diarization
            diarization_result = self._diarize_audio(wav_path, num_speakers)
            
            # Perform transcription
            transcription_result = self._transcribe_audio(
                wav_path,
                language=language,
                translate=translate,
                prompt=prompt
            )
            
            # Combine results
            result = self._combine_results(
                diarization_result,
                transcription_result,
                language
            )
            
            # Clean up temporary WAV file if it was created
            if wav_path != file_path:
                os.remove(wav_path)
            
            processing_time = time.time() - start_time
            logger.info(f"Sequential processing completed in {processing_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}", exc_info=True)
            raise
    
    def _convert_to_wav(self, input_path: str) -> str:
        """Convert audio file to WAV format if needed"""
        if input_path.lower().endswith('.wav'):
            return input_path
            
        output_path = f"{os.path.splitext(input_path)[0]}.wav"
        
        try:
            subprocess.run([
                'ffmpeg',
                '-i', input_path,
                '-ar', '16000',
                '-ac', '1',
                '-c:a', 'pcm_s16le',
                '-y',  # Overwrite output file if it exists
                output_path
            ], check=True, capture_output=True)
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
            raise RuntimeError(f"Failed to convert audio file: {str(e)}")
    
    def _diarize_audio(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None
    ) -> Dict[str, Any]:
        """Perform speaker diarization on the audio file"""
        logger.info("Starting diarization...")
        start_time = time.time()
        
        # Run diarization with progress hook
        with ProgressHook() as hook:
            diarization = self.diarization_model(
                audio_path,
                num_speakers=num_speakers,
                hook=hook
            )
        
        # Convert to list of segments
        segments = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'start': segment.start,
                'end': segment.end,
                'speaker': speaker
            })
        
        diarization_time = time.time() - start_time
        logger.info(f"Diarization completed in {diarization_time:.2f} seconds")
        
        return {
            'segments': segments,
            'num_speakers': len(set(seg['speaker'] for seg in segments)),
            'duration': diarization.get_timeline().duration()
        }
    
    def _transcribe_audio(
        self,
        audio_path: str,
        language: Optional[str] = None,
        translate: bool = False,
        prompt: str = ""
    ) -> Dict[str, Any]:
        """Transcribe audio using Whisper with optimized VAD settings for call recordings"""
        logger.info("Starting transcription...")
        start_time = time.time()
        
        # Import VAD options (should be available in container)
        from faster_whisper.vad import VadOptions
        
        # Optimized transcription options for call recordings between two people
        options = dict(
            language=language,
            beam_size=5,  # Good balance of speed and accuracy
            vad_filter=True,
            vad_parameters=VadOptions(
                max_speech_duration_s=30.0,  # Reasonable max for call segments
                min_speech_duration_ms=100,  # Filter very short segments
                speech_pad_ms=100,  # Small padding for speech boundaries
                threshold=0.25,  # Optimized for call quality audio
                neg_threshold=0.2,  # Good for detecting silence in calls
            ),
            word_timestamps=True,
            initial_prompt=prompt,
            language_detection_segments=1,  # Minimal language detection for speed
            task="translate" if translate else "transcribe",
        )
        
        # Run Whisper model with optimized parameters
        segments, info = self.whisper_model.transcribe(audio_path, **options)
        
        # Convert segments to list and extract detailed information
        segments = list(segments)
        result_segments = [
            {
                "avg_logprob": s.avg_logprob,  # Quality metric
                "start": float(s.start),
                "end": float(s.end),
                "text": s.text.strip(),
                "words": [
                    {
                        "start": float(w.start),
                        "end": float(w.end),
                        "word": w.word,
                        "probability": w.probability,
                    }
                    for w in s.words or []
                ]
            }
            for s in segments
        ]
        
        transcription_time = time.time() - start_time
        logger.info(f"Transcription completed in {transcription_time:.2f} seconds")
        
        return {
            'segments': result_segments,
            'language': info.language,
            'language_probability': info.language_probability,
            'duration': info.duration,
            'avg_logprob': sum(s.avg_logprob for s in segments) / len(segments) if segments else 0.0
        }
    
    def _combine_results(
        self,
        diarization_result: Dict[str, Any],
        transcription_result: Dict[str, Any],
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Combine diarization and transcription results"""
        logger.info("Combining diarization and transcription results...")
        
        # Get the language from transcription if not provided
        if language is None:
            language = transcription_result.get('language', 'en')
        
        # Create the final result structure
        result = {
            'language': language,
            'num_speakers': diarization_result.get('num_speakers', 0),
            'duration': diarization_result.get('duration', 0),
            'segments': []
        }
        
        # For each transcription segment, find the corresponding speaker
        for seg in transcription_result['segments']:
            # Find overlapping diarization segments
            matching_speakers = []
            for diar_seg in diarization_result['segments']:
                # Check for overlap
                overlap_start = max(seg['start'], diar_seg['start'])
                overlap_end = min(seg['end'], diar_seg['end'])
                
                if overlap_start < overlap_end:  # If there's an overlap
                    overlap_duration = overlap_end - overlap_start
                    matching_speakers.append((diar_seg['speaker'], overlap_duration))
            
            # Determine the most likely speaker for this segment
            speaker = None
            if matching_speakers:
                # Group by speaker and sum durations
                speaker_durations = {}
                for spk, dur in matching_speakers:
                    speaker_durations[spk] = speaker_durations.get(spk, 0) + dur
                
                # Get speaker with maximum duration
                speaker = max(speaker_durations.items(), key=lambda x: x[1])[0]
            
            # Add to results
            result['segments'].append({
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text'],
                'speaker': speaker,
                'words': seg.get('words', [])
            })
        
        return result
