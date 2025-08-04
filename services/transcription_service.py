"""
Transcription service for ToneBridge Backend
Supports multiple speech-to-text engines
"""

import os
import io
import base64
from typing import Dict, Any, Optional, List
import speech_recognition as sr

# Optional imports for transformer models
try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: PyTorch/Transformers not available. Using fallback speech recognition.")

from utils.logger import setup_logger
from utils.error_handlers import AudioProcessingError, ModelError
from utils.audio_utils import audio_processor
from config import Config

logger = setup_logger(__name__)

class TranscriptionService:
    """Service for handling speech-to-text transcription"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.whisper_pipeline = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize speech recognition models"""
        import importlib
        logger.info(f"TRANSFORMERS_AVAILABLE: {TRANSFORMERS_AVAILABLE}")
        if TRANSFORMERS_AVAILABLE:
            try:
                torch_version = importlib.import_module('torch').__version__
                transformers_version = importlib.import_module('transformers').__version__
                logger.info(f"torch version: {torch_version}")
                logger.info(f"transformers version: {transformers_version}")
            except Exception as e:
                logger.warning(f"Could not get torch/transformers version: {e}")
        if not TRANSFORMERS_AVAILABLE:
            logger.info("PyTorch/Transformers not available. Using speech_recognition library only.")
            return
        try:
            # Initialize Whisper pipeline for better accuracy
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("Using CUDA for transcription")
            else:
                device = "cpu"
                logger.info("Using CPU for transcription")
            logger.info(f"Initializing Whisper pipeline with model: {Config.TRANSCRIPTION_MODEL_NAME}")
            self.whisper_pipeline = pipeline(
                "automatic-speech-recognition",
                model=Config.TRANSCRIPTION_MODEL_NAME,
                device=device
            )
            logger.info("Transcription models initialized successfully")
            logger.info(f"Whisper pipeline created: {self.whisper_pipeline is not None}")
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Failed to initialize Whisper pipeline: {str(e)}\n{tb}")
            raise ModelError(f"Failed to initialize Whisper pipeline: {str(e)}")
    
    def transcribe_audio(self, audio_data: bytes, audio_format: str = 'wav') -> Dict[str, Any]:
        """
        Transcribe audio to text using available models
        
        Args:
            audio_data: Audio data as bytes
            audio_format: Format of the audio data
        
        Returns:
            Dictionary with transcription results
        """
        try:
            # Preprocess audio
            audio_array, sample_rate = audio_processor.load_audio(audio_data, format=audio_format)
            audio_array = audio_processor.preprocess_audio(audio_array, sample_rate)
            
            # Try Whisper first (better accuracy)
            if TRANSFORMERS_AVAILABLE and self.whisper_pipeline:
                logger.info("Using Whisper for transcription")
                result = self._transcribe_with_whisper(audio_array, sample_rate)
            else:
                logger.info("Using speech_recognition fallback")
                # Fallback to speech_recognition
                result = self._transcribe_with_speech_recognition(audio_data, audio_format)
            
            logger.info(f"Transcription completed: '{result['text']}' ({len(result['text'])} characters)")
            logger.info(f"Confidence: {result.get('confidence', 'N/A')}")
            logger.info(f"Model used: {result.get('model', 'N/A')}")
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise AudioProcessingError(f"Transcription failed: {str(e)}")
    
    def _transcribe_with_whisper(self, audio_array, sample_rate: int) -> Dict[str, Any]:
        """Transcribe using Whisper model"""
        try:
            # Convert audio array to format expected by Whisper
            audio_input = {
                "array": audio_array,
                "sampling_rate": sample_rate
            }
            
            result = self.whisper_pipeline(audio_input)
            
            # Calculate confidence based on text quality and length
            text = result['text'].strip()
            confidence = self._calculate_whisper_confidence(text, audio_array)
            
            return {
                'text': text,
                'confidence': confidence,
                'language': result.get('language', 'en'),
                'model': 'whisper'
            }
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {str(e)}")
            raise ModelError(f"Whisper transcription failed: {str(e)}")
    
    def _calculate_whisper_confidence(self, text: str, audio_array) -> float:
        """Calculate confidence score for Whisper transcription"""
        if not text:
            return 0.0
        
        # Base confidence on text length and quality
        base_confidence = 0.7  # Whisper is generally quite accurate
        
        # Boost confidence for longer, more complete sentences
        if len(text.split()) >= 3:
            base_confidence += 0.1
        
        # Boost confidence if text contains common words/patterns
        common_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        word_count = sum(1 for word in text.lower().split() if word in common_words)
        if word_count > 0:
            base_confidence += 0.1
        
        # Reduce confidence for very short or unclear text
        if len(text) < 5:
            base_confidence -= 0.2
        
        # Cap confidence between 0.5 and 0.95
        return max(0.5, min(0.95, base_confidence))
    
    def _transcribe_with_speech_recognition(self, audio_data: bytes, audio_format: str) -> Dict[str, Any]:
        """Transcribe using speech_recognition library"""
        try:
            # Convert audio data to AudioData object
            audio = sr.AudioData(
                audio_data,
                sample_rate=Config.AUDIO_SAMPLE_RATE,
                sample_width=2  # 16-bit audio
            )
            
            # Try multiple recognition services
            text = None
            confidence = 0.0
            
            # Try Google Speech Recognition
            try:
                result = self.recognizer.recognize_google(audio, show_all=True)
                if result and 'alternative' in result:
                    text = result['alternative'][0]['transcript']
                    confidence = result['alternative'][0].get('confidence', 0.0)
                    logger.info("Used Google Speech Recognition")
            except Exception as e:
                logger.warning(f"Google Speech Recognition failed: {str(e)}")
            
            # Fallback to Sphinx if Google fails
            if not text:
                try:
                    text = self.recognizer.recognize_sphinx(audio)
                    confidence = 0.5  # Sphinx doesn't provide confidence scores
                    logger.info("Used Sphinx recognition")
                except Exception as e:
                    logger.error(f"Sphinx recognition failed: {str(e)}")
                    raise AudioProcessingError("All speech recognition methods failed")
            
            return {
                'text': text.strip(),
                'confidence': confidence,
                'language': 'en',
                'model': 'speech_recognition'
            }
            
        except Exception as e:
            logger.error(f"Speech recognition failed: {str(e)}")
            raise AudioProcessingError(f"Speech recognition failed: {str(e)}")
    
    def transcribe_chunks(self, audio_chunks: List[bytes], audio_format: str = 'wav') -> List[Dict[str, Any]]:
        """
        Transcribe multiple audio chunks
        
        Args:
            audio_chunks: List of audio data chunks
            audio_format: Format of the audio data
        
        Returns:
            List of transcription results
        """
        results = []
        
        for i, chunk in enumerate(audio_chunks):
            try:
                result = self.transcribe_audio(chunk, audio_format)
                result['chunk_index'] = i
                results.append(result)
                logger.info(f"Transcribed chunk {i+1}/{len(audio_chunks)}")
            except Exception as e:
                logger.error(f"Failed to transcribe chunk {i}: {str(e)}")
                # Continue with other chunks
                results.append({
                    'text': '',
                    'confidence': 0.0,
                    'error': str(e),
                    'chunk_index': i
                })
        
        return results
    
    def get_transcription_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics from transcription results
        
        Args:
            results: List of transcription results
        
        Returns:
            Dictionary with statistics
        """
        if not results:
            return {}
        
        total_text = ' '.join([r.get('text', '') for r in results])
        avg_confidence = sum([r.get('confidence', 0.0) for r in results]) / len(results)
        successful_chunks = len([r for r in results if r.get('text', '').strip()])
        
        return {
            'total_characters': len(total_text),
            'total_words': len(total_text.split()),
            'average_confidence': avg_confidence,
            'successful_chunks': successful_chunks,
            'total_chunks': len(results),
            'success_rate': successful_chunks / len(results) if results else 0.0
        }

# Global transcription service instance
transcription_service = TranscriptionService() 