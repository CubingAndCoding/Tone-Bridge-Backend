"""
Audio processing utilities for ToneBridge Backend (Deployment Version)
Handles audio processing without requiring ffmpeg system installation
"""

import os
import tempfile
import subprocess
import numpy as np
from typing import Tuple, Dict, Any
from utils.logger import setup_logger
from utils.error_handlers import AudioProcessingError

logger = setup_logger(__name__)

class AudioProcessor:
    """Audio processing utilities for deployment"""
    
    def __init__(self):
        self.sample_rate = 16000
        self.channels = 1
        logger.info("Initializing deployment audio processor")
    
    def load_audio(self, audio_data: bytes, format: str = 'wav') -> Tuple[np.ndarray, int]:
        """
        Load audio data and convert to numpy array
        
        Args:
            audio_data: Raw audio data as bytes
            format: Audio format (wav, webm, mp3, etc.)
        
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            logger.info(f"Loading audio: {len(audio_data)} bytes, format: {format}")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=f'.{format}', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                # Try to load with soundfile first (works for wav, flac)
                if format in ['wav', 'flac']:
                    import soundfile as sf
                    audio_array, sample_rate = sf.read(temp_file_path)
                    logger.info(f"Loaded with soundfile: {sample_rate}Hz, {len(audio_array)} samples")
                    return audio_array, sample_rate
                
                # For other formats, try pydub
                else:
                    from pydub import AudioSegment
                    
                    # Load audio with pydub
                    if format == 'webm':
                        audio = AudioSegment.from_file(temp_file_path, format="webm")
                    elif format == 'mp3':
                        audio = AudioSegment.from_file(temp_file_path, format="mp3")
                    elif format == 'mp4':
                        audio = AudioSegment.from_file(temp_file_path, format="mp4")
                    else:
                        audio = AudioSegment.from_file(temp_file_path)
                    
                    # Convert to mono and target sample rate
                    audio = audio.set_channels(1)
                    audio = audio.set_frame_rate(self.sample_rate)
                    
                    # Convert to numpy array
                    samples = np.array(audio.get_array_of_samples())
                    
                    # Normalize to float
                    if audio.sample_width == 2:
                        samples = samples.astype(np.float32) / 32768.0
                    elif audio.sample_width == 4:
                        samples = samples.astype(np.float32) / 2147483648.0
                    else:
                        samples = samples.astype(np.float32) / 128.0
                    
                    logger.info(f"Loaded with pydub: {self.sample_rate}Hz, {len(samples)} samples")
                    return samples, self.sample_rate
                    
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            logger.error(f"Failed to load audio: {str(e)}")
            raise AudioProcessingError(f"Failed to load audio: {str(e)}")
    
    def preprocess_audio(self, audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Preprocess audio for transcription
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of audio
        
        Returns:
            Preprocessed audio array
        """
        try:
            # Resample if needed
            if sample_rate != self.sample_rate:
                audio_array = self._resample_audio(audio_array, sample_rate, self.sample_rate)
            
            # Normalize audio
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            
            # Ensure mono
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            logger.info(f"Preprocessed audio: {len(audio_array)} samples")
            return audio_array
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {str(e)}")
            raise AudioProcessingError(f"Audio preprocessing failed: {str(e)}")
    
    def extract_audio_features(self, audio_array: np.ndarray, sample_rate: int) -> Dict[str, np.ndarray]:
        """
        Extract basic audio features (simplified for deployment)
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of audio
        
        Returns:
            Dictionary of audio features
        """
        try:
            features = {}
            
            # RMS Energy
            features['rms_energy'] = np.sqrt(np.mean(audio_array**2))
            
            # Zero crossing rate
            features['zero_crossing_rate'] = np.sum(np.diff(np.sign(audio_array)) != 0) / len(audio_array)
            
            # Spectral centroid (simplified)
            fft = np.fft.fft(audio_array)
            freqs = np.fft.fftfreq(len(audio_array), 1/sample_rate)
            magnitude = np.abs(fft)
            features['spectral_centroid'] = np.sum(freqs * magnitude) / np.sum(magnitude)
            
            logger.info("Audio features extracted successfully")
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return {}
    
    def _resample_audio(self, audio_array: np.ndarray, old_rate: int, new_rate: int) -> np.ndarray:
        """
        Simple resampling (for deployment)
        
        Args:
            audio_array: Audio data
            old_rate: Original sample rate
            new_rate: Target sample rate
        
        Returns:
            Resampled audio array
        """
        if old_rate == new_rate:
            return audio_array
        
        # Simple linear interpolation
        ratio = new_rate / old_rate
        new_length = int(len(audio_array) * ratio)
        indices = np.linspace(0, len(audio_array) - 1, new_length)
        
        return np.interp(indices, np.arange(len(audio_array)), audio_array)

# Create global instance
audio_processor = AudioProcessor() 