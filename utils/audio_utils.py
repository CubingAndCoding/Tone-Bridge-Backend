"""
Audio processing utilities for ToneBridge Backend
"""

import os
import io
import tempfile
import subprocess
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional, Union
from pydub import AudioSegment
from pydub.utils import make_chunks
import wave
from utils.logger import setup_logger
from utils.error_handlers import AudioProcessingError

logger = setup_logger(__name__)

class AudioProcessor:
    """Centralized audio processing utilities"""
    
    def __init__(self, target_sample_rate: int = 16000, target_channels: int = 1):
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
    
    def load_audio(self, audio_data: Union[bytes, str], format: str = None) -> Tuple[np.ndarray, int]:
        """
        Load audio from various sources and formats

        Args:
            audio_data: Audio data as bytes or file path
            format: Audio format (e.g., 'wav', 'webm', 'mp3'). Used for explicit conversion.

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            logger.info(f"Loading audio: {len(audio_data)} bytes, format: {format}")
            
            if isinstance(audio_data, bytes):
                # If format is webm, convert to wav using ffmpeg
                if format == 'webm':
                    logger.info("Converting webm to wav using ffmpeg...")
                    
                    # Create temporary files
                    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_in:
                        temp_in.write(audio_data)
                        temp_in.flush()
                        temp_in_path = temp_in.name
                    
                    temp_out_path = temp_in_path.replace('.webm', '.wav')
                    
                    try:
                        # Run ffmpeg to convert webm to wav
                        cmd = [
                            'ffmpeg', '-y', '-i', temp_in_path,
                            '-ar', str(self.target_sample_rate),
                            '-ac', str(self.target_channels),
                            temp_out_path
                        ]
                        logger.info(f"Running ffmpeg: {' '.join(cmd)}")
                        
                        result = subprocess.run(
                            cmd, 
                            check=True, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE,
                            text=True
                        )
                        logger.info("ffmpeg conversion completed successfully")
                        
                        # Load the wav file with librosa
                        audio_array, sample_rate = librosa.load(
                            temp_out_path,
                            sr=self.target_sample_rate,
                            mono=(self.target_channels == 1)
                        )
                        
                    except subprocess.CalledProcessError as ffmpeg_error:
                        logger.error(f"ffmpeg conversion failed: {ffmpeg_error}")
                        logger.error(f"ffmpeg stderr: {ffmpeg_error.stderr}")
                        raise AudioProcessingError(f"ffmpeg conversion failed: {ffmpeg_error}")
                    except Exception as e:
                        logger.error(f"Unexpected error during ffmpeg conversion: {str(e)}")
                        raise AudioProcessingError(f"ffmpeg conversion error: {str(e)}")
                    finally:
                        # Clean up temp files
                        try:
                            os.remove(temp_in_path)
                            if os.path.exists(temp_out_path):
                                os.remove(temp_out_path)
                        except Exception as cleanup_error:
                            logger.warning(f"Failed to cleanup temp files: {cleanup_error}")
                else:
                    # For other formats, try using pydub first for better format support
                    try:
                        logger.info(f"Attempting to load audio with pydub, format: {format}")
                        
                        # Use pydub to load and convert to wav
                        audio = AudioSegment.from_file(io.BytesIO(audio_data), format=format)
                        
                        # Export to wav format in memory
                        wav_buffer = io.BytesIO()
                        audio.export(wav_buffer, format='wav')
                        wav_buffer.seek(0)
                        
                        # Load with librosa
                        audio_array, sample_rate = librosa.load(
                            wav_buffer,
                            sr=self.target_sample_rate,
                            mono=(self.target_channels == 1)
                        )
                        
                        logger.info("Audio loaded successfully with pydub + librosa")
                        
                    except Exception as pydub_error:
                        logger.warning(f"Pydub loading failed: {str(pydub_error)}, trying librosa directly")
                        
                        # Fallback to direct librosa loading
                        try:
                            audio_array, sample_rate = librosa.load(
                                io.BytesIO(audio_data),
                                sr=self.target_sample_rate,
                                mono=(self.target_channels == 1)
                            )
                            logger.info("Audio loaded successfully with librosa directly")
                        except Exception as librosa_error:
                            logger.error(f"Both pydub and librosa loading failed")
                            logger.error(f"Pydub error: {str(pydub_error)}")
                            logger.error(f"Librosa error: {str(librosa_error)}")
                            raise AudioProcessingError(f"Failed to load audio with any method. Pydub: {str(pydub_error)}, Librosa: {str(librosa_error)}")
            else:
                # Load from file path
                audio_array, sample_rate = librosa.load(
                    audio_data,
                    sr=self.target_sample_rate,
                    mono=(self.target_channels == 1)
                )
            
            logger.info(f"Audio loaded successfully: {len(audio_array)} samples, {sample_rate}Hz")
            return audio_array, sample_rate
            
        except Exception as e:
            logger.error(f"Failed to load audio: {str(e)}")
            logger.error(f"Audio data length: {len(audio_data) if isinstance(audio_data, bytes) else 'N/A'}")
            logger.error(f"Audio format: {format}")
            raise AudioProcessingError(f"Failed to load audio: {str(e)}")
    
    def preprocess_audio(self, audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Preprocess audio for model input
        
        Args:
            audio_array: Input audio array
            sample_rate: Original sample rate
        
        Returns:
            Preprocessed audio array
        """
        try:
            # Normalize audio
            audio_array = librosa.util.normalize(audio_array)
            
            # Apply noise reduction (optional)
            # audio_array = self._reduce_noise(audio_array, sample_rate)
            
            # Trim silence
            audio_array, _ = librosa.effects.trim(audio_array, top_db=20)
            
            logger.info(f"Audio preprocessed: {len(audio_array)} samples")
            return audio_array
            
        except Exception as e:
            logger.error(f"Failed to preprocess audio: {str(e)}")
            raise AudioProcessingError(f"Failed to preprocess audio: {str(e)}")
    
    def convert_audio_format(self, audio_data: bytes, input_format: str, output_format: str = 'wav') -> bytes:
        """
        Convert audio between formats
        
        Args:
            audio_data: Input audio data
            input_format: Input format (mp3, wav, etc.)
            output_format: Output format (wav, mp3, etc.)
        
        Returns:
            Converted audio data as bytes
        """
        try:
            # Load audio with pydub
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format=input_format)
            
            # Convert to target format
            output_buffer = io.BytesIO()
            audio.export(output_buffer, format=output_format)
            
            return output_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Failed to convert audio format: {str(e)}")
            raise AudioProcessingError(f"Failed to convert audio format: {str(e)}")
    
    def split_audio_into_chunks(self, audio_array: np.ndarray, chunk_duration: float = 10.0) -> list:
        """
        Split audio into chunks for processing
        
        Args:
            audio_array: Input audio array
            chunk_duration: Duration of each chunk in seconds
        
        Returns:
            List of audio chunks
        """
        try:
            samples_per_chunk = int(self.target_sample_rate * chunk_duration)
            chunks = []
            
            for i in range(0, len(audio_array), samples_per_chunk):
                chunk = audio_array[i:i + samples_per_chunk]
                if len(chunk) > 0:
                    chunks.append(chunk)
            
            logger.info(f"Audio split into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to split audio: {str(e)}")
            raise AudioProcessingError(f"Failed to split audio: {str(e)}")
    
    def extract_audio_features(self, audio_array: np.ndarray, sample_rate: int) -> dict:
        """
        Extract audio features for emotion analysis
        
        Args:
            audio_array: Input audio array
            sample_rate: Sample rate
        
        Returns:
            Dictionary of audio features
        """
        try:
            features = {}
            
            # MFCC features
            mfcc = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=13)
            features['mfcc'] = mfcc
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_array, sr=sample_rate)[0]
            features['spectral_centroid'] = spectral_centroids
            
            # Pitch features
            pitches, magnitudes = librosa.piptrack(y=audio_array, sr=sample_rate)
            features['pitch'] = pitches
            features['pitch_magnitude'] = magnitudes
            
            # Energy features
            rms = librosa.feature.rms(y=audio_array)[0]
            features['rms_energy'] = rms
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_array)[0]
            features['zero_crossing_rate'] = zcr
            
            logger.info(f"Extracted {len(features)} audio features")
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract audio features: {str(e)}")
            raise AudioProcessingError(f"Failed to extract audio features: {str(e)}")
    
    def _reduce_noise(self, audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply noise reduction to audio
        
        Args:
            audio_array: Input audio array
            sample_rate: Sample rate
        
        Returns:
            Noise-reduced audio array
        """
        # Simple noise reduction using spectral gating
        # This is a basic implementation - could be enhanced with more sophisticated methods
        
        # Compute spectrogram
        D = librosa.stft(audio_array)
        
        # Estimate noise from first and last frames
        noise_spectrum = np.mean(np.abs(D[:, :10]), axis=1, keepdims=True)
        
        # Apply spectral gating
        gate_threshold = 2.0
        gain = np.maximum(1 - gate_threshold * noise_spectrum / (np.abs(D) + 1e-8), 0)
        D_filtered = D * gain
        
        # Convert back to time domain
        audio_filtered = librosa.istft(D_filtered)
        
        return audio_filtered

# Global audio processor instance
audio_processor = AudioProcessor() 