"""
Emotion detection service for ToneBridge Backend
Supports multiple emotion detection models
"""

import os
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

# Optional imports for transformer models
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: PyTorch/Transformers not available. Using rule-based emotion detection.")

from sklearn.preprocessing import StandardScaler
from utils.logger import setup_logger
from utils.error_handlers import ModelError
from utils.audio_utils import audio_processor
from config import Config

logger = setup_logger(__name__)

class EmotionService:
    """Service for detecting emotions in text and audio"""
    
    def __init__(self):
        self.text_emotion_pipeline = None
        self.audio_emotion_model = None
        self.scaler = StandardScaler()
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize emotion detection models"""
        logger.info(f"TRANSFORMERS_AVAILABLE for emotion: {TRANSFORMERS_AVAILABLE}")
        
        if not TRANSFORMERS_AVAILABLE:
            logger.info("PyTorch/Transformers not available. Using rule-based emotion detection.")
            return
            
        try:
            # Initialize text-based emotion detection
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("Using CUDA for emotion detection")
            else:
                device = "cpu"
                logger.info("Using CPU for emotion detection")
            
            # Use a reliable, public emotion model
            emotion_model = "SamLowe/roberta-base-go_emotions"
            logger.info(f"Initializing emotion pipeline with model: {emotion_model}")
            self.text_emotion_pipeline = pipeline(
                "text-classification",
                model=emotion_model,
                device=device
            )
            
            logger.info("Emotion detection models initialized successfully")
            logger.info(f"Emotion pipeline created: {self.text_emotion_pipeline is not None}")
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Failed to initialize emotion models: {str(e)}\n{tb}")
            logger.info("Falling back to rule-based emotion detection")
    
    def detect_emotion_from_text(self, text: str) -> Dict[str, Any]:
        """
        Detect emotion from text using transformer model or rule-based detection
        
        Args:
            text: Input text to analyze
        
        Returns:
            Dictionary with emotion detection results
        """
        try:
            if not text or not text.strip():
                return {
                    'emotion': 'neutral',
                    'confidence': 0.0,
                    'emoji': Config.get_emotion_emoji('neutral'),
                    'model': 'rule_based'
                }
            
            # Use transformer model if available
            if TRANSFORMERS_AVAILABLE and self.text_emotion_pipeline:
                logger.info("Using AI-based emotion detection (transformer model)")
                # Get emotion prediction
                result = self.text_emotion_pipeline(text)
                
                # Extract emotion and confidence
                emotion = result[0]['label'].lower()
                confidence = result[0]['score']
                
                # Map to supported emotions
                emotion = self._map_emotion(emotion)
                
                # Get emoji for emotion
                emoji = Config.get_emotion_emoji(emotion)
                
                logger.info(f"Text emotion detected: {emotion} (confidence: {confidence:.3f})")
                
                return {
                    'emotion': emotion,
                    'confidence': confidence,
                    'emoji': emoji,
                    'model': 'text_classification',
                    'original_text': text
                }
            else:
                logger.info("Using rule-based emotion detection (fallback)")
                # Fallback to rule-based emotion detection
                return self._detect_emotion_rule_based(text)
            
        except Exception as e:
            logger.error(f"Text emotion detection failed: {str(e)}")
            return self._detect_emotion_rule_based(text)
    
    def detect_emotion_from_audio(self, audio_data: bytes, audio_format: str = 'wav') -> Dict[str, Any]:
        """
        Detect emotion from audio using audio features
        
        Args:
            audio_data: Audio data as bytes
            audio_format: Format of the audio data
        
        Returns:
            Dictionary with emotion detection results
        """
        try:
            # Load and preprocess audio
            audio_array, sample_rate = audio_processor.load_audio(audio_data)
            audio_array = audio_processor.preprocess_audio(audio_array, sample_rate)
            
            # Extract audio features
            features = audio_processor.extract_audio_features(audio_array, sample_rate)
            
            # Analyze features for emotion
            emotion_result = self._analyze_audio_features(features, audio_array, sample_rate)
            
            logger.info(f"Audio emotion detected: {emotion_result['emotion']} (confidence: {emotion_result['confidence']:.3f})")
            
            return emotion_result
            
        except Exception as e:
            logger.error(f"Audio emotion detection failed: {str(e)}")
            raise ModelError(f"Audio emotion detection failed: {str(e)}")
    
    def _analyze_audio_features(self, features: Dict[str, np.ndarray], audio_array: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Analyze audio features to determine emotion
        
        Args:
            features: Extracted audio features
            audio_array: Audio array
            sample_rate: Sample rate
        
        Returns:
            Emotion detection result
        """
        try:
            # Calculate feature statistics
            feature_stats = {}
            
            # MFCC statistics
            if 'mfcc' in features:
                mfcc_mean = np.mean(features['mfcc'], axis=1)
                mfcc_std = np.std(features['mfcc'], axis=1)
                feature_stats['mfcc_mean'] = np.mean(mfcc_mean)
                feature_stats['mfcc_std'] = np.mean(mfcc_std)
            
            # Spectral centroid statistics
            if 'spectral_centroid' in features:
                feature_stats['spectral_centroid_mean'] = np.mean(features['spectral_centroid'])
                feature_stats['spectral_centroid_std'] = np.std(features['spectral_centroid'])
            
            # Energy statistics
            if 'rms_energy' in features:
                feature_stats['energy_mean'] = np.mean(features['rms_energy'])
                feature_stats['energy_std'] = np.std(features['rms_energy'])
            
            # Zero crossing rate
            if 'zero_crossing_rate' in features:
                feature_stats['zcr_mean'] = np.mean(features['zero_crossing_rate'])
            
            # Determine emotion based on features
            emotion, confidence = self._classify_emotion_from_features(feature_stats)
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'emoji': Config.get_emotion_emoji(emotion),
                'model': 'audio_features',
                'feature_stats': feature_stats
            }
            
        except Exception as e:
            logger.error(f"Audio feature analysis failed: {str(e)}")
            raise ModelError(f"Audio feature analysis failed: {str(e)}")
    
    def _classify_emotion_from_features(self, feature_stats: Dict[str, float]) -> Tuple[str, float]:
        """
        Classify emotion based on audio features
        
        Args:
            feature_stats: Audio feature statistics
        
        Returns:
            Tuple of (emotion, confidence)
        """
        # Simple rule-based classification
        # This could be enhanced with a trained ML model
        
        energy_mean = feature_stats.get('energy_mean', 0.0)
        spectral_centroid_mean = feature_stats.get('spectral_centroid_mean', 0.0)
        zcr_mean = feature_stats.get('zcr_mean', 0.0)
        
        # High energy + high spectral centroid = excited/happy
        if energy_mean > 0.1 and spectral_centroid_mean > 2000:
            return 'excited', 0.7
        
        # Low energy + low spectral centroid = sad/calm
        elif energy_mean < 0.05 and spectral_centroid_mean < 1000:
            return 'sad', 0.6
        
        # High zero crossing rate = angry/frustrated
        elif zcr_mean > 0.1:
            return 'frustrated', 0.6
        
        # Medium energy + medium spectral centroid = neutral
        else:
            return 'neutral', 0.5
    
    def _map_emotion(self, emotion: str) -> str:
        """
        Map detected emotion to supported emotion categories
        Maps Go Emotions model outputs to our supported emotions
        
        Args:
            emotion: Raw emotion from model
        
        Returns:
            Mapped emotion
        """
        # Go Emotions model has 27 emotions, map to our 11 supported emotions
        emotion_mapping = {
            # Positive emotions
            'joy': 'happy',
            'amusement': 'happy',
            'excitement': 'excited',
            'gratitude': 'happy',
            'love': 'happy',
            'optimism': 'excited',
            'relief': 'calm',
            'pride': 'excited',
            'admiration': 'happy',
            'desire': 'excited',
            'caring': 'calm',
            'approval': 'happy',
            'realization': 'surprise',
            'curiosity': 'excited',
            'interest': 'excited',
            'surprise': 'surprise',
            
            # Negative emotions
            'sadness': 'sad',
            'grief': 'sad',
            'disappointment': 'sad',
            'remorse': 'sad',
            'embarrassment': 'fear',
            'nervousness': 'fear',
            'fear': 'fear',
            'disgust': 'disgust',
            'anger': 'angry',
            'annoyance': 'frustrated',
            'disapproval': 'frustrated',
            'confusion': 'frustrated',
            'neutral': 'neutral'
        }
        
        return emotion_mapping.get(emotion.lower(), 'neutral')
    
    def _detect_emotion_rule_based(self, text: str) -> Dict[str, Any]:
        """
        Enhanced rule-based emotion detection as fallback when transformer models aren't available
        
        Args:
            text: Input text to analyze
        
        Returns:
            Dictionary with emotion detection results
        """
        text_lower = text.lower()
        
        # Enhanced emotion keywords with more context
        emotion_keywords = {
            'happy': [
                'happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'fantastic', 'awesome', 
                'love', 'like', 'good', 'nice', 'beautiful', 'perfect', 'excellent', 'brilliant',
                'delighted', 'pleased', 'satisfied', 'content', 'cheerful', 'glad', 'thrilled'
            ],
            'sad': [
                'sad', 'sorrow', 'depressed', 'unhappy', 'miserable', 'terrible', 'awful', 
                'hate', 'dislike', 'bad', 'horrible', 'worst', 'disappointed', 'upset', 'crying',
                'tears', 'lonely', 'alone', 'miss', 'lost', 'gone', 'died', 'death'
            ],
            'angry': [
                'angry', 'mad', 'furious', 'rage', 'hate', 'terrible', 'awful', 'horrible',
                'annoyed', 'irritated', 'frustrated', 'upset', 'disappointed', 'outraged',
                'livid', 'enraged', 'fuming', 'seething', 'disgusted', 'repulsed'
            ],
            'fear': [
                'afraid', 'scared', 'fear', 'terrified', 'worried', 'anxious', 'nervous',
                'panic', 'horror', 'dread', 'terror', 'frightened', 'alarmed', 'concerned',
                'stress', 'stressed', 'tense', 'uneasy', 'uncomfortable'
            ],
            'surprise': [
                'surprised', 'shocked', 'amazed', 'wow', 'incredible', 'unbelievable',
                'astonished', 'stunned', 'flabbergasted', 'baffled', 'confused', 'puzzled'
            ],
            'excited': [
                'excited', 'thrilled', 'pumped', 'energetic', 'enthusiastic', 'eager',
                'motivated', 'inspired', 'passionate', 'zealous', 'animated', 'lively'
            ],
            'calm': [
                'calm', 'peaceful', 'relaxed', 'serene', 'tranquil', 'quiet', 'gentle',
                'soft', 'smooth', 'easy', 'comfortable', 'soothing', 'restful'
            ],
            'frustrated': [
                'frustrated', 'annoyed', 'irritated', 'upset', 'disappointed', 'bothered',
                'troubled', 'disturbed', 'agitated', 'restless', 'impatient', 'exasperated'
            ]
        }
        
        # Count emotion keywords with weighted scoring
        emotion_scores = {}
        words = text_lower.split()
        
        for emotion, keywords in emotion_keywords.items():
            score = 0
            for keyword in keywords:
                # Exact word match gets higher score
                if keyword in words:
                    score += 2
                # Partial match gets lower score
                elif keyword in text_lower:
                    score += 1
            if score > 0:
                emotion_scores[emotion] = score
        
        # Analyze text patterns for additional context
        # Exclamation marks indicate excitement/anger
        if '!' in text:
            if any(emotion in emotion_scores for emotion in ['angry', 'frustrated']):
                emotion_scores['angry'] = emotion_scores.get('angry', 0) + 1
            elif any(emotion in emotion_scores for emotion in ['happy', 'excited']):
                emotion_scores['excited'] = emotion_scores.get('excited', 0) + 1
        
        # Question marks might indicate confusion/fear
        if '?' in text and len(words) < 5:
            emotion_scores['fear'] = emotion_scores.get('fear', 0) + 1
        
        # Determine primary emotion
        if emotion_scores:
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            # Calculate confidence based on score strength and text length
            max_possible_score = len(words) * 2  # Maximum possible score
            confidence = min(emotion_scores[primary_emotion] / max_possible_score, 0.85)
            confidence = max(0.3, confidence)  # Minimum confidence of 0.3
        else:
            primary_emotion = 'neutral'
            confidence = 0.4  # Slightly higher default confidence for neutral
        
        # Get emoji for emotion
        emoji = Config.get_emotion_emoji(primary_emotion)
        
        logger.info(f"Enhanced rule-based emotion detected: {primary_emotion} (confidence: {confidence:.3f})")
        
        return {
            'emotion': primary_emotion,
            'confidence': confidence,
            'emoji': emoji,
            'model': 'rule_based',
            'original_text': text
        }
    
    def detect_emotion_combined(self, text: str, audio_data: bytes = None, audio_format: str = 'wav') -> Dict[str, Any]:
        """
        Combine text and audio emotion detection for better accuracy
        
        Args:
            text: Input text
            audio_data: Audio data (optional)
            audio_format: Audio format
        
        Returns:
            Combined emotion detection result
        """
        try:
            # Get text emotion
            text_result = self.detect_emotion_from_text(text)
            
            # Get audio emotion if available
            audio_result = None
            if audio_data:
                audio_result = self.detect_emotion_from_audio(audio_data, audio_format)
            
            # Combine results
            if audio_result:
                # Weighted combination (text: 60%, audio: 40%)
                text_weight = 0.6
                audio_weight = 0.4
                
                # Use text emotion as primary, audio as secondary
                final_emotion = text_result['emotion']
                final_confidence = (text_result['confidence'] * text_weight + 
                                  audio_result['confidence'] * audio_weight)
                
                # If audio confidence is much higher, use audio emotion
                if audio_result['confidence'] > text_result['confidence'] + 0.2:
                    final_emotion = audio_result['emotion']
                
                return {
                    'emotion': final_emotion,
                    'confidence': final_confidence,
                    'emoji': Config.get_emotion_emoji(final_emotion),
                    'text_emotion': text_result,
                    'audio_emotion': audio_result,
                    'model': 'combined'
                }
            else:
                return text_result
                
        except Exception as e:
            logger.error(f"Combined emotion detection failed: {str(e)}")
            raise ModelError(f"Combined emotion detection failed: {str(e)}")

# Global emotion service instance
emotion_service = EmotionService() 