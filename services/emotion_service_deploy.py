"""
Emotion detection service for ToneBridge Backend (Deployment Version)
Uses rule-based emotion detection without heavy ML dependencies
"""

import os
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import re
from utils.logger import setup_logger
from utils.error_handlers import ModelError
from utils.audio_utils import audio_processor

logger = setup_logger(__name__)

class EmotionService:
    """Service for detecting emotions in text and audio (rule-based)"""
    
    def __init__(self):
        self.scaler = None
        logger.info("Initializing rule-based emotion detection service")
    
    def detect_emotion_from_text(self, text: str) -> Dict[str, Any]:
        """
        Detect emotion from text using rule-based detection
        
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
                    'emoji': '😐',
                    'model': 'rule_based'
                }
            
            logger.info("Using rule-based emotion detection")
            return self._detect_emotion_rule_based(text)
            
        except Exception as e:
            logger.error(f"Error in emotion detection: {str(e)}")
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'emoji': '😐',
                'model': 'rule_based'
            }
    
    def detect_emotion_from_audio(self, audio_data: bytes, audio_format: str = 'wav') -> Dict[str, Any]:
        """
        Detect emotion from audio using basic audio analysis
        
        Args:
            audio_data: Audio data as bytes
            audio_format: Audio format (wav, mp3, etc.)
        
        Returns:
            Dictionary with emotion detection results
        """
        try:
            logger.info("Audio emotion detection not available in deployment version")
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'emoji': '😐',
                'model': 'rule_based'
            }
            
        except Exception as e:
            logger.error(f"Error in audio emotion detection: {str(e)}")
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'emoji': '😐',
                'model': 'rule_based'
            }
    
    def detect_emotion_combined(self, text: str, audio_data: bytes = None, audio_format: str = 'wav') -> Dict[str, Any]:
        """
        Detect emotion from combined text and audio (text-only in deployment)
        
        Args:
            text: Input text to analyze
            audio_data: Audio data as bytes (optional)
            audio_format: Audio format (optional)
        
        Returns:
            Dictionary with emotion detection results
        """
        try:
            # In deployment version, we only use text-based detection
            return self.detect_emotion_from_text(text)
            
        except Exception as e:
            logger.error(f"Error in combined emotion detection: {str(e)}")
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'emoji': '😐',
                'model': 'rule_based'
            }
    
    def _detect_emotion_rule_based(self, text: str) -> Dict[str, Any]:
        """
        Rule-based emotion detection using keyword analysis
        
        Args:
            text: Input text to analyze
        
        Returns:
            Dictionary with emotion detection results
        """
        # Convert to lowercase for analysis
        text_lower = text.lower()
        
        # Define emotion keywords and their weights
        emotion_keywords = {
            'happy': {
                'keywords': ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'fantastic', 'love', 'like', 'good', 'nice', 'beautiful', 'awesome'],
                'emoji': '😊',
                'weight': 1.0
            },
            'sad': {
                'keywords': ['sad', 'depressed', 'unhappy', 'miserable', 'terrible', 'awful', 'bad', 'hate', 'dislike', 'sorry', 'regret'],
                'emoji': '😢',
                'weight': 1.0
            },
            'angry': {
                'keywords': ['angry', 'mad', 'furious', 'rage', 'hate', 'terrible', 'awful', 'horrible', 'disgusting'],
                'emoji': '😠',
                'weight': 1.0
            },
            'excited': {
                'keywords': ['excited', 'thrilled', 'amazing', 'incredible', 'wow', 'awesome', 'fantastic', 'brilliant'],
                'emoji': '🤩',
                'weight': 0.8
            },
            'calm': {
                'keywords': ['calm', 'peaceful', 'relaxed', 'quiet', 'gentle', 'soft', 'smooth'],
                'emoji': '😌',
                'weight': 0.7
            },
            'surprised': {
                'keywords': ['surprised', 'shocked', 'amazed', 'wow', 'incredible', 'unbelievable'],
                'emoji': '😲',
                'weight': 0.8
            }
        }
        
        # Calculate emotion scores
        emotion_scores = {}
        words = re.findall(r'\b\w+\b', text_lower)
        
        for emotion, config in emotion_keywords.items():
            score = 0
            for keyword in config['keywords']:
                if keyword in text_lower:
                    score += config['weight']
            emotion_scores[emotion] = score
        
        # Find the emotion with highest score
        if emotion_scores:
            max_emotion = max(emotion_scores, key=emotion_scores.get)
            max_score = emotion_scores[max_emotion]
            
            # Calculate confidence (normalize by text length)
            confidence = min(max_score / max(len(words), 1), 1.0)
            
            # If no strong emotion detected, default to neutral
            if confidence < 0.1:
                return {
                    'emotion': 'neutral',
                    'confidence': 0.5,
                    'emoji': '😐',
                    'model': 'rule_based'
                }
            
            return {
                'emotion': max_emotion,
                'confidence': confidence,
                'emoji': emotion_keywords[max_emotion]['emoji'],
                'model': 'rule_based'
            }
        
        # Default to neutral if no emotions detected
        return {
            'emotion': 'neutral',
            'confidence': 0.5,
            'emoji': '😐',
            'model': 'rule_based'
        }

# Create global instance
emotion_service = EmotionService() 