"""
Configuration settings for ToneBridge Backend
"""

import os
from typing import Optional

class Config:
    """Base configuration class"""
    
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Server settings
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    
    # CORS settings
    ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '').split(',') if os.getenv('ALLOWED_ORIGINS') else []
    
    # API settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    REQUEST_TIMEOUT = 30  # seconds
    
    # Security settings
    DISABLE_LOGGING = os.getenv('DISABLE_LOGGING', 'False').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Privacy settings - ensure no data persistence
    NO_DATA_STORAGE = True  # Never store user data
    NO_AUDIO_STORAGE = True  # Never store audio files
    NO_TRANSCRIPTION_STORAGE = True  # Never store transcriptions
    NO_ANALYTICS_STORAGE = True  # Never store analytics data
    
    # Model settings
    TRANSCRIPTION_MODEL = os.getenv('TRANSCRIPTION_MODEL', 'whisper-1')
    EMOTION_MODEL = os.getenv('EMOTION_MODEL', 'emotion-english-distilroberta-base')
    TRANSCRIPTION_MODEL_NAME = os.getenv('TRANSCRIPTION_MODEL_NAME', 'openai/whisper-base')
    
    # Free tier optimizations
    USE_LIGHTWEIGHT_MODELS = os.getenv('USE_LIGHTWEIGHT_MODELS', 'True').lower() == 'true'
    MAX_AUDIO_SIZE = int(os.getenv('MAX_AUDIO_SIZE', 5 * 1024 * 1024))  # 5MB for free tier
    
    # Rate limiting (optional)
    RATE_LIMIT_ENABLED = os.getenv('RATE_LIMIT_ENABLED', 'False').lower() == 'true'
    RATE_LIMIT_REQUESTS = int(os.getenv('RATE_LIMIT_REQUESTS', 100))  # requests per minute
    RATE_LIMIT_WINDOW = int(os.getenv('RATE_LIMIT_WINDOW', 60))  # seconds

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    
    # Development-specific settings
    ALLOWED_ORIGINS = [
        'http://localhost:3000',
        'http://localhost:8100',
        'https://localhost:8100',
        'http://127.0.0.1:3000',
        'http://127.0.0.1:8100',
        'https://127.0.0.1:8100'
    ]

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    DISABLE_LOGGING = True  # Disable file logging in production
    
    # Production security
    SECRET_KEY = os.getenv('SECRET_KEY')  # Must be set in production
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY environment variable must be set in production")
    
    # Production CORS - more restrictive
    ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '').split(',') if os.getenv('ALLOWED_ORIGINS') else []
    
    # Production rate limiting
    RATE_LIMIT_ENABLED = True

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    
    # Test-specific settings
    ALLOWED_ORIGINS = ['http://localhost:3000', 'http://localhost:8100']

# Configuration mapping
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(config_name: Optional[str] = None) -> Config:
    """
    Get configuration based on environment
    
    Args:
        config_name: Configuration name (development, production, testing)
    
    Returns:
        Configuration instance
    """
    if not config_name:
        config_name = os.getenv('FLASK_ENV', 'development')
    
    config_class = config_map.get(config_name, config_map['default'])
    return config_class() 