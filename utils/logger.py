"""
Centralized logging utility for ToneBridge Backend
"""

import logging
import sys
from typing import Optional
from datetime import datetime
import os

def setup_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Setup a logger with consistent formatting and handlers
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level (defaults to INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Set log level
    log_level = level or logging.INFO
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (only in development)
    if not os.getenv('DISABLE_LOGGING', 'False').lower() == 'true':
        try:
            file_handler = logging.FileHandler('tonebridge.log')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except (PermissionError, OSError):
            # If we can't write to file, just use console
            pass
    
    return logger

def log_request(logger: logging.Logger, request_data: dict, response_data: dict, duration: float):
    """
    Log API request/response details (privacy-aware)
    
    Args:
        logger: Logger instance
        request_data: Request information (sanitized)
        response_data: Response information (sanitized)
        duration: Request duration in seconds
    """
    # Sanitize sensitive data
    sanitized_request = sanitize_log_data(request_data)
    sanitized_response = sanitize_log_data(response_data)
    
    logger.info(
        f"API Request - Duration: {duration:.3f}s | "
        f"Request: {sanitized_request} | Response: {sanitized_response}"
    )

def log_error(logger: logging.Logger, error: Exception, context: dict = None):
    """
    Log error with context (privacy-aware)
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Additional context information (sanitized)
    """
    sanitized_context = sanitize_log_data(context) if context else None
    context_str = f" | Context: {sanitized_context}" if sanitized_context else ""
    logger.error(f"Error: {str(error)}{context_str}", exc_info=True)

def sanitize_log_data(data: dict) -> dict:
    """
    Remove sensitive information from log data
    
    Args:
        data: Data to sanitize
    
    Returns:
        Sanitized data dictionary
    """
    if not isinstance(data, dict):
        return data
    
    sensitive_keys = [
        'audio', 'audio_b64', 'audio_data', 'transcript', 'text',
        'emotion', 'confidence', 'user_data', 'personal_info'
    ]
    
    sanitized = {}
    for key, value in data.items():
        if key.lower() in [k.lower() for k in sensitive_keys]:
            if isinstance(value, str) and len(value) > 10:
                sanitized[key] = f"[REDACTED - {len(value)} chars]"
            elif isinstance(value, (list, dict)):
                sanitized[key] = f"[REDACTED - {type(value).__name__}]"
            else:
                sanitized[key] = "[REDACTED]"
        else:
            sanitized[key] = value
    
    return sanitized

def log_privacy_notice(logger: logging.Logger):
    """
    Log privacy notice to confirm no data storage
    
    Args:
        logger: Logger instance
    """
    logger.info("🔒 PRIVACY NOTICE: No user data, audio, or transcriptions are stored on the server")
    logger.info("📱 All transcriptions are stored only in the user's browser localStorage")
    logger.info("🗑️  Audio data is processed in memory and immediately discarded") 