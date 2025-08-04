"""
Centralized error handlers for ToneBridge Backend
"""

from flask import jsonify, request
from werkzeug.exceptions import HTTPException
import traceback
from utils.logger import setup_logger

logger = setup_logger(__name__)

class ToneBridgeError(Exception):
    """Base exception class for ToneBridge"""
    def __init__(self, message: str, status_code: int = 400, error_code: str = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or self.__class__.__name__

class AudioProcessingError(ToneBridgeError):
    """Raised when audio processing fails"""
    def __init__(self, message: str = "Audio processing failed"):
        super().__init__(message, status_code=422, error_code="AUDIO_PROCESSING_ERROR")

class ModelError(ToneBridgeError):
    """Raised when ML model operations fail"""
    def __init__(self, message: str = "Model processing failed"):
        super().__init__(message, status_code=500, error_code="MODEL_ERROR")

class ValidationError(ToneBridgeError):
    """Raised when input validation fails"""
    def __init__(self, message: str = "Validation failed"):
        super().__init__(message, status_code=400, error_code="VALIDATION_ERROR")

def create_error_response(error: Exception, status_code: int = 400) -> tuple:
    """
    Create standardized error response
    
    Args:
        error: Exception that occurred
        status_code: HTTP status code
    
    Returns:
        Tuple of (response_dict, status_code)
    """
    error_response = {
        'error': {
            'message': str(error),
            'type': error.__class__.__name__,
            'status_code': status_code
        },
        'success': False,
        'timestamp': None  # Will be set by middleware
    }
    
    # Add additional context for ToneBridgeError
    if isinstance(error, ToneBridgeError):
        error_response['error']['code'] = error.error_code
    
    return error_response, status_code

def register_error_handlers(app):
    """Register all error handlers with the Flask app"""
    
    @app.errorhandler(ToneBridgeError)
    def handle_tonebridge_error(error):
        """Handle custom ToneBridge exceptions"""
        logger.error(f"ToneBridge Error: {error.message}", exc_info=True)
        return jsonify(*create_error_response(error, error.status_code))
    
    @app.errorhandler(HTTPException)
    def handle_http_error(error):
        """Handle HTTP exceptions"""
        logger.error(f"HTTP Error: {error.description}", exc_info=True)
        return jsonify(*create_error_response(error, error.code))
    
    @app.errorhandler(ValueError)
    def handle_value_error(error):
        """Handle ValueError exceptions"""
        logger.error(f"Value Error: {str(error)}", exc_info=True)
        return jsonify(*create_error_response(error, 400))
    
    @app.errorhandler(TypeError)
    def handle_type_error(error):
        """Handle TypeError exceptions"""
        logger.error(f"Type Error: {str(error)}", exc_info=True)
        return jsonify(*create_error_response(error, 400))
    
    @app.errorhandler(Exception)
    def handle_generic_error(error):
        """Handle all other exceptions"""
        logger.error(f"Unexpected Error: {str(error)}", exc_info=True)
        return jsonify(*create_error_response(error, 500))
    
    @app.before_request
    def log_request_start():
        """Log request start for timing"""
        request.start_time = None  # Will be set by middleware
    
    @app.after_request
    def log_request_end(response):
        """Log request completion and timing"""
        if hasattr(request, 'start_time') and request.start_time:
            duration = None  # Calculate duration if needed
            logger.info(f"Request completed: {request.method} {request.path} - {response.status_code}")
        return response

def validate_audio_file(file) -> bool:
    """
    Validate uploaded audio file
    
    Args:
        file: FileStorage object from Flask
    
    Returns:
        True if valid, raises ValidationError if not
    """
    if not file:
        raise ValidationError("No audio file provided")
    
    if file.filename == '':
        raise ValidationError("No file selected")
    
    # Check file size (16MB limit)
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning
    
    if file_size > 16 * 1024 * 1024:  # 16MB
        raise ValidationError("File size exceeds 16MB limit")
    
    # Check file extension
    allowed_extensions = {'wav', 'mp3', 'm4a', 'flac'}
    if not file.filename.lower().endswith(tuple(f'.{ext}' for ext in allowed_extensions)):
        raise ValidationError(f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}")
    
    return True 