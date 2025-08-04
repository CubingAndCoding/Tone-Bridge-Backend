"""
API routes for ToneBridge Backend
"""

import time
import base64
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os

from services.transcription_service import transcription_service
from services.emotion_service import emotion_service
from utils.logger import setup_logger, log_request
from utils.error_handlers import ValidationError, validate_audio_file
from config import Config

logger = setup_logger(__name__)

# Create blueprint
api_bp = Blueprint('api', __name__)

def create_success_response(data: dict, message: str = "Success") -> dict:
    """
    Create standardized success response
    
    Args:
        data: Response data
        message: Success message
    
    Returns:
        Standardized response dictionary
    """
    return {
        'success': True,
        'message': message,
        'data': data,
        'timestamp': time.time()
    }

@api_bp.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """
    Transcribe audio and detect emotion
    
    Expected request:
    - audio: base64 encoded audio data
    - format: audio format (optional, defaults to 'wav')
    - include_emotion: boolean (optional, defaults to True)
    """
    start_time = time.time()
    
    try:
        # Get request data
        data = request.get_json()
        if not data:
            raise ValidationError("No JSON data provided")
        
        # Extract parameters
        audio_b64 = data.get('audio')
        audio_format = data.get('format', 'wav')
        include_emotion = data.get('include_emotion', True)
        
        # Normalize and validate audio format
        if audio_format:
            # Map common MIME types to format names
            format_mapping = {
                'audio/webm': 'webm',
                'audio/mp4': 'mp4',
                'audio/wav': 'wav',
                'audio/mpeg': 'mp3',
                'audio/ogg': 'ogg',
                'audio/flac': 'flac',
                'audio/aac': 'aac'
            }
            
            # If format looks like a MIME type, convert it
            if audio_format.startswith('audio/'):
                audio_format = format_mapping.get(audio_format, audio_format.split('/')[-1])
            
            # Clean up format name
            audio_format = audio_format.lower().strip()
            
            # Validate format
            supported_formats = ['wav', 'webm', 'mp4', 'mp3', 'ogg', 'flac', 'aac', 'm4a']
            if audio_format not in supported_formats:
                logger.warning(f"Unsupported format '{audio_format}', defaulting to 'wav'")
                audio_format = 'wav'
        
        if not audio_b64:
            raise ValidationError("No audio data provided")
        
        # Decode base64 audio
        try:
            audio_data = base64.b64decode(audio_b64)
            logger.info(f"Audio data decoded successfully: {len(audio_data)} bytes, format: {audio_format}")
            
            # Validate audio data
            if len(audio_data) == 0:
                raise ValidationError("Audio data is empty after base64 decoding")
            if len(audio_data) < 100:  # Very small audio files are suspicious
                logger.warning(f"Audio data seems very small: {len(audio_data)} bytes")
                
        except Exception as e:
            logger.error(f"Base64 decoding failed: {str(e)}")
            raise ValidationError(f"Invalid base64 audio data: {str(e)}")
        
        # Transcribe audio
        transcription_result = transcription_service.transcribe_audio(audio_data, audio_format)
        
        # Initialize response
        response_data = {
            'transcript': transcription_result['text'],
            'confidence': transcription_result['confidence'],
            'model': transcription_result['model'],
            'language': transcription_result.get('language', 'en')
        }
        
        # Detect emotion if requested
        if include_emotion and transcription_result['text'].strip():
            emotion_result = emotion_service.detect_emotion_from_text(transcription_result['text'])
            
            response_data.update({
                'emotion': emotion_result['emotion'],
                'emotion_confidence': emotion_result['confidence'],
                'emotion_emoji': emotion_result['emoji'],
                'emotion_model': emotion_result['model']
            })
        
        # Log request
        duration = time.time() - start_time
        log_request(logger, {
            'audio_length': len(audio_data),
            'format': audio_format,
            'include_emotion': include_emotion
        }, response_data, duration)
        
        return jsonify(create_success_response(response_data, "Transcription completed"))
        
    except Exception as e:
        logger.error(f"Transcription endpoint error: {str(e)}")
        raise

@api_bp.route('/emotion', methods=['POST'])
def detect_emotion():
    """
    Detect emotion from text or audio
    
    Expected request:
    - text: text to analyze (optional if audio provided)
    - audio: base64 encoded audio data (optional if text provided)
    - format: audio format (optional, defaults to 'wav')
    - method: 'text', 'audio', or 'combined' (optional, defaults to 'combined')
    """
    start_time = time.time()
    
    try:
        # Get request data
        data = request.get_json()
        if not data:
            raise ValidationError("No JSON data provided")
        
        # Extract parameters
        text = data.get('text', '').strip()
        audio_b64 = data.get('audio')
        audio_format = data.get('format', 'wav')
        method = data.get('method', 'combined')
        
        if not text and not audio_b64:
            raise ValidationError("Either text or audio must be provided")
        
        # Detect emotion based on method
        if method == 'text':
            if not text:
                raise ValidationError("Text is required for text-only emotion detection")
            emotion_result = emotion_service.detect_emotion_from_text(text)
            
        elif method == 'audio':
            if not audio_b64:
                raise ValidationError("Audio is required for audio-only emotion detection")
            
            audio_data = base64.b64decode(audio_b64)
            emotion_result = emotion_service.detect_emotion_from_audio(audio_data, audio_format)
            
        else:  # combined
            audio_data = None
            if audio_b64:
                audio_data = base64.b64decode(audio_b64)
            
            emotion_result = emotion_service.detect_emotion_combined(text, audio_data, audio_format)
        
        # Prepare response
        response_data = {
            'emotion': emotion_result['emotion'],
            'confidence': emotion_result['confidence'],
            'emoji': emotion_result['emoji'],
            'model': emotion_result['model']
        }
        
        # Add additional data if available
        if 'original_text' in emotion_result:
            response_data['text'] = emotion_result['original_text']
        
        if 'text_emotion' in emotion_result:
            response_data['text_emotion'] = emotion_result['text_emotion']
        
        if 'audio_emotion' in emotion_result:
            response_data['audio_emotion'] = emotion_result['audio_emotion']
        
        # Log request
        duration = time.time() - start_time
        log_request(logger, {
            'text_length': len(text),
            'audio_length': len(audio_b64) if audio_b64 else 0,
            'method': method
        }, response_data, duration)
        
        return jsonify(create_success_response(response_data, "Emotion detection completed"))
        
    except Exception as e:
        logger.error(f"Emotion detection endpoint error: {str(e)}")
        raise

@api_bp.route('/upload', methods=['POST'])
def upload_audio():
    """
    Upload audio file for processing
    
    Expected request:
    - file: audio file (multipart/form-data)
    - include_emotion: boolean (optional, defaults to True)
    """
    start_time = time.time()
    
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            raise ValidationError("No file uploaded")
        
        file = request.files['file']
        
        # Validate file
        validate_audio_file(file)
        
        # Get parameters
        include_emotion = request.form.get('include_emotion', 'true').lower() == 'true'
        
        # Read file data
        audio_data = file.read()
        audio_format = file.filename.split('.')[-1].lower()
        
        # Transcribe audio
        transcription_result = transcription_service.transcribe_audio(audio_data, audio_format)
        
        # Initialize response
        response_data = {
            'transcript': transcription_result['text'],
            'confidence': transcription_result['confidence'],
            'model': transcription_result['model'],
            'language': transcription_result.get('language', 'en'),
            'filename': secure_filename(file.filename)
        }
        
        # Detect emotion if requested
        if include_emotion and transcription_result['text'].strip():
            emotion_result = emotion_service.detect_emotion_from_text(transcription_result['text'])
            
            response_data.update({
                'emotion': emotion_result['emotion'],
                'emotion_confidence': emotion_result['confidence'],
                'emotion_emoji': emotion_result['emoji'],
                'emotion_model': emotion_result['model']
            })
        
        # Log request
        duration = time.time() - start_time
        log_request(logger, {
            'filename': file.filename,
            'file_size': len(audio_data),
            'include_emotion': include_emotion
        }, response_data, duration)
        
        return jsonify(create_success_response(response_data, "File processed successfully"))
        
    except Exception as e:
        logger.error(f"Upload endpoint error: {str(e)}")
        raise

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'ToneBridge API',
        'version': '1.0.0',
        'timestamp': time.time()
    })

@api_bp.route('/debug/audio', methods=['POST'])
def debug_audio():
    """Debug endpoint to test audio data"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        audio_b64 = data.get('audio')
        audio_format = data.get('format', 'wav')
        
        if not audio_b64:
            return jsonify({'error': 'No audio data provided'}), 400
        
        # Decode base64
        try:
            audio_data = base64.b64decode(audio_b64)
        except Exception as e:
            return jsonify({'error': f'Invalid base64: {str(e)}'}), 400
        
        # Return debug info
        return jsonify({
            'success': True,
            'audio_length': len(audio_data),
            'format': audio_format,
            'base64_length': len(audio_b64),
            'first_100_chars': audio_b64[:100] if len(audio_b64) > 100 else audio_b64
        })
        
    except Exception as e:
        return jsonify({'error': f'Debug failed: {str(e)}'}), 500

@api_bp.route('/models', methods=['GET'])
def get_models():
    """Get available models and their status"""
    return jsonify(create_success_response({
        'transcription_models': {
            'whisper': transcription_service.whisper_pipeline is not None,
            'speech_recognition': True
        },
        'emotion_models': {
            'text_classification': emotion_service.text_emotion_pipeline is not None,
            'audio_features': True
        },
        'supported_emotions': Config.SUPPORTED_EMOTIONS,
        'emotion_emojis': Config.EMOTION_EMOJIS
    }, "Model information retrieved")) 

@api_bp.route('/tts', methods=['POST'])
def text_to_speech():
    """
    Convert text to speech with voice options
    
    Expected request:
    - text: text to convert to speech
    - voice: voice ID (optional, defaults to 'en-US-Neural2-A')
    - speed: playback speed (optional, defaults to 1.0)
    - pitch: pitch adjustment (optional, defaults to 1.0)
    - volume: volume level (optional, defaults to 1.0)
    """
    start_time = time.time()
    
    try:
        # Get request data
        data = request.get_json()
        if not data:
            raise ValidationError("No JSON data provided")
        
        # Extract parameters
        text = data.get('text', '').strip()
        voice = data.get('voice', 'en-US-Neural2-A')
        speed = float(data.get('speed', 1.0))
        pitch = float(data.get('pitch', 1.0))
        volume = float(data.get('volume', 1.0))
        
        if not text:
            raise ValidationError("No text provided")
        
        # Validate parameters
        if len(text) > 5000:
            raise ValidationError("Text too long (max 5000 characters)")
        
        if speed < 0.5 or speed > 2.0:
            raise ValidationError("Speed must be between 0.5 and 2.0")
        
        if pitch < 0.5 or pitch > 2.0:
            raise ValidationError("Pitch must be between 0.5 and 2.0")
        
        if volume < 0.0 or volume > 1.0:
            raise ValidationError("Volume must be between 0.0 and 1.0")
        
        # Available voices (simulated for now)
        available_voices = [
            'en-US-Neural2-A', 'en-US-Neural2-B', 'en-US-Neural2-C',
            'en-US-Neural2-D', 'en-US-Neural2-E', 'en-US-Neural2-F'
        ]
        
        if voice not in available_voices:
            raise ValidationError(f"Voice '{voice}' not available")
        
        # Simulate TTS processing (in a real implementation, you'd use a TTS service)
        import time
        time.sleep(0.5)  # Simulate processing time
        
        # Calculate estimated duration (rough estimate: 150 words per minute)
        word_count = len(text.split())
        estimated_duration = (word_count / 150) * 60  # seconds
        
        # Generate a mock audio URL (in real implementation, this would be the actual audio file)
        audio_url = f"/api/tts/audio/{int(time.time())}_{hash(text) % 10000}"
        
        # Prepare response
        response_data = {
            'audio_url': audio_url,
            'duration': estimated_duration,
            'word_count': word_count,
            'voice_used': voice,
            'text_length': len(text),
            'speed': speed,
            'pitch': pitch,
            'volume': volume
        }
        
        # Log request
        duration = time.time() - start_time
        log_request(logger, {
            'text_length': len(text),
            'voice': voice,
            'speed': speed,
            'pitch': pitch,
            'volume': volume
        }, response_data, duration)
        
        return jsonify(create_success_response(response_data, "Text-to-speech conversion completed"))
        
    except Exception as e:
        logger.error(f"TTS endpoint error: {str(e)}")
        raise

@api_bp.route('/tts/voices', methods=['GET'])
def get_voices():
    """Get available TTS voices"""
    voices = [
        {
            'id': 'en-US-Neural2-A',
            'name': 'Emma',
            'language': 'English (US)',
            'gender': 'female',
            'description': 'Clear and friendly female voice'
        },
        {
            'id': 'en-US-Neural2-B',
            'name': 'James',
            'language': 'English (US)',
            'gender': 'male',
            'description': 'Professional male voice'
        },
        {
            'id': 'en-US-Neural2-C',
            'name': 'Sophia',
            'language': 'English (US)',
            'gender': 'female',
            'description': 'Warm and expressive voice'
        },
        {
            'id': 'en-US-Neural2-D',
            'name': 'Michael',
            'language': 'English (US)',
            'gender': 'male',
            'description': 'Deep and authoritative voice'
        },
        {
            'id': 'en-US-Neural2-E',
            'name': 'Olivia',
            'language': 'English (US)',
            'gender': 'female',
            'description': 'Young and energetic voice'
        },
        {
            'id': 'en-US-Neural2-F',
            'name': 'David',
            'language': 'English (US)',
            'gender': 'male',
            'description': 'Calm and soothing voice'
        }
    ]
    
    return jsonify(create_success_response({
        'voices': voices,
        'total': len(voices)
    }, "Voice list retrieved")) 