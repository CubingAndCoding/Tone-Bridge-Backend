"""
API tests for ToneBridge Backend
"""

import pytest
import json
import base64
from io import BytesIO
from unittest.mock import patch, MagicMock

from app import create_app
from config import TestingConfig

@pytest.fixture
def app():
    """Create test app"""
    app = create_app(TestingConfig)
    return app

@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()

@pytest.fixture
def mock_audio_data():
    """Mock audio data for testing"""
    # Create a simple WAV file in memory
    import wave
    import numpy as np
    
    # Create a simple sine wave
    sample_rate = 16000
    duration = 1  # 1 second
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t) * 0.5
    
    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Create WAV file in memory
    buffer = BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    return buffer.getvalue()

class TestHealthCheck:
    """Test health check endpoint"""
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert data['service'] == 'ToneBridge Backend'

class TestTranscribeAPI:
    """Test transcription endpoint"""
    
    def test_transcribe_missing_audio(self, client):
        """Test transcription with missing audio data"""
        response = client.post('/api/transcribe', 
                             json={'format': 'wav'})
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert not data['success']
        assert 'No audio data provided' in data['error']['message']
    
    def test_transcribe_invalid_base64(self, client):
        """Test transcription with invalid base64 data"""
        response = client.post('/api/transcribe', 
                             json={'audio': 'invalid-base64'})
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert not data['success']
    
    @patch('services.transcription_service.transcription_service.transcribe_audio')
    def test_transcribe_success(self, mock_transcribe, client, mock_audio_data):
        """Test successful transcription"""
        # Mock transcription service
        mock_transcribe.return_value = {
            'text': 'Hello, how are you?',
            'confidence': 0.95,
            'model': 'whisper',
            'language': 'en'
        }
        
        # Encode audio data
        audio_b64 = base64.b64encode(mock_audio_data).decode('utf-8')
        
        response = client.post('/api/transcribe', 
                             json={'audio': audio_b64, 'format': 'wav'})
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['success']
        assert data['data']['transcript'] == 'Hello, how are you?'
        assert data['data']['confidence'] == 0.95

class TestEmotionAPI:
    """Test emotion detection endpoint"""
    
    def test_emotion_missing_input(self, client):
        """Test emotion detection with no input"""
        response = client.post('/api/emotion', json={})
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert not data['success']
        assert 'Either text or audio must be provided' in data['error']['message']
    
    @patch('services.emotion_service.emotion_service.detect_emotion_from_text')
    def test_emotion_text_only(self, mock_emotion, client):
        """Test emotion detection with text only"""
        # Mock emotion service
        mock_emotion.return_value = {
            'emotion': 'happy',
            'confidence': 0.87,
            'emoji': '😊',
            'model': 'text_classification'
        }
        
        response = client.post('/api/emotion', 
                             json={'text': 'I am happy today!'})
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['success']
        assert data['data']['emotion'] == 'happy'
        assert data['data']['emoji'] == '😊'
    
    @patch('services.emotion_service.emotion_service.detect_emotion_from_audio')
    def test_emotion_audio_only(self, mock_emotion, client, mock_audio_data):
        """Test emotion detection with audio only"""
        # Mock emotion service
        mock_emotion.return_value = {
            'emotion': 'excited',
            'confidence': 0.75,
            'emoji': '🤩',
            'model': 'audio_features'
        }
        
        # Encode audio data
        audio_b64 = base64.b64encode(mock_audio_data).decode('utf-8')
        
        response = client.post('/api/emotion', 
                             json={'audio': audio_b64, 'format': 'wav', 'method': 'audio'})
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['success']
        assert data['data']['emotion'] == 'excited'
        assert data['data']['emoji'] == '🤩'

class TestUploadAPI:
    """Test file upload endpoint"""
    
    def test_upload_no_file(self, client):
        """Test upload with no file"""
        response = client.post('/api/upload')
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert not data['success']
        assert 'No file uploaded' in data['error']['message']
    
    def test_upload_invalid_file(self, client):
        """Test upload with invalid file"""
        data = {'file': (BytesIO(b'invalid data'), 'test.txt')}
        response = client.post('/api/upload', data=data)
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert not data['success']
    
    @patch('services.transcription_service.transcription_service.transcribe_audio')
    def test_upload_success(self, mock_transcribe, client, mock_audio_data):
        """Test successful file upload"""
        # Mock transcription service
        mock_transcribe.return_value = {
            'text': 'Hello from uploaded file',
            'confidence': 0.92,
            'model': 'whisper',
            'language': 'en'
        }
        
        # Create file upload
        data = {'file': (BytesIO(mock_audio_data), 'test.wav')}
        response = client.post('/api/upload', data=data)
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['success']
        assert data['data']['transcript'] == 'Hello from uploaded file'
        assert data['data']['filename'] == 'test.wav'

class TestModelsAPI:
    """Test models information endpoint"""
    
    def test_models_info(self, client):
        """Test models information endpoint"""
        response = client.get('/api/models')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['success']
        assert 'transcription_models' in data['data']
        assert 'emotion_models' in data['data']
        assert 'supported_emotions' in data['data']
        assert 'emotion_emojis' in data['data']

class TestErrorHandling:
    """Test error handling"""
    
    def test_404_error(self, client):
        """Test 404 error handling"""
        response = client.get('/nonexistent')
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test method not allowed error"""
        response = client.get('/api/transcribe')
        assert response.status_code == 405
    
    def test_invalid_json(self, client):
        """Test invalid JSON handling"""
        response = client.post('/api/transcribe', 
                             data='invalid json',
                             content_type='application/json')
        assert response.status_code == 400 