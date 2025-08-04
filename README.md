# ToneBridge Backend

Real-time speech-to-text with emotion detection for accessibility.

## 🚀 Features

- **Real-time Transcription**: Convert speech to text using Whisper or speech_recognition
- **Emotion Detection**: Detect emotions from text using transformer models
- **Audio Processing**: Support for multiple audio formats (WAV, MP3, M4A, FLAC)
- **RESTful API**: Clean, documented API endpoints
- **Error Handling**: Comprehensive error handling and logging
- **Scalable Architecture**: Modular design following DRY principles

## 📋 Requirements

- Python 3.8+
- Flask 2.3+
- PyTorch (for transformer models)
- Audio processing libraries (librosa, soundfile, pydub)

## 🛠️ Installation

### **Option 1: Quick Setup (Recommended)**
```bash
cd ToneBridge/backend
python install.py
```

### **Option 2: Manual Installation**

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ToneBridge/backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install basic dependencies (without PyTorch)**
   ```bash
   pip install -r requirements.txt
   ```

4. **Optional: Install PyTorch for AI models**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install transformers
   ```

5. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

6. **Create uploads directory**
   ```bash
   mkdir uploads
   ```

## 🚀 Running the Application

### Development Mode
```bash
python app.py
```

### Production Mode
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:create_app()
```

## 📚 API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "ToneBridge Backend",
  "version": "1.0.0"
}
```

#### 2. Transcribe Audio
```http
POST /api/transcribe
```

**Request Body:**
```json
{
  "audio": "base64_encoded_audio_data",
  "format": "wav",
  "include_emotion": true
}
```

**Response:**
```json
{
  "success": true,
  "message": "Transcription completed",
  "data": {
    "transcript": "Hello, how are you?",
    "confidence": 0.95,
    "model": "whisper",
    "language": "en",
    "emotion": "happy",
    "emotion_confidence": 0.87,
    "emotion_emoji": "😊",
    "emotion_model": "text_classification"
  },
  "timestamp": 1234567890.123
}
```

#### 3. Detect Emotion
```http
POST /api/emotion
```

**Request Body:**
```json
{
  "text": "I'm feeling great today!",
  "audio": "base64_encoded_audio_data",
  "format": "wav",
  "method": "combined"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Emotion detection completed",
  "data": {
    "emotion": "excited",
    "confidence": 0.92,
    "emoji": "🤩",
    "model": "combined",
    "text_emotion": {...},
    "audio_emotion": {...}
  },
  "timestamp": 1234567890.123
}
```

#### 4. Upload Audio File
```http
POST /api/upload
```

**Request:** Multipart form data
- `file`: Audio file (WAV, MP3, M4A, FLAC)
- `include_emotion`: Boolean (optional)

**Response:**
```json
{
  "success": true,
  "message": "File processed successfully",
  "data": {
    "transcript": "Hello, how are you?",
    "confidence": 0.95,
    "model": "whisper",
    "language": "en",
    "filename": "audio.wav",
    "emotion": "happy",
    "emotion_confidence": 0.87,
    "emotion_emoji": "😊",
    "emotion_model": "text_classification"
  },
  "timestamp": 1234567890.123
}
```

#### 5. Get Models Information
```http
GET /api/models
```

**Response:**
```json
{
  "success": true,
  "message": "Model information retrieved",
  "data": {
    "transcription_models": {
      "whisper": true,
      "speech_recognition": true
    },
    "emotion_models": {
      "text_classification": true,
      "audio_features": true
    },
    "supported_emotions": ["happy", "sad", "angry", ...],
    "emotion_emojis": {
      "happy": "😊",
      "sad": "😢",
      ...
    }
  },
  "timestamp": 1234567890.123
}
```

## 🏗️ Architecture

### Directory Structure
```
backend/
├── app.py                 # Main Flask application
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── env.example          # Environment variables template
├── README.md            # This file
├── services/            # Business logic services
│   ├── __init__.py
│   ├── transcription_service.py
│   └── emotion_service.py
├── utils/               # Utility functions
│   ├── __init__.py
│   ├── logger.py
│   ├── error_handlers.py
│   └── audio_utils.py
├── routes/              # API routes
│   ├── __init__.py
│   └── api.py
└── uploads/             # File upload directory
```

### Key Components

1. **Services Layer**: Business logic for transcription and emotion detection
2. **Utils Layer**: Reusable utilities for logging, error handling, and audio processing
3. **Routes Layer**: API endpoint definitions
4. **Config**: Centralized configuration management

### Design Principles

- **DRY (Don't Repeat Yourself)**: Shared utilities and centralized configuration
- **Separation of Concerns**: Clear separation between services, routes, and utilities
- **Error Handling**: Comprehensive error handling with custom exceptions
- **Logging**: Structured logging throughout the application
- **Modularity**: Easy to extend and modify individual components

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASK_ENV` | Flask environment | `development` |
| `SECRET_KEY` | Flask secret key | `dev-secret-key-change-in-production` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `5000` |
| `EMOTION_MODEL_NAME` | Emotion detection model | `m3hrdadfi/emotion-english-distilroberta-base` |
| `TRANSCRIPTION_MODEL_NAME` | Transcription model | `openai/whisper-base` |
| `UPLOAD_FOLDER` | File upload directory | `uploads` |
| `MAX_CONTENT_LENGTH` | Max file size (bytes) | `16777216` (16MB) |

### Supported Emotions

- `happy`, `sad`, `angry`, `fear`, `surprise`, `disgust`
- `neutral`, `sarcastic`, `excited`, `calm`, `frustrated`

## 🧪 Testing

Run tests with pytest:
```bash
pytest tests/
```

## 🚀 Deployment

### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:create_app()"]
```

### Render/Heroku
The application is ready for deployment on Render, Heroku, or similar platforms.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions, please open an issue on GitHub. 