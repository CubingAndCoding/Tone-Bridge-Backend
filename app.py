"""
ToneBridge Backend - Main Application
Real-time speech-to-text with emotion detection
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from config import Config
from services.transcription_service import TranscriptionService
from services.emotion_service import EmotionService
from utils.logger import setup_logger
from utils.error_handlers import register_error_handlers
from routes.api import api_bp

# Load environment variables
load_dotenv()

# Setup logging
logger = setup_logger(__name__)

def create_app(config_class=Config):
    """Application factory pattern for Flask app creation"""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize CORS with dynamic origins for local development
    allowed_origins = [
        "http://localhost:3000", 
        "http://localhost:8100", 
        "https://localhost:8100",
        "https://tonebridge.vercel.app"
    ]
    
    # Add environment variable origins if specified
    env_origins = os.getenv('ALLOWED_ORIGINS', '')
    if env_origins:
        allowed_origins.extend([origin.strip() for origin in env_origins.split(',')])
    
    # Add specific IP addresses for common local development
    specific_ips = [
        "192.168.1.210",  # Common local network IP
        "192.168.1.100",  # Another common local network IP
        "10.0.0.1",       # Common router IP
    ]
    
    for ip in specific_ips:
        allowed_origins.extend([
            f"http://{ip}:3000",
            f"http://{ip}:8100", 
            f"https://{ip}:3000",
            f"https://{ip}:8100"
        ])
    
    # Add local IP addresses dynamically
    import socket
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        allowed_origins.extend([
            f"http://{local_ip}:3000",
            f"http://{local_ip}:8100", 
            f"https://{local_ip}:3000",
            f"https://{local_ip}:8100"
        ])
        logger.info(f"Detected local IP: {local_ip}")
    except Exception as e:
        logger.warning(f"Failed to detect local IP: {e}")
    
    # Add mobile-specific origins for better compatibility
    mobile_origins = [
        "capacitor://localhost",  # Capacitor apps
        "ionic://localhost",      # Ionic apps
        "http://localhost",       # Generic localhost
        "https://localhost",      # Generic localhost HTTPS
    ]
    allowed_origins.extend(mobile_origins)
    
    # Log all allowed origins for debugging
    logger.info(f"Configured CORS origins: {allowed_origins}")
    
    CORS(app, resources={
        r"/*": {
            "origins": allowed_origins,
            "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"],
            "allow_headers": [
                "Content-Type", 
                "Authorization", 
                "X-Requested-With",
                "Accept",
                "Origin",
                "Access-Control-Request-Method",
                "Access-Control-Request-Headers"
            ],
            "expose_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True,
            "max_age": 86400  # Cache preflight requests for 24 hours
        }
    })
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Register error handlers
    register_error_handlers(app)
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        return jsonify({
            'status': 'healthy',
            'service': 'ToneBridge Backend',
            'version': '1.0.0'
        })
    
    # Root endpoint
    @app.route('/')
    def root():
        return jsonify({
            'message': 'ToneBridge Backend API',
            'endpoints': {
                'health': '/health',
                'transcribe': '/api/transcribe',
                'emotion': '/api/emotion'
            }
        })
    
    logger.info("ToneBridge Backend initialized successfully")
    logger.info(f"CORS allowed origins: {allowed_origins}")
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(
        host=os.getenv('HOST', '0.0.0.0'),
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('FLASK_ENV') == 'development'
    ) 