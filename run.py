#!/usr/bin/env python3
"""
ToneBridge Backend Runner
"""

import os
from app import app
from config import get_config

def main():
    """Main application entry point"""
    # Get configuration
    config = get_config()
    
    # Set up environment
    host = os.getenv('HOST', config.HOST)
    port = int(os.getenv('PORT', config.PORT))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Production settings
    if config.DEBUG:
        print("🚀 Starting ToneBridge Backend (Development)")
        print(f"📍 Server: http://{host}:{port}")
        print(f"🔧 Debug: {debug}")
    else:
        print("🚀 Starting ToneBridge Backend (Production)")
        print(f"📍 Server: https://{host}:{port}")
        print("🔒 Privacy: No data storage enabled")
        print("📱 All transcriptions stored in browser localStorage only")
    
    # Start the application
    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True
    )

if __name__ == '__main__':
    main() 