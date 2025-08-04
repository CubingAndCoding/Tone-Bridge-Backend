#!/usr/bin/env python3
"""
ToneBridge Backend HTTPS Runner Script
"""

import os
import sys
import ssl
from app import create_app

def main():
    """Main entry point for the HTTPS application"""
    try:
        # Create the Flask app
        app = create_app()
        
        # Get configuration
        host = os.getenv('HOST', '0.0.0.0')
        port = int(os.getenv('PORT', 5000))
        debug = os.getenv('FLASK_ENV') == 'development'
        
        # SSL certificate paths
        cert_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'certs', 'localhost.pem')
        key_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'certs', 'localhost-key.pem')
        
        # Check if certificates exist
        if not os.path.exists(cert_path):
            print("❌ SSL certificate not found:", cert_path)
            print("Please run fix-ssl-certificates.bat first.")
            sys.exit(1)
            
        if not os.path.exists(key_path):
            print("❌ SSL key not found:", key_path)
            print("Please run fix-ssl-certificates.bat first.")
            sys.exit(1)
        
        print(f"🚀 Starting ToneBridge Backend with HTTPS...")
        print(f"📍 Server: {host}:{port}")
        print(f"🔧 Debug Mode: {debug}")
        print(f"🔐 SSL Certificate: {cert_path}")
        print(f"🔑 SSL Key: {key_path}")
        print(f"🌐 Health Check: https://{host}:{port}/health")
        print(f"📚 API Docs: https://{host}:{port}/")
        print("=" * 50)
        
        # Create SSL context
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert_path, key_path)
        
        # Run the application with HTTPS
        app.run(
            host=host,
            port=port,
            debug=debug,
            ssl_context=context
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down ToneBridge Backend...")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting ToneBridge Backend: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 