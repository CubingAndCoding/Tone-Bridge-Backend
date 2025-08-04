#!/usr/bin/env python3
"""
ToneBridge Backend Setup Script
"""

import os
import sys
import subprocess
import shutil

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ['uploads', 'logs']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")

def setup_environment():
    """Set up environment file"""
    env_file = '.env'
    env_example = 'env.example'
    
    if not os.path.exists(env_file):
        if os.path.exists(env_example):
            shutil.copy(env_example, env_file)
            print(f"📄 Created .env file from {env_example}")
        else:
            print("⚠️  No env.example file found. Please create .env manually.")
    else:
        print("📄 .env file already exists")

def check_python_version():
    """Check Python version"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_dependencies():
    """Install Python dependencies"""
    return run_command("pip install -r requirements.txt", "Installing dependencies")

def main():
    """Main setup function"""
    print("🚀 ToneBridge Backend Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Setup environment
    setup_environment()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed. Please check the errors above.")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your configuration")
    print("2. Run: python run.py")
    print("3. Or run: python app.py")
    print("4. Test the API: curl http://localhost:5000/health")
    print("\n📚 For more information, see README.md")

if __name__ == '__main__':
    main() 