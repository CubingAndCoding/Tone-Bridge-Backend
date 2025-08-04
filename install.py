#!/usr/bin/env python3
"""
ToneBridge Backend Installation Script
"""

import subprocess
import sys
import os

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

def install_basic_dependencies():
    """Install basic dependencies without PyTorch"""
    print("📦 Installing basic dependencies...")
    
    # Read requirements and filter out PyTorch/transformers
    with open('requirements.txt', 'r') as f:
        lines = f.readlines()
    
    basic_requirements = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and 'torch' not in line and 'transformers' not in line:
            basic_requirements.append(line)
    
    # Create temporary requirements file
    with open('requirements_basic.txt', 'w') as f:
        f.write('\n'.join(basic_requirements))
    
    success = run_command("pip install -r requirements_basic.txt", "Installing basic dependencies")
    
    # Clean up
    if os.path.exists('requirements_basic.txt'):
        os.remove('requirements_basic.txt')
    
    return success

def install_pytorch():
    """Install PyTorch and transformers"""
    print("🧠 Installing PyTorch and transformers...")
    
    # Install PyTorch (CPU version for compatibility)
    pytorch_success = run_command(
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
        "Installing PyTorch (CPU version)"
    )
    
    if pytorch_success:
        transformers_success = run_command(
            "pip install transformers",
            "Installing transformers"
        )
        return transformers_success
    
    return False

def main():
    """Main installation function"""
    print("🚀 ToneBridge Backend Installation")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Ask user about PyTorch installation
    print("\n🤔 Installation Options:")
    print("1. Basic installation (faster, uses rule-based emotion detection)")
    print("2. Full installation with PyTorch (slower, uses AI models for better accuracy)")
    
    while True:
        choice = input("\nChoose installation type (1 or 2): ").strip()
        if choice in ['1', '2']:
            break
        print("Please enter 1 or 2")
    
    # Install basic dependencies
    if not install_basic_dependencies():
        print("❌ Basic installation failed. Please check your Python environment.")
        sys.exit(1)
    
    # Install PyTorch if requested
    if choice == '2':
        if not install_pytorch():
            print("⚠️  PyTorch installation failed. Continuing with basic installation.")
            print("   The app will use rule-based emotion detection instead of AI models.")
        else:
            print("✅ Full installation completed successfully!")
    
    # Create necessary directories
    directories = ['uploads', 'logs']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")
    
    # Set up environment file
    if not os.path.exists('.env'):
        if os.path.exists('env.example'):
            import shutil
            shutil.copy('env.example', '.env')
            print("📄 Created .env file from env.example")
        else:
            print("⚠️  No env.example file found. Please create .env manually.")
    
    print("\n🎉 Installation completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your configuration")
    print("2. Run: python app.py")
    print("3. Or run: flask run")
    print("4. Test the API: curl http://localhost:5000/health")
    
    if choice == '1':
        print("\n💡 Note: Using rule-based emotion detection.")
        print("   For better accuracy, you can install PyTorch later by running:")
        print("   pip install torch transformers")
    
    print("\n📚 For more information, see README.md")

if __name__ == '__main__':
    main() 