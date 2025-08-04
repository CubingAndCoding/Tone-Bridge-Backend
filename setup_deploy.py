#!/usr/bin/env python3
"""
Deployment setup script for ToneBridge Backend
Pre-downloads models to avoid timeout issues during deployment
"""

import os
import sys
from pathlib import Path

def setup_deployment():
    """Setup deployment environment"""
    print("🚀 Setting up ToneBridge Backend for deployment...")
    
    try:
        # Import required libraries
        print("📦 Importing required libraries...")
        import torch
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        # Pre-download emotion model
        print("🤖 Pre-downloading emotion detection model...")
        emotion_model = "SamLowe/roberta-base-go_emotions"
        
        # Download tokenizer
        print(f"📥 Downloading tokenizer: {emotion_model}")
        tokenizer = AutoTokenizer.from_pretrained(emotion_model)
        
        # Download model
        print(f"📥 Downloading model: {emotion_model}")
        model = AutoModelForSequenceClassification.from_pretrained(emotion_model)
        
        # Test the pipeline
        print("🧪 Testing emotion detection pipeline...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        
        # Test with sample text
        test_text = "I am very happy today!"
        result = pipeline(test_text)
        print(f"✅ Test successful! Emotion: {result[0]['label']}")
        
        print("🎉 Deployment setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = setup_deployment()
    sys.exit(0 if success else 1) 