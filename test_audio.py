#!/usr/bin/env python3
"""
Simple test script to debug audio data issues
"""

import base64
import json
import requests

def test_audio_endpoint():
    """Test the debug audio endpoint"""
    
    # Test with empty data
    print("Testing with empty data...")
    response = requests.post('http://localhost:5000/api/debug/audio', 
                           json={'audio': '', 'format': 'webm'})
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()
    
    # Test with invalid base64
    print("Testing with invalid base64...")
    response = requests.post('http://localhost:5000/api/debug/audio', 
                           json={'audio': 'invalid_base64!', 'format': 'webm'})
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()
    
    # Test with valid base64 but not audio
    print("Testing with valid base64 (not audio)...")
    test_data = base64.b64encode(b"this is not audio data").decode('utf-8')
    response = requests.post('http://localhost:5000/api/debug/audio', 
                           json={'audio': test_data, 'format': 'webm'})
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

if __name__ == "__main__":
    test_audio_endpoint() 