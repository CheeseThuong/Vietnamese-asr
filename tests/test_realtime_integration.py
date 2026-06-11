"""
Integration test for real-time speech recognition routing.
Loads a local WAV file, converts it, and runs transcribe_realtime_array.
"""
import os
import sys
import numpy as np
import librosa

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app.app as app_module
from app.app import transcribe_realtime_array, transcribe_audio_array, REALTIME_ASR_ENGINE

def test_realtime_routing():
    print("=" * 60)
    print("INTEGRATION TEST: REAL-TIME ROUTING & ENGINE TEST")
    print("=" * 60)
    print(f"Current Configured Real-time Engine (from .env): {REALTIME_ASR_ENGINE}")
    
    # Load audio
    wav_path = r"Data\Data\vlsp2020_train_p1\database_sa1_Jan08_Mar19_cleaned_utt_0000000005-1.wav"
    if not os.path.exists(wav_path):
        print(f"Error: WAV file {wav_path} not found.")
        sys.exit(1)
        
    print(f"Loading audio from {wav_path}...")
    speech, sr = librosa.load(wav_path, sr=16000)
    print(f"Loaded audio: length={len(speech)}, sample_rate={sr}, duration={len(speech)/sr:.2f}s")
    
    # Read ground truth
    txt_path = wav_path.replace(".wav", ".txt")
    ground_truth = ""
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            ground_truth = f.read().strip()
        print(f"Ground Truth: {ground_truth}")
    else:
        print("Warning: Ground truth text file not found.")
        
    print("-" * 60)
    print("Running transcribe_realtime_array...")
    
    # Run real-time transcription through the local-only realtime route.
    rt_res = transcribe_realtime_array(speech, sampling_rate=sr)
    
    print("\n[REALTIME ASR RESULTS]")
    print(f"Success: {rt_res.get('success')}")
    if rt_res.get('success'):
        print(f"Engine output: {rt_res.get('transcription')}")
        print(f"Original output: {rt_res.get('original_transcription')}")
        print(f"Metrics: {rt_res.get('asr_metrics')}")
    else:
        print(f"Error: {rt_res.get('error')}")
        
    print("-" * 60)
    print("Loading local Wav2Vec2 model...")
    loaded = app_module.load_model()
    if loaded:
        app_module.MODEL_LOAD_STATUS = "ready"
    else:
        app_module.MODEL_LOAD_STATUS = "error"
        
    print("Running local Wav2Vec2 (transcribe_audio_array)...")
    local_res = transcribe_audio_array(speech, sampling_rate=sr)
    
    print("\n[LOCAL WAV2VEC2 RESULTS]")
    print(f"Success: {local_res.get('success')}")
    if local_res.get('success'):
        print(f"Output: {local_res.get('transcription')}")
        print(f"Metrics: {local_res.get('asr_metrics')}")
    else:
        print(f"Error: {local_res.get('error')}")
        
    print("=" * 60)
    if rt_res.get('success') and local_res.get('success'):
        print("TEST COMPLETED SUCCESSFULLY!")
    else:
        print("TEST FAILED OR COMPLETED WITH ERRORS.")
        sys.exit(1)

if __name__ == "__main__":
    test_realtime_routing()
