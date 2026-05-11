"""
Test script for Vietnamese ASR model
Nhận dạng giọng nói tiếng Việt với model đã train
"""

import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_model(model_path="results/final_model"):
    """Load model và processor"""
    print("=" * 60)
    print("Loading Model...")
    print("=" * 60)
    
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"✓ Model loaded from: {model_path}")
    print(f"✓ Device: {device}")
    print("=" * 60)
    
    return model, processor, device

def transcribe_audio(audio_path, model, processor, device):
    """Nhận dạng giọng nói từ file audio"""
    print(f"\nProcessing: {audio_path}")
    
    # Load audio
    speech, sr = librosa.load(audio_path, sr=16000)
    
    # Process audio
    input_values = processor(
        speech, 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_values.to(device)
    
    # Inference
    with torch.no_grad():
        logits = model(input_values).logits
    
    # Decode
    pred_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(pred_ids)[0]
    
    return transcription

def test_single_file(audio_path, model_path="results/final_model"):
    """Test với một file audio"""
    # Load model
    model, processor, device = load_model(model_path)
    
    # Transcribe
    transcription = transcribe_audio(audio_path, model, processor, device)
    
    print("\n" + "=" * 60)
    print("RESULT:")
    print("=" * 60)
    print(f"File: {audio_path}")
    print(f"Transcription: {transcription}")
    print("=" * 60)
    
    return transcription

def test_batch_files(audio_dir, model_path="results/final_model"):
    """Test với nhiều files trong thư mục"""
    audio_dir = Path(audio_dir)
    audio_files = list(audio_dir.glob("*.wav"))
    
    if not audio_files:
        print(f"No .wav files found in {audio_dir}")
        return
    
    print(f"\nFound {len(audio_files)} audio files")
    
    # Load model once
    model, processor, device = load_model(model_path)
    
    results = []
    for audio_file in audio_files:
        transcription = transcribe_audio(str(audio_file), model, processor, device)
        results.append({
            'file': audio_file.name,
            'transcription': transcription
        })
    
    # Print results
    print("\n" + "=" * 60)
    print("BATCH RESULTS:")
    print("=" * 60)
    for result in results:
        print(f"\n{result['file']}:")
        print(f"  → {result['transcription']}")
    print("=" * 60)
    
    return results

def test_with_ground_truth(test_jsonl="processed_data_vivos/test.jsonl", 
                           model_path="results/final_model",
                           num_samples=10):
    """Test với ground truth để tính WER/CER"""
    import json
    from jiwer import wer, cer
    
    # Load model
    model, processor, device = load_model(model_path)
    
    # Read test data
    with open(test_jsonl, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
    
    # Sample random
    import random
    samples = random.sample(test_data, min(num_samples, len(test_data)))
    
    predictions = []
    references = []
    
    print("\n" + "=" * 60)
    print("Testing with Ground Truth:")
    print("=" * 60)
    
    for i, sample in enumerate(samples, 1):
        audio_path = sample['audio_path']
        reference = sample['transcript']
        
        print(f"\n[{i}/{len(samples)}] {audio_path}")
        
        # Transcribe
        prediction = transcribe_audio(audio_path, model, processor, device)
        
        predictions.append(prediction)
        references.append(reference)
        
        print(f"  Ground truth: {reference}")
        print(f"  Prediction:   {prediction}")
    
    # Calculate metrics
    wer_score = wer(references, predictions)
    cer_score = cer(references, predictions)
    
    print("\n" + "=" * 60)
    print("METRICS:")
    print("=" * 60)
    print(f"Word Error Rate (WER): {wer_score:.4f} ({wer_score*100:.2f}%)")
    print(f"Character Error Rate (CER): {cer_score:.4f} ({cer_score*100:.2f}%)")
    print("=" * 60)
    
    return wer_score, cer_score

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║   Vietnamese ASR Model - Test Script                  ║
    ║   Nhận dạng giọng nói tiếng Việt                      ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    # Option 1: Test single file
    print("\n[Option 1] Test một file audio:")
    print("  Example: test_single_file('path/to/audio.wav')")
    
    # Option 2: Test batch
    print("\n[Option 2] Test nhiều files:")
    print("  Example: test_batch_files('path/to/audio/folder')")
    
    # Option 3: Test with ground truth
    print("\n[Option 3] Test với ground truth (tính WER/CER):")
    print("  Example: test_with_ground_truth('processed_data_vivos/test.jsonl', num_samples=10)")
    
    print("\n" + "=" * 60)
    print("Quick Test - Chọn một file test ngẫu nhiên:")
    print("=" * 60)
    
    # Try to find a test file
    test_paths = [
        "Data/vivos/vivos/test/waves",
        "processed_data_vivos"
    ]
    
    test_file = None
    for path in test_paths:
        test_path = Path(path)
        if test_path.exists():
            wav_files = list(test_path.rglob("*.wav"))
            if wav_files:
                test_file = wav_files[0]
                break
    
    if test_file:
        print(f"\nTìm thấy file test: {test_file}")
        print("Running test...")
        test_single_file(str(test_file))
    else:
        print("\nKhông tìm thấy file audio để test.")
        print("Vui lòng sử dụng:")
        print("  test_single_file('path/to/your/audio.wav')")
    
    print("\n" + "=" * 60)
    print("Để test thủ công, import và gọi functions:")
    print("  from test_model import test_single_file")
    print("  test_single_file('your_audio.wav')")
    print("=" * 60)
