"""
Test Kaggle-trained Model vs Old Model
So sánh chất lượng model mới (Kaggle, WER ~13.3%) với model cũ (WER = 100%)
"""

import torch
import numpy as np
import os
import json
import time
from pathlib import Path
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

try:
    from jiwer import wer as compute_wer, cer as compute_cer
except ImportError:
    print("⚠ jiwer not installed. Run: pip install jiwer")
    compute_wer = None
    compute_cer = None

try:
    from datasets import load_from_disk
except ImportError:
    print("⚠ datasets not installed. Run: pip install datasets")
    load_from_disk = None

try:
    import soundfile as sf
    from io import BytesIO
except ImportError:
    sf = None

import warnings
warnings.filterwarnings('ignore')


PROJECT_ROOT = Path(__file__).resolve().parent

# Model paths
NEW_MODEL_PATH = PROJECT_ROOT / "final_model"
OLD_MODEL_PATH = PROJECT_ROOT / "final_model_OLD_WER100"
DATASET_PATH = PROJECT_ROOT / "Data" / "train"


def load_model_and_processor(model_path):
    """Load model và processor từ path"""
    print(f"  Loading from: {model_path}")
    
    processor = Wav2Vec2Processor.from_pretrained(str(model_path))
    model = Wav2Vec2ForCTC.from_pretrained(str(model_path))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    return model, processor, device


def transcribe(model, processor, device, speech, sr=16000):
    """Transcribe audio array"""
    speech = np.asarray(speech, dtype=np.float32)
    if speech.ndim > 1:
        speech = np.mean(speech, axis=1)
    
    # Normalize
    speech = speech / (np.max(np.abs(speech)) + 1e-6)
    
    input_values = processor(
        speech,
        sampling_rate=sr,
        return_tensors="pt"
    ).input_values.to(device)
    
    with torch.no_grad():
        logits = model(input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0].strip().lower()
    
    return transcription


def extract_audio(audio_sample):
    """Extract audio from dataset sample"""
    if not audio_sample:
        return None, None
    
    # If audio has 'array' key (loaded dataset)
    if isinstance(audio_sample, dict):
        if 'array' in audio_sample:
            return np.array(audio_sample['array'], dtype=np.float32), audio_sample.get('sampling_rate', 16000)
        
        audio_bytes = audio_sample.get('bytes')
        if audio_bytes and sf:
            return sf.read(BytesIO(audio_bytes))
        
        audio_path = audio_sample.get('path')
        if audio_path and Path(audio_path).exists() and sf:
            return sf.read(audio_path)
    
    return None, None


def test_model(model_path, samples, model_name="Model"):
    """Test a model on given samples and return results"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    
    if not model_path.exists():
        print(f"  ❌ Model not found at {model_path}")
        return None
    
    try:
        model, processor, device = load_model_and_processor(model_path)
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        return None
    
    print(f"  ✓ Model loaded on {device}")
    print(f"  Testing {len(samples)} samples...\n")
    
    predictions = []
    references = []
    times = []
    
    for i, sample in enumerate(samples):
        audio = sample.get('audio')
        transcript = sample.get('transcription') or sample.get('text') or ''
        
        speech, sr = extract_audio(audio)
        if speech is None:
            print(f"  [{i+1}] ⚠ Could not read audio, skipping")
            continue
        
        start_time = time.perf_counter()
        prediction = transcribe(model, processor, device, speech, sr)
        elapsed = time.perf_counter() - start_time
        
        predictions.append(prediction)
        references.append(transcript.lower().strip())
        times.append(elapsed)
        
        # Show first 10 results in detail
        if i < 10:
            print(f"  [{i+1}] Ground truth: {transcript}")
            print(f"       Prediction:   {prediction}")
            print(f"       Time: {elapsed:.2f}s")
            print()
    
    # Calculate metrics
    results = {
        "model_name": model_name,
        "model_path": str(model_path),
        "num_samples": len(predictions),
        "avg_time": np.mean(times) if times else 0,
    }
    
    if compute_wer and predictions and references:
        wer_score = compute_wer(references, predictions)
        cer_score = compute_cer(references, predictions)
        results["wer"] = wer_score
        results["cer"] = cer_score
        
        print(f"  {'─'*40}")
        print(f"  📊 RESULTS for {model_name}:")
        print(f"  {'─'*40}")
        print(f"  WER: {wer_score:.4f} ({wer_score*100:.2f}%)")
        print(f"  CER: {cer_score:.4f} ({cer_score*100:.2f}%)")
        print(f"  Avg inference time: {np.mean(times):.2f}s")
        print(f"  Samples tested: {len(predictions)}")
    else:
        print(f"  ⚠ Could not compute WER (jiwer not available or no predictions)")
    
    # Cleanup GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results


def main():
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║  🎯 Kaggle Model vs Old Model - Comparison Test          ║
    ║  So sánh model mới (WER ~13.3%) với model cũ (WER=100%)  ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Load dataset
    print("📂 Loading test dataset...")
    
    if load_from_disk is None:
        print("❌ 'datasets' library not available")
        return
    
    if not DATASET_PATH.exists():
        print(f"❌ Dataset not found at {DATASET_PATH}")
        return
    
    try:
        dataset = load_from_disk(str(DATASET_PATH))
        print(f"✓ Dataset loaded: {dataset}")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    # Use test split if available, otherwise validation, otherwise train
    split_name = 'test' if 'test' in dataset else ('validation' if 'validation' in dataset else 'train')
    split_data = dataset[split_name]
    
    # Sample random items for testing
    import random
    random.seed(42)  # Reproducible
    num_samples = min(20, len(split_data))
    indices = random.sample(range(len(split_data)), num_samples)
    samples = [split_data[i] for i in indices]
    
    print(f"✓ Using {num_samples} samples from '{split_name}' split")
    
    # Test NEW model (Kaggle-trained)
    new_results = test_model(NEW_MODEL_PATH, samples, "🆕 Kaggle Model (checkpoint-15000)")
    
    # Test OLD model
    old_results = test_model(OLD_MODEL_PATH, samples, "🔴 Old Model (WER=100%)")
    
    # Comparison
    print(f"\n{'='*60}")
    print(f"📊 COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    if new_results and old_results:
        new_wer = new_results.get('wer', float('inf'))
        old_wer = old_results.get('wer', float('inf'))
        new_cer = new_results.get('cer', float('inf'))
        old_cer = old_results.get('cer', float('inf'))
        
        print(f"\n  {'Metric':<20} {'Old Model':>15} {'New Model':>15} {'Change':>15}")
        print(f"  {'─'*65}")
        print(f"  {'WER':<20} {old_wer*100:>14.2f}% {new_wer*100:>14.2f}% {(old_wer-new_wer)*100:>+14.2f}%")
        print(f"  {'CER':<20} {old_cer*100:>14.2f}% {new_cer*100:>14.2f}% {(old_cer-new_cer)*100:>+14.2f}%")
        print(f"  {'Avg Time':<20} {old_results['avg_time']:>14.2f}s {new_results['avg_time']:>14.2f}s")
        
        if new_wer < old_wer:
            improvement = (old_wer - new_wer) / old_wer * 100
            print(f"\n  🎉 Model mới TỐT HƠN {improvement:.1f}% so với model cũ!")
        elif new_wer == old_wer:
            print(f"\n  ⚠ Hai model có WER tương đương")
        else:
            print(f"\n  ❌ Model mới tệ hơn model cũ")
    
    elif new_results:
        new_wer = new_results.get('wer', 'N/A')
        print(f"\n  🆕 Kaggle Model WER: {new_wer*100:.2f}% (Old model could not be tested)")
        print(f"  📝 Old model had WER = 100% (from training report)")
        if isinstance(new_wer, float) and new_wer < 1.0:
            print(f"\n  🎉 Model mới TỐT HƠN NHIỀU! WER giảm từ 100% xuống {new_wer*100:.2f}%")
    
    print(f"\n{'='*60}")
    print("Done!")


if __name__ == "__main__":
    main()
