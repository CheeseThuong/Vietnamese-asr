"""
Script demo để test nhanh model inference
"""
from pathlib import Path
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from src.training.language_model import LanguageModelDecoder

def transcribe_audio_file(
    audio_path: str,
    model_path: str,
    lm_path: str = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Transcribe một file audio
    
    Args:
        audio_path: Đường dẫn tới file audio
        model_path: Đường dẫn tới model
        lm_path: Đường dẫn tới language model (optional)
        device: Device để chạy inference
    
    Returns:
        Transcript text
    """
    print(f"Loading model from: {model_path}")
    model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device)
    model.eval()
    
    print(f"Loading processor from: {model_path}")
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    
    # Initialize decoder
    decoder = LanguageModelDecoder(processor, kenlm_model_path=lm_path)
    
    # Load audio
    print(f"\nLoading audio: {audio_path}")
    audio_array, sampling_rate = torchaudio.load(audio_path)
    
    # Resample if needed
    if sampling_rate != 16000:
        print(f"Resampling from {sampling_rate}Hz to 16000Hz")
        resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
        audio_array = resampler(audio_array)
    
    # Convert to mono
    if audio_array.shape[0] > 1:
        audio_array = torch.mean(audio_array, dim=0, keepdim=True)
    
    audio_array = audio_array.squeeze().numpy()
    duration = len(audio_array) / 16000
    print(f"Audio duration: {duration:.2f}s")
    
    # Process
    print("\nTranscribing...")
    input_values = processor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_values.to(device)
    
    # Inference
    with torch.no_grad():
        logits = model(input_values).logits
    
    # Decode
    if decoder.decoder is not None:
        print("Using Language Model decoder")
        transcription = decoder.decode(logits, beam_width=100)
        if isinstance(transcription, list):
            transcription = transcription[0]
    else:
        print("Using greedy decoder")
        pred_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(pred_ids)[0]
    
    return transcription

def main():
    """
    Demo usage
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Transcribe audio file với Wav2Vec2")
    parser.add_argument("audio_file", type=str, help="Path to audio file")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model directory"
    )
    parser.add_argument(
        "--lm",
        type=str,
        default=None,
        help="Path to language model binary"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # Default paths
    base_dir = Path(__file__).parent
    if args.model is None:
        args.model = str(base_dir / 'models' / 'wav2vec2-vietnamese-asr' / 'final_model')
    if args.lm is None:
        lm_path = base_dir / 'language_models' / 'vietnamese_5gram.bin'
        args.lm = str(lm_path) if lm_path.exists() else None
    
    # Check model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        print("Please train the model first using train_wav2vec2.py")
        return
    
    # Check audio file exists
    if not Path(args.audio_file).exists():
        print(f"Error: Audio file not found: {args.audio_file}")
        return
    
    print("="*60)
    print("Vietnamese Speech Recognition Demo")
    print("="*60)
    
    # Transcribe
    transcription = transcribe_audio_file(
        args.audio_file,
        args.model,
        args.lm,
        args.device
    )
    
    # Print result
    print("\n" + "="*60)
    print("TRANSCRIPTION RESULT")
    print("="*60)
    print(transcription)
    print("="*60)

if __name__ == "__main__":
    main()
