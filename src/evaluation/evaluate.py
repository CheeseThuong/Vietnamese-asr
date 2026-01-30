"""
Evaluation script để đánh giá model với WER và CER metrics
"""
import torch
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_from_disk
import evaluate
from src.data.preprocessing import load_and_prepare_datasets
from src.training.language_model import LanguageModelDecoder

# Load metrics
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

class ASREvaluator:
    """
    Evaluator class để đánh giá ASR model
    """
    def __init__(
        self,
        model_path: str,
        processor_path: str = None,
        lm_path: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            model_path: Path to trained Wav2Vec2 model
            processor_path: Path to processor (if None, use model_path)
            lm_path: Path to KenLM binary file (optional)
            device: Device to run inference
        """
        print(f"Loading model from: {model_path}")
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device)
        self.model.eval()
        
        processor_path = processor_path or model_path
        print(f"Loading processor from: {processor_path}")
        self.processor = Wav2Vec2Processor.from_pretrained(processor_path)
        
        self.device = device
        
        # Initialize decoder with LM
        self.decoder = LanguageModelDecoder(
            self.processor,
            kenlm_model_path=lm_path
        )
    
    def transcribe(self, audio_input, sampling_rate: int = 16000):
        """
        Transcribe audio input
        
        Args:
            audio_input: Audio array or path to audio file
            sampling_rate: Sampling rate of audio
        """
        import torchaudio
        
        # Load audio if path is provided
        if isinstance(audio_input, str):
            audio_array, sr = torchaudio.load(audio_input)
            if sr != sampling_rate:
                resampler = torchaudio.transforms.Resample(sr, sampling_rate)
                audio_array = resampler(audio_array)
            audio_array = audio_array.squeeze().numpy()
        else:
            audio_array = audio_input
        
        # Process audio
        input_values = self.processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        ).input_values.to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(input_values).logits
        
        # Decode
        if self.decoder.decoder is not None:
            # Use LM decoder
            transcription = self.decoder.decode(logits, beam_width=100)
            if isinstance(transcription, list):
                transcription = transcription[0]
        else:
            # Greedy decoding
            pred_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(pred_ids)[0]
        
        return transcription
    
    def evaluate_dataset(
        self,
        test_dataset,
        batch_size: int = 8,
        save_predictions: bool = True,
        output_file: str = None
    ) -> Dict:
        """
        Evaluate model trên dataset
        
        Returns:
            Dictionary chứa WER, CER và các metrics khác
        """
        print("Evaluating model...")
        
        predictions = []
        references = []
        errors = []
        
        # Process dataset
        for i in tqdm(range(0, len(test_dataset), batch_size)):
            batch = test_dataset[i:i+batch_size]
            
            # Get input values and labels
            input_values = torch.tensor(batch['input_values']).to(self.device)
            labels = batch['labels']
            
            # Inference
            with torch.no_grad():
                logits = self.model(input_values).logits
            
            # Decode predictions
            if self.decoder.decoder is not None:
                batch_predictions = []
                for logit in logits:
                    pred = self.decoder.decode(logit.unsqueeze(0), beam_width=100)
                    if isinstance(pred, list):
                        pred = pred[0]
                    batch_predictions.append(pred)
            else:
                pred_ids = torch.argmax(logits, dim=-1)
                batch_predictions = self.processor.batch_decode(pred_ids)
            
            # Decode references
            labels_tensor = torch.tensor(labels)
            labels_tensor[labels_tensor == -100] = self.processor.tokenizer.pad_token_id
            batch_references = self.processor.batch_decode(labels_tensor, group_tokens=False)
            
            predictions.extend(batch_predictions)
            references.extend(batch_references)
            
            # Track errors for analysis
            for pred, ref in zip(batch_predictions, batch_references):
                if pred.lower() != ref.lower():
                    errors.append({
                        'prediction': pred,
                        'reference': ref
                    })
        
        # Compute metrics
        wer = wer_metric.compute(predictions=predictions, references=references)
        cer = cer_metric.compute(predictions=predictions, references=references)
        
        results = {
            'wer': wer,
            'cer': cer,
            'num_samples': len(predictions),
            'num_errors': len(errors),
            'accuracy': 1 - (len(errors) / len(predictions))
        }
        
        # Save predictions
        if save_predictions and output_file:
            output_data = {
                'results': results,
                'predictions': [
                    {'prediction': pred, 'reference': ref}
                    for pred, ref in zip(predictions, references)
                ],
                'errors': errors[:100]  # Save first 100 errors for analysis
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n✓ Predictions saved to: {output_file}")
        
        return results
    
    def analyze_errors(self, errors: List[Dict], top_n: int = 20):
        """
        Phân tích các lỗi thường gặp
        """
        from collections import Counter
        
        print("\n" + "="*50)
        print(f"TOP {top_n} COMMON ERRORS")
        print("="*50)
        
        # Count error patterns
        error_patterns = []
        for error in errors:
            pred_words = error['prediction'].split()
            ref_words = error['reference'].split()
            
            # Simple word-level alignment
            max_len = max(len(pred_words), len(ref_words))
            for i in range(max_len):
                pred_word = pred_words[i] if i < len(pred_words) else '<empty>'
                ref_word = ref_words[i] if i < len(ref_words) else '<empty>'
                
                if pred_word != ref_word:
                    error_patterns.append(f"{ref_word} -> {pred_word}")
        
        # Count and display
        error_counts = Counter(error_patterns)
        for pattern, count in error_counts.most_common(top_n):
            print(f"{count:3d}x  {pattern}")

def main():
    """
    Main evaluation pipeline
    """
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'processed_data'
    model_dir = base_dir / 'models' / 'wav2vec2-vietnamese-asr' / 'final_model'
    lm_dir = base_dir / 'language_models'
    results_dir = base_dir / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # Check if model exists
    if not model_dir.exists():
        print(f"✗ Model not found at: {model_dir}")
        print("Please train the model first using train_wav2vec2.py")
        return
    
    # Load processor
    processor = Wav2Vec2Processor.from_pretrained(str(model_dir))
    
    # Load test dataset
    print("Loading test dataset...")
    from data_preprocessing import VietnameseASRDataset
    
    test_dataset = VietnameseASRDataset(str(data_dir / 'test.jsonl')).to_hf_dataset()
    test_dataset = test_dataset.map(
        lambda batch: {
            'input_values': processor(batch['audio']['array'], sampling_rate=16000).input_values[0],
            'labels': processor(batch['transcript']).input_ids
        },
        remove_columns=test_dataset.column_names
    )
    
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # Evaluate without LM
    print("\n" + "="*50)
    print("EVALUATION WITHOUT LANGUAGE MODEL")
    print("="*50)
    
    evaluator_no_lm = ASREvaluator(
        model_path=str(model_dir),
        lm_path=None
    )
    
    results_no_lm = evaluator_no_lm.evaluate_dataset(
        test_dataset,
        batch_size=8,
        save_predictions=True,
        output_file=str(results_dir / 'predictions_no_lm.json')
    )
    
    print(f"\nResults (No LM):")
    print(f"  WER: {results_no_lm['wer']:.4f} ({results_no_lm['wer']*100:.2f}%)")
    print(f"  CER: {results_no_lm['cer']:.4f} ({results_no_lm['cer']*100:.2f}%)")
    print(f"  Accuracy: {results_no_lm['accuracy']:.4f} ({results_no_lm['accuracy']*100:.2f}%)")
    
    # Evaluate with LM (if available)
    lm_file = lm_dir / 'vietnamese_5gram.bin'
    if lm_file.exists():
        print("\n" + "="*50)
        print("EVALUATION WITH LANGUAGE MODEL")
        print("="*50)
        
        evaluator_with_lm = ASREvaluator(
            model_path=str(model_dir),
            lm_path=str(lm_file)
        )
        
        results_with_lm = evaluator_with_lm.evaluate_dataset(
            test_dataset,
            batch_size=8,
            save_predictions=True,
            output_file=str(results_dir / 'predictions_with_lm.json')
        )
        
        print(f"\nResults (With LM):")
        print(f"  WER: {results_with_lm['wer']:.4f} ({results_with_lm['wer']*100:.2f}%)")
        print(f"  CER: {results_with_lm['cer']:.4f} ({results_with_lm['cer']*100:.2f}%)")
        print(f"  Accuracy: {results_with_lm['accuracy']:.4f} ({results_with_lm['accuracy']*100:.2f}%)")
        
        # Compare results
        print("\n" + "="*50)
        print("IMPROVEMENT WITH LANGUAGE MODEL")
        print("="*50)
        wer_improvement = (results_no_lm['wer'] - results_with_lm['wer']) / results_no_lm['wer'] * 100
        cer_improvement = (results_no_lm['cer'] - results_with_lm['cer']) / results_no_lm['cer'] * 100
        print(f"  WER improvement: {wer_improvement:+.2f}%")
        print(f"  CER improvement: {cer_improvement:+.2f}%")
    else:
        print(f"\n⚠ Language model not found at: {lm_file}")
        print("Run language_model.py to build the language model")
    
    # Save final results
    final_results = {
        'no_lm': results_no_lm,
        'with_lm': results_with_lm if lm_file.exists() else None
    }
    
    with open(results_dir / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n✓ Final results saved to: {results_dir / 'final_results.json'}")

if __name__ == "__main__":
    main()
