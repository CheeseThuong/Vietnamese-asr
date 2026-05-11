"""
Training script cho Wav2Vec2 ASR với BitNet quantization
"""
import os
import torch
import json
from typing import Optional
from dataclasses import dataclass
from typing import Dict, List, Union
from pathlib import Path

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import numpy as np
from src.data.preprocessing import load_and_prepare_datasets, load_and_prepare_hf_datasets, create_processor
import evaluate

# Load WER and CER metrics
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def get_latest_checkpoint(output_dir: str):
    """Return latest checkpoint path in output_dir, or None if not found."""
    if not os.path.isdir(output_dir):
        return None
    checkpoints = []
    for name in os.listdir(output_dir):
        if name.startswith("checkpoint-"):
            path = os.path.join(output_dir, name)
            if os.path.isdir(path):
                try:
                    step = int(name.replace("checkpoint-", ""))
                except ValueError:
                    step = -1
                checkpoints.append((step, path))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def get_hf_dataset_source() -> Optional[str]:
    """Return a configured Hugging Face dataset source if one is set."""
    for env_name in ("HF_DATASET_SOURCE", "HF_DATASET_URL", "HF_DATASET_ID"):
        value = os.getenv(env_name)
        if value:
            return value.strip()
    return None

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator để pad input và labels
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Union[int, None] = None
    max_length_labels: Union[int, None] = None
    pad_to_multiple_of: Union[int, None] = None
    pad_to_multiple_of_labels: Union[int, None] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad input features
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Pad labels - Dùng tokenizer trực tiếp thay vì as_target_processor()
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            max_length=self.max_length_labels,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # Replace padding with -100 to ignore loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

def apply_bitnet_quantization(model, use_cpu=False):
    """
    Áp dụng quantization để giảm kích thước model
    
    Args:
        model: Model cần quantize
        use_cpu: Nếu True, dùng PyTorch native quantization (tốt cho CPU)
                 Nếu False, thử dùng bitsandbytes (tốt cho GPU)
    """
    if use_cpu or not torch.cuda.is_available():
        # CPU: Dùng PyTorch dynamic quantization
        try:
            print("Applying PyTorch dynamic quantization (optimized for CPU)...")
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},  # Quantize Linear layers
                dtype=torch.qint8
            )
            print("✓ Applied PyTorch int8 quantization (CPU-optimized)")
            print("  - Model size reduced ~75%")
            print("  - Inference speed up ~2-3x on CPU")
            return quantized_model
        except Exception as e:
            print(f"⚠ PyTorch quantization failed: {e}")
            return model
    else:
        # GPU: Thử dùng bitsandbytes
        try:
            import bitsandbytes as bnb
            
            # Quantize model to 8-bit
            model = bnb.nn.Linear8bitLt(
                model,
                has_fp16_weights=False,
                threshold=6.0
            )
            print("✓ Applied bitsandbytes 8-bit quantization (GPU-optimized)")
            return model
        except ImportError:
            print("⚠ bitsandbytes not available, falling back to PyTorch quantization")
            return apply_bitnet_quantization(model, use_cpu=True)
        except Exception as e:
            print(f"⚠ BitNet quantization failed: {e}, falling back to PyTorch quantization")
            return apply_bitnet_quantization(model, use_cpu=True)

def create_model(vocab_size: int, processor: Wav2Vec2Processor, pretrained_model: str = None):
    """
    Tạo hoặc load pre-trained Wav2Vec2 model
    
    Args:
        vocab_size: Kích thước vocabulary
        processor: Wav2Vec2Processor instance (cần để lấy pad_token_id)
        pretrained_model: Tên model pre-trained (optional)
    """
    if pretrained_model:
        print(f"Loading pre-trained model: {pretrained_model}")
        model = Wav2Vec2ForCTC.from_pretrained(
            pretrained_model,
            attention_dropout=0.1,
            hidden_dropout=0.1,
            feat_proj_dropout=0.0,
            mask_time_prob=0.05,
            layerdrop=0.1,
            ctc_loss_reduction="mean",
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=vocab_size,
            ignore_mismatched_sizes=True
        )
    else:
        print("Creating model from scratch")
        from transformers import Wav2Vec2Config
        config = Wav2Vec2Config(
            vocab_size=vocab_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            pad_token_id=processor.tokenizer.pad_token_id
        )
        model = Wav2Vec2ForCTC(config)
    
    # Freeze feature encoder (chỉ fine-tune CTC head)
    model.freeze_feature_encoder()
    
    return model

def train_model(
    model,
    train_dataset,
    val_dataset,
    processor,
    output_dir: str,
    num_train_epochs: int = 30,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 3e-4,
    warmup_steps: int = 500,
    use_fp16: bool = True,
    resume_from_checkpoint: str = None
):
    """
    Train Wav2Vec2 model
    """
    # Define compute_metrics INSIDE train_model so it can access processor
    def compute_metrics(pred):
        """
        Compute WER and CER metrics
        """
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer, "cer": cer}
    
    # Data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    
    # Training arguments - Tự động điều chỉnh cho CPU/GPU
    is_gpu = torch.cuda.is_available()
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        group_by_length=False,  # Tắt vì dataset có 'input_values' thay vì 'input_ids'
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=learning_rate,
        weight_decay=0.005,
        num_train_epochs=num_train_epochs,
        warmup_steps=warmup_steps,
        fp16=use_fp16 and is_gpu,  # FP16 chỉ trên GPU
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        dataloader_num_workers=2 if is_gpu else 0,  # GPU cho phép workers, CPU không
        dataloader_pin_memory=is_gpu,  # Pin memory chỉ với GPU
        report_to=["tensorboard"],
        gradient_checkpointing=True,
        remove_unused_columns=False,  # CRITICAL: Dataset đã được preprocess, không remove columns
    )
    
    print(f"\n✓ Training config: {'GPU' if is_gpu else 'CPU'} mode")
    print(f"  - Batch size: {batch_size}")
    print(f"  - FP16: {training_args.fp16}")
    print(f"  - Workers: {training_args.dataloader_num_workers}")
    
    # Early stopping
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=5,
        early_stopping_threshold=0.001
    )
    
    # Trainer (không cần tokenizer parameter trong transformers >= 4.30)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping]
    )
    
    # Train
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    if resume_from_checkpoint:
        print(f"\n↩ Resuming from checkpoint: {resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save final model
    trainer.save_model(f"{output_dir}/final_model")
    processor.save_pretrained(f"{output_dir}/final_model")
    
    print(f"\n✓ Training completed! Model saved to: {output_dir}/final_model")
    
    return trainer

def main():
    """
    Main training pipeline
    """
    # Paths - từ root của project
    project_root = Path(__file__).parent.parent.parent
    hf_raw_dir = project_root / 'Data' / 'train'
    data_dir = project_root / 'processed_data_vivos'
    output_dir = project_root / 'models' / 'wav2vec2-vietnamese-asr'
    vocab_path = project_root / 'vocab.json'
    output_dir.mkdir(parents=True, exist_ok=True)

    remote_hf_source = get_hf_dataset_source()

    local_hf_dataset_ready = (hf_raw_dir / 'dataset_dict.json').exists()

    # If a remote HF dataset is configured, use it directly as well.
    if remote_hf_source:
        print(f"\nDetected remote Hugging Face dataset source: {remote_hf_source}")
        print("Will load it directly after the processor is ready.")
    
    # Configuration
    config = {
        'pretrained_model': 'nguyenvulebinh/wav2vec2-base-vietnamese-250h',  # Hoặc None để train from scratch
        'num_train_epochs': 30,
        'batch_size': 8,
        'gradient_accumulation_steps': 2,
        'learning_rate': 3e-4,
        'use_fp16': True,
        'apply_quantization': True
    }
    
    # Save config
    with open(output_dir / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Resume training (auto-detect latest checkpoint)
    resume_env = os.getenv("RESUME_FROM_CHECKPOINT", "auto").strip()
    if resume_env.lower() == "none":
        resume_from_checkpoint = None
    elif resume_env.lower() == "auto":
        resume_from_checkpoint = get_latest_checkpoint(str(output_dir))
    else:
        resume_from_checkpoint = resume_env
    
    # Load processor
    global processor
    print("Loading processor...")
    if config['pretrained_model']:
        processor = Wav2Vec2Processor.from_pretrained(config['pretrained_model'])
    else:
        processor = create_processor(str(vocab_path))

    # Resolve dataset source after the processor exists.
    train_dataset = val_dataset = test_dataset = None
    token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if remote_hf_source:
        print(f"\nLoading remote Hugging Face dataset directly: {remote_hf_source}")
        train_dataset, val_dataset, test_dataset = load_and_prepare_hf_datasets(
            remote_hf_source,
            processor,
            token=token,
        )
    elif local_hf_dataset_ready:
        print(f"\nLoading local Hugging Face DatasetDict directly: {hf_raw_dir}")
        train_dataset, val_dataset, test_dataset = load_and_prepare_hf_datasets(
            str(hf_raw_dir),
            processor,
            token=token,
        )
    
    if train_dataset is None or val_dataset is None or test_dataset is None:
        # Fallback to legacy JSONL pipeline.
        required_files = ['train.jsonl', 'validation.jsonl', 'test.jsonl']
        missing_files = [f for f in required_files if not (data_dir / f).exists()]

        if missing_files:
            print(f"\n❌ Lỗi: Không tìm thấy dataset files: {missing_files}")
            print(f"📁 Đường dẫn tìm kiếm: {data_dir}")
            print(f"\n💡 Giải pháp:")
            print(f"   1. Chạy lệnh: python prepare_vivos.py")
            print(f"   2. Hoặc: python prepare_full_dataset.py (để gộp VIVOS + VinBigData)")
            print(f"   3. Sau đó chạy lại: python train.py")
            return

        print("\nLoading datasets from JSONL fallback...")
        train_dataset, val_dataset, test_dataset = load_and_prepare_datasets(
            str(data_dir / 'train.jsonl'),
            str(data_dir / 'validation.jsonl'),
            str(data_dir / 'test.jsonl'),
            processor
        )
    
    # Create model
    print("\nCreating model...")
    vocab_size = len(processor.tokenizer)
    model = create_model(vocab_size, config['pretrained_model'])
    
    # QUAN TRỌNG: KHÔNG quantize khi training!
    # Quantization chỉ dùng cho inference sau khi train xong
    # Nếu quantize, model không thể backprop gradient (RuntimeError: leaf Variable requires grad)
    if config['apply_quantization']:
        print("\n⚠ Warning: Quantization is DISABLED during training.")
        print("  Quantization should only be applied AFTER training for inference.")
        print("  Use src/utils/optimization.py after training completes.")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Train
    trainer = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        processor=processor,
        output_dir=str(output_dir),
        num_train_epochs=config['num_train_epochs'],
        batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],
        use_fp16=config['use_fp16'],
        resume_from_checkpoint=resume_from_checkpoint
    )
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("Evaluating on test set...")
    print("="*50)
    
    test_results = trainer.evaluate(test_dataset)
    print(f"\nTest Results:")
    print(f"  WER: {test_results['eval_wer']:.4f}")
    print(f"  CER: {test_results['eval_cer']:.4f}")
    
    # Save test results
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)

if __name__ == "__main__":
    main()
