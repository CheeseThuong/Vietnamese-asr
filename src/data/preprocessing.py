"""
Data preprocessing pipeline cho Wav2Vec2 ASR
"""
import os
import torch
import torchaudio
from io import BytesIO
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
import json
from pathlib import Path
from transformers import Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
import re
from typing import Dict, List
import soundfile as sf
import numpy as np


def normalize_hf_dataset_source(source):
    """Convert a Hugging Face dataset URL to a dataset repo id."""
    if not source:
        return None

    source = source.strip()
    if not source:
        return None

    match = re.search(r"huggingface\.co/datasets/([^/?#]+/[^/?#]+)", source)
    if match:
        return match.group(1)

    return source

class VietnameseASRDataset:
    """
    Dataset class để load và preprocess dữ liệu ASR tiếng Việt
    """
    def __init__(self, jsonl_file: str, processor: Wav2Vec2Processor = None):
        self.data = self.load_data(jsonl_file)
        self.processor = processor
        
    def load_data(self, jsonl_file: str) -> List[Dict]:
        """Load dữ liệu từ JSONL file"""
        data = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def normalize_text(self, text: str) -> str:
        return normalize_text(text)
    
    def create_vocab(self, texts: List[str]) -> Dict[str, int]:
        """
        Tạo vocabulary từ dữ liệu
        """
        vocab = set()
        for text in texts:
            text = self.normalize_text(text)
            vocab.update(list(text))
        
        # Thêm các token đặc biệt
        vocab_dict = {char: idx for idx, char in enumerate(sorted(vocab))}
        vocab_dict['[UNK]'] = len(vocab_dict)
        vocab_dict['[PAD]'] = len(vocab_dict)
        
        return vocab_dict
    
    def to_hf_dataset(self) -> Dataset:
        """
        Chuyển đổi sang HuggingFace Dataset format
        """
        dataset_dict = {
            'audio_path': [item['audio_path'] for item in self.data],
            'transcript': [self.normalize_text(item['transcript']) for item in self.data],
            'speaker_id': [item['speaker_id'] for item in self.data],
            'dataset': [item['dataset'] for item in self.data]
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        # Không dùng Audio feature - sẽ load audio thủ công trong prepare_dataset
        
        return dataset

def prepare_dataset(batch, processor):
    """
    Hàm preprocessing cho batch data
    """
    audio_path = batch["audio_path"]
    
    try:
        # Check file exists BEFORE attempting to load
        if not os.path.exists(audio_path):
            abs_path = os.path.abspath(audio_path)
            print(f"❌ File not found: {audio_path}")
            print(f"   Working dir: {os.getcwd()}")
            print(f"   Absolute path: {abs_path}")
            print(f"   Exists: {os.path.exists(abs_path)}")
            return None  # Return None to filter out this sample
        
        # Load audio file
        speech_array, sampling_rate = sf.read(audio_path)
        
        # Convert to mono if stereo
        if len(speech_array.shape) > 1:
            speech_array = speech_array.mean(axis=1)
        
        # Resample if needed
        if sampling_rate != 16000:
            speech_tensor = torch.FloatTensor(speech_array)
            resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
            speech_array = resampler(speech_tensor).numpy()
        
        # Convert to float32 if needed
        if speech_array.dtype != np.float32:
            speech_array = speech_array.astype(np.float32)
        
        # Process audio to input values
        input_values = processor(
            speech_array, 
            sampling_rate=16000
        ).input_values[0]
        
        # Encode target text
        labels = processor.tokenizer(
            normalize_text(batch["transcript"]),
            padding=False,
            truncation=True
        ).input_ids
        
        # Return ONLY required columns
        return {
            "input_values": input_values,
            "labels": labels
        }
        
    except Exception as e:
        print(f"❌ Error loading audio {audio_path}: {e}")
        return None  # Return None to filter out this sample


def normalize_text(text: str) -> str:
    """Chuẩn hóa văn bản tiếng Việt."""
    text = (text or "").lower()
    text = re.sub(r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', text)
    text = re.sub(r'<unk>|<s>|</s>|<pad>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_hf_dataset_source(dataset_source, data_dir="data", token=None):
    """Load a local saved DatasetDict or a remote Hugging Face dataset repo."""
    dataset_path = Path(dataset_source)
    if dataset_path.exists():
        loaded_dataset = load_from_disk(str(dataset_path))
        if isinstance(loaded_dataset, DatasetDict):
            return loaded_dataset
        return DatasetDict({"train": loaded_dataset})

    repo_id = normalize_hf_dataset_source(dataset_source)
    if not repo_id:
        raise ValueError("dataset_source is empty")

    load_kwargs = {"data_dir": data_dir}
    if token:
        load_kwargs["token"] = token

    print(f"Loading Hugging Face dataset source: {repo_id}...")
    loaded_dataset = load_dataset(repo_id, **load_kwargs)
    if isinstance(loaded_dataset, DatasetDict):
        return loaded_dataset
    return DatasetDict({"train": loaded_dataset})


def _load_audio_from_sample(audio_sample):
    """Load audio samples from bytes or a local file path."""
    if not audio_sample:
        return None, None

    audio_bytes = audio_sample.get("bytes")
    audio_path = audio_sample.get("path")

    if audio_bytes:
        return sf.read(BytesIO(audio_bytes))

    if audio_path and Path(audio_path).exists():
        return sf.read(audio_path)

    return None, None


def prepare_hf_audio_sample(batch, processor):
    """Convert a Hugging Face audio row into input_values + labels."""
    try:
        audio = batch.get("audio") or {}
        transcript = normalize_text(batch.get("transcription") or batch.get("text") or "")
        if not transcript:
            return None

        speech_array, sampling_rate = _load_audio_from_sample(audio)
        if speech_array is None or sampling_rate is None:
            return None

        if len(speech_array.shape) > 1:
            speech_array = speech_array.mean(axis=1)

        if sampling_rate != 16000:
            speech_tensor = torch.FloatTensor(speech_array)
            resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
            speech_array = resampler(speech_tensor).numpy()

        if speech_array.dtype != np.float32:
            speech_array = speech_array.astype(np.float32)

        input_values = processor(
            speech_array,
            sampling_rate=16000
        ).input_values[0]

        labels = processor.tokenizer(
            transcript,
            padding=False,
            truncation=True
        ).input_ids

        return {
            "input_values": input_values,
            "labels": labels
        }
    except Exception as e:
        print(f"❌ Error processing HF audio sample: {e}")
        return None


def _prepare_dataset_split(dataset_split, processor, split_name: str):
    """Map and filter one Dataset split."""
    original_length = len(dataset_split)
    prepared_split = dataset_split.map(
        lambda batch: prepare_hf_audio_sample(batch, processor),
        remove_columns=dataset_split.column_names
    )
    prepared_split = prepared_split.filter(lambda x: x.get("input_values") is not None)
    skipped = original_length - len(prepared_split)
    print(f"{split_name.capitalize()}: {len(prepared_split)} samples (skipped: {skipped})")
    return prepared_split


def load_and_prepare_hf_datasets(
    dataset_source: str,
    processor: Wav2Vec2Processor,
    data_dir: str = "data",
    token: str = None,
):
    """Load and preprocess a local saved DatasetDict or a remote HF dataset."""
    raw_dataset = load_hf_dataset_source(dataset_source, data_dir=data_dir, token=token)

    if "train" in raw_dataset and ("validation" not in raw_dataset or "test" not in raw_dataset):
        if "validation" not in raw_dataset and "test" not in raw_dataset:
            split_once = raw_dataset["train"].train_test_split(test_size=0.1, seed=42)
            split_twice = split_once["test"].train_test_split(test_size=0.5, seed=42)
            raw_dataset = DatasetDict({
                "train": split_once["train"],
                "validation": split_twice["train"],
                "test": split_twice["test"],
            })
        elif "validation" not in raw_dataset:
            split_once = raw_dataset["train"].train_test_split(test_size=0.1, seed=42)
            raw_dataset = DatasetDict({
                "train": split_once["train"],
                "validation": split_once["test"],
                "test": raw_dataset["test"],
            })
        elif "test" not in raw_dataset:
            split_once = raw_dataset["train"].train_test_split(test_size=0.1, seed=42)
            raw_dataset = DatasetDict({
                "train": split_once["train"],
                "validation": raw_dataset["validation"],
                "test": split_once["test"],
            })

    train_dataset = _prepare_dataset_split(raw_dataset["train"], processor, "train")
    val_dataset = _prepare_dataset_split(raw_dataset["validation"], processor, "validation")
    test_dataset = _prepare_dataset_split(raw_dataset["test"], processor, "test")

    print("\n" + "="*60)
    print("Dataset processing completed")
    print("="*60)

    return train_dataset, val_dataset, test_dataset

def create_vocabulary_file(train_jsonl: str, output_file: str = "vocab.json"):
    """
    Tạo file vocabulary từ dữ liệu training
    """
    # Load training data
    dataset = VietnameseASRDataset(train_jsonl)
    
    # Extract all texts
    texts = [item['transcript'] for item in dataset.data]
    
    # Create vocabulary
    vocab = dataset.create_vocab(texts)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    print(f"Vocabulary created with {len(vocab)} characters")
    print(f"Saved to: {output_file}")
    
    return vocab

def create_processor(vocab_file: str, model_name: str = None):
    """
    Tạo Wav2Vec2Processor từ vocabulary file
    """
    if model_name:
        # Use pre-trained processor
        processor = Wav2Vec2Processor.from_pretrained(model_name)
    else:
        # Create new processor from vocab
        tokenizer = Wav2Vec2CTCTokenizer(
            vocab_file,
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|"
        )
        
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True
        )
        
        processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer
        )
    
    return processor

def load_and_prepare_datasets(
    train_file: str,
    val_file: str,
    test_file: str,
    processor: Wav2Vec2Processor
):
    """
    Load và prepare tất cả datasets (không dùng multiprocessing để tránh WinError 87 trên Windows)
    """
    print("Loading datasets...")
    
    # Load datasets
    train_dataset = VietnameseASRDataset(train_file).to_hf_dataset()
    val_dataset = VietnameseASRDataset(val_file).to_hf_dataset()
    test_dataset = VietnameseASRDataset(test_file).to_hf_dataset()
    
    print(f"Train: {len(train_dataset)} samples")
    print(f"Validation: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    
    # Apply preprocessing - Không dùng num_proc (single process, không multiprocessing)
    print("\nPreprocessing datasets...")
    
    # Process datasets with error handling
    print("Processing train dataset...")
    train_orig_len = len(train_dataset)
    train_dataset = train_dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        remove_columns=train_dataset.column_names
    )
    # Filter out None/invalid samples (check for input_values presence)
    train_dataset = train_dataset.filter(lambda x: x.get('input_values') is not None)
    train_skipped = train_orig_len - len(train_dataset)
    
    print("Processing validation dataset...")
    val_orig_len = len(val_dataset)
    val_dataset = val_dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        remove_columns=val_dataset.column_names
    )
    val_dataset = val_dataset.filter(lambda x: x.get('input_values') is not None)
    val_skipped = val_orig_len - len(val_dataset)
    
    print("Processing test dataset...")
    test_orig_len = len(test_dataset)
    test_dataset = test_dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        remove_columns=test_dataset.column_names
    )
    test_dataset = test_dataset.filter(lambda x: x.get('input_values') is not None)
    test_skipped = test_orig_len - len(test_dataset)
    
    # Summary report
    total_skipped = train_skipped + val_skipped + test_skipped
    print("\n" + "="*60)
    print("📊 Dataset Processing Summary")
    print("="*60)
    print(f"Train:      {len(train_dataset):,} samples (skipped: {train_skipped})")
    print(f"Validation: {len(val_dataset):,} samples (skipped: {val_skipped})")
    print(f"Test:       {len(test_dataset):,} samples (skipped: {test_skipped})")
    print("="*60)
    
    if total_skipped > 0:
        print(f"\n⚠️ Warning: {total_skipped} files were skipped due to errors")
        print("💡 Common causes:")
        print("   1. Audio files not found (check Cell 13 - dataset copy/symlink)")
        print("   2. Incorrect working directory (should be /content/Vietnamese-asr)")
        print("   3. Wrong paths in JSONL files (should be relative: Data/vivos/...)")
    
    if len(train_dataset) == 0:
        print("\n❌ CRITICAL: No training samples loaded!")
        print("🔍 Troubleshooting steps:")
        print("   1. Check if Cell 13 completed (dataset copy/symlink)")
        print("   2. Verify audio files exist: ls -la Data/vivos/vivos/train/waves/")
        print("   3. Check working directory: pwd")
        print("   4. Verify JSONL paths with Cell 12 output")
        raise ValueError("No training data available")
    
    return train_dataset, val_dataset, test_dataset

def main():
    """
    Main function để test preprocessing pipeline
    """
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'processed_data'
    
    # Create vocabulary
    print("Creating vocabulary...")
    vocab = create_vocabulary_file(
        str(data_dir / 'train.jsonl'),
        str(base_dir / 'vocab.json')
    )
    
    # Create processor
    print("\nCreating processor...")
    # Option 1: Use pre-trained Vietnamese model
    # processor = create_processor(None, model_name="nguyenvulebinh/wav2vec2-base-vietnamese-250h")
    
    # Option 2: Create from scratch with our vocab
    processor = create_processor(str(base_dir / 'vocab.json'))
    
    # Save processor
    processor.save_pretrained(str(base_dir / 'processor'))
    print(f"Processor saved to: {base_dir / 'processor'}")
    
    # Test loading a sample
    print("\nTesting data loading...")
    dataset = VietnameseASRDataset(str(data_dir / 'train.jsonl'))
    print(f"Sample transcript: {dataset.data[0]['transcript']}")
    print(f"Normalized: {dataset.normalize_text(dataset.data[0]['transcript'])}")

if __name__ == "__main__":
    main()
