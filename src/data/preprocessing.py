"""
Data preprocessing pipeline cho Wav2Vec2 ASR
"""
import torch
import torchaudio
from datasets import Dataset
import json
from pathlib import Path
from transformers import Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
import re
from typing import Dict, List
import soundfile as sf
import numpy as np

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
        """
        Chuẩn hóa văn bản tiếng Việt
        - Chuyển về chữ thường
        - Loại bỏ dấu câu không cần thiết
        - Chuẩn hóa khoảng trắng
        """
        # Chuyển về chữ thường
        text = text.lower()
        
        # Loại bỏ các ký tự đặc biệt, giữ lại chữ cái tiếng Việt và số
        # Giữ lại khoảng trắng
        text = re.sub(r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', text)
        
        # Loại bỏ <unk> và các token đặc biệt khác
        text = re.sub(r'<unk>|<s>|</s>|<pad>', ' ', text)
        
        # Chuẩn hóa khoảng trắng
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
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
    # Load audio file thủ công bằng soundfile
    audio_path = batch["audio_path"]
    
    try:
        # Resolve path & check exists
        audio_path_obj = Path(audio_path)
        if not audio_path_obj.exists():
            # Debug: In ra thông tin để check
            import os
            print(f"❌ File not found: {audio_path}")
            print(f"   Working dir: {os.getcwd()}")
            print(f"   Absolute path: {audio_path_obj.absolute()}")
            print(f"   Exists: {audio_path_obj.exists()}")
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio với soundfile (use file object to avoid Windows unicode path issues)
        with open(audio_path_obj, "rb") as f:
            speech_array, sampling_rate = sf.read(f)
        
        # Resample nếu cần (soundfile không tự động resample)
        if sampling_rate != 16000:
            # Dùng torchaudio để resample
            speech_tensor = torch.FloatTensor(speech_array)
            if len(speech_tensor.shape) > 1:  # Stereo to mono
                speech_tensor = speech_tensor.mean(dim=1)
            resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
            speech_array = resampler(speech_tensor).numpy()
        
        # Convert to float32 nếu cần
        if speech_array.dtype != np.float32:
            speech_array = speech_array.astype(np.float32)
        
        # Compute input values (mel spectrogram features)
        batch["input_values"] = processor(
            speech_array, 
            sampling_rate=16000
        ).input_values[0]
        
    except Exception as e:
        print(f"❌ Error loading audio {audio_path}: {e}")
        # Return None to filter out this sample
        return None
    
    # Encode target text - Dùng tokenizer trực tiếp (không dùng as_target_processor)
    batch["labels"] = processor.tokenizer(
        batch["transcript"],
        padding=False,
        truncation=True
    ).input_ids
    
    return batch

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
    # Filter out None (failed samples)
    train_dataset = train_dataset.filter(lambda x: x is not None)
    train_skipped = train_orig_len - len(train_dataset)
    
    print("Processing validation dataset...")
    val_orig_len = len(val_dataset)
    val_dataset = val_dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        remove_columns=val_dataset.column_names
    )
    val_dataset = val_dataset.filter(lambda x: x is not None)
    val_skipped = val_orig_len - len(val_dataset)
    
    print("Processing test dataset...")
    test_orig_len = len(test_dataset)
    test_dataset = test_dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        remove_columns=test_dataset.column_names
    )
    test_dataset = test_dataset.filter(lambda x: x is not None)
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
