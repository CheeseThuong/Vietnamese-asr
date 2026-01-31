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
        # Fallback: torchaudio.load (handles unicode paths better on Windows)
        try:
            speech_tensor, sampling_rate = torchaudio.load(str(audio_path_obj))
            if speech_tensor.ndim > 1:
                speech_tensor = speech_tensor.mean(dim=0)
            speech_array = speech_tensor.numpy()

            if sampling_rate != 16000:
                resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
                speech_array = resampler(torch.FloatTensor(speech_array)).numpy()

            if speech_array.dtype != np.float32:
                speech_array = speech_array.astype(np.float32)

            batch["input_values"] = processor(
                speech_array,
                sampling_rate=16000
            ).input_values[0]
        except Exception as e2:
            print(f"Error loading audio {audio_path}: {e}")
            print(f"Fallback failed for {audio_path}: {e2}")
            # Return empty audio on error
            batch["input_values"] = processor(
                np.zeros(16000, dtype=np.float32), 
                sampling_rate=16000
            ).input_values[0]
    
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
    
    # Track corrupted files
    corrupted_files = []
    
    def prepare_and_filter(batch, processor, corrupted_list):
        """Wrapper để track corrupted files"""
        try:
            result = prepare_dataset(batch, processor)
            # Kiểm tra nếu audio bị lỗi (input_values = zeros)
            if result["input_values"] is not None and len(result["input_values"]) > 0:
                # Check if it's all zeros (corrupted)
                if not np.all(result["input_values"] == 0):
                    return result
                else:
                    corrupted_list.append(batch["audio_path"])
            else:
                corrupted_list.append(batch["audio_path"])
        except Exception as e:
            corrupted_list.append(batch["audio_path"])
            print(f"⚠️ Skipping corrupted file: {batch['audio_path']}")
        return None
    
    # Process with filter
    train_dataset = train_dataset.map(
        lambda batch: prepare_and_filter(batch, processor, corrupted_files),
        remove_columns=train_dataset.column_names
    )
    # Remove None entries (corrupted files)
    train_dataset = train_dataset.filter(lambda x: x is not None and "input_values" in x)
    
    val_dataset = val_dataset.map(
        lambda batch: prepare_and_filter(batch, processor, corrupted_files),
        remove_columns=val_dataset.column_names
    )
    val_dataset = val_dataset.filter(lambda x: x is not None and "input_values" in x)
    
    test_dataset = test_dataset.map(
        lambda batch: prepare_and_filter(batch, processor, corrupted_files),
        remove_columns=test_dataset.column_names
    )
    test_dataset = test_dataset.filter(lambda x: x is not None and "input_values" in x)
    
    # Report corrupted files
    if corrupted_files:
        print(f"\n⚠️ Skipped {len(corrupted_files)} corrupted audio files")
        print(f"Final counts:")
        print(f"  - Train: {len(train_dataset)} samples")
        print(f"  - Validation: {len(val_dataset)} samples")
        print(f"  - Test: {len(test_dataset)} samples")
    
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
