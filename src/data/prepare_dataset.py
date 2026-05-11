"""
Script để gộp và xử lý dữ liệu VIVOS và VinBigData cho ASR tiếng Việt
"""
import os
import json
import shutil
import random
import re
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset


HF_DATASET_DEFAULT_SOURCE = "https://huggingface.co/datasets/linhtran92/viet_bud500/tree/main/data"


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


def load_hf_dataset_split(dataset_source, split_name="train", data_dir="data", token=None):
    """Load a split from a gated Hugging Face dataset repo."""
    repo_id = normalize_hf_dataset_source(dataset_source)
    if not repo_id:
        raise ValueError("dataset_source is empty")

    load_kwargs = {
        "data_dir": data_dir,
        "split": split_name,
    }
    if token:
        load_kwargs["token"] = token

    print(f"Loading Hugging Face dataset: {repo_id} [{split_name}]...")
    return load_dataset(repo_id, **load_kwargs)


def process_hf_parquet_data(parquet_path, output_dir):
    """
    Xử lý dữ liệu Parquet từ Hugging Face dataset export.

    Schema hỗ trợ:
    - audio.bytes: bytes audio nhúng trong parquet
    - audio.path: tên file gốc
    - transcription: transcript tiếng Việt
    """
    from pyarrow import parquet as pq

    print("Processing Hugging Face Parquet dataset...")
    parquet_path = Path(parquet_path)
    output_dir = Path(output_dir)
    audio_output_dir = output_dir / "audio"
    audio_output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(parquet_path.glob("*.parquet"))
    if not parquet_files:
        print(f"Warning: no parquet files found in {parquet_path}")
        return []

    data = []
    seen_paths = set()

    for parquet_file in parquet_files:
        print(f"Reading {parquet_file.name}...")
        table = pq.read_table(parquet_file, columns=["audio", "transcription"])
        for row in table.to_pylist():
            audio = row.get("audio") or {}
            transcript = (row.get("transcription") or "").strip()
            if not transcript:
                continue

            source_name = audio.get("path") or "unknown.wav"
            audio_bytes = audio.get("bytes")
            target_name = Path(source_name).name
            target_audio_path = audio_output_dir / target_name

            if target_audio_path.as_posix() in seen_paths:
                # Avoid duplicate filenames across parquet shards.
                stem = target_audio_path.stem
                suffix = target_audio_path.suffix or ".wav"
                counter = 1
                while True:
                    candidate = audio_output_dir / f"{stem}_{counter}{suffix}"
                    if candidate.as_posix() not in seen_paths:
                        target_audio_path = candidate
                        break
                    counter += 1

            if audio_bytes:
                target_audio_path.write_bytes(audio_bytes)
                seen_paths.add(target_audio_path.as_posix())
            else:
                if Path(source_name).exists():
                    target_audio_path = Path(source_name).resolve()
                else:
                    continue

            data.append({
                "audio_path": str(target_audio_path),
                "transcript": transcript,
                "speaker_id": "unknown",
                "dataset": "hf_parquet"
            })

    print(f"Hugging Face Parquet: Found {len(data)} samples")
    return data


def process_hf_dataset_repo(dataset_source, output_dir, split_name="train", token=None):
    """Load a gated Hugging Face dataset repo directly and convert it to our JSONL format."""
    output_dir = Path(output_dir)
    audio_output_dir = output_dir / "audio"
    audio_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_hf_dataset_split(dataset_source, split_name=split_name, data_dir="data", token=token)
    data = []
    seen_paths = set()

    for row in dataset:
        audio = row.get("audio") or {}
        transcript = (row.get("transcription") or row.get("text") or "").strip()
        if not transcript:
            continue

        source_name = audio.get("path") or "unknown.wav"
        audio_bytes = audio.get("bytes")
        target_name = Path(source_name).name
        target_audio_path = audio_output_dir / target_name

        if target_audio_path.as_posix() in seen_paths:
            stem = target_audio_path.stem
            suffix = target_audio_path.suffix or ".wav"
            counter = 1
            while True:
                candidate = audio_output_dir / f"{stem}_{counter}{suffix}"
                if candidate.as_posix() not in seen_paths:
                    target_audio_path = candidate
                    break
                counter += 1

        if audio_bytes:
            target_audio_path.write_bytes(audio_bytes)
            seen_paths.add(target_audio_path.as_posix())
        else:
            if Path(source_name).exists():
                target_audio_path = Path(source_name).resolve()
            else:
                continue

        data.append({
            "audio_path": str(target_audio_path),
            "transcript": transcript,
            "speaker_id": row.get("speaker_id", "unknown"),
            "dataset": "hf_remote"
        })

    print(f"Hugging Face repo split '{split_name}': Found {len(data)} samples")
    return data

def process_vivos_data(vivos_path, output_dir):
    """
    Xử lý dữ liệu VIVOS
    Format: prompts.txt chứa ID và transcript, audio files trong waves/
    """
    print("Processing VIVOS dataset...")
    data = []
    
    for split in ['train', 'test']:
        split_path = Path(vivos_path) / split
        prompts_file = split_path / 'prompts.txt'
        waves_dir = split_path / 'waves'
        
        if not prompts_file.exists():
            print(f"Warning: {prompts_file} not found")
            continue
            
        # Đọc prompts
        with open(prompts_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) != 2:
                    continue
                    
                audio_id, transcript = parts
                # Format: VIVOSSPK01_R001
                speaker_id = audio_id.split('_')[0]
                
                # Tìm file audio tương ứng
                audio_file = waves_dir / speaker_id / f"{audio_id}.wav"
                
                if audio_file.exists():
                    data.append({
                        'audio_path': str(audio_file),
                        'transcript': transcript,
                        'speaker_id': speaker_id,
                        'dataset': 'vivos',
                        'split': split
                    })
                else:
                    print(f"Warning: Audio file not found: {audio_file}")
    
    print(f"VIVOS: Found {len(data)} samples")
    return data

def process_vinbigdata(vinbigdata_path, output_dir):
    """
    Xử lý dữ liệu VinBigData 
    Format: Mỗi file .txt chứa transcript, file audio cùng tên
    """
    print("Processing VinBigData dataset...")
    data = []
    vinbigdata_path = Path(vinbigdata_path)
    
    # Duyệt qua tất cả các phần (p1, p2, p3, ...)
    for part_dir in vinbigdata_path.glob('vlsp2020_train_p*'):
        print(f"Processing {part_dir.name}...")
        
        # Tìm tất cả file .txt
        txt_files = list(part_dir.glob('*.txt'))
        
        for txt_file in tqdm(txt_files, desc=f"Processing {part_dir.name}"):
            # Đọc transcript
            with open(txt_file, 'r', encoding='utf-8') as f:
                transcript = f.read().strip()
            
            # Tìm file audio tương ứng (thường là .wav)
            audio_file = txt_file.with_suffix('.wav')
            if not audio_file.exists():
                # Thử .flac
                audio_file = txt_file.with_suffix('.flac')
            if not audio_file.exists():
                # Thử .mp3
                audio_file = txt_file.with_suffix('.mp3')
            
            if audio_file.exists():
                data.append({
                    'audio_path': str(audio_file),
                    'transcript': transcript,
                    'speaker_id': 'unknown',  # VinBigData không có thông tin speaker
                    'dataset': 'vinbigdata',
                    'split': 'train'  # Tất cả đều là train data
                })
            else:
                print(f"Warning: Audio file not found for {txt_file}")
    
    print(f"VinBigData: Found {len(data)} samples")
    return data

def split_data(data, train_ratio=0.9):
    """
    Chia dữ liệu thành train/validation/test
    """
    import random
    random.seed(42)
    
    # Tách dữ liệu theo dataset gốc
    vivos_data = [d for d in data if d['dataset'] == 'vivos']
    vinbig_data = [d for d in data if d['dataset'] == 'vinbigdata']
    
    # VIVOS đã có train/test split
    vivos_train = [d for d in vivos_data if d['split'] == 'train']
    vivos_test = [d for d in vivos_data if d['split'] == 'test']
    
    # Chia VIVOS train thành train/val (90/10)
    random.shuffle(vivos_train)
    split_idx = int(len(vivos_train) * train_ratio)
    vivos_train_final = vivos_train[:split_idx]
    vivos_val = vivos_train[split_idx:]
    
    # Chia VinBigData thành train/val (90/10)
    random.shuffle(vinbig_data)
    split_idx = int(len(vinbig_data) * train_ratio)
    vinbig_train = vinbig_data[:split_idx]
    vinbig_val = vinbig_data[split_idx:]
    
    # Gộp lại
    train_data = vivos_train_final + vinbig_train
    val_data = vivos_val + vinbig_val
    test_data = vivos_test
    
    # Shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    return train_data, val_data, test_data

def save_dataset(data, output_file):
    """
    Lưu dataset dưới dạng JSON Lines
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


def split_hf_data(data, train_ratio=0.9, val_ratio=0.05):
    """Split Parquet dataset into train/validation/test."""
    random.seed(42)
    shuffled = data[:]
    random.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_data = shuffled[:train_end]
    val_data = shuffled[train_end:val_end]
    test_data = shuffled[val_end:]

    return train_data, val_data, test_data

def create_dataset_summary(train_data, val_data, test_data, output_dir):
    """
    Tạo file summary thống kê dataset
    """
    def count_by_dataset(rows):
        counts = {}
        for row in rows:
            dataset_name = row.get('dataset', 'unknown')
            counts[dataset_name] = counts.get(dataset_name, 0) + 1
        return counts

    summary = {
        'train': {
            'total': len(train_data),
            'by_dataset': count_by_dataset(train_data)
        },
        'validation': {
            'total': len(val_data),
            'by_dataset': count_by_dataset(val_data)
        },
        'test': {
            'total': len(test_data),
            'by_dataset': count_by_dataset(test_data)
        }
    }
    
    with open(output_dir / 'dataset_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"Train: {summary['train']['total']} samples")
    for name, count in summary['train']['by_dataset'].items():
        print(f"  - {name}: {count}")
    print(f"\nValidation: {summary['validation']['total']} samples")
    for name, count in summary['validation']['by_dataset'].items():
        print(f"  - {name}: {count}")
    print(f"\nTest: {summary['test']['total']} samples")
    for name, count in summary['test']['by_dataset'].items():
        print(f"  - {name}: {count}")
    print("="*50)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prepare ASR datasets")
    parser.add_argument(
        "--hf_train_path",
        type=str,
        default=None,
        help="Path to Hugging Face parquet train folder (contains *.parquet files)"
    )
    parser.add_argument(
        "--hf_output_dir",
        type=str,
        default="processed_data_hf",
        help="Output directory for Hugging Face parquet dataset"
    )
    parser.add_argument(
        "--skip_vivos_vinbig",
        action="store_true",
        help="Only prepare Hugging Face parquet data"
    )
    parser.add_argument(
        "--hf_dataset_source",
        type=str,
        default=os.getenv("HF_DATASET_SOURCE", HF_DATASET_DEFAULT_SOURCE),
        help="Hugging Face dataset URL or repo id (default: linhtran92/viet_bud500 data folder)"
    )
    parser.add_argument(
        "--hf_dataset_split",
        type=str,
        default="train",
        help="Split name to load from the remote Hugging Face dataset"
    )

    args = parser.parse_args()

    # Đường dẫn
    base_dir = Path(__file__).parent
    vivos_path = base_dir / 'Data' / 'vivos' / 'vivos'
    vinbigdata_path = base_dir / 'Data' / 'Data'
    output_dir = base_dir / 'processed_data'
    
    # Tạo thư mục output
    output_dir.mkdir(exist_ok=True)

    if args.hf_train_path:
        hf_output_dir = Path(args.hf_output_dir)
        hf_output_dir.mkdir(parents=True, exist_ok=True)

        hf_data = process_hf_parquet_data(args.hf_train_path, hf_output_dir)
        if not hf_data:
            print("Error: No Hugging Face parquet data found!")
            return

        train_data, val_data, test_data = split_hf_data(hf_data)

        print("\nSaving Hugging Face parquet datasets...")
        save_dataset(train_data, hf_output_dir / 'train.jsonl')
        save_dataset(val_data, hf_output_dir / 'validation.jsonl')
        save_dataset(test_data, hf_output_dir / 'test.jsonl')
        create_dataset_summary(train_data, val_data, test_data, hf_output_dir)

        print(f"\nHugging Face datasets saved to: {hf_output_dir}")
        print("Files created:")
        print("  - train.jsonl")
        print("  - validation.jsonl")
        print("  - test.jsonl")
        print("  - dataset_summary.json")

        if args.skip_vivos_vinbig:
            return

    if args.hf_dataset_source:
        hf_output_dir = Path(args.hf_output_dir)
        hf_output_dir.mkdir(parents=True, exist_ok=True)
        token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

        try:
            hf_data = process_hf_dataset_repo(
                args.hf_dataset_source,
                hf_output_dir,
                split_name=args.hf_dataset_split,
                token=token,
            )
        except Exception as e:
            print(f"Error loading remote Hugging Face dataset: {e}")
            print("Hint: the dataset is gated. Log in to Hugging Face and accept the access conditions first.")
            return

        if not hf_data:
            print("Error: No samples were loaded from the remote Hugging Face dataset!")
            return

        train_data, val_data, test_data = split_hf_data(hf_data)
        print("\nSaving remote Hugging Face datasets...")
        save_dataset(train_data, hf_output_dir / 'train.jsonl')
        save_dataset(val_data, hf_output_dir / 'validation.jsonl')
        save_dataset(test_data, hf_output_dir / 'test.jsonl')
        create_dataset_summary(train_data, val_data, test_data, hf_output_dir)
        print(f"\nRemote Hugging Face datasets saved to: {hf_output_dir}")
        print("Run training using the generated JSONL files or set the output directory as your data source.")

        if args.skip_vivos_vinbig:
            return
    
    # Xử lý VIVOS
    vivos_data = process_vivos_data(vivos_path, output_dir)
    
    # Xử lý VinBigData
    vinbigdata_data = process_vinbigdata(vinbigdata_path, output_dir)
    
    # Gộp tất cả dữ liệu
    all_data = vivos_data + vinbigdata_data
    print(f"\nTotal samples: {len(all_data)}")
    
    # Chia train/val/test
    train_data, val_data, test_data = split_data(all_data)
    
    # Lưu datasets
    print("\nSaving datasets...")
    save_dataset(train_data, output_dir / 'train.jsonl')
    save_dataset(val_data, output_dir / 'validation.jsonl')
    save_dataset(test_data, output_dir / 'test.jsonl')
    
    # Tạo summary
    create_dataset_summary(train_data, val_data, test_data, output_dir)
    
    print(f"\nDatasets saved to: {output_dir}")
    print("Files created:")
    print("  - train.jsonl")
    print("  - validation.jsonl")
    print("  - test.jsonl")
    print("  - dataset_summary.json")

if __name__ == "__main__":
    main()
