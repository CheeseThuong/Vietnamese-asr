"""
Script đơn giản để chuẩn bị dữ liệu VIVOS (không gộp VinBigData)
Sử dụng để test quy trình nhanh hơn
"""
import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import random

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
            
        print(f"\nProcessing {split} split...")
        
        # Đọc prompts
        with open(prompts_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in tqdm(lines, desc=f"Processing {split}"):
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
    
    print(f"\nVIVOS: Found {len(data)} samples")
    return data

def split_data(data, val_ratio=0.1):
    """
    Chia dữ liệu thành train/validation/test
    """
    random.seed(42)
    
    # VIVOS đã có train/test split
    train_data = [d for d in data if d['split'] == 'train']
    test_data = [d for d in data if d['split'] == 'test']
    
    # Chia train thành train/val
    random.shuffle(train_data)
    split_idx = int(len(train_data) * (1 - val_ratio))
    train_final = train_data[:split_idx]
    val_data = train_data[split_idx:]
    
    return train_final, val_data, test_data

def save_dataset(data, output_file):
    """
    Lưu dataset dưới dạng JSON Lines
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def create_dataset_summary(train_data, val_data, test_data, output_dir):
    """
    Tạo file summary thống kê dataset
    """
    # Count speakers
    train_speakers = len(set(d['speaker_id'] for d in train_data))
    val_speakers = len(set(d['speaker_id'] for d in val_data))
    test_speakers = len(set(d['speaker_id'] for d in test_data))
    
    summary = {
        'dataset': 'VIVOS only',
        'train': {
            'total': len(train_data),
            'speakers': train_speakers
        },
        'validation': {
            'total': len(val_data),
            'speakers': val_speakers
        },
        'test': {
            'total': len(test_data),
            'speakers': test_speakers
        },
        'total_samples': len(train_data) + len(val_data) + len(test_data),
        'total_speakers': len(set(d['speaker_id'] for d in train_data + val_data + test_data))
    }
    
    with open(output_dir / 'dataset_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*50)
    print("DATASET SUMMARY (VIVOS ONLY)")
    print("="*50)
    print(f"Train: {summary['train']['total']} samples ({summary['train']['speakers']} speakers)")
    print(f"Validation: {summary['validation']['total']} samples ({summary['validation']['speakers']} speakers)")
    print(f"Test: {summary['test']['total']} samples ({summary['test']['speakers']} speakers)")
    print(f"\nTotal: {summary['total_samples']} samples")
    print(f"Total speakers: {summary['total_speakers']}")
    print("="*50)

def check_audio_files(data, sample_size=10):
    """
    Kiểm tra một số audio files để đảm bảo chúng có thể đọc được
    """
    import torchaudio
    
    print(f"\nChecking {sample_size} random audio files...")
    
    random_samples = random.sample(data, min(sample_size, len(data)))
    
    for item in random_samples:
        try:
            info = torchaudio.info(item['audio_path'])
            print(f"✓ {Path(item['audio_path']).name}: {info.sample_rate}Hz, {info.num_channels}ch, {info.num_frames/info.sample_rate:.2f}s")
        except Exception as e:
            print(f"✗ {item['audio_path']}: {e}")

def main():
    """
    Main function
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu VIVOS (không gộp VinBigData)")
    parser.add_argument(
        "--vivos_path",
        type=str,
        default="Data/vivos/vivos",
        help="Path to VIVOS dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="processed_data_vivos",
        help="Output directory"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--check-audio",
        action="store_true",
        help="Check audio files before processing"
    )
    
    args = parser.parse_args()
    
    # Đường dẫn
    vivos_path = Path(args.vivos_path)
    output_dir = Path(args.output_dir)
    
    if not vivos_path.exists():
        print(f"Error: VIVOS path not found: {vivos_path}")
        return
    
    # Tạo thư mục output
    output_dir.mkdir(exist_ok=True)
    
    print("="*50)
    print("VIVOS DATASET PREPARATION")
    print("="*50)
    print(f"Input: {vivos_path}")
    print(f"Output: {output_dir}")
    print("="*50)
    
    # Xử lý VIVOS
    vivos_data = process_vivos_data(vivos_path, output_dir)
    
    if len(vivos_data) == 0:
        print("Error: No data found!")
        return
    
    # Check audio files if requested
    if args.check_audio:
        check_audio_files(vivos_data)
    
    # Chia train/val/test
    train_data, val_data, test_data = split_data(vivos_data, args.val_ratio)
    
    # Lưu datasets
    print("\nSaving datasets...")
    save_dataset(train_data, output_dir / 'train.jsonl')
    save_dataset(val_data, output_dir / 'validation.jsonl')
    save_dataset(test_data, output_dir / 'test.jsonl')
    
    # Tạo summary
    create_dataset_summary(train_data, val_data, test_data, output_dir)
    
    print(f"\n✓ Datasets saved to: {output_dir}")
    print("\nFiles created:")
    print(f"  - {output_dir}/train.jsonl")
    print(f"  - {output_dir}/validation.jsonl")
    print(f"  - {output_dir}/test.jsonl")
    print(f"  - {output_dir}/dataset_summary.json")
    
    print("\n" + "="*50)
    print("NEXT STEPS:")
    print("="*50)
    print("1. Check audio normalization:")
    print("   python normalize_audio.py --analyze-only")
    print("\n2. Normalize if needed:")
    print("   python normalize_audio.py --output_dir Data/vivos_normalized")
    print("\n3. Run preprocessing:")
    print("   python data_preprocessing.py")
    print("\n4. Train model:")
    print("   python train_wav2vec2.py")
    print("\n5. Evaluate model:")
    print("   python run_evaluation.py")
    print("="*50)

if __name__ == "__main__":
    main()
