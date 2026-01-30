"""
Script để gộp và xử lý dữ liệu VIVOS và VinBigData cho ASR tiếng Việt
"""
import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import pandas as pd

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

def create_dataset_summary(train_data, val_data, test_data, output_dir):
    """
    Tạo file summary thống kê dataset
    """
    summary = {
        'train': {
            'total': len(train_data),
            'vivos': len([d for d in train_data if d['dataset'] == 'vivos']),
            'vinbigdata': len([d for d in train_data if d['dataset'] == 'vinbigdata'])
        },
        'validation': {
            'total': len(val_data),
            'vivos': len([d for d in val_data if d['dataset'] == 'vivos']),
            'vinbigdata': len([d for d in val_data if d['dataset'] == 'vinbigdata'])
        },
        'test': {
            'total': len(test_data),
            'vivos': len([d for d in test_data if d['dataset'] == 'vivos']),
            'vinbigdata': len([d for d in test_data if d['dataset'] == 'vinbigdata'])
        }
    }
    
    with open(output_dir / 'dataset_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"Train: {summary['train']['total']} samples")
    print(f"  - VIVOS: {summary['train']['vivos']}")
    print(f"  - VinBigData: {summary['train']['vinbigdata']}")
    print(f"\nValidation: {summary['validation']['total']} samples")
    print(f"  - VIVOS: {summary['validation']['vivos']}")
    print(f"  - VinBigData: {summary['validation']['vinbigdata']}")
    print(f"\nTest: {summary['test']['total']} samples")
    print(f"  - VIVOS: {summary['test']['vivos']}")
    print(f"  - VinBigData: {summary['test']['vinbigdata']}")
    print("="*50)

def main():
    # Đường dẫn
    base_dir = Path(__file__).parent
    vivos_path = base_dir / 'Data' / 'vivos' / 'vivos'
    vinbigdata_path = base_dir / 'Data' / 'Data'
    output_dir = base_dir / 'processed_data'
    
    # Tạo thư mục output
    output_dir.mkdir(exist_ok=True)
    
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
