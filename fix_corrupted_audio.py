"""
Script để tìm và loại bỏ các file audio bị lỗi khỏi dataset JSONL
"""
import json
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

def check_audio_file(audio_path):
    """Kiểm tra xem file audio có đọc được không"""
    try:
        speech, sr = sf.read(audio_path)
        if len(speech) == 0:
            return False, "Empty audio"
        return True, None
    except Exception as e:
        return False, str(e)

def clean_jsonl_file(input_file, output_file=None):
    """
    Đọc JSONL file, kiểm tra từng audio file, loại bỏ file lỗi
    """
    if output_file is None:
        output_file = input_file.replace('.jsonl', '_cleaned.jsonl')
    
    print(f"Processing: {input_file}")
    
    # Load data
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f"Total samples: {len(data)}")
    
    # Check each audio file
    valid_data = []
    corrupted_files = []
    
    for item in tqdm(data, desc="Checking audio files"):
        audio_path = item['audio_path']
        is_valid, error = check_audio_file(audio_path)
        
        if is_valid:
            valid_data.append(item)
        else:
            corrupted_files.append({
                'path': audio_path,
                'error': error,
                'transcript': item.get('transcript', '')
            })
    
    # Save cleaned data
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Report
    print(f"\n✅ Results:")
    print(f"   - Valid files: {len(valid_data)}")
    print(f"   - Corrupted files: {len(corrupted_files)}")
    print(f"   - Output: {output_file}")
    
    # Save corrupted files list
    if corrupted_files:
        corrupted_log = output_file.replace('.jsonl', '_corrupted.json')
        with open(corrupted_log, 'w', encoding='utf-8') as f:
            json.dump(corrupted_files, f, ensure_ascii=False, indent=2)
        print(f"   - Corrupted files list: {corrupted_log}")
        
        # Print first 10 corrupted files
        print(f"\n⚠️ First 10 corrupted files:")
        for item in corrupted_files[:10]:
            print(f"   - {item['path']}")
            print(f"     Error: {item['error']}")
    
    return valid_data, corrupted_files

def main():
    """Clean all JSONL files"""
    import sys
    
    if len(sys.argv) > 1:
        # Clean specific file
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        clean_jsonl_file(input_file, output_file)
    else:
        # Clean all files in processed_data_vivos/
        data_dir = Path('processed_data_vivos')
        
        if not data_dir.exists():
            print(f"❌ Directory not found: {data_dir}")
            print("\nUsage:")
            print(f"  python {__file__} <input_file.jsonl> [output_file.jsonl]")
            return
        
        for jsonl_file in ['train.jsonl', 'validation.jsonl', 'test.jsonl']:
            input_path = data_dir / jsonl_file
            if input_path.exists():
                print(f"\n{'='*60}")
                clean_jsonl_file(str(input_path))
            else:
                print(f"⚠️ File not found: {input_path}")

if __name__ == "__main__":
    main()
