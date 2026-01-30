"""
Script để chuyển đường dẫn relative sang absolute trong JSONL files
"""
import json
from pathlib import Path
import os

def convert_to_absolute_paths(input_file, output_file=None):
    """
    Đọc JSONL file và chuyển audio_path từ relative sang absolute
    """
    if output_file is None:
        output_file = input_file.replace('.jsonl', '_absolute.jsonl')
    
    # Get current working directory
    base_dir = Path.cwd()
    
    print(f"Processing: {input_file}")
    print(f"Base directory: {base_dir}")
    
    # Load data
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f"Total samples: {len(data)}")
    
    # Convert paths
    updated_count = 0
    for item in data:
        audio_path = item['audio_path']
        
        # Check if already absolute
        if not os.path.isabs(audio_path):
            # Convert to absolute
            abs_path = (base_dir / audio_path).resolve()
            item['audio_path'] = str(abs_path)
            updated_count += 1
    
    # Save updated data
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Results:")
    print(f"   - Updated paths: {updated_count}")
    print(f"   - Output: {output_file}")
    
    # Show sample paths
    print(f"\n📄 Sample paths (first 3):")
    for item in data[:3]:
        print(f"   {item['audio_path']}")
    
    return data

def main():
    """Convert all JSONL files to use absolute paths"""
    import sys
    
    if len(sys.argv) > 1:
        # Convert specific file
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        convert_to_absolute_paths(input_file, output_file)
    else:
        # Convert all files in processed_data_vivos/
        data_dir = Path('processed_data_vivos')
        
        if not data_dir.exists():
            print(f"❌ Directory not found: {data_dir}")
            print("\nUsage:")
            print(f"  python {Path(__file__).name} <input_file.jsonl> [output_file.jsonl]")
            return
        
        for jsonl_file in ['train.jsonl', 'validation.jsonl', 'test.jsonl']:
            input_path = data_dir / jsonl_file
            if input_path.exists():
                print(f"\n{'='*60}")
                # Overwrite original file with absolute paths
                output_path = input_path  # Overwrite directly
                convert_to_absolute_paths(str(input_path), str(output_path))
            else:
                print(f"⚠️ File not found: {input_path}")

if __name__ == "__main__":
    main()
