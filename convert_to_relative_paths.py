"""
Script để chuyển absolute paths về relative paths cho Colab
"""
import json
from pathlib import Path
import os

def convert_to_relative_paths(input_file, output_file=None, base_path=None):
    """
    Chuyển absolute paths về relative paths (cho Colab)
    """
    if output_file is None:
        output_file = input_file
    
    if base_path is None:
        # Tự động detect base path từ absolute paths
        base_path = Path.cwd()
    
    print(f"Processing: {input_file}")
    print(f"Base directory: {base_path}")
    
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
        
        # Convert to Path object
        path_obj = Path(audio_path)
        
        # If absolute, convert to relative
        if path_obj.is_absolute():
            # Extract relative path from "Data/vivos/..." onwards
            parts = path_obj.parts
            
            # Find "Data" in path
            try:
                data_idx = parts.index('Data')
                # Take from "Data" onwards
                relative_path = Path(*parts[data_idx:])
                item['audio_path'] = str(relative_path).replace('\\', '/')
                updated_count += 1
            except ValueError:
                # "Data" not in path, try to make relative to base
                try:
                    relative_path = path_obj.relative_to(base_path)
                    item['audio_path'] = str(relative_path).replace('\\', '/')
                    updated_count += 1
                except ValueError:
                    print(f"⚠️ Cannot convert: {audio_path}")
    
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
    """Convert all JSONL files to use relative paths"""
    import sys
    
    if len(sys.argv) > 1:
        # Convert specific file
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        convert_to_relative_paths(input_file, output_file)
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
                # Overwrite original file with relative paths
                convert_to_relative_paths(str(input_path))
            else:
                print(f"⚠️ File not found: {input_path}")

if __name__ == "__main__":
    main()
