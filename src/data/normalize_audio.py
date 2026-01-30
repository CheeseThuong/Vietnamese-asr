"""
Script để chuẩn hóa audio files về 16kHz, mono
Sử dụng cho VIVOS dataset
"""
import os
import torchaudio
from pathlib import Path
from tqdm import tqdm
import json

def check_audio_info(audio_path):
    """
    Kiểm tra thông tin audio file
    """
    try:
        info = torchaudio.info(audio_path)
        return {
            'path': str(audio_path),
            'sample_rate': info.sample_rate,
            'num_channels': info.num_channels,
            'num_frames': info.num_frames,
            'duration': info.num_frames / info.sample_rate
        }
    except Exception as e:
        print(f"Error reading {audio_path}: {e}")
        return None

def normalize_audio(input_path, output_path=None, target_sr=16000):
    """
    Chuẩn hóa audio file:
    - Resample về 16kHz
    - Convert sang mono
    - Normalize amplitude
    
    Args:
        input_path: Path to input audio file
        output_path: Path to save normalized audio (if None, overwrite input)
        target_sr: Target sample rate (default 16000)
    """
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(input_path)
        
        modified = False
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            modified = True
        
        # Resample if needed
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=target_sr
            )
            waveform = resampler(waveform)
            modified = True
        
        # Normalize amplitude to [-1, 1]
        if waveform.abs().max() > 0:
            waveform = waveform / waveform.abs().max()
            modified = True
        
        # Save
        if modified:
            save_path = output_path if output_path else input_path
            torchaudio.save(save_path, waveform, target_sr)
            return True, "normalized"
        else:
            return False, "already_ok"
            
    except Exception as e:
        return False, f"error: {e}"

def scan_audio_files(base_dir, extensions=['.wav', '.mp3', '.flac', '.ogg']):
    """
    Scan tất cả audio files trong directory
    """
    audio_files = []
    base_path = Path(base_dir)
    
    for ext in extensions:
        audio_files.extend(list(base_path.rglob(f'*{ext}')))
    
    return audio_files

def analyze_dataset(base_dir):
    """
    Phân tích dataset và đưa ra thống kê
    """
    print(f"Scanning audio files in: {base_dir}")
    audio_files = scan_audio_files(base_dir)
    
    print(f"Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print("No audio files found!")
        return None
    
    # Analyze sample
    print("\nAnalyzing audio files...")
    stats = {
        'total': len(audio_files),
        'sample_rates': {},
        'channels': {},
        'total_duration': 0,
        'files_info': []
    }
    
    for audio_file in tqdm(audio_files, desc="Analyzing"):
        info = check_audio_info(audio_file)
        if info:
            sr = info['sample_rate']
            ch = info['num_channels']
            
            stats['sample_rates'][sr] = stats['sample_rates'].get(sr, 0) + 1
            stats['channels'][ch] = stats['channels'].get(ch, 0) + 1
            stats['total_duration'] += info['duration']
            stats['files_info'].append(info)
    
    return stats

def print_stats(stats):
    """
    In thống kê dataset
    """
    print("\n" + "="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    print(f"Total files: {stats['total']}")
    print(f"Total duration: {stats['total_duration']/3600:.2f} hours")
    
    print("\nSample Rates:")
    for sr, count in sorted(stats['sample_rates'].items()):
        percentage = (count / stats['total']) * 100
        print(f"  {sr} Hz: {count} files ({percentage:.1f}%)")
    
    print("\nChannels:")
    for ch, count in sorted(stats['channels'].items()):
        percentage = (count / stats['total']) * 100
        ch_name = "Mono" if ch == 1 else f"{ch} channels"
        print(f"  {ch_name}: {count} files ({percentage:.1f}%)")
    
    # Check if normalization needed
    needs_norm = False
    if len(stats['sample_rates']) > 1 or 16000 not in stats['sample_rates']:
        needs_norm = True
        print("\n⚠ Some files need resampling to 16kHz")
    
    if any(ch > 1 for ch in stats['channels'].keys()):
        needs_norm = True
        print("⚠ Some files need conversion to mono")
    
    if not needs_norm:
        print("\n✓ All files are already normalized (16kHz, mono)")
    
    print("="*60)
    
    return needs_norm

def normalize_dataset(base_dir, inplace=False, output_dir=None):
    """
    Chuẩn hóa tất cả audio files trong dataset
    
    Args:
        base_dir: Base directory chứa audio files
        inplace: Nếu True, ghi đè file gốc. Nếu False, lưu vào output_dir
        output_dir: Directory để lưu normalized files
    """
    audio_files = scan_audio_files(base_dir)
    
    if not audio_files:
        print("No audio files found!")
        return
    
    print(f"\nNormalizing {len(audio_files)} audio files...")
    
    results = {
        'normalized': 0,
        'already_ok': 0,
        'errors': 0
    }
    
    for audio_file in tqdm(audio_files, desc="Normalizing"):
        if inplace:
            output_path = None
        else:
            # Preserve directory structure
            rel_path = audio_file.relative_to(base_dir)
            output_path = Path(output_dir) / rel_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        success, status = normalize_audio(audio_file, output_path)
        
        if success:
            results['normalized'] += 1
        elif status == 'already_ok':
            results['already_ok'] += 1
            # Copy file if not inplace
            if not inplace and output_path:
                import shutil
                shutil.copy2(audio_file, output_path)
        else:
            results['errors'] += 1
            print(f"\nError with {audio_file}: {status}")
    
    print("\n" + "="*60)
    print("NORMALIZATION RESULTS")
    print("="*60)
    print(f"Normalized: {results['normalized']}")
    print(f"Already OK: {results['already_ok']}")
    print(f"Errors: {results['errors']}")
    print("="*60)

def main():
    """
    Main function
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Chuẩn hóa audio files về 16kHz, mono")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="Data/vivos/vivos",
        help="Directory chứa audio files"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Chỉ phân tích, không normalize"
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Ghi đè file gốc (CẢNH BÁO: Không thể hoàn tác!)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Data/vivos_normalized",
        help="Directory để lưu normalized files (nếu không dùng --inplace)"
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.data_dir)
    
    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        return
    
    print("="*60)
    print("AUDIO NORMALIZATION TOOL")
    print("="*60)
    print(f"Data directory: {base_dir}")
    
    # Analyze dataset
    stats = analyze_dataset(str(base_dir))
    
    if stats is None:
        return
    
    needs_norm = print_stats(stats)
    
    # Save stats
    stats_file = Path("audio_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        # Remove files_info to reduce size
        stats_to_save = {k: v for k, v in stats.items() if k != 'files_info'}
        json.dump(stats_to_save, f, indent=2)
    print(f"\nStats saved to: {stats_file}")
    
    # Normalize if requested
    if not args.analyze_only:
        if not needs_norm:
            print("\n✓ No normalization needed!")
            return
        
        print("\n" + "="*60)
        if args.inplace:
            print("⚠ WARNING: This will OVERWRITE original files!")
            response = input("Continue? (yes/no): ")
            if response.lower() != 'yes':
                print("Cancelled.")
                return
        else:
            print(f"Normalized files will be saved to: {args.output_dir}")
        print("="*60)
        
        normalize_dataset(
            str(base_dir),
            inplace=args.inplace,
            output_dir=args.output_dir
        )
        
        print("\n✓ Normalization complete!")
    else:
        print("\n(Use without --analyze-only to normalize files)")

if __name__ == "__main__":
    main()
