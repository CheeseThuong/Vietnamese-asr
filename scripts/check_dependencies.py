"""
Script kiểm tra dependencies đã được cài đặt chưa
"""
import sys

def check_dependencies():
    """
    Kiểm tra các thư viện cần thiết
    """
    required_packages = {
        'torch': 'PyTorch',
        'torchaudio': 'TorchAudio',
        'transformers': 'HuggingFace Transformers',
        'datasets': 'HuggingFace Datasets',
        'accelerate': 'Accelerate',
        'librosa': 'Librosa',
        'soundfile': 'SoundFile',
        'jiwer': 'JiWER',
        'evaluate': 'Evaluate',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'tqdm': 'tqdm',
    }
    
    missing = []
    installed = []
    
    print("Checking dependencies...\n")
    print("="*60)
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name:30s} [OK]")
            installed.append(name)
        except ImportError:
            print(f"✗ {name:30s} [MISSING]")
            missing.append(package)
    
    print("="*60)
    
    if missing:
        print(f"\n❌ Missing {len(missing)} package(s):")
        for pkg in missing:
            print(f"   - {pkg}")
        
        print("\n" + "="*60)
        print("INSTALLATION INSTRUCTIONS")
        print("="*60)
        print("\nOption 1: Install all dependencies from requirements.txt")
        print("  pip install -r requirements.txt")
        
        print("\nOption 2: Install missing packages only")
        print(f"  pip install {' '.join(missing)}")
        
        print("\n" + "="*60)
        return False
    else:
        print(f"\n✓ All {len(installed)} required packages are installed!")
        return True

def check_torch_cuda():
    """
    Kiểm tra CUDA availability
    """
    try:
        import torch
        
        print("\n" + "="*60)
        print("PYTORCH & CUDA CHECK")
        print("="*60)
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
            print("\n✓ GPU training is available!")
        else:
            print("\n⚠ GPU not available, will use CPU")
            print("  (Training will be slower)")
        print("="*60)
        
    except ImportError:
        print("\n⚠ PyTorch not installed, skipping CUDA check")

def main():
    print("="*60)
    print("DEPENDENCY CHECKER")
    print("="*60)
    print()
    
    success = check_dependencies()
    
    if success:
        check_torch_cuda()
        print("\n✓ All checks passed! You're ready to go!")
        return 0
    else:
        print("\n❌ Please install missing dependencies first!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
