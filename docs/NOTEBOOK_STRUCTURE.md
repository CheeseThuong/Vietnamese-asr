# Cấu trúc Notebook Đúng - Vietnamese ASR Training

## Thứ tự thực hiện cells (QUAN TRỌNG!)

---

## CELL 1: Header (Markdown)
```markdown
# Vietnamese ASR Training - Google Colab

**Nhận dạng giọng nói tiếng Việt với Wav2Vec2**

## Setup Checklist
- [ ] Runtime -> Change runtime type -> GPU (T4)
- [ ] Mount Google Drive
- [ ] Upload dataset lên Drive
- [ ] Run all cells theo thứ tự

## Latest Updates (2026-01-31)
- Fixed: Audio loading errors with better error handling
- Fixed: processor NameError in model creation
- Fixed: Dataset columns mismatch in training
- Optimized: Symlink instead of copy (< 1s vs 5-15 min)
```

---

## CELL 2: Check GPU (Markdown)
```markdown
## 1. Check GPU & Environment
```

---

## CELL 3: Check GPU (Python)
```python
import torch
import sys

print("="*60)
print("Environment Info")
print("="*60)
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("\nGPU Ready!")
else:
    print("\nWARNING: GPU not available!")
    print("Go to: Runtime -> Change runtime type -> Hardware accelerator -> GPU")
```

---

## CELL 4: Mount Drive (Markdown)
```markdown
## 2. Mount Google Drive
```

---

## CELL 5: Mount Drive (Python)
```python
from google.colab import drive
import os

# Mount Drive
drive.mount('/content/drive')

# Create working directory
DRIVE_ROOT = "/content/drive/MyDrive/VietnameseASR"
os.makedirs(DRIVE_ROOT, exist_ok=True)

print(f"\nDrive mounted at: {DRIVE_ROOT}")
print("\nRecommended folder structure:")
print(f"{DRIVE_ROOT}/")
print("  |- data/               # Dataset files")
print("  |   |- train.jsonl")
print("  |   |- validation.jsonl")
print("  |   |- test.jsonl")
print("  |- vivos/              # Audio files folder")
print("  |- models/             # Checkpoints (auto-created)")
print("  |- final_model/        # Final output (auto-created)")
```

---

## CELL 6: Install Dependencies (Markdown)
```markdown
## 3. Install Dependencies
```

---

## CELL 7: Install Dependencies (Python)
```python
%%capture
# Install packages (silent mode)
!pip install -q transformers datasets evaluate jiwer soundfile librosa accelerate tensorboard
```

---

## CELL 8: Verify Installation (Python)
```python
# Verify installation
import transformers
import datasets
import evaluate

print("All packages installed successfully!")
print(f"   - transformers: {transformers.__version__}")
print(f"   - datasets: {datasets.__version__}")
print(f"   - evaluate: {evaluate.__version__}")
```

---

## CELL 9: Clone Source Code (Markdown)
```markdown
## 4. Clone Source Code from GitHub

This cell will:
- Clone repository if not exists
- Pull latest code if already exists
- Add to Python path
```

---

## CELL 10: Clone Source Code (Python)
```python
import os
import sys

# Check if repo already exists
if os.path.exists('/content/Vietnamese-asr'):
    print("Repository already exists, updating...")
    os.chdir('/content/Vietnamese-asr')
    !git pull origin main
else:
    print("Cloning repository...")
    !git clone https://github.com/CheeseThuong/Vietnamese-asr.git
    os.chdir('/content/Vietnamese-asr')

# CRITICAL: Add to Python path
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    
print(f"\nRepository ready")
print(f"Location: {os.getcwd()}")
print(f"Python path: {sys.path[0]}")

# Verify structure
print(f"\nRepository structure:")
!ls -la
print(f"\nsrc/ contents:")
!ls -la src/
```

---

## CELL 11: Verify Imports (Python)
```python
# Verify Python imports (DEBUG)
print("="*60)
print("Verifying imports...")
print("="*60)

try:
    from src.data.preprocessing import load_and_prepare_datasets
    from src.training.train_wav2vec2 import create_model, train_model
    print("All imports successful!")
except ModuleNotFoundError as e:
    print(f"Import failed: {e}")
    print("\nTroubleshooting:")
    print(f"  Current dir: {os.getcwd()}")
    print(f"  Python path: {sys.path[0]}")
    print(f"  src/ exists: {os.path.exists('src')}")
    
print("="*60)
```

---

## CELL 12: Check Dataset (Markdown)
```markdown
## 5. Check Dataset

**Upload instructions:**
1. **On local machine**, run: `python convert_to_relative_paths.py`
2. Upload 3 files from `processed_data_vivos/` to Google Drive
3. Put in: `MyDrive/VietnameseASR/data/`

**IMPORTANT**: Files must have **relative paths** (e.g., `Data/vivos/vivos/train/...`) NOT absolute paths (`D:\Projects\...`)
```

---

## CELL 13: Check Dataset Files (Python)
```python
from pathlib import Path
import json

# Dataset path
DATA_DIR = Path(f"{DRIVE_ROOT}/data")

# Check files
required_files = ['train.jsonl', 'validation.jsonl', 'test.jsonl']
missing = [f for f in required_files if not (DATA_DIR / f).exists()]

if missing:
    print("Missing dataset files:")
    for f in missing:
        print(f"   - {f}")
    print(f"\nExpected location: {DATA_DIR}")
    print("\nUpload dataset files to Google Drive first!")
else:
    print("All dataset files found!")
    # Count samples and check paths
    for file in required_files:
        filepath = DATA_DIR / file
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            count = len(lines)
            
        # Check sample path
        if lines:
            sample = json.loads(lines[0])
            audio_path = sample['audio_path']
            is_relative = not os.path.isabs(audio_path)
            path_type = "relative" if is_relative else "ABSOLUTE"
            
            print(f"   - {file}: {count:,} samples ({path_type})")
            print(f"     Sample path: {audio_path[:80]}...")
            
            # Warning if absolute paths found
            if not is_relative:
                print(f"     WARNING: Absolute paths detected! Will fail on Colab.")
                print(f"     Run 'python convert_to_relative_paths.py' locally first!")
        else:
            print(f"   - {file}: {count:,} samples")
```

---

## CELL 14: Setup Audio Dataset (Python)
```python
# Setup dataset in Colab workspace
print("Setting up dataset in Colab workspace...")

# Create Data folder in Colab
COLAB_DATA_DIR = Path("/content/Vietnamese-asr/Data")
COLAB_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Symlink vivos folder from Drive (FAST - instant access)
DRIVE_VIVOS = Path(f"{DRIVE_ROOT}/vivos")
COLAB_VIVOS = COLAB_DATA_DIR / "vivos"

if not DRIVE_VIVOS.exists():
    print(f"VIVOS audio folder not found at: {DRIVE_VIVOS}")
    print(f"Make sure to upload the 'vivos' folder to Drive")
    print(f"   Expected structure: {DRIVE_ROOT}/vivos/vivos/train/waves/...")
    raise FileNotFoundError(f"Audio dataset not found: {DRIVE_VIVOS}")

# Remove old copy/symlink if exists
if COLAB_VIVOS.exists() or COLAB_VIVOS.is_symlink():
    if COLAB_VIVOS.is_symlink():
        COLAB_VIVOS.unlink()  # Remove symlink
        print("Removed old symlink")
    else:
        import shutil
        shutil.rmtree(COLAB_VIVOS)  # Remove directory
        print("Removed old copy")

# Create symlink (instant - no copy needed!)
print(f"Creating symlink...")
print(f"   Source: {DRIVE_VIVOS}")
print(f"   Destination: {COLAB_VIVOS}")

COLAB_VIVOS.symlink_to(DRIVE_VIVOS)
print(f"Symlink created successfully!")

# Verify audio files
if COLAB_VIVOS.exists():
    # Count audio files
    wav_files = list(COLAB_VIVOS.rglob("*.wav"))
    print(f"\nAudio files ready: {len(wav_files):,} WAV files")
    print(f"Location: {COLAB_VIVOS}")
    
    # Show sample structure
    print(f"\nSample structure:")
    sample_files = list(COLAB_VIVOS.rglob("*.wav"))[:3]
    for f in sample_files:
        print(f"   {f.relative_to(COLAB_DATA_DIR)}")
else:
    print(f"\nAudio files verification failed!")
    raise FileNotFoundError(f"Cannot access: {COLAB_VIVOS}")
```

---

## CELL 15: Training Configuration (Markdown)
```markdown
## 6. Training Configuration
```

---

## CELL 16: Training Configuration (Python)
```python
import json

# Configuration - Optimized for Colab GPU
config = {
    'pretrained_model': 'nguyenvulebinh/wav2vec2-base-vietnamese-250h',
    'num_train_epochs': 30,          # Number of epochs
    'batch_size': 16,                # GPU T4 ~ 16GB RAM
    'gradient_accumulation_steps': 1,
    'learning_rate': 3e-4,
    'use_fp16': True,                # Mixed precision training
    'apply_quantization': False,     # Don't quantize during training
    'save_steps': 500,               # Save checkpoint every 500 steps
    'eval_steps': 500,               # Evaluate every 500 steps
}

# Output directories
OUTPUT_DIR = Path(f"{DRIVE_ROOT}/models/wav2vec2-vietnamese")
FINAL_MODEL_DIR = Path(f"{DRIVE_ROOT}/final_model")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Save config
with open(OUTPUT_DIR / 'config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("Configuration:")
for key, value in config.items():
    print(f"   - {key}: {value}")
print(f"\nOutput: {OUTPUT_DIR}")
```

---

## CELL 17: RELOAD Modules (Markdown)
```markdown
## 7. Reload Latest Code (if needed)

**IMPORTANT**: Run this cell if you:
- Updated code on GitHub
- Need to use latest bug fixes
- Got errors about missing columns

This will:
1. Pull latest code from GitHub
2. Clear old Python modules from memory
3. Reload modules with new code
```

---

## CELL 18: RELOAD Modules (Python)
```python
# Pull latest code from GitHub and reload modules
import os
import sys

os.chdir('/content/Vietnamese-asr')

print("Pulling latest code from GitHub...")
!git pull origin main

print("\nReloading Python modules...")

# Remove old modules from cache
modules_to_reload = [
    'src.data.preprocessing',
    'src.training.train_wav2vec2'
]

for mod in modules_to_reload:
    if mod in sys.modules:
        del sys.modules[mod]
        print(f"Cleared: {mod}")

# Re-import with new code
from src.data.preprocessing import load_and_prepare_datasets
from src.training.train_wav2vec2 import create_model, train_model

print("\nModules reloaded with new code!")
print("\n" + "="*60)
print("NEXT STEP: Run Cell 20 (Load Processor & Datasets)")
print("="*60)
```

---

## CELL 19: Load Processor & Datasets (Markdown)
```markdown
## 8. Load Processor & Datasets

**CRITICAL**: This cell MUST be run:
- After pulling new code (Cell 18)
- Before training (Cell 23)

This loads data with the correct format for training.
```

---

## CELL 20: Load Processor & Datasets (Python)
```python
from transformers import Wav2Vec2Processor
from src.data.preprocessing import load_and_prepare_datasets

print("Loading processor...")
processor = Wav2Vec2Processor.from_pretrained(config['pretrained_model'])

print("\nLoading datasets...")
print("This may take 5-10 minutes...")

# Change to Vietnamese-asr directory to use relative paths
os.chdir('/content/Vietnamese-asr')
print(f"Working directory: {os.getcwd()}")

train_dataset, val_dataset, test_dataset = load_and_prepare_datasets(
    str(DATA_DIR / 'train.jsonl'),
    str(DATA_DIR / 'validation.jsonl'),
    str(DATA_DIR / 'test.jsonl'),
    processor
)

print(f"\nDatasets loaded:")
print(f"   - Train: {len(train_dataset):,} samples")
print(f"   - Validation: {len(val_dataset):,} samples")
print(f"   - Test: {len(test_dataset):,} samples")
```

---

## CELL 21: Verify Dataset Columns (Python)
```python
# DEBUG: Verify dataset columns
print("=" * 60)
print("Dataset Columns Verification")
print("=" * 60)

# Check train dataset
print("\nTrain Dataset:")
print(f"   Columns: {train_dataset.column_names}")
print(f"   Features: {train_dataset.features}")

# Check sample
if len(train_dataset) > 0:
    sample = train_dataset[0]
    print(f"\nSample keys: {list(sample.keys())}")
    print(f"   - input_values shape: {len(sample['input_values']) if 'input_values' in sample else 'MISSING!'}")
    print(f"   - labels shape: {len(sample['labels']) if 'labels' in sample else 'MISSING!'}")
    
    # Verify NO old columns
    old_columns = ['audio_path', 'transcript', 'speaker_id', 'dataset']
    found_old = [col for col in old_columns if col in sample]
    if found_old:
        print(f"\nWARNING: Old columns still present: {found_old}")
        print("SOLUTION: Re-run Cell 18 (Reload modules) then Cell 20 (Load datasets)")
    else:
        print(f"\nOK: No old columns found")
        print("Dataset is ready for training!")

print("=" * 60)
```

---

## CELL 22: Create Model (Markdown)
```markdown
## 9. Create Model
```

---

## CELL 23: Create Model (Python)
```python
from src.training.train_wav2vec2 import create_model

print("Creating model...")
vocab_size = len(processor.tokenizer)
model = create_model(vocab_size, processor, config['pretrained_model'])

# Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Model info
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nModel ready on {device}")
print(f"   - Total parameters: {total_params:,}")
print(f"   - Trainable: {trainable_params:,}")
print(f"   - Frozen: {total_params - trainable_params:,}")
```

---

## CELL 24: Start Training (Markdown)
```markdown
## 10. Start Training

**Estimated time: 15-20 hours on T4 GPU**

IMPORTANT:
- Keep this tab open
- Colab free timeout: ~12 hours
- Checkpoints auto-saved to Drive every 500 steps
```

---

## CELL 25: Start Training (Python)
```python
from src.training.train_wav2vec2 import train_model

print("="*60)
print("Starting Training...")
print("="*60)
print("\nIMPORTANT:")
print("   - Keep this tab open!")
print("   - Colab timeout: ~12 hours")
print("   - Checkpoints auto-saved to Drive every 500 steps")
print("\n" + "="*60 + "\n")

# Train
trainer = train_model(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    processor=processor,
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=config['num_train_epochs'],
    batch_size=config['batch_size'],
    gradient_accumulation_steps=config['gradient_accumulation_steps'],
    learning_rate=config['learning_rate'],
    use_fp16=config['use_fp16']
)
```

---

## CELL 26: Save Final Model (Markdown)
```markdown
## 11. Save Final Model
```

---

## CELL 27: Save Final Model (Python)
```python
print("Saving final model...")

# Save model
trainer.save_model(str(FINAL_MODEL_DIR))
processor.save_pretrained(str(FINAL_MODEL_DIR))

# Save training history
import pandas as pd
if hasattr(trainer.state, 'log_history'):
    history_df = pd.DataFrame(trainer.state.log_history)
    history_df.to_csv(f"{DRIVE_ROOT}/training_history.csv", index=False)
    print(f"Training history saved")

print(f"\nTraining completed!")
print(f"Final model: {FINAL_MODEL_DIR}")
print(f"\nModel saved to Google Drive, you can:")
print(f"   1. Download to local machine from Drive")
print(f"   2. Use directly from Drive in other notebooks")
print(f"   3. Upload to HuggingFace Hub")
```

---

## CELL 28-32: Monitor Training (Optional)

### CELL 28 (Markdown)
```markdown
## 12. Monitor Training (Optional)

Run these cells during training to monitor progress
```

### CELL 29: TensorBoard (Python)
```python
# TensorBoard
%load_ext tensorboard
%tensorboard --logdir {OUTPUT_DIR}/runs
```

### CELL 30: GPU Monitoring (Python)
```python
# GPU monitoring
!nvidia-smi
```

### CELL 31: Check Checkpoints (Python)
```python
# Check latest checkpoint
!ls -lh {OUTPUT_DIR}/checkpoint-*/ | tail -5
```

---

## CELL 32-34: Test Model (After Training)

### CELL 32 (Markdown)
```markdown
## 13. Test Model (After Training)
```

### CELL 33: Load Model (Python)
```python
# Load trained model
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf

model = Wav2Vec2ForCTC.from_pretrained(str(FINAL_MODEL_DIR))
processor = Wav2Vec2Processor.from_pretrained(str(FINAL_MODEL_DIR))
model = model.to(device)
model.eval()

print("Model loaded for inference")
```

### CELL 34: Transcribe Function (Python)
```python
# Transcribe audio file
def transcribe(audio_path):
    # Load audio
    speech, sr = sf.read(audio_path)
    
    # Process
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Decode
    pred_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(pred_ids)[0]
    
    return transcription

# Test example
# audio_file = "/path/to/audio.wav"
# result = transcribe(audio_file)
# print(f"Transcription: {result}")
```

---

## CELL 35: Tips (Markdown)
```markdown
## Tips

### Avoid Colab timeout:
- Training takes ~15-20h, Colab free timeout after ~12h
- **Solution:** Split training into multiple sessions
  ```python
  # Session 1: Train 10 epochs
  config['num_train_epochs'] = 10
  
  # Session 2: Resume from checkpoint, train additional 10 epochs
  config['resume_from_checkpoint'] = str(OUTPUT_DIR / 'checkpoint-5000')
  config['num_train_epochs'] = 20
  ```

### Colab Pro:
- Timeout: ~24h
- Better GPU: A100/V100
- Training time: ~8-10h

### Auto-save to Drive:
Checkpoints auto-saved to Drive every 500 steps, safe if Colab disconnects!
```

---

## CRITICAL EXECUTION ORDER

**First Time Setup (Cells 1-16):**
```
1-16: Setup environment, mount Drive, install packages, clone repo, configure
```

**Every Time You Train (Cells 17-25):**
```
17-18: Pull latest code & reload modules (if code was updated)
19-20: Load processor & datasets (MUST run after Cell 18)
21:    Verify dataset columns (check before training)
22-23: Create model
24-25: Start training
```

**After Training (Cells 26-34):**
```
26-27: Save final model
28-31: Monitor (optional, during training)
32-34: Test model
```

---

## COMMON MISTAKES TO AVOID

1. **Running Cell 20 BEFORE Cell 18**
   - Cell 18 reloads code
   - Cell 20 loads dataset
   - MUST reload code FIRST, then load dataset

2. **Not re-running Cell 20 after Cell 18**
   - Cell 18 updates code in memory
   - Cell 20 creates NEW dataset with new code
   - Old dataset in memory still has old format

3. **Skipping Cell 21 (Verify)**
   - Always check columns before training
   - Should see ONLY: ['input_values', 'labels']
   - If see: ['audio_path', 'transcript'...] -> Re-run Cell 18 then Cell 20

4. **Training with old dataset**
   - If you get KeyError during training
   - Re-run: Cell 18 -> Cell 20 -> Cell 21 (verify) -> Cell 23 -> Cell 25

---

## TROUBLESHOOTING

### If Cell 21 shows OLD columns:
```
WARNING: Old columns still present: ['audio_path', 'transcript', 'speaker_id', 'dataset']
```
**Solution:**
1. Re-run Cell 18 (Reload modules)
2. Re-run Cell 20 (Load datasets)
3. Re-run Cell 21 (Verify) - should now show ONLY ['input_values', 'labels']
4. Continue to Cell 23

### If training shows KeyError:
```
KeyError: 'input_values'
```
**Solution:** Same as above - reload dataset

### If audio files missing:
```
Error loading audio Data/vivos/vivos/test/waves/...
```
**Solution:** This is OK! Dataset has a few missing files, they are automatically skipped.

---

## SUMMARY CHECKLIST

Before training, verify:
- [ ] Cell 13: All JSONL files found with RELATIVE paths
- [ ] Cell 14: Audio files symlinked successfully (~11,000+ files)
- [ ] Cell 18: Code reloaded (if updated)
- [ ] Cell 20: Datasets loaded successfully
- [ ] Cell 21: Columns are ONLY ['input_values', 'labels']
- [ ] Cell 23: Model created successfully
- [ ] Cell 25: Training starts without KeyError

If ANY checkbox fails, re-run from Cell 18.
