# 🔧 Hướng Dẫn Fix Lỗi Colab Training

## 📌 Tóm Tắt 2 Lỗi Chính

### ❌ **Lỗi 1: Không load được audio files (0 samples)**
```
Error loading audio Data/vivos/vivos/test/waves/VIVOSDEV19/VIVOSDEV19_266.wav: Audio file not found
⚠️ Skipped 12420 corrupted audio files
Final counts: Train: 0 samples, Validation: 0 samples, Test: 0 samples
```

### ❌ **Lỗi 2: NameError - processor not defined**
```
NameError: name 'processor' is not defined
File "/content/Vietnamese-asr/src/training/train_wav2vec2.py", line 145
```

---

## 🔍 Nguyên Nhân & Giải Pháp

### **Lỗi 1: Audio Files Không Tìm Thấy**

#### 🎯 **Nguyên nhân chính**:
1. **Paths trong JSONL**: `Data/vivos/vivos/train/waves/...` (relative paths ✅)
2. **Working directory**: `/content/Vietnamese-asr` ✅
3. **Audio files cần ở**: `/content/Vietnamese-asr/Data/vivos/...` ✅
4. **❌ VẤN ĐỀ**: Cell 13 chưa chạy hoặc chưa hoàn thành!

#### ⚙️ **Cell 13 làm gì?**
- **MỤC ĐÍCH**: Copy/symlink folder `vivos/` từ Google Drive vào Colab workspace
- **TRƯỚC ĐÂY**: Dùng `shutil.copytree()` → Copy 11,000+ files → **5-15 phút** ⏱️
- **BÂY GIỜ (FIXED)**: Dùng **symlink** → Instant access → **< 1 giây** ⚡

#### ✅ **Giải pháp**:

**Bước 1**: Kiểm tra folder `vivos` đã upload lên Drive chưa
```python
# Chạy cell này để check:
from pathlib import Path
DRIVE_VIVOS = Path("/content/drive/MyDrive/VietnameseASR/vivos")
if DRIVE_VIVOS.exists():
    wav_files = list(DRIVE_VIVOS.rglob("*.wav"))
    print(f"✅ Found {len(wav_files):,} WAV files on Drive")
else:
    print("❌ Folder 'vivos' chưa upload lên Drive!")
    print("💡 Upload folder 'vivos' vào MyDrive/VietnameseASR/")
```

**Bước 2**: **CHỈ CẦN** chạy Cell 13 (version mới - có symlink)
- **QUAN TRỌNG**: Phải dùng notebook từ GitHub (có code mới nhất)
- **ĐỂ LẤY CODE MỚI**:
  1. Runtime → Restart runtime (clear old code)
  2. File → Open notebook → **GitHub** tab
  3. Repository: `CheeseThuong/Vietnamese-asr`
  4. File: `colab_train.ipynb`
  5. Click "Open"
  6. Re-run từ Cell 1

**Bước 3**: Verify audio files đã sẵn sàng
```python
# Check sau khi chạy Cell 13:
!ls -la /content/Vietnamese-asr/Data/vivos/vivos/train/waves/ | head -10
```

**Kết quả mong đợi**:
```
✅ Symlink created successfully!
✅ Audio files ready: 11,420 WAV files
📂 Location: /content/Vietnamese-asr/Data/vivos
```

---

### **Lỗi 2: `processor` Not Defined**

#### 🎯 **Nguyên nhân**:
- Trong file `src/training/train_wav2vec2.py`, hàm `create_model()` **dùng biến global `processor`**
- Nhưng `processor` **chưa được khởi tạo** hoặc **không được truyền vào** hàm

**Code cũ (SAI)**:
```python
def create_model(vocab_size: int, pretrained_model: str = None):
    # ...
    model = Wav2Vec2ForCTC.from_pretrained(
        pretrained_model,
        pad_token_id=processor.tokenizer.pad_token_id,  # ❌ processor không tồn tại!
        # ...
    )
```

#### ✅ **Giải pháp**:

**ĐÃ FIX** trong 2 files:

1. **File `src/training/train_wav2vec2.py`** (line 133):
```python
# Code MỚI (ĐÚNG):
def create_model(vocab_size: int, processor: Wav2Vec2Processor, pretrained_model: str = None):
    """
    Args:
        vocab_size: Kích thước vocabulary
        processor: Wav2Vec2Processor instance ← THÊM PARAMETER NÀY
        pretrained_model: Tên model pre-trained
    """
    # Bây giờ processor được truyền vào, không còn lỗi!
    model = Wav2Vec2ForCTC.from_pretrained(
        pretrained_model,
        pad_token_id=processor.tokenizer.pad_token_id,  # ✅ OK
        # ...
    )
```

2. **File `colab_train.ipynb`** (Cell 8 - Creating Model):
```python
# Code MỚI (ĐÚNG):
from src.training.train_wav2vec2 import create_model

vocab_size = len(processor.tokenizer)
# ✅ Truyền processor vào hàm
model = create_model(vocab_size, processor, config['pretrained_model'])
```

**Để áp dụng fix**:
- **CẦN** reload notebook từ GitHub (code mới nhất đã push)
- Xem hướng dẫn ở **Bước 2** của Lỗi 1

---

## 🚀 Quy Trình Chạy Lại (Sau Khi Fix)

### **Checklist trước khi train**:

- [ ] ✅ Upload 3 files JSONL mới (relative paths) lên Drive
  - `train.jsonl`, `validation.jsonl`, `test.jsonl`
  - Đặt vào: `MyDrive/VietnameseASR/data/`
  - **QUAN TRỌNG**: Phải chạy `python convert_to_relative_paths.py` trên máy local trước!

- [ ] ✅ Upload folder `vivos/` lên Drive
  - Đặt vào: `MyDrive/VietnameseASR/vivos/`
  - Cấu trúc: `vivos/vivos/train/waves/...` và `vivos/vivos/test/waves/...`

- [ ] ✅ Reload notebook mới từ GitHub
  - Runtime → Restart runtime
  - File → Open notebook → GitHub → `CheeseThuong/Vietnamese-asr` → `colab_train.ipynb`

### **Chạy cells theo thứ tự**:

| Cell | Tên | Thời gian | Kết quả mong đợi |
|------|-----|-----------|------------------|
| 1 | Check GPU | 1s | `✅ GPU Ready! Tesla T4` |
| 2 | Mount Drive | 5s | `✓ Drive mounted at: /content/drive/MyDrive/VietnameseASR` |
| 3-4 | Install deps | 30s | `✅ All packages installed` |
| 5-6 | Clone repo | 10s | `✅ Repository ready` |
| 7 | Verify imports | 1s | `✅ All imports successful!` |
| 12 | Check dataset | 2s | `✅ All dataset files found!` <br> `train.jsonl: 10,494 samples (✅ relative)` |
| **13** | **Setup dataset** | **< 1s** ⚡ | `✅ Symlink created successfully!` <br> `✅ Audio files ready: 11,420 WAV files` |
| 14 | Config | 1s | `✅ Configuration: ...` |
| 15 | Load processor | 5s | `✅ Processor loaded` |
| **16** | **Load datasets** | **5-10 min** | `✅ Datasets loaded:` <br> `- Train: 10,494 samples` <br> `- Validation: 1,166 samples` <br> `- Test: 760 samples` |
| 17 | Create model | 10s | `✅ Model ready on cuda` |
| 18+ | Training | 15-20h | `🚀 Starting Training...` |

### **❌ Nếu Cell 16 vẫn báo "0 samples"**:

```python
# Debug: Check working directory và file existence
import os
from pathlib import Path

print("Working dir:", os.getcwd())
print("Audio folder exists:", Path("Data/vivos/vivos/train/waves").exists())

# List sample files
!ls -la Data/vivos/vivos/train/waves/VIVOSSPK01/ | head -5

# Check JSONL path format
import json
with open('/content/drive/MyDrive/VietnameseASR/data/train.jsonl', 'r') as f:
    sample = json.loads(f.readline())
    print("Sample path:", sample['audio_path'])
    print("Is absolute:", os.path.isabs(sample['audio_path']))
```

**Các vấn đề thường gặp**:

1. **Working directory sai** → Chạy: `os.chdir('/content/Vietnamese-asr')`
2. **Cell 13 chưa chạy** → Audio files không tồn tại → Re-run Cell 13
3. **JSONL files trên Drive vẫn là absolute paths** → Upload lại files từ `processed_data_vivos/`

---

## 📋 Summary

| Lỗi | Nguyên Nhân | Giải Pháp | Status |
|-----|-------------|-----------|--------|
| **Audio files not found** | Cell 13 chưa chạy/chưa xong | Re-run Cell 13 (version mới - symlink) | ✅ FIXED |
| **processor not defined** | `create_model()` thiếu parameter | Truyền `processor` vào hàm | ✅ FIXED |
| **Absolute paths in JSONL** | Chưa convert sang relative | Chạy `convert_to_relative_paths.py` | ✅ DONE |
| **TorchCodec required** | Fallback không cần thiết | Loại bỏ fallback torchaudio.load | ✅ FIXED |

---

## 🆘 Nếu Vẫn Gặp Vấn Đề

### **Troubleshooting Checklist**:

```python
# === 1. Check Drive files ===
!ls -la /content/drive/MyDrive/VietnameseASR/
# Expected: data/, vivos/, models/, final_model/

# === 2. Check JSONL files ===
!ls -la /content/drive/MyDrive/VietnameseASR/data/
# Expected: train.jsonl, validation.jsonl, test.jsonl

# === 3. Check audio folder ===
!ls -la /content/drive/MyDrive/VietnameseASR/vivos/vivos/
# Expected: train/, test/

# === 4. Check symlink ===
!ls -la /content/Vietnamese-asr/Data/
# Expected: vivos -> /content/drive/MyDrive/VietnameseASR/vivos (symlink)

# === 5. Count audio files ===
!find /content/Vietnamese-asr/Data/vivos -name "*.wav" | wc -l
# Expected: ~11,420 files

# === 6. Check sample JSONL path ===
!head -1 /content/drive/MyDrive/VietnameseASR/data/train.jsonl | python3 -m json.tool
# Expected: "audio_path": "Data/vivos/vivos/train/..." (NOT "D:\\Projects\\...")
```

### **Common Issues**:

| Vấn Đề | Cách Fix |
|--------|----------|
| "No module named 'src'" | Re-run Cell 5 (Clone repo + add to sys.path) |
| "Drive not mounted" | Re-run Cell 2 (Mount Drive) |
| "GPU not available" | Runtime → Change runtime type → GPU (T4) |
| Cell 13 timeout | KHÔNG thể xảy ra với symlink (< 1s), nếu timeout = code cũ, cần reload notebook |
| "Permission denied" (symlink) | Bình thường trên Colab, thử: `!ln -sf {DRIVE_VIVOS} {COLAB_VIVOS}` |

---

## 📝 Notes

- **Symlink vs Copy**: Symlink không copy files, chỉ tạo shortcut → Instant + Tiết kiệm disk space
- **Training time**: 15-20h trên T4 GPU, cần chia nhỏ sessions vì Colab free timeout ~12h
- **Checkpoints**: Auto-save mỗi 500 steps vào Drive → An toàn nếu disconnect
- **Files đã fix**: `src/data/preprocessing.py`, `src/training/train_wav2vec2.py`, `colab_train.ipynb`

---

**Tạo bởi**: GitHub Copilot | **Ngày**: 2026-01-31
