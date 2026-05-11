# 🚀 Hướng dẫn Training trên Google Colab

## 📋 Giới thiệu

Google Colab cung cấp **GPU miễn phí** (Tesla T4, 15GB RAM) lý tưởng cho training ASR model.

**So sánh:**
- **Local CPU**: ~6 ngày ❌
- **Colab GPU**: ~15-20 giờ ✅
- **Chi phí**: Miễn phí (Colab free) hoặc $10/tháng (Colab Pro)

---

## 🎯 Quick Start

### Bước 1: Chuẩn bị Dataset (Trên máy local)

```bash
# Chạy script chuẩn bị dataset
python prepare_vivos.py
```

Sẽ tạo folder `processed_data_vivos/` với:
- `train.jsonl` (~10,494 samples)
- `validation.jsonl` (~1,166 samples)
- `test.jsonl` (~760 samples)

### Bước 2: Upload Dataset lên Google Drive

1. Mở Google Drive
2. Tạo folder: `MyDrive/VietnameseASR/data/`
3. Upload 3 files `.jsonl` vào folder này
4. *(Optional)* Upload folder `src/` vào `MyDrive/VietnameseASR/code/`

**Cấu trúc khuyến nghị:**
```
MyDrive/
└── VietnameseASR/
    ├── data/              # Dataset
    │   ├── train.jsonl
    │   ├── validation.jsonl
    │   └── test.jsonl
    ├── code/              # Source code (optional nếu dùng GitHub)
    │   └── src/
    ├── models/            # Auto-created khi training
    └── final_model/       # Auto-created sau training
```

### Bước 3: Mở Notebook trên Colab

**Option 1: Upload file .ipynb**
1. Mở https://colab.research.google.com
2. File → Upload notebook
3. Chọn `colab_train.ipynb`

**Option 2: Từ Google Drive**
1. Upload `colab_train.ipynb` vào Drive
2. Double-click file → Open with → Google Colaboratory

**Option 3: Từ GitHub**
1. Push code lên GitHub repo
2. Colab → File → Open notebook → GitHub
3. Nhập repo URL

### Bước 4: Bật GPU

**QUAN TRỌNG!** Phải bật GPU trước khi chạy:

1. Runtime → Change runtime type
2. Hardware accelerator → **GPU**
3. GPU type → **T4** (free) hoặc **A100** (Pro)
4. Save

Verify GPU:
```python
import torch
print(torch.cuda.is_available())  # Phải là True
print(torch.cuda.get_device_name(0))  # Tesla T4
```

### Bước 5: Chạy từng cell

**Chạy tuần tự từ trên xuống:**

1. **Cell 1**: Check GPU ✅
2. **Cell 2**: Mount Google Drive ✅
3. **Cell 3-4**: Install dependencies ✅
4. **Cell 5**: Upload/Clone source code ✅
5. **Cell 6**: Check dataset ✅
6. **Cell 7**: Config ✅
7. **Cell 8**: Load processor & datasets ✅
8. **Cell 9**: Create model ✅
9. **Cell 10**: **START TRAINING** 🚀
10. **Cell 11**: Save final model ✅

---

## ⚙️ Configuration

### Config mặc định (GPU T4):

```python
config = {
    'pretrained_model': 'nguyenvulebinh/wav2vec2-base-vietnamese-250h',
    'num_train_epochs': 30,
    'batch_size': 16,              # T4 (15GB) ~ batch 16
    'gradient_accumulation_steps': 1,
    'learning_rate': 3e-4,
    'use_fp16': True,              # Mixed precision
    'save_steps': 500,
    'eval_steps': 500,
}
```

### Điều chỉnh cho GPU khác nhau:

| GPU | VRAM | Batch Size | Time (30 epochs) |
|-----|------|------------|------------------|
| T4 (free) | 15GB | 12-16 | ~18-20h |
| V100 (Pro) | 16GB | 16-20 | ~12-15h |
| A100 (Pro) | 40GB | 24-32 | ~8-10h |

**Nếu Out of Memory:**
```python
config['batch_size'] = 8
config['gradient_accumulation_steps'] = 2
```

---

## 📊 Monitoring Training

### TensorBoard (Real-time)

```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/VietnameseASR/models/wav2vec2-vietnamese/runs
```

### GPU Usage

```python
!nvidia-smi
```

### Training Progress

```python
# View logs
!tail -100 /content/drive/MyDrive/VietnameseASR/models/wav2vec2-vietnamese/trainer_state.json

# List checkpoints
!ls -lh /content/drive/MyDrive/VietnameseASR/models/wav2vec2-vietnamese/checkpoint-*/
```

### Training History

```python
import pandas as pd
history = pd.read_csv("/content/drive/MyDrive/VietnameseASR/training_history.csv")

# Plot WER over time
import matplotlib.pyplot as plt
plt.plot(history['eval_wer'])
plt.title('Word Error Rate')
plt.xlabel('Step')
plt.ylabel('WER')
plt.show()
```

---

## 🔄 Xử lý Timeout

### Vấn đề:
- **Colab Free**: Timeout sau ~12 giờ
- **Training cần**: ~15-20 giờ
- → Bị ngắt giữa chừng!

### Giải pháp 1: Chia nhỏ training

**Session 1 (10 epochs):**
```python
config['num_train_epochs'] = 10
# Run training...
# Sau 10 epochs, checkpoint auto-saved
```

**Session 2 (epochs 11-20):**
```python
# Mở notebook mới hoặc restart
config['num_train_epochs'] = 20
config['resume_from_checkpoint'] = '/content/drive/MyDrive/VietnameseASR/models/wav2vec2-vietnamese/checkpoint-6000'
# Continue training...
```

**Session 3 (epochs 21-30):**
```python
config['num_train_epochs'] = 30
config['resume_from_checkpoint'] = '/content/drive/.../checkpoint-12000'
```

### Giải pháp 2: Keep-alive script

```python
# Cell riêng - Chạy song song với training
import time
from google.colab import output

while True:
    time.sleep(60)  # Mỗi 1 phút
    output.clear()
    print("Keep-alive ping")
```

### Giải pháp 3: Colab Pro

- $10/tháng
- Timeout: ~24 giờ (đủ cho 1 lần chạy)
- Better GPU: V100/A100
- Priority access

---

## 💾 Checkpointing Strategy

### Auto-save mỗi 500 steps:

```python
training_args = TrainingArguments(
    save_steps=500,              # Lưu mỗi 500 steps
    save_total_limit=2,          # Chỉ giữ 2 checkpoint gần nhất
    load_best_model_at_end=True, # Load model tốt nhất
)
```

### Manual checkpoint:

```python
# Lưu tại bất kỳ lúc nào
trainer.save_model("/content/drive/MyDrive/VietnameseASR/checkpoint_manual")
```

### Resume từ checkpoint:

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    # ... other args ...
    resume_from_checkpoint="/content/drive/.../checkpoint-5000"
)
```

---

## 📥 Download Model

### Option 1: Download từ Drive

1. Training xong → Model saved to Drive
2. Vào Drive → Download folder `final_model/`
3. Giải nén trên máy local

### Option 2: Download trực tiếp từ Colab

```python
# Zip model
!zip -r final_model.zip /content/drive/MyDrive/VietnameseASR/final_model/

# Download
from google.colab import files
files.download('final_model.zip')
```

### Option 3: Upload lên HuggingFace Hub

```python
from huggingface_hub import HfApi, create_repo

# Login (cần HF token)
!huggingface-cli login

# Create repo
repo_name = "your-username/wav2vec2-vietnamese-asr"
create_repo(repo_name, private=False)

# Upload
model.push_to_hub(repo_name)
processor.push_to_hub(repo_name)

print(f"✅ Uploaded to: https://huggingface.co/{repo_name}")
```

---

## 🧪 Test Model

### Test trên notebook:

```python
# Load model
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import torch

model = Wav2Vec2ForCTC.from_pretrained("/content/drive/.../final_model")
processor = Wav2Vec2Processor.from_pretrained("/content/drive/.../final_model")
model.eval()

# Transcribe
def transcribe(audio_path):
    speech, sr = sf.read(audio_path)
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]

# Test
result = transcribe("/path/to/audio.wav")
print(f"Kết quả: {result}")
```

### Upload audio để test:

```python
from google.colab import files

# Upload file
uploaded = files.upload()

# Get filename
audio_file = list(uploaded.keys())[0]

# Transcribe
result = transcribe(audio_file)
print(f"Transcription: {result}")
```

---

## ⚠️ Common Issues

### Issue 1: Runtime disconnected

**Nguyên nhân:** Colab timeout hoặc mất kết nối

**Giải pháp:**
- Checkpoints đã lưu vào Drive → An toàn!
- Resume từ checkpoint gần nhất:
```python
config['resume_from_checkpoint'] = '/content/drive/.../checkpoint-XXXX'
```

### Issue 2: Out of Memory (OOM)

**Nguyên nhân:** Batch size quá lớn

**Giải pháp:**
```python
# Giảm batch size
config['batch_size'] = 8

# Tăng gradient accumulation (giữ effective batch size)
config['gradient_accumulation_steps'] = 2

# Gradient checkpointing (tiết kiệm memory)
config['gradient_checkpointing'] = True
```

### Issue 3: Dataset not found

**Nguyên nhân:** Chưa mount Drive hoặc đường dẫn sai

**Giải pháp:**
```python
# Re-mount Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Check path
!ls /content/drive/MyDrive/VietnameseASR/data/
```

### Issue 4: Training quá chậm

**Nguyên nhân:** Không dùng GPU hoặc chạy trên CPU

**Giải pháp:**
```python
# Check GPU
assert torch.cuda.is_available(), "GPU not enabled!"

# Verify runtime
# Runtime → Change runtime type → GPU
```

---

## 📈 Expected Results

### Training Metrics:

| Metric | Initial | After 10 epochs | After 30 epochs |
|--------|---------|-----------------|-----------------|
| **WER** | ~50% | ~25-30% | ~15-20% |
| **CER** | ~30% | ~15-18% | ~8-12% |
| **Loss** | ~10 | ~2-3 | ~0.5-1.0 |

### Training Time:

- **GPU T4**: ~18-20 giờ (30 epochs)
- **GPU V100**: ~12-15 giờ
- **GPU A100**: ~8-10 giờ

### Model Size:

- **Original**: ~400MB
- **After quantization**: ~100MB (75% reduction)

---

## 🎯 Tips & Best Practices

### 1. Start Small, Scale Up

```python
# Test với subset nhỏ trước (5-10 phút)
train_dataset = train_dataset.select(range(100))
config['num_train_epochs'] = 2

# Sau khi confirm code chạy OK → Train full
```

### 2. Monitor Regularly

- Check TensorBoard mỗi 1-2 giờ
- Theo dõi WER/Loss trends
- Stop sớm nếu overfit

### 3. Use Early Stopping

```python
from transformers import EarlyStoppingCallback

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,  # Stop sau 3 evals không improve
    early_stopping_threshold=0.01
)

trainer = Trainer(..., callbacks=[early_stopping])
```

### 4. Save Intermediate Results

```python
# Lưu mỗi 1000 steps thay vì 500 nếu muốn tiết kiệm disk
training_args.save_steps = 1000
```

### 5. Backup to Multiple Locations

```python
# Sau training xong, copy sang nơi khác
!cp -r /content/drive/MyDrive/VietnameseASR/final_model /content/drive/MyDrive/Backups/
```

---

## 📚 Resources

- **Colab Docs**: https://colab.research.google.com/notebooks/intro.ipynb
- **HuggingFace Wav2Vec2**: https://huggingface.co/docs/transformers/model_doc/wav2vec2
- **Transformers Trainer**: https://huggingface.co/docs/transformers/main_classes/trainer

---

## 🆘 Need Help?

**Gặp vấn đề?** Check:
1. Runtime → Change runtime type → GPU ✅
2. Drive mounted ✅
3. Dataset files tồn tại ✅
4. GPU memory không quá tải ✅

**Still stuck?** Share error log để debug!

---

**Chúc bạn training thành công! 🚀**
