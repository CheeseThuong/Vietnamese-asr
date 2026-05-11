# Hướng dẫn Training trên Kaggle

## Tại sao chuyển sang Kaggle?

✅ **Ưu điểm vượt trội:**
- **Chạy background**: Tắt máy vẫn train! Không cần giữ tab mở
- **GPU miễn phí**: 30 giờ/tuần (T4 hoặc P100)
- **Auto-save**: Tất cả outputs tự động lưu
- **Email thông báo**: Khi notebook chạy xong
- **Stable**: Ít bị disconnect hơn Colab

❌ **Colab limitations:**
- Phải giữ tab mở suốt 15-20 giờ
- Free tier timeout ~12 giờ
- Tắt máy = mất tiến trình

---

## Bước 1: Chuẩn bị Dataset

### 1.1. Chạy script tạo relative paths (máy local)

```bash
cd "D:\Projects\do_an_tri_tue_nhan_tao"
python convert_to_relative_paths.py
```

Kết quả: 3 files JSONL với relative paths trong `processed_data_vivos/`

### 1.2. Upload Dataset lên Kaggle Datasets

**Đây là bước QUAN TRỌNG nhất!**

1. Truy cập: https://www.kaggle.com/datasets
2. Click **"New Dataset"** (góc trên bên phải)
3. Upload các files sau:
   - `processed_data_vivos/train.jsonl`
   - `processed_data_vivos/validation.jsonl`
   - `processed_data_vivos/test.jsonl`
   - **Toàn bộ folder** `Data/vivos/` (chứa audio files)

4. Đặt tên dataset: **"vietnamese-vivos-asr"** (hoặc tên bạn thích)
5. Title: "Vietnamese VIVOS ASR Dataset"
6. Subtitle: "Audio dataset for Vietnamese speech recognition"
7. Description (tùy chọn):
   ```
   Vietnamese speech recognition dataset from VIVOS corpus.
   
   Contains:
   - train.jsonl: 10,494 training samples
   - validation.jsonl: 1,166 validation samples
   - test.jsonl: 760 test samples
   - vivos/: Audio WAV files
   
   For training Wav2Vec2 Vietnamese ASR model.
   ```
8. Privacy: **Public** (để dễ chia sẻ) hoặc **Private** (nếu không muốn public)
9. Click **"Create"**

**Thời gian upload:** ~30 phút - 2 giờ (tùy tốc độ mạng)

**Cấu trúc dataset sau khi upload:**
```
/kaggle/input/vietnamese-vivos-asr/
├── train.jsonl
├── validation.jsonl
├── test.jsonl
└── vivos/
    └── vivos/
        ├── train/
        │   └── waves/
        │       └── *.wav (10,000+ files)
        └── test/
            └── waves/
                └── *.wav (800+ files)
```

---

## Bước 2: Tạo Kaggle Notebook

### 2.1. Upload notebook đã tạo

1. Truy cập: https://www.kaggle.com/code
2. Click **"New Notebook"**
3. Click menu **File → Import Notebook**
4. Upload file: `kaggle_train.ipynb` (file tôi vừa tạo)

### 2.2. Cấu hình Notebook Settings

**Bật các settings sau (rất quan trọng!):**

1. **Accelerator**: GPU T4 x2 (hoặc GPU P100)
   - Click **Settings** (icon bánh răng bên phải)
   - Accelerator → Chọn **GPU T4 x2**
   
2. **Internet**: ON
   - Settings → Internet → **Toggle ON**
   - Cần để pull code từ GitHub

3. **Persistence**: Session-only persistence
   - Mặc định đã bật

### 2.3. Add Dataset vào Notebook

1. Click **"Add Data"** (góc phải, icon +)
2. Search: "vietnamese-vivos-asr" (hoặc tên dataset bạn đặt)
3. Tìm dataset của bạn → Click **"Add"**
4. Dataset sẽ xuất hiện ở `/kaggle/input/vietnamese-vivos-asr/`

---

## Bước 3: Chạy Training

### 3.1. Chạy tất cả cells

**QUAN TRỌNG: Chạy từng cell theo thứ tự!**

```
Cell 1:  Check GPU & Environment (30s)
Cell 2:  Install Dependencies (2 phút)
Cell 3:  Clone Code from GitHub (30s) - INTERNET PHẢI BẬT!
Cell 4:  Setup Dataset (1 phút)
Cell 5:  Setup Audio Files (30s)
Cell 6:  Training Config (5s)
Cell 7:  Load Processor & Datasets (5-10 phút) ← QUAN TRỌNG!
Cell 8:  Verify Dataset (10s)
Cell 9:  Create Model (30s)
Cell 10: Pre-Training Check (10s) ← KIỂM TRA KỸ!
Cell 11: Start Training (15-20 giờ!) ← ĐÂY LÀ CELL CHÍNH
Cell 12: Save Final Model (2 phút)
```

### 3.2. Kiểm tra Pre-Training Check (Cell 10)

**PHẢI PASS TẤT CẢ CHECKS trước khi chạy Cell 11!**

Expected output:
```
======================================================================
PRE-TRAINING VERIFICATION
======================================================================

[1/4] Checking datasets...
      [✓] All datasets loaded
          Train: 10,494 samples
          Val:   1,166 samples
          Test:  760 samples

[2/4] Checking processor...
      [✓] Processor loaded
          Vocab size: 112

[3/4] Checking model...
      [✓] Model loaded
          Device: cuda:0

[4/4] Checking dataset structure...
      [✓] Correct structure

======================================================================
✓ VERIFICATION PASSED!
======================================================================

All checks passed! Ready for training.
Expected time: 15-20 hours on T4 GPU
```

Nếu có lỗi, **DỪNG NGAY** và fix trước khi tiếp tục!

### 3.3. Start Training (Cell 11)

Khi chạy cell này:
```python
trainer = train_model(...)  # Bắt đầu train 15-20 giờ!
```

**Bạn có thể:**
- ✅ Đóng tab này
- ✅ Tắt máy tính
- ✅ Đi ngủ
- ✅ Đi làm việc khác

**Kaggle sẽ:**
- Tiếp tục train tự động
- Lưu checkpoints mỗi 500 steps
- Email cho bạn khi xong
- Lưu tất cả outputs

---

## Bước 4: Monitoring (Tùy chọn)

### 4.1. Xem tiến trình

Quay lại notebook sau vài giờ:
- Click **"View Output"** để xem logs
- Training progress bar
- Loss values (should decrease)

### 4.2. GPU monitoring

Chạy cell 13 (optional):
```python
!nvidia-smi
```

Xem:
- GPU usage (should be ~90-100%)
- Memory usage
- Temperature

---

## Bước 5: Lấy Trained Model

### 5.1. Sau khi training xong

1. Kaggle gửi email: "Your notebook has finished running"
2. Truy cập notebook: https://www.kaggle.com/code/[your-username]/[notebook-name]
3. Click tab **"Output"** (bên phải)

### 5.2. Download model

Trong Output tab, bạn sẽ thấy:
```
/kaggle/working/
├── final_model/          ← Model đã train xong
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── preprocessor_config.json
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── models/               ← Checkpoints
│   └── wav2vec2-vietnamese/
│       ├── checkpoint-500/
│       ├── checkpoint-1000/
│       └── ...
└── training_history.csv  ← Training logs
```

**Download:**
1. Click vào `final_model` folder
2. Click **"Download"** button
3. File sẽ được nén thành `final_model.zip`

### 5.3. Sử dụng model

**Option 1: Local testing**
```python
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf

# Load model
model = Wav2Vec2ForCTC.from_pretrained("./final_model")
processor = Wav2Vec2Processor.from_pretrained("./final_model")

# Transcribe audio
audio, sr = sf.read("test.wav")
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
pred_ids = torch.argmax(logits, dim=-1)
transcript = processor.batch_decode(pred_ids)[0]
print(transcript)
```

**Option 2: Upload to HuggingFace Hub**
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./final_model",
    repo_id="your-username/wav2vec2-vietnamese-vivos",
    repo_type="model"
)
```

---

## Troubleshooting

### ❌ Lỗi: "Dataset not found"

**Nguyên nhân:** Chưa add dataset vào notebook

**Giải pháp:**
1. Click "Add Data" bên phải
2. Search dataset name
3. Click "Add"
4. Re-run cell 4

### ❌ Lỗi: "Internet disabled"

**Nguyên nhân:** Chưa bật Internet trong Settings

**Giải pháp:**
1. Settings → Internet → Toggle ON
2. Re-run cell 3 (Clone code)

### ❌ Lỗi: "GPU not available"

**Nguyên nhân:** Chưa chọn GPU accelerator

**Giải pháp:**
1. Settings → Accelerator → GPU T4 x2
2. Click "Save"
3. Notebook sẽ restart
4. Re-run tất cả cells

### ❌ Lỗi: "Old columns found: ['audio_path', 'transcript'...]"

**Nguyên nhân:** Code trên GitHub chưa được update

**Giải pháp:**
1. Local machine: 
   ```bash
   git add src/data/preprocessing.py
   git commit -m "Fix dataset preprocessing"
   git push
   ```
2. Kaggle cell 3: Re-run để pull latest code
3. Re-run cell 7 (Load datasets)
4. Re-run cell 10 (Pre-training check) → Should PASS now!

### ❌ Kaggle timeout sau 9 giờ

**Nguyên nhân:** Free tier có limit

**Giải pháp:**
- Checkpoints đã được lưu ở `/kaggle/working/models/.../checkpoint-XXXX/`
- Tạo notebook mới
- Resume từ latest checkpoint:
  ```python
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=val_dataset,
  )
  
  # Resume from checkpoint
  trainer.train(resume_from_checkpoint="/kaggle/working/models/wav2vec2-vietnamese/checkpoint-5000")
  ```

---

## So sánh Colab vs Kaggle

| Feature | Google Colab | Kaggle |
|---------|-------------|--------|
| **Background execution** | ❌ Không (phải giữ tab) | ✅ Có (tắt máy được) |
| **GPU Free** | T4 (~12h timeout) | T4/P100 (30h/week) |
| **Training time** | 15-20h (bị timeout!) | 15-20h (đủ!) |
| **Auto-save outputs** | ❌ Không | ✅ Có |
| **Email notification** | ❌ Không | ✅ Có |
| **Storage** | Google Drive | Kaggle Output |
| **Internet** | ✅ Có | ✅ Có (phải bật) |
| **Dataset upload** | Drive (chậm) | Datasets (nhanh hơn) |

**Kết luận:** Kaggle phù hợp hơn cho training dài hạn!

---

## Checklist Trước Khi Chạy

- [ ] Dataset đã upload lên Kaggle Datasets
- [ ] Notebook có Internet enabled
- [ ] Notebook có GPU accelerator
- [ ] Dataset đã được add vào notebook
- [ ] Code mới nhất đã push lên GitHub
- [ ] Cell 10 (Pre-Training Check) PASS tất cả checks

**Nếu tất cả đều OK → Start training! 🚀**

---

## Câu hỏi thường gặp

**Q: Training mất bao lâu?**  
A: 15-20 giờ trên T4 GPU, 10-12 giờ trên P100 GPU

**Q: Tôi có cần giữ tab mở không?**  
A: KHÔNG! Kaggle chạy background, bạn tắt máy được.

**Q: Nếu mất điện thì sao?**  
A: Không sao! Kaggle server vẫn train. Bạn chỉ cần máy để check kết quả sau.

**Q: Free tier có đủ không?**  
A: Có! 30 GPU hours/week đủ cho 1 lần training hoàn chỉnh.

**Q: Làm sao biết training xong?**  
A: Kaggle gửi email thông báo.

**Q: Model lưu ở đâu?**  
A: `/kaggle/working/final_model/` - download từ Output tab.

**Q: Có thể resume từ checkpoint không?**  
A: Có! Checkpoints lưu ở `/kaggle/working/models/.../checkpoint-XXXX/`

---

**Good luck with training! 🎉**

Nếu có vấn đề gì, check lại các bước trong troubleshooting section.
