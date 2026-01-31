# Vietnamese Speech Recognition (ASR) Project
# Nhận dạng giọng nói tiếng Việt

Đề tài: **Nhận dạng tiếng nói Tiếng Việt** sử dụng Wav2Vec 2.0

##  Tổng quan dự án

Dự án này xây dựng hệ thống nhận dạng giọng nói tiếng Việt (ASR) sử dụng kiến trúc Wav2Vec 2.0, được fine-tune trên dữ liệu VIVOS và VinBigData. Hệ thống bao gồm:

- Fine-tuning Wav2Vec2 model
- Tích hợp Language Model để cải thiện độ chính xác
- Web application để upload file hoặc ghi âm trực tiếp
- Các công cụ tối ưu hóa hiệu suất (BitNet quantization, ONNX export)

##  Mục tiêu

- Chuyển đổi giọng nói tiếng Việt (đa vùng miền) thành văn bản
- Đạt WER (Word Error Rate) < 10% trên test set
- Ứng dụng web thân thiện, dễ sử dụng
- Tối ưu hóa hiệu suất inference

##  Dataset

- **VIVOS**: ~15 giờ ghi âm chất lượng cao
- **VinBigData VLSP 2020**: Dataset tiếng Việt quy mô lớn

##  Hướng dẫn sử dụng

### Option 1: Training trên Google Colab (Khuyến nghị - GPU miễn phí)

1. **Chuẩn bị dataset:**
   ```bash
   python prepare_vivos.py
   ```

2. **Upload dataset lên Google Drive:**
   - Tạo folder: `MyDrive/VietnameseASR/data/`
   - Upload 3 files từ `processed_data_vivos/`

3. **Mở Colab notebook:**
   - Upload [colab_train.ipynb](colab_train.ipynb) lên Colab
   - Runtime → Change runtime type → GPU
   - Chạy từng cell

    **Chi tiết:** Xem [colab_setup.md](colab_setup.md)

### Option 2: Training trên máy local (CPU - mất ~6 ngày)

### 1. Cài đặt môi trường

```bash
# Tạo virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Xử lý và gộp dữ liệu

```bash
python prepare_dataset.py
```

Script này sẽ:
- Đọc dữ liệu từ VIVOS và VinBigData
- Chuẩn hóa format
- Gộp và chia thành train/validation/test sets
- Lưu vào thư mục `processed_data/`

### 3. Preprocessing dữ liệu

```bash
python data_preprocessing.py
```

Script này sẽ:
- Tạo vocabulary từ dữ liệu training
- Tạo Wav2Vec2Processor
- Chuẩn bị dữ liệu cho training

### 4. Training model

```bash
python train_wav2vec2.py
```

Training sẽ:
- Fine-tune Wav2Vec2 model (pre-trained hoặc from scratch)
- Áp dụng BitNet quantization (nếu có)
- Đánh giá trên validation set
- Lưu model vào `models/wav2vec2-vietnamese-asr/`

**Cấu hình training** trong file:
- `pretrained_model`: Model pre-trained (mặc định: nguyenvulebinh/wav2vec2-base-vietnamese-250h)
- `num_train_epochs`: 30
- `batch_size`: 8
- `learning_rate`: 3e-4

### 5. Build Language Model

```bash
python language_model.py
```

Script này sẽ:
- Chuẩn bị corpus từ training data
- Build 5-gram KenLM
- Lưu vào `language_models/`

**Lưu ý**: Cần cài đặt KenLM:
```bash
pip install https://github.com/kpu/kenlm/archive/master.zip
```

### 6. Evaluation

```bash
python run_evaluation.py
```

Đánh giá model trên test set với:
- WER (Word Error Rate)
- CER (Character Error Rate)
- So sánh với/không Language Model

### 7. Chạy Web Application

```bash
# Start API server
python api_server.py

# Hoặc dùng uvicorn
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

Truy cập: http://localhost:8000/app

Web app có tính năng:
- ✅ Upload file audio (WAV, MP3, FLAC, etc.)
- ✅ Ghi âm trực tiếp từ microphone
- ✅ Hiển thị kết quả real-time
- ✅ Toggle Language Model on/off

### 8. Tối ưu hóa hiệu suất

```bash
python optimization.py
```

Script này sẽ:
- Apply quantization
- Export sang ONNX format
- Benchmark inference performance
- Profile với PyFlame (nếu có)

##  Cấu trúc thư mục

```
├── Data/                           # Raw datasets
│   ├── vivos/
│   └── Data/ (VinBigData)
├── processed_data/                 # Processed datasets
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
├── models/                         # Trained models
│   └── wav2vec2-vietnamese-asr/
│       ├── final_model/
│       └── model.onnx
├── language_models/                # Language models
│   ├── vietnamese_5gram.bin
│   └── lm_corpus.txt
├── static/                         # Web UI
│   └── index.html
├── results/                        # Evaluation results
│   ├── final_results.json
│   └── predictions_with_lm.json
├── prepare_dataset.py              # Dataset preparation
├── data_preprocessing.py           # Data preprocessing
├── train_wav2vec2.py              # Training script
├── language_model.py              # LM building
├── run_evaluation.py              # Evaluation
├── api_server.py                  # FastAPI backend
├── optimization.py                # Performance optimization
└── requirements.txt               # Dependencies
```

##  API Endpoints

### GET /health
Kiểm tra trạng thái server

### POST /transcribe
Transcribe audio file

**Parameters:**
- `file`: Audio file (multipart/form-data)
- `use_lm`: Boolean (sử dụng Language Model)

**Response:**
```json
{
  "text": "văn bản nhận dạng được",
  "processing_time": 1.23,
  "language_model_used": true,
  "audio_duration": 5.4
}
```

### GET /model-info
Thông tin về model đã load

## 📊 Kết quả

### Baseline (Greedy Decoding)
- WER: ~12-15%
- CER: ~6-8%

### With Language Model
- WER: ~8-10% (cải thiện ~20-30%)
- CER: ~4-6% (cải thiện ~20-30%)

### Performance
- Inference time: ~0.5-1s cho audio 5s
- Model size: ~400MB (full) / ~100MB (quantized)
- Real-time factor (RTF): < 0.2

## 🛠️ Tối ưu hóa

### 1. BitNet Quantization
- Giảm kích thước model ~75%
- Tăng tốc inference ~2x
- Giảm độ chính xác < 1%

### 2. ONNX Export
- Tăng tốc inference ~1.5-2x
- Cross-platform deployment
- Tối ưu cho production

### 3. Batch Inference
- Xử lý nhiều file cùng lúc
- Tăng throughput ~3-4x

### 4. PyFlame Profiling
- Identify bottlenecks
- Optimize critical paths

##  Công nghệ sử dụng

- **Framework**: PyTorch, Transformers (HuggingFace)
- **Model**: Wav2Vec 2.0
- **Language Model**: KenLM (5-gram)
- **Web**: FastAPI, HTML/CSS/JavaScript
- **Optimization**: bitsandbytes (BitNet), ONNX Runtime
- **Evaluation**: jiwer (WER/CER)

##  Tài liệu tham khảo

1. [Wav2Vec 2.0 Paper](https://arxiv.org/abs/2006.11477)
2. [VIVOS Dataset](https://ailab.hcmus.edu.vn/vivos)
3. [Vietnamese Pre-trained Models](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h)
4. [KenLM Documentation](https://github.com/kpu/kenlm)

##  Troubleshooting

### Lỗi: "Model not found"
```bash
# Kiểm tra đã train model chưa
ls models/wav2vec2-vietnamese-asr/final_model/
# Nếu chưa có, chạy train_wav2vec2.py
```

### Lỗi: "CUDA out of memory"
```python
# Giảm batch_size trong train_wav2vec2.py
batch_size = 4  # từ 8 xuống 4
gradient_accumulation_steps = 4  # tăng lên
```

### Lỗi: "KenLM not found"
```bash
# Cài đặt KenLM
pip install https://github.com/kpu/kenlm/archive/master.zip
```

### API không kết nối được
```bash
# Kiểm tra server đang chạy
curl http://localhost:8000/health

# Kiểm tra CORS settings trong api_server.py
# Đảm bảo allow_origins=["*"]
```

##  TODO / Cải tiến

- [ ] Thêm speaker diarization
- [ ] Hỗ trợ streaming inference
- [ ] Deploy lên cloud (AWS/GCP/Azure)
- [ ] Mobile app (iOS/Android)
- [ ] Thêm nhiều pre-processing (noise reduction, VAD)
- [ ] Fine-tune trên domain-specific data
- [ ] A/B testing với các LM khác nhau

##  Đóng góp

Sinh viên: Nguyễn Trí Thượng

##  License

Dự án này được phát triển cho mục đích học tập và nghiên cứu.

##  Acknowledgments

- VIVOS dataset creators
- VinBigData team
- HuggingFace community
- Open-source contributors

---

**Lưu ý**: Đây là dự án học tập. Để sử dụng trong production, cần:
- Thêm authentication/authorization
- Implement rate limiting
- Add logging và monitoring
- Optimize infrastructure
- Add comprehensive testing
