<![CDATA[<div align="center">

# 🎙️ VietASR Pro

### Hệ thống Nhận dạng Tiếng nói Tiếng Việt | Vietnamese Speech Recognition System

**Wav2Vec 2.0 · Fine-tuned on VIVOS + VLSP 2020 · Flask & FastAPI**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![WER](https://img.shields.io/badge/Best%20WER-13.3%25-brightgreen)
![License](https://img.shields.io/badge/License-Academic%20Research-blue)
![Flask](https://img.shields.io/badge/Demo-Flask-000000?logo=flask&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)

</div>

---

## 1. Giới thiệu 🇻🇳 | Introduction 🇬🇧

### 🇻🇳 Giới thiệu

**VietASR Pro** là hệ thống nhận dạng giọng nói tiếng Việt (Automatic Speech Recognition — ASR) được xây dựng dựa trên kiến trúc **Wav2Vec 2.0** của Facebook AI. Hệ thống được fine-tune trên hai bộ dữ liệu lớn: **VIVOS** (~15 giờ) và **VLSP 2020 / viet_bud500** (quy mô lớn), đạt **Word Error Rate (WER) 13,3%** tại checkpoint tốt nhất.

Dự án bao gồm:
- Pipeline huấn luyện hoàn chỉnh trên Kaggle GPU
- Ứng dụng web demo Flask với giao diện hiện đại (dark mode, ghi âm trực tiếp, upload file)
- Server API FastAPI với tài liệu Swagger tự động
- Bộ công cụ đánh giá WER/CER và tối ưu hóa model

**Đối tượng sử dụng:** Sinh viên, nhà nghiên cứu NLP/ASR tiếng Việt, và lập trình viên muốn tích hợp nhận dạng giọng nói tiếng Việt vào ứng dụng.

### 🇬🇧 Introduction

**VietASR Pro** is a Vietnamese Automatic Speech Recognition (ASR) system built on **Wav2Vec 2.0** architecture by Facebook AI. The model is fine-tuned on two major datasets: **VIVOS** (~15 hours) and **VLSP 2020 / viet_bud500** (large-scale), achieving a **Word Error Rate (WER) of 13.3%** at the best checkpoint.

The project includes:
- Complete training pipeline on Kaggle GPU
- Modern Flask web demo (dark mode, live recording, file upload)
- FastAPI REST server with automatic Swagger documentation
- WER/CER evaluation toolkit and model optimization utilities

**Target users:** Students, Vietnamese NLP/ASR researchers, and developers integrating Vietnamese speech recognition into applications.

---

## 2. Tính năng nổi bật 🇻🇳 | Key Features 🇬🇧

### 🇻🇳 Tính năng nổi bật

- 🔬 **Fine-tuning Wav2Vec 2.0** — Huấn luyện trên VIVOS + VLSP 2020, đạt WER 13,3%
- 🧠 **CTC Decoding** — Sử dụng Connectionist Temporal Classification cho quá trình giải mã
- 🎨 **Web Demo Flask** — Giao diện hiện đại với sidebar, dark mode, ghi âm micro, upload file audio
- ⚡ **FastAPI Backend** — REST API thay thế với Swagger UI, hỗ trợ upload và transcribe
- 📊 **Đánh giá tự động** — Tính toán WER, CER trên tập test, so sánh có/không Language Model
- 🔧 **Tối ưu hóa model** — Dynamic quantization (int8), ONNX export, batch inference
- ☁️ **Kaggle GPU Training** — Pipeline huấn luyện sẵn cho Kaggle (T4/P100)
- 📓 **Notebook Colab/Kaggle** — Sẵn sàng sử dụng trên Google Colab hoặc Kaggle
- 🗣️ **Hỗ trợ đa vùng miền** — Dialect mapping, chuẩn hóa từ vùng miền
- 🔠 **Phiên âm IPA** — Chuyển đổi tiếng Việt sang IPA (tích hợp trong chatbot)

### 🇬🇧 Key Features

- 🔬 **Wav2Vec 2.0 Fine-tuning** — Trained on VIVOS + VLSP 2020, achieving 13.3% WER
- 🧠 **CTC Decoding** — Connectionist Temporal Classification for sequence decoding
- 🎨 **Flask Web Demo** — Modern UI with sidebar, dark mode, mic recording, file upload
- ⚡ **FastAPI Backend** — Alternative REST API with Swagger UI
- 📊 **Automated Evaluation** — WER, CER computation on test set
- 🔧 **Model Optimization** — Dynamic quantization (int8), ONNX export, batch inference
- ☁️ **Kaggle GPU Training** — Ready-to-use training pipeline on Kaggle (T4/P100)
- 📓 **Colab/Kaggle Notebooks** — Pre-configured for Google Colab or Kaggle
- 🗣️ **Dialect Support** — Dialect mapping, regional word normalization
- 🔠 **IPA Transcription** — Vietnamese-to-IPA conversion in built-in chatbot

---

## 3. Kiến trúc hệ thống 🇻🇳 | System Architecture 🇬🇧

### 🇻🇳 Kiến trúc hệ thống

Hệ thống sử dụng kiến trúc **Wav2Vec 2.0 base** (12 lớp Transformer, 768 hidden dimensions, 12 attention heads) được pre-train bởi `nguyenvulebinh/wav2vec2-base-vietnamese-250h`, sau đó fine-tune với CTC loss trên dữ liệu VIVOS + VLSP 2020.

### 🇬🇧 System Architecture

The system uses **Wav2Vec 2.0 base** architecture (12 Transformer layers, 768 hidden dimensions, 12 attention heads) pre-trained by `nguyenvulebinh/wav2vec2-base-vietnamese-250h`, then fine-tuned with CTC loss on VIVOS + VLSP 2020 data.

```
╔═══════════════════════════════════════════════════════════════════╗
║                     VietASR Pro — Data Flow                      ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  🎤 Audio Input (16kHz WAV)                                      ║
║       │                                                           ║
║       ▼                                                           ║
║  ┌────────────────────┐                                           ║
║  │  Feature Encoder   │  7 CNN layers → raw audio → latent repr. ║
║  │  (Frozen)          │                                           ║
║  └────────┬───────────┘                                           ║
║           ▼                                                       ║
║  ┌────────────────────┐                                           ║
║  │  Context Network   │  12 Transformer layers → context repr.   ║
║  │  (Fine-tuned)      │                                           ║
║  └────────┬───────────┘                                           ║
║           ▼                                                       ║
║  ┌────────────────────┐                                           ║
║  │  CTC Head          │  Linear → 110 vocab tokens               ║
║  │  (Fine-tuned)      │                                           ║
║  └────────┬───────────┘                                           ║
║           ▼                                                       ║
║  ┌────────────────────┐                                           ║
║  │  Greedy / Beam     │  argmax or LM-assisted beam search       ║
║  │  Decoding          │                                           ║
║  └────────┬───────────┘                                           ║
║           ▼                                                       ║
║  📝 Vietnamese Text Output                                       ║
║                                                                   ║
║  ═════════════════════════════════════════════════                ║
║  Serving Layer:                                                   ║
║  ┌──────────────┐    ┌──────────────┐                            ║
║  │  Flask App   │    │  FastAPI     │                             ║
║  │  (port 5000) │    │  (port 8000) │                             ║
║  │  Web UI Demo │    │  REST API    │                             ║
║  └──────────────┘    └──────────────┘                            ║
╚═══════════════════════════════════════════════════════════════════╝
```

---

## 4. Cấu trúc thư mục 🇻🇳 | Project Structure 🇬🇧

### 🇻🇳 Cấu trúc thư mục / 🇬🇧 Project Structure

```
VietASR-Pro/
├── README.md                          # Tài liệu chính (song ngữ)
├── PAPER_VietASR_Pro.md               # Bài báo khoa học (tiếng Việt)
├── LICENSE                            # Giấy phép
├── requirements.txt                   # Thư viện Python (hợp nhất)
├── .gitignore                         # Danh sách file bỏ qua Git
├── train.py                           # Entry point — huấn luyện
├── run_server.py                      # Entry point — FastAPI server
├── run_evaluation.py                  # Entry point — đánh giá
│
├── configs/                           # Cấu hình huấn luyện
│   └── training_config.yaml
│
├── data/                              # Xử lý dữ liệu
│   ├── README.md                      # Mô tả cấu trúc dataset
│   ├── preprocessing/                 # Scripts chuẩn bị dữ liệu
│   │   ├── __init__.py
│   │   ├── check_dataset.py           # Kiểm tra chất lượng dataset
│   │   ├── prepare_dataset.py         # Gộp VIVOS + VLSP
│   │   ├── prepare_vivos.py           # Chỉ VIVOS
│   │   ├── prepare_full_dataset.py    # Gộp đầy đủ
│   │   ├── create_vlsp_jsonl.py       # Tạo VLSP JSONL
│   │   ├── merge_vivos_vlsp_jsonl.py  # Gộp JSONL
│   │   ├── convert_to_absolute_paths.py
│   │   └── convert_to_relative_paths.py
│   └── raw/                           # Dữ liệu thô (.gitignore'd)
│
├── src/                               # Mã nguồn chính
│   ├── __init__.py
│   ├── api/                           # FastAPI server
│   │   ├── __init__.py
│   │   └── server.py
│   ├── model/                         # Định nghĩa model
│   │   └── __init__.py
│   ├── training/                      # Pipeline huấn luyện
│   │   ├── __init__.py
│   │   ├── train_wav2vec2.py          # Training Wav2Vec2 + CTC
│   │   └── language_model.py          # KenLM / LM decoder
│   ├── evaluation/                    # Đánh giá model
│   │   ├── __init__.py
│   │   └── evaluate.py               # WER/CER evaluation
│   ├── data/                          # Data processing module
│   │   ├── __init__.py
│   │   ├── preprocessing.py           # Core preprocessing pipeline
│   │   ├── prepare_dataset.py
│   │   ├── prepare_vivos_only.py
│   │   └── normalize_audio.py
│   └── utils/                         # Tiện ích
│       ├── __init__.py
│       ├── optimization.py            # Quantization, ONNX export
│       ├── profiling.py               # Performance profiling
│       └── demo.py                    # Demo script
│
├── notebooks/                         # Jupyter notebooks
│   ├── colab_train_FINAL.ipynb        # Notebook huấn luyện Colab
│   └── kaggle_train.ipynb             # Notebook huấn luyện Kaggle
│
├── app/                               # Flask Web Demo
│   ├── __init__.py
│   ├── app.py                         # Flask application (port 5000)
│   ├── templates/
│   │   └── index.html                 # Giao diện chính
│   └── static/
│       ├── css/style.css              # Stylesheet
│       └── js/main.js                 # JavaScript logic
│
├── tests/                             # Kiểm thử
│   ├── __init__.py
│   ├── test_kaggle_model.py           # So sánh model Kaggle vs cũ
│   ├── test_model.py                  # Test model inference
│   ├── test_retry.py                  # Test retry logic
│   └── test_upload_api.py             # Test API upload
│
├── scripts/                           # Scripts tiện ích
│   ├── check_audio_path.py
│   ├── check_training.py
│   ├── check_vocab.py
│   ├── check_dependencies.py
│   ├── quick_commands.bat
│   ├── run_pipeline.bat
│   ├── run_pipeline.sh
│   ├── demo_viet_bud500.py
│   ├── setup/                         # Scripts cài đặt
│   │   ├── install_dependencies.bat
│   │   └── quick_start.bat
│   ├── migration/                     # Scripts di chuyển project
│   │   ├── copy_project_from_c.ps1
│   │   ├── update_c_paths.ps1
│   │   └── verify_migration.ps1
│   └── profiling/
│       └── flamegraph_guide.py
│
├── results/                           # Kết quả & logs
│   ├── training_history.csv           # Lịch sử huấn luyện
│   ├── audio_stats.json
│   ├── dataset_errors.csv
│   ├── unknown_chars.txt
│   └── checkpoints/
│       └── old/
│           └── final_model_OLD_WER100/  # Model thất bại (WER=100%)
│
├── docs/                              # Tài liệu bổ sung
│   ├── PROJECT_STRUCTURE.md
│   ├── TRAINING_ANALYSIS_REPORT.md
│   ├── FIX_AUDIO_ERRORS.md
│   ├── FIX_COLAB_ERRORS.md
│   ├── KAGGLE_SETUP_GUIDE.md
│   ├── NOTEBOOK_STRUCTURE.md
│   ├── colab_setup.md
│   └── old_ui_reference.html
│
├── third_party/                       # Công cụ bên thứ ba
│   ├── README.md
│   ├── PyFlame/                       # C++ profiling tool
│   └── pyflame_demos/                 # Ví dụ PyFlame
│
├── final_model/                       # Model Kaggle (WER 13.3%)
│   ├── config.json
│   ├── model.safetensors              # ~377MB (gitignored)
│   ├── vocab.json
│   ├── tokenizer_config.json
│   └── trainer_state.json
│
└── token/                             # HF token (gitignored)
```

---

## 5. Yêu cầu hệ thống 🇻🇳 | Requirements 🇬🇧

### 🇻🇳 Yêu cầu hệ thống

| Thành phần | Yêu cầu |
|---|---|
| **Python** | 3.10 trở lên |
| **PyTorch** | ≥ 2.0.0 (hỗ trợ CUDA 12.x) |
| **Transformers** | ≥ 4.35.0 |
| **GPU (huấn luyện)** | Kaggle T4 (16GB) hoặc P100 (16GB) |
| **GPU (inference)** | NVIDIA RTX 4060 8GB hoặc tương đương; CPU cũng hoạt động |
| **RAM** | ≥ 16GB |
| **Dung lượng đĩa** | ~2GB (model + dependencies) |

### 🇬🇧 Requirements

| Component | Requirement |
|---|---|
| **Python** | 3.10 or later |
| **PyTorch** | ≥ 2.0.0 (CUDA 12.x supported) |
| **Transformers** | ≥ 4.35.0 |
| **GPU (training)** | Kaggle T4 (16GB) or P100 (16GB) |
| **GPU (inference)** | NVIDIA RTX 4060 8GB or equivalent; CPU also works |
| **RAM** | ≥ 16GB |
| **Disk space** | ~2GB (model + dependencies) |

**Thư viện chính / Key libraries:**

```
torch>=2.0.0        torchaudio>=2.0.0      transformers>=4.35.0
librosa>=0.10.0     soundfile>=0.12.0      jiwer>=3.0.0
flask>=3.0.0        flask-cors>=4.0.0      fastapi>=0.104.0
uvicorn>=0.24.0     datasets>=2.14.0       accelerate>=0.24.0
pydub>=0.25.1       numpy>=1.24.0          pandas>=2.0.0
```

---

## 6. Hướng dẫn cài đặt 🇻🇳 | Installation 🇬🇧

### 🇻🇳 Hướng dẫn cài đặt

```bash
# 1. Clone repository
git clone https://github.com/CheeseThuong/Vietnamese-asr.git
cd Vietnamese-asr

# 2. Tạo virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 3. Cài đặt dependencies
pip install -r requirements.txt

# 4. Tải model (đã train trên Kaggle)
# Model nằm trong thư mục final_model/
# Hoặc tải từ Hugging Face Hub nếu cần
```

### 🇬🇧 Installation

```bash
# 1. Clone repository
git clone https://github.com/CheeseThuong/Vietnamese-asr.git
cd Vietnamese-asr

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download model (trained on Kaggle)
# The model is located in final_model/
# Or download from Hugging Face Hub if needed
```

---

## 7. Hướng dẫn sử dụng 🇻🇳 | Usage 🇬🇧

### 🇻🇳 Hướng dẫn sử dụng

#### 🔬 Huấn luyện
```bash
# Trên Kaggle — mở notebook và chạy
notebooks/kaggle_train.ipynb

# Trên Google Colab
notebooks/colab_train_FINAL.ipynb

# Hoặc chạy trực tiếp
python train.py
```

#### 📊 Đánh giá model
```bash
python run_evaluation.py
# → Kết quả WER/CER được lưu tại results/
```

#### 🌐 Flask Web Demo
```bash
python app/app.py
# → Mở trình duyệt: http://localhost:5000
```

Tính năng web demo:
- ✅ Ghi âm trực tiếp từ microphone
- ✅ Upload file audio (WAV, MP3, FLAC, OGG, M4A, ...)
- ✅ Hiển thị kết quả real-time
- ✅ Dark mode / Light mode
- ✅ Lịch sử nhận dạng
- ✅ AI Chatbot (phiên âm IPA, chuẩn hóa vùng miền)

#### ⚡ FastAPI Server
```bash
python run_server.py
# Hoặc:
uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
# → Swagger UI: http://localhost:8000/docs
```

### 🇬🇧 Usage

#### 🔬 Training
```bash
# On Kaggle — open notebook and run
notebooks/kaggle_train.ipynb

# On Google Colab
notebooks/colab_train_FINAL.ipynb

# Or run directly
python train.py
```

#### 📊 Evaluation
```bash
python run_evaluation.py
# → WER/CER results saved to results/
```

#### 🌐 Flask Web Demo
```bash
python app/app.py
# → Open browser: http://localhost:5000
```

#### ⚡ FastAPI Server
```bash
python run_server.py
# Or:
uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
# → Swagger UI: http://localhost:8000/docs
```

---

## 8. Kết quả thực nghiệm 🇻🇳 | Experimental Results 🇬🇧

### 🇻🇳 Kết quả thực nghiệm

#### Model Kaggle (Fine-tuned trên VIVOS + VLSP 2020)

Training: 15.000 steps, batch size 32, Kaggle GPU (T4/P100), learning rate 1e-4 → cosine decay

| Checkpoint | Steps | Eval Loss | WER (%) | Ghi chú |
|---|---|---|---|---|
| checkpoint-1000 | 1.000 | 0.1363 | 15,57 | — |
| checkpoint-2000 | 2.000 | 0.1207 | 15,07 | — |
| checkpoint-3000 | 3.000 | 0.1140 | 14,82 | — |
| checkpoint-5000 | 5.000 | 0.1071 | 14,59 | — |
| checkpoint-7000 | 7.000 | 0.0965 | 14,27 | — |
| checkpoint-9000 | 9.000 | 0.0879 | 13,96 | — |
| checkpoint-10000 | 10.000 | 0.0844 | 13,90 | — |
| checkpoint-11000 | 11.000 | 0.0818 | 13,72 | — |
| checkpoint-12000 | 12.000 | 0.0756 | 13,49 | — |
| checkpoint-14000 | 14.000 | 0.0732 | 13,41 | — |
| **checkpoint-15000** | **15.000** | **0.0664** | **13,31** | **🏆 Best** |

#### So sánh với model cũ (Local Training — FAILED)

| Model | Steps | WER (%) | Ghi chú |
|---|---|---|---|
| Local Training (CPU) | 3.000 | **100,0** | ❌ Hoàn toàn không học |
| **Kaggle Training (GPU)** | **15.000** | **13,3** | ✅ Thành công |

> **Bài học:** Model cũ thất bại (WER=100%) do data preprocessing sai, training configuration không phù hợp, và dataset quá nhỏ khi train riêng VIVOS trên CPU. Chuyển sang Kaggle GPU + dataset lớn hơn (viet_bud500) đã giải quyết vấn đề.

### 🇬🇧 Experimental Results

See table above. Key takeaway: **Kaggle fine-tuning achieved 13.3% WER** compared to the failed local training attempt (100% WER). The improvement came from using GPU training, larger dataset (VLSP 2020), and proper hyperparameter configuration.

---

## 9. Dataset 🇻🇳🇬🇧

### 🇻🇳 Dữ liệu

| Dataset | Mô tả | Số giờ | Nguồn |
|---|---|---|---|
| **VIVOS** | Bộ dữ liệu tiếng Việt chất lượng cao, 46 người nói | ~15 giờ | [ailab.hcmus.edu.vn/vivos](https://ailab.hcmus.edu.vn/vivos) |
| **VLSP 2020 / viet_bud500** | Bộ dữ liệu tiếng Việt quy mô lớn | Lớn | [HuggingFace](https://huggingface.co/datasets/doof-ferb/vlsp2020_vinai_100h) |

### 🇬🇧 Dataset

| Dataset | Description | Hours | Source |
|---|---|---|---|
| **VIVOS** | High-quality Vietnamese speech, 46 speakers | ~15h | [ailab.hcmus.edu.vn/vivos](https://ailab.hcmus.edu.vn/vivos) |
| **VLSP 2020 / viet_bud500** | Large-scale Vietnamese speech | Large | [HuggingFace](https://huggingface.co/datasets/doof-ferb/vlsp2020_vinai_100h) |

**Vocabulary:** 110 tokens — bao gồm đầy đủ ký tự tiếng Việt (có dấu), chữ số, và các token đặc biệt (`<pad>`, `<unk>`, `|`).

---

## 10. Hướng phát triển 🇻🇳 | Future Work 🇬🇧

### 🇻🇳 Hướng phát triển

- [ ] 🗣️ Tích hợp Language Model (KenLM 5-gram) để cải thiện WER thêm 20-30%
- [ ] 📡 Streaming inference — nhận dạng real-time
- [ ] ☁️ Deploy lên cloud (AWS/GCP/Azure)
- [ ] 📱 Mobile app (iOS/Android)
- [ ] 🎛️ Noise reduction & Voice Activity Detection (VAD)
- [ ] 🎯 Fine-tune trên domain-specific data (y tế, pháp luật...)
- [ ] 👥 Speaker diarization — nhận dạng nhiều người nói
- [ ] 🔍 Beam search decoding với Language Model

### 🇬🇧 Future Work

- [ ] 🗣️ Language Model integration (KenLM 5-gram) for 20-30% WER improvement
- [ ] 📡 Streaming inference — real-time recognition
- [ ] ☁️ Cloud deployment (AWS/GCP/Azure)
- [ ] 📱 Mobile app (iOS/Android)
- [ ] 🎛️ Noise reduction & Voice Activity Detection (VAD)
- [ ] 🎯 Domain-specific fine-tuning (medical, legal, etc.)
- [ ] 👥 Speaker diarization — multi-speaker recognition
- [ ] 🔍 Beam search decoding with Language Model

---

## 11. Tác giả 🇻🇳 | Author 🇬🇧

### 🇻🇳 Tác giả

| | Thông tin |
|---|---|
| **Họ và tên** | Nguyễn Trí Thượng |
| **Đề tài** | Nhận dạng tiếng nói Tiếng Việt sử dụng Wav2Vec 2.0 |
| **Loại đề tài** | Đồ án Trí tuệ Nhân tạo |
| **Năm học** | 2025–2026 |

### 🇬🇧 Author

| | Info |
|---|---|
| **Name** | Nguyễn Trí Thượng |
| **Topic** | Vietnamese Speech Recognition using Wav2Vec 2.0 |
| **Type** | AI Course Project |
| **Academic Year** | 2025–2026 |

---

## 12. Giấy phép 🇻🇳 | License 🇬🇧

### 🇻🇳 Giấy phép

Dự án này được phát triển cho mục đích **học tập và nghiên cứu**. Mã nguồn có thể sử dụng tự do cho mục đích phi thương mại.

### 🇬🇧 License

This project is developed for **academic and research purposes**. The source code is freely available for non-commercial use.

---

## 13. Lời cảm ơn 🇻🇳 | Acknowledgements 🇬🇧

### 🇻🇳 Lời cảm ơn

- 🤗 **Hugging Face** — Thư viện Transformers, Datasets, và cộng đồng mã nguồn mở
- 🎓 **VIVOS** — Nhóm tác giả bộ dữ liệu VIVOS tại AILAB HCMUS
- 📊 **VLSP** — Ban tổ chức cuộc thi VLSP 2020 và nhóm VinBigData
- ☁️ **Kaggle** — Hạ tầng GPU miễn phí cho huấn luyện model
- 🇻🇳 **nguyenvulebinh** — Pre-trained Wav2Vec2 Vietnamese model
- 🌐 **Flask & FastAPI** — Framework web demo và API

### 🇬🇧 Acknowledgements

- 🤗 **Hugging Face** — Transformers library, Datasets, and open-source community
- 🎓 **VIVOS** — VIVOS dataset team at AILAB HCMUS
- 📊 **VLSP** — VLSP 2020 organizers and VinBigData team
- ☁️ **Kaggle** — Free GPU infrastructure for model training
- 🇻🇳 **nguyenvulebinh** — Pre-trained Wav2Vec2 Vietnamese model
- 🌐 **Flask & FastAPI** — Web demo and API frameworks

---

## 📚 Tài liệu tham khảo / References

1. Baevski, A., et al. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*. [arXiv:2006.11477](https://arxiv.org/abs/2006.11477)
2. [VIVOS Dataset](https://ailab.hcmus.edu.vn/vivos)
3. [Wav2Vec2 Vietnamese Pre-trained](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h)
4. [KenLM Documentation](https://github.com/kpu/kenlm)
5. [Hugging Face Transformers](https://huggingface.co/docs/transformers)
6. [Flask Documentation](https://flask.palletsprojects.com/)
7. [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

<div align="center">

**VietASR Pro** — Made with ❤️ for Vietnamese Speech Recognition

*Nguyễn Trí Thượng · 2025–2026*

</div>
]]>
