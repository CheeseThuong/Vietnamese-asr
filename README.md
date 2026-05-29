<div align="center">

# VietASR Pro

### Hệ thống nhận dạng tiếng nói tiếng Việt | Vietnamese Automatic Speech Recognition System

**Wav2Vec 2.0 · VIVOS · VLSP 2020 / viet_bud500 · Flask · FastAPI · Kaggle GPU**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![Flask](https://img.shields.io/badge/Web%20Demo-Flask-000000?logo=flask&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![Best WER](https://img.shields.io/badge/Best%20WER-13.31%25-brightgreen)
![License](https://img.shields.io/badge/License-Academic%20Research-blue)

</div>

---

## Mục lục | Table of Contents

1. [Giới thiệu | Introduction](#1-giới-thiệu--introduction)
2. [Tính năng chính | Key Features](#2-tính-năng-chính--key-features)
3. [Kiến trúc hệ thống | System Architecture](#3-kiến-trúc-hệ-thống--system-architecture)
4. [Công nghệ sử dụng | Technology Stack](#4-công-nghệ-sử-dụng--technology-stack)
5. [Cấu trúc thư mục | Project Structure](#5-cấu-trúc-thư-mục--project-structure)
6. [Yêu cầu hệ thống | System Requirements](#6-yêu-cầu-hệ-thống--system-requirements)
7. [Hướng dẫn cài đặt chi tiết | Detailed Installation Guide](#7-hướng-dẫn-cài-đặt-chi-tiết--detailed-installation-guide)
8. [Hướng dẫn sử dụng | Usage Guide](#8-hướng-dẫn-sử-dụng--usage-guide)
9. [API Endpoints](#9-api-endpoints)
10. [Kết quả thực nghiệm | Experimental Results](#10-kết-quả-thực-nghiệm--experimental-results)
11. [Dữ liệu | Datasets](#11-dữ-liệu--datasets)
12. [Cấu hình | Configuration](#12-cấu-hình--configuration)
13. [Kiểm thử | Testing](#13-kiểm-thử--testing)
14. [Giới hạn hiện tại | Current Limitations](#14-giới-hạn-hiện-tại--current-limitations)
15. [Hướng phát triển | Roadmap](#15-hướng-phát-triển--roadmap)
16. [Tác giả | Author](#16-tác-giả--author)
17. [Giấy phép | License](#17-giấy-phép--license)
18. [Lời cảm ơn | Acknowledgements](#18-lời-cảm-ơn--acknowledgements)
19. [Tài liệu tham khảo | References](#19-tài-liệu-tham-khảo--references)

---

## 1. Giới thiệu | Introduction

### Tiếng Việt

**VietASR Pro** là hệ thống nhận dạng tiếng nói tiếng Việt được xây dựng trên kiến trúc **Wav2Vec 2.0**. Dự án tập trung vào việc fine-tune mô hình pre-trained tiếng Việt, xây dựng pipeline xử lý dữ liệu, huấn luyện trên GPU, đánh giá bằng các chỉ số ASR tiêu chuẩn và triển khai thành web demo/API có thể sử dụng thực tế.

Mô hình hiện tại được fine-tune trên **VIVOS** và **VLSP 2020 / viet_bud500**, đạt **WER tốt nhất 13,31%** tại checkpoint 15.000 steps trong quá trình huấn luyện trên Kaggle GPU.

Dự án phù hợp cho:

- Sinh viên thực hiện đồ án trí tuệ nhân tạo hoặc xử lý ngôn ngữ tự nhiên.
- Nhà nghiên cứu cần pipeline ASR tiếng Việt có thể tái sử dụng.
- Lập trình viên muốn tích hợp chuyển giọng nói tiếng Việt thành văn bản vào ứng dụng web hoặc backend API.

### English

**VietASR Pro** is a Vietnamese Automatic Speech Recognition system built on the **Wav2Vec 2.0** architecture. The project focuses on fine-tuning a Vietnamese pre-trained model, building a complete data-processing pipeline, training on GPU, evaluating with standard ASR metrics, and deploying the model through a usable web demo and REST API.

The current model is fine-tuned on **VIVOS** and **VLSP 2020 / viet_bud500**, achieving a best **WER of 13.31%** at checkpoint 15,000 during Kaggle GPU training.

This project is suitable for:

- Students working on AI, NLP, or speech-processing course projects.
- Researchers who need a reusable Vietnamese ASR pipeline.
- Developers who want to integrate Vietnamese speech-to-text into web applications or backend APIs.

---

## 2. Tính năng chính | Key Features

### Tiếng Việt

- **Fine-tuning Wav2Vec 2.0**: Huấn luyện mô hình ASR tiếng Việt với CTC loss.
- **Pipeline dữ liệu hoàn chỉnh**: Chuẩn bị, kiểm tra, gộp và chuẩn hóa dữ liệu âm thanh.
- **Huấn luyện trên Kaggle GPU**: Tối ưu cho môi trường Kaggle, Google Colab hoặc Jupyter Notebook.
- **Web demo bằng Flask**: Hỗ trợ ghi âm micro, upload audio và xem kết quả nhận dạng.
- **REST API bằng FastAPI**: Cung cấp endpoint cho upload file và nhận kết quả transcription.
- **Đánh giá tự động**: Tính WER, CER và lưu kết quả vào thư mục `results/`.
- **Tối ưu hóa mô hình**: Hỗ trợ dynamic quantization, ONNX export và batch inference.
- **Hậu xử lý văn bản**: Có thể kết hợp LLM để chuẩn hóa câu, dấu câu và ngữ cảnh.
- **Hỗ trợ mở rộng**: Có cấu trúc rõ ràng để bổ sung language model, streaming inference, diarization hoặc VAD.

### English

- **Wav2Vec 2.0 fine-tuning**: Train a Vietnamese ASR model with CTC loss.
- **Complete data pipeline**: Prepare, validate, merge, and normalize speech datasets.
- **Kaggle GPU training**: Designed for Kaggle, Google Colab, or Jupyter Notebook environments.
- **Flask web demo**: Supports microphone recording, audio upload, and transcription display.
- **FastAPI REST API**: Provides upload and transcription endpoints for integration.
- **Automated evaluation**: Computes WER and CER and saves results under `results/`.
- **Model optimization**: Supports dynamic quantization, ONNX export, and batch inference.
- **Text post-processing**: Can integrate LLM-based correction for punctuation, formatting, and context refinement.
- **Extensible design**: Structured for future language-model decoding, streaming inference, diarization, or VAD.

---

## 3. Kiến trúc hệ thống | System Architecture

### Tiếng Việt

Hệ thống sử dụng mô hình **Wav2Vec 2.0 base** được pre-train cho tiếng Việt, sau đó fine-tune bằng **Connectionist Temporal Classification (CTC)**. Dữ liệu âm thanh được chuẩn hóa về 16 kHz, đưa qua feature encoder, transformer context network, CTC classification head và decoder để tạo văn bản tiếng Việt.

### English

The system uses a Vietnamese **Wav2Vec 2.0 base** model and fine-tunes it with **Connectionist Temporal Classification (CTC)**. Audio is normalized to 16 kHz, processed by the feature encoder, transformer context network, CTC classification head, and decoder to generate Vietnamese text.

```text
Audio Input
    |
    v
Audio Loading and Normalization
    |
    v
Wav2Vec 2.0 Feature Encoder
    |
    v
Transformer Context Network
    |
    v
CTC Classification Head
    |
    v
Greedy Decoding / Beam Search Decoder
    |
    v
Vietnamese Text Output
    |
    +-------------------------+
    |                         |
    v                         v
Flask Web Demo           FastAPI REST API
Port 5000                Port 8000
```

---

## 4. Công nghệ sử dụng | Technology Stack

| Layer | Công nghệ | Purpose |
|---|---|---|
| Model | Wav2Vec 2.0, CTC | Vietnamese speech recognition |
| Deep Learning | PyTorch, Torchaudio | Training and inference |
| NLP/ASR Toolkit | Hugging Face Transformers, Datasets | Model loading, tokenizer, training utilities |
| Audio Processing | Librosa, SoundFile, Pydub | Audio loading and conversion |
| Evaluation | JiWER | WER/CER calculation |
| Web Demo | Flask, Flask-CORS | Browser-based demo interface |
| API Server | FastAPI, Uvicorn | REST API and Swagger documentation |
| Data Analysis | NumPy, Pandas | Dataset statistics and result analysis |
| Notebook Runtime | Kaggle, Google Colab, Jupyter | Training and experimentation |
| Optimization | ONNX, Dynamic Quantization | Deployment and inference optimization |

---

## 5. Cấu trúc thư mục | Project Structure

```text
VietASR-Pro/
├── README.md                          # Main bilingual project documentation
├── README.md                  # Detailed bilingual installation guide
├── PAPER_VietASR_Pro.md               # Research/project report
├── LICENSE                            # License information
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignored files
├── train.py                           # Training entry point
├── run_server.py                      # FastAPI server entry point
├── run_evaluation.py                  # Evaluation entry point
│
├── configs/                           # Training and runtime configuration
│   └── training_config.yaml
│
├── data/                              # Dataset preparation and raw data folder
│   ├── README.md
│   ├── preprocessing/
│   │   ├── check_dataset.py
│   │   ├── prepare_dataset.py
│   │   ├── prepare_vivos.py
│   │   ├── prepare_full_dataset.py
│   │   ├── create_vlsp_jsonl.py
│   │   ├── merge_vivos_vlsp_jsonl.py
│   │   ├── convert_to_absolute_paths.py
│   │   └── convert_to_relative_paths.py
│   └── raw/                           # Raw datasets; should not be committed
│
├── src/                               # Main source code
│   ├── api/                           # FastAPI implementation
│   │   └── server.py
│   ├── model/                         # Model-related modules
│   ├── training/                      # Training pipeline
│   │   ├── train_wav2vec2.py
│   │   └── language_model.py
│   ├── evaluation/                    # WER/CER evaluation
│   │   └── evaluate.py
│   ├── data/                          # Core preprocessing modules
│   │   ├── preprocessing.py
│   │   ├── prepare_dataset.py
│   │   ├── prepare_vivos_only.py
│   │   └── normalize_audio.py
│   └── utils/                         # Optimization, profiling, demo utilities
│       ├── optimization.py
│       ├── profiling.py
│       └── demo.py
│
├── notebooks/                         # Colab/Kaggle notebooks
│   ├── colab_train_FINAL.ipynb
│   └── kaggle_train.ipynb
│
├── app/                               # Flask web demo
│   ├── app.py
│   ├── post_processing_config.json
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── css/style.css
│       └── js/main.js
│
├── tests/                             # Unit and integration tests
│   ├── test_kaggle_model.py
│   ├── test_model.py
│   ├── test_retry.py
│   └── test_upload_api.py
│
├── scripts/                           # Helper scripts
│   ├── check_audio_path.py
│   ├── check_training.py
│   ├── check_vocab.py
│   ├── check_dependencies.py
│   ├── quick_commands.bat
│   ├── run_pipeline.bat
│   ├── run_pipeline.sh
│   ├── setup/
│   ├── migration/
│   └── profiling/
│
├── results/                           # Evaluation output and logs
│   ├── training_history.csv
│   ├── audio_stats.json
│   ├── dataset_errors.csv
│   ├── unknown_chars.txt
│   └── checkpoints/
│
├── docs/                              # Additional documentation
├── third_party/                       # Third-party tools and references
├── final_model/                       # Fine-tuned model files; large files are gitignored
└── token/                             # Local tokens; must not be committed
```

---

## 6. Yêu cầu hệ thống | System Requirements

| Component | Minimum | Recommended |
|---|---:|---:|
| Python | 3.9+ | 3.10+ |
| RAM | 4 GB | 8-16 GB |
| Disk Space | 2 GB | 5 GB or more |
| GPU for inference | Not required | NVIDIA CUDA, AMD ROCm, or Apple MPS |
| GPU for training | Not recommended on CPU | Kaggle T4/P100 or stronger |
| OS | Windows 10/11, macOS 12+, Ubuntu 20.04+ | Latest stable OS version |

Core dependencies:

```text
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
accelerate>=0.24.0
librosa>=0.10.0
soundfile>=0.12.0
jiwer>=3.0.0
flask>=3.0.0
flask-cors>=4.0.0
fastapi>=0.104.0
uvicorn>=0.24.0
pydub>=0.25.1
numpy>=1.24.0
pandas>=2.0.0
```

---
## 7. Hướng dẫn cài đặt chi tiết | Detailed Installation Guide

Phần này gộp nội dung từ `README.md` vào tài liệu chính để project chỉ cần duy trì một file README duy nhất.

This section merges the installation guide into the main README so the project can be maintained with a single README file.

### 7.0 Cài đặt nhanh | Quick Start

### Tiếng Việt

```bash
# 1. Clone repository
git clone https://github.com/CheeseThuong/Vietnamese-asr.git
cd Vietnamese-asr

# 2. Tạo môi trường ảo
python -m venv .venv

# 3. Kích hoạt môi trường ảo
# Windows
.venv\Scripts\activate

# Linux/macOS
# source .venv/bin/activate

# 4. Cài đặt thư viện
pip install --upgrade pip
pip install -r requirements.txt

# 5. Chạy Flask web demo
python -m app.app
```

Mở trình duyệt tại:

```text
http://localhost:5000
```

### English

```bash
# 1. Clone repository
git clone https://github.com/CheeseThuong/Vietnamese-asr.git
cd Vietnamese-asr

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the environment
# Windows
.venv\Scripts\activate

# Linux/macOS
# source .venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Run the Flask web demo
python -m app.app
```

Open the browser at:

```text
http://localhost:5000
```


---

### 7.0.1 Ghi chú nhanh | Quick Notes

- Nếu chỉ chạy demo, CPU vẫn có thể sử dụng được, nhưng GPU giúp inference nhanh hơn.
- Nếu fine-tune đầy đủ Wav2Vec 2.0, nên dùng Kaggle GPU T4/P100 hoặc GPU tương đương.
- Không commit các file model lớn như `model.safetensors`, checkpoint hoặc Hugging Face token lên Git.

- CPU inference is supported, but GPU acceleration is recommended for faster transcription.
- Full Wav2Vec 2.0 fine-tuning should be done on Kaggle T4/P100 or an equivalent GPU.
- Do not commit large model files such as `model.safetensors`, checkpoints, or Hugging Face tokens to Git.

### 7.1 Yêu cầu hệ thống | System Requirements

| Component | Minimum | Recommended |
|---|---:|---:|
| Python | 3.9+ | 3.10+ |
| RAM | 4 GB | 8-16 GB |
| Disk Space | 2 GB | 5 GB or more |
| GPU | Not required for inference | CUDA/ROCm/Apple MPS for faster inference |
| Training GPU | Not recommended on CPU | Kaggle T4/P100 or stronger |
| OS | Windows 10/11, macOS 12+, Ubuntu 20.04+ | Latest stable version |

#### Notes

- Inference can run on CPU, but GPU is recommended for faster processing.
- Full Wav2Vec 2.0 fine-tuning should be done on GPU.
- Large model files such as `model.safetensors` should not be committed to Git.

---

### 7.2 Cài đặt tự động | Automatic Installation

#### 2.1 Windows

Run the provided installer if it exists in the project root:

```bat
install.bat
```

Expected behavior:

- Create or reuse a Python virtual environment.
- Install required Python packages.
- Start the Flask web demo.
- Open the local web interface at `http://localhost:5000`, if the script supports auto-open.

#### 2.2 Linux/macOS

```bash
chmod +x install.sh
./install.sh
```

If the script is not available or fails, use the manual installation method below.

---

### 7.3 Cài đặt thủ công | Manual Installation

#### 3.1 Clone Repository

```bash
git clone https://github.com/CheeseThuong/Vietnamese-asr.git
cd Vietnamese-asr
```

#### 3.2 Create Virtual Environment

Windows:

```bat
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3.3 Upgrade pip

```bash
python -m pip install --upgrade pip setuptools wheel
```

#### 3.4 Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3.5 Verify Installation

```bash
python --version
pip list
```

Optional dependency check:

```bash
python scripts/check_dependencies.py
```

---

### 7.4 Cài đặt PyTorch theo thiết bị | Install PyTorch by Device

If `pip install -r requirements.txt` cannot install the correct PyTorch build, install PyTorch manually first.

#### 4.1 NVIDIA CUDA

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 4.2 CPU Only

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 4.3 AMD ROCm on Linux

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

#### 4.4 Apple Silicon / macOS

```bash
pip install torch torchaudio
```

After installation, test PyTorch:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

### 7.5 Cấu hình model | Model Configuration

#### 5.1 Use Local Model

Place the trained model inside `final_model/`:

```text
VietASR-Pro/
└── final_model/
    ├── config.json
    ├── preprocessor_config.json
    ├── tokenizer_config.json
    ├── vocab.json
    ├── model.safetensors
    └── trainer_state.json
```

Recommended checks:

```bash
ls final_model
```

Windows PowerShell:

```powershell
Get-ChildItem final_model
```

#### 5.2 Use Hugging Face Model

If a local model is not available, set an environment variable to load a Hugging Face model.

Windows Command Prompt:

```bat
set ASR_MODEL_SOURCE=nguyenvulebinh/wav2vec2-base-vietnamese-250h
```

Windows PowerShell:

```powershell
$env:ASR_MODEL_SOURCE="nguyenvulebinh/wav2vec2-base-vietnamese-250h"
```

Linux/macOS:

```bash
export ASR_MODEL_SOURCE="nguyenvulebinh/wav2vec2-base-vietnamese-250h"
```

#### 5.3 Hugging Face Cache

Downloaded models are usually cached at:

```text
~/.cache/huggingface/
```

To use a custom cache directory:

```bash
export HF_HOME="/path/to/huggingface_cache"
```

Windows PowerShell:

```powershell
$env:HF_HOME="D:\huggingface_cache"
```

---

### 7.6 Cấu hình hậu xử lý văn bản | Text Post-processing Configuration

Optional post-processing can be configured in:

```text
app/post_processing_config.json
```

Example:

```json
{
  "llm": {
    "provider": "gemini",
    "api_key": "YOUR_GEMINI_API_KEY",
    "model": "gemini-flash-latest"
  },
  "post_processing": {
    "enabled": true,
    "normalize_text": true,
    "restore_punctuation": true
  }
}
```

Security recommendations:

- Do not commit API keys to GitHub.
- Add local config files containing secrets to `.gitignore`.
- Prefer environment variables for private credentials.

Example using environment variables:

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

Linux/macOS:

```bash
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

---

### 7.7 Chạy ứng dụng | Run the Application

#### 7.1 Flask Web Demo

```bash
python -m app.app
```

Open:

```text
http://localhost:5000
```

#### 7.2 FastAPI Server

```bash
python run_server.py
```

Alternative command:

```bash
uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

#### 7.3 Evaluation

```bash
python run_evaluation.py
```

Expected output folder:

```text
results/
```

#### 7.4 Training

For full training, use Kaggle or Google Colab notebooks:

```text
notebooks/kaggle_train.ipynb
notebooks/colab_train_FINAL.ipynb
```

Direct entry point:

```bash
python train.py
```

---

### 7.8 Chạy trên Kaggle hoặc Google Colab | Run on Kaggle or Google Colab

#### Kaggle

1. Create a new Kaggle Notebook.
2. Enable GPU in Notebook Settings.
3. Upload or attach the project dataset.
4. Open `notebooks/kaggle_train.ipynb`.
5. Make sure output paths point to Kaggle's writable directory, for example:

```text
/kaggle/working/
```

Recommended outputs:

```text
/kaggle/working/final_model/
/kaggle/working/results/
/kaggle/working/checkpoints/
```

#### Google Colab

1. Open `notebooks/colab_train_FINAL.ipynb`.
2. Select GPU runtime.
3. Mount Google Drive if needed.
4. Update dataset paths before running training cells.

---

### 7.9 Kiểm thử sau cài đặt | Post-installation Tests

Run all tests:

```bash
pytest tests/
```

Run selected tests:

```bash
pytest tests/test_model.py
pytest tests/test_upload_api.py
```

Test model loading:

```bash
python -c "from transformers import Wav2Vec2ForCTC; print('Transformers import OK')"
```

Test audio dependencies:

```bash
python -c "import librosa, soundfile; print('Audio dependencies OK')"
```

---

### 7.10 Lỗi thường gặp | Troubleshooting

#### 10.1 Python command not found

Windows:

- Install Python from `python.org`.
- During installation, enable **Add Python to PATH**.
- Restart the terminal after installation.

Ubuntu:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip
```

macOS:

```bash
brew install python@3.11
```

#### 10.2 Virtual environment activation is blocked on Windows

PowerShell may block script execution. Run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
.venv\Scripts\Activate.ps1
```

#### 10.3 Torch installation error

Install the PyTorch build that matches your device. See [Section 4](#4-cài-đặt-pytorch-theo-thiết-bị--install-pytorch-by-device).

#### 10.4 Port 5000 is already in use

Use a different port if the application supports the `--port` argument:

```bash
python -m app.app --port 8080
```

Or find and close the process using port 5000.

Windows:

```bat
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

Linux/macOS:

```bash
lsof -i :5000
kill -9 <PID>
```

#### 10.5 Model download is slow

Possible solutions:

- Use a stable internet connection.
- Pre-download the model from Hugging Face.
- Place the model manually in `final_model/`.
- Use `HF_HOME` to point to a persistent cache folder.

#### 10.6 Missing FFmpeg or audio conversion error

Some audio formats require FFmpeg.

Ubuntu:

```bash
sudo apt install ffmpeg
```

macOS:

```bash
brew install ffmpeg
```

Windows:

- Install FFmpeg.
- Add the FFmpeg `bin` folder to the system PATH.
- Restart the terminal.

#### 10.7 CUDA is not available

Check driver and CUDA availability:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

If CUDA is still unavailable:

- Confirm that the NVIDIA driver is installed.
- Install the correct PyTorch CUDA build.
- Use CPU mode if GPU is not required.

---

### 7.11 Cập nhật project | Update the Project

```bash
git pull
pip install -r requirements.txt --upgrade
```

If dependencies conflict, recreate the virtual environment:

```bash
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows equivalent:

```bat
rmdir /s /q .venv
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

### 7.12 Gỡ cài đặt | Uninstall

To remove the local virtual environment:

Windows:

```bat
rmdir /s /q .venv
```

Linux/macOS:

```bash
rm -rf .venv
```

To remove Hugging Face cache, delete the cache directory only when you no longer need downloaded models:

```text
~/.cache/huggingface/
```

---

### 7.13 Hỗ trợ | Support

If you encounter an issue:

1. Check this installation guide.
2. Confirm that Python, PyTorch, Transformers, and audio libraries are installed correctly.
3. Check whether `final_model/` contains all required model files.
4. Run the test commands in [Section 9](#9-kiểm-thử-sau-cài-đặt--post-installation-tests).
5. Create a GitHub Issue with the error log, operating system, Python version, and command used.

---

### 7.14 Checklist cài đặt nhanh | Quick Installation Checklist

- [ ] Python installed and added to PATH.
- [ ] Virtual environment created and activated.
- [ ] `pip` upgraded.
- [ ] Dependencies installed from `requirements.txt`.
- [ ] PyTorch version matches CPU/GPU setup.
- [ ] Model files are available in `final_model/` or `ASR_MODEL_SOURCE` is configured.
- [ ] Flask app runs at `http://localhost:5000`.
- [ ] FastAPI docs open at `http://localhost:8000/docs`, if using API server.
- [ ] Test audio file can be transcribed successfully.

---
## 8. Hướng dẫn sử dụng | Usage Guide

### 8.1 Chạy Flask Web Demo | Run Flask Web Demo

```bash
python -m app.app
```

Default URL:

```text
http://localhost:5000
```

Main functions:

- Record audio from the microphone.
- Upload audio files such as WAV, MP3, FLAC, OGG, or M4A.
- View Vietnamese transcription output.
- Enable or disable text post-processing.
- Review recognition history if supported by the current UI version.

### 8.2 Chạy FastAPI Server | Run FastAPI Server

```bash
python run_server.py
```

Alternative command:

```bash
uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

### 8.3 Huấn luyện mô hình | Train the Model

Recommended notebook workflows:

```text
notebooks/kaggle_train.ipynb
notebooks/colab_train_FINAL.ipynb
```

Direct Python entry point:

```bash
python train.py
```

Training outputs should be saved to:

```text
results/
final_model/
```

### 8.4 Đánh giá mô hình | Evaluate the Model

```bash
python run_evaluation.py
```

Expected outputs:

```text
results/training_history.csv
results/evaluation_results.csv
results/audio_stats.json
```

### 8.5 Tối ưu hóa mô hình | Optimize the Model

Optimization utilities are located in:

```text
src/utils/optimization.py
```

Supported directions:

- Dynamic quantization for smaller CPU inference.
- ONNX export for deployment.
- Batch inference for multiple audio files.

---

## 9. API Endpoints

The exact route names may depend on the current implementation in `app/app.py` and `src/api/server.py`. The project commonly supports the following endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serve the web interface |
| POST | `/api/transcribe` | Transcribe a recorded or uploaded audio file |
| POST | `/api/upload` | Upload audio and return transcription |
| POST | `/api/chatbot` | Run helper functions such as normalization or IPA conversion, if enabled |
| POST | `/api/summarize` | Apply post-processing or text refinement, if enabled |
| GET | `/api/status` | Return model/server status |
| GET | `/api/device-info` | Return available device information |
| POST | `/api/device-select` | Select or reload inference device, if supported |
| GET | `/health` | Health-check endpoint for deployment |

Example request with `curl`:

```bash
curl -X POST "http://localhost:8000/api/transcribe" \
  -F "file=@sample.wav"
```

---

## 10. Kết quả thực nghiệm | Experimental Results

### Tiếng Việt

Mô hình tốt nhất được huấn luyện trên Kaggle GPU với dữ liệu VIVOS + VLSP 2020 / viet_bud500. Kết quả cho thấy việc sử dụng GPU, dữ liệu lớn hơn và cấu hình huấn luyện phù hợp đã cải thiện đáng kể so với lần huấn luyện local trước đó.

### English

The best model was trained on Kaggle GPU with VIVOS + VLSP 2020 / viet_bud500. The results show that GPU training, a larger dataset, and a more suitable training configuration significantly improved performance compared with the previous local training attempt.

| Checkpoint | Steps | Eval Loss | WER (%) | Note |
|---|---:|---:|---:|---|
| checkpoint-1000 | 1,000 | 0.1363 | 15.57 | Baseline checkpoint |
| checkpoint-2000 | 2,000 | 0.1207 | 15.07 | Improved |
| checkpoint-3000 | 3,000 | 0.1140 | 14.82 | Improved |
| checkpoint-5000 | 5,000 | 0.1071 | 14.59 | Improved |
| checkpoint-7000 | 7,000 | 0.0965 | 14.27 | Improved |
| checkpoint-9000 | 9,000 | 0.0879 | 13.96 | Improved |
| checkpoint-10000 | 10,000 | 0.0844 | 13.90 | Improved |
| checkpoint-11000 | 11,000 | 0.0818 | 13.72 | Improved |
| checkpoint-12000 | 12,000 | 0.0756 | 13.49 | Improved |
| checkpoint-14000 | 14,000 | 0.0732 | 13.41 | Near best |
| **checkpoint-15000** | **15,000** | **0.0664** | **13.31** | **Best checkpoint** |

Comparison with the previous local training attempt:

| Model | Training Setup | WER (%) | Result |
|---|---|---:|---|
| Local Training | CPU, smaller data, earlier preprocessing | 100.00 | Failed to learn effectively |
| Kaggle Training | GPU, larger dataset, improved configuration | 13.31 | Successful fine-tuning |

Main lessons learned:

- Data preprocessing quality strongly affects ASR performance.
- CPU-only training is not practical for full Wav2Vec 2.0 fine-tuning.
- Larger and cleaner Vietnamese speech data improves model generalization.
- Frequent checkpoint evaluation helps identify the best model before overfitting.

---

## 11. Dữ liệu | Datasets

| Dataset | Vietnamese Description | English Description | Approx. Size | Source |
|---|---|---|---:|---|
| VIVOS | Bộ dữ liệu tiếng Việt chất lượng cao với nhiều người nói | High-quality Vietnamese speech dataset | About 15 hours | [VIVOS](https://ailab.hcmus.edu.vn/vivos) |
| VLSP 2020 / viet_bud500 | Bộ dữ liệu tiếng Việt quy mô lớn dùng để mở rộng huấn luyện | Larger Vietnamese speech dataset for training expansion | Large-scale | [Hugging Face Dataset](https://huggingface.co/datasets/doof-ferb/vlsp2020_vinai_100h) |

Vocabulary configuration:

- Vietnamese characters with diacritics.
- Digits and basic punctuation when supported by tokenizer configuration.
- Special tokens such as `<pad>`, `<unk>`, and word separator `|`.
- Current vocabulary size: approximately 110 tokens, depending on the tokenizer file used.

Dataset notes:

- Raw audio data should be placed under `data/raw/`.
- Large datasets and model weights should not be committed to Git.
- Use JSONL/CSV manifests with absolute or relative paths depending on the training environment.
- Always validate audio paths before starting long training jobs.

---

## 12. Cấu hình | Configuration

### 12.1 Local Model

Place the trained model inside `final_model/`:

```text
final_model/
├── config.json
├── preprocessor_config.json
├── tokenizer_config.json
├── vocab.json
├── model.safetensors
└── trainer_state.json
```

### 12.2 Hugging Face Model Source

If no local model exists, the application can be configured to load a model from Hugging Face.

Windows PowerShell:

```powershell
$env:ASR_MODEL_SOURCE="nguyenvulebinh/wav2vec2-base-vietnamese-250h"
```

Windows Command Prompt:

```bat
set ASR_MODEL_SOURCE=nguyenvulebinh/wav2vec2-base-vietnamese-250h
```

Linux/macOS:

```bash
export ASR_MODEL_SOURCE="nguyenvulebinh/wav2vec2-base-vietnamese-250h"
```

### 12.3 Post-processing API Configuration

Optional text post-processing can be configured in:

```text
app/post_processing_config.json
```

Example:

```json
{
  "llm": {
    "provider": "gemini",
    "api_key": "YOUR_API_KEY_HERE",
    "model": "gemini-flash-latest"
  },
  "post_processing": {
    "enabled": true,
    "normalize_text": true,
    "restore_punctuation": true
  }
}
```

Security note:

- Do not commit API keys, Hugging Face tokens, or private credentials.
- Keep local secrets in `.env`, environment variables, or ignored config files.

---

## 13. Kiểm thử | Testing

Run all tests:

```bash
pytest tests/
```

Run selected tests:

```bash
pytest tests/test_model.py
pytest tests/test_upload_api.py
pytest tests/test_kaggle_model.py
```

Recommended checks before submission or deployment:

- Confirm that `requirements.txt` installs successfully in a clean environment.
- Confirm that the Flask app starts without missing model files.
- Test at least one short WAV file and one uploaded compressed audio file.
- Check that WER/CER evaluation runs and saves output to `results/`.
- Make sure private tokens and model weights are not accidentally committed.

---

## 14. Giới hạn hiện tại | Current Limitations

### Tiếng Việt

- Mô hình có thể nhận dạng sai khi âm thanh nhiều nhiễu, người nói quá nhanh hoặc audio có nhiều người nói chồng lên nhau.
- Kết quả có thể thiếu dấu câu nếu không bật hậu xử lý văn bản.
- Streaming inference và speaker diarization chưa phải là chức năng ổn định trong phiên bản hiện tại.
- Kết quả WER phụ thuộc mạnh vào chất lượng tập test, cách chuẩn hóa transcript và cấu hình decoder.

### English

- The model may produce errors with noisy audio, very fast speech, or overlapping speakers.
- Output may lack punctuation when text post-processing is disabled.
- Streaming inference and speaker diarization are not yet stable production features in the current version.
- WER depends heavily on test-set quality, transcript normalization, and decoder configuration.

---

## 15. Hướng phát triển | Roadmap

- [ ] Integrate KenLM or another Vietnamese language model for beam-search decoding.
- [ ] Improve punctuation restoration and text normalization.
- [ ] Add Voice Activity Detection before transcription.
- [ ] Add streaming inference for real-time speech recognition.
- [ ] Add speaker diarization for multi-speaker audio.
- [ ] Export and benchmark ONNX inference.
- [ ] Add Docker support for easier deployment.
- [ ] Add CI/CD workflow for tests and linting.
- [ ] Prepare a hosted demo version with clear usage limits.
- [ ] Expand domain-specific fine-tuning for education, customer service, legal, or medical speech.

---

## 16. Tác giả | Author

| Field | Information |
|---|---|
| Name | Nguyễn Trí Thượng |
| Project | Vietnamese Speech Recognition using Wav2Vec 2.0 |
| Category | Artificial Intelligence Course Project |
| Academic Year | 2025-2026 |

---

## 17. Giấy phép | License

### Tiếng Việt

Dự án được phát triển cho mục đích học tập và nghiên cứu. Mã nguồn có thể được sử dụng cho mục đích phi thương mại, học thuật hoặc thử nghiệm. Khi sử dụng lại, vui lòng ghi rõ nguồn dự án và tôn trọng giấy phép của các dataset/model bên thứ ba.

### English

This project is developed for academic and research purposes. The source code may be used for non-commercial, educational, or experimental use. When reusing this project, please provide proper attribution and respect the licenses of all third-party datasets and models.

---

## 18. Lời cảm ơn | Acknowledgements

- **Hugging Face** for Transformers, Datasets, model hosting, and open-source tooling.
- **PyTorch** and **Torchaudio** for deep learning and audio-processing support.
- **VIVOS dataset team** at AILAB HCMUS for the Vietnamese speech dataset.
- **VLSP 2020 community and related dataset providers** for Vietnamese speech resources.
- **nguyenvulebinh** for the Vietnamese Wav2Vec2 pre-trained model.
- **Kaggle** for accessible GPU resources used during experimentation.
- **Flask** and **FastAPI** for web demo and API deployment frameworks.

---

## 19. Tài liệu tham khảo | References

1. Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*. arXiv:2006.11477.
2. VIVOS Corpus: https://ailab.hcmus.edu.vn/vivos
3. Wav2Vec2 Vietnamese Pre-trained Model: https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h
4. Hugging Face Transformers Documentation: https://huggingface.co/docs/transformers
5. Hugging Face Datasets Documentation: https://huggingface.co/docs/datasets
6. PyTorch Documentation: https://pytorch.org/docs/stable/index.html
7. Torchaudio Documentation: https://pytorch.org/audio/stable/index.html
8. JiWER Documentation: https://jitsi.github.io/jiwer/
9. Flask Documentation: https://flask.palletsprojects.com/
10. FastAPI Documentation: https://fastapi.tiangolo.com/
11. KenLM Repository: https://github.com/kpu/kenlm

---

<div align="center">

**VietASR Pro**  
Vietnamese Automatic Speech Recognition System  
Nguyễn Trí Thượng · 2025-2026

</div>
