# 📘 Hướng Dẫn Cài Đặt VietASR Pro

## 📋 Yêu cầu hệ thống

| Thành phần | Yêu cầu tối thiểu |
|---|---|
| **Python** | 3.9 trở lên |
| **RAM** | 4 GB (khuyến nghị 8 GB) |
| **Ổ cứng** | 2 GB trống (cho model + dependencies) |
| **GPU** | Không bắt buộc (hỗ trợ CUDA, ROCm, Apple MPS) |
| **OS** | Windows 10/11, macOS 12+, Ubuntu 20.04+ |

---

## 🚀 Cài đặt 1-click

### Windows

1. **Double-click** file `install.bat`
2. Chờ quá trình cài đặt hoàn tất (~5-10 phút lần đầu)
3. Trình duyệt sẽ tự động mở tại `http://localhost:5000`

### Linux / macOS

```bash
# Cấp quyền thực thi (chỉ cần lần đầu)
chmod +x install.sh

# Chạy installer
./install.sh
```

---

## 🔧 Cài đặt thủ công (nếu cần)

```bash
# 1. Tạo virtual environment
python -m venv .venv

# 2. Kích hoạt
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

# 3. Cài dependencies
pip install -r requirements.txt

# 4. Chạy server
python -m app.app
```

---

## 📦 Cấu hình Model

### Sử dụng model local

Đặt model đã train vào thư mục `final_model/`:

```
VietASR-Pro/
└── final_model/
    ├── config.json
    ├── preprocessor_config.json
    ├── tokenizer_config.json
    ├── vocab.json
    └── model.safetensors
```

### Sử dụng model từ Hugging Face

Nếu không có model local, hệ thống tự động tải model từ Hugging Face.
Có thể cấu hình qua biến môi trường:

```bash
set ASR_MODEL_SOURCE=nguyenvulebinh/wav2vec2-large-vi-vlsp2020
```

---

## 🔑 Cấu hình API (Post-processing)

### Gemini API (mặc định)

Mở file `app/post_processing_config.json` và cập nhật API key:

```json
{
    "llm": {
        "provider": "gemini",
        "api_key": "YOUR_GEMINI_API_KEY",
        "model": "gemini-flash-latest"
    }
}
```

### Bật/tắt post-processing

Có thể cấu hình trực tiếp trong giao diện web tại mục **Cài Đặt** → **Hậu xử lý văn bản**.

---

## ❓ Xử lý lỗi thường gặp

### Python không tìm thấy

- **Windows**: Tải từ [python.org](https://www.python.org/downloads/), tick **"Add Python to PATH"** khi cài
- **Ubuntu**: `sudo apt install python3 python3-venv python3-pip`
- **macOS**: `brew install python@3.11`

### Lỗi cài torch (GPU)

```bash
# CUDA (NVIDIA):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# ROCm (AMD):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# CPU only:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Port 5000 bị chiếm

```bash
# Đổi port:
python -m app.app --port 8080
```

### Model tải quá lâu

Model Hugging Face được cache tại `~/.cache/huggingface/`. Lần tải đầu tiên có thể mất 5-15 phút tùy tốc độ mạng.

---

## 📞 Liên hệ hỗ trợ

Nếu gặp vấn đề, vui lòng tạo Issue trên GitHub repository hoặc liên hệ nhóm phát triển.
