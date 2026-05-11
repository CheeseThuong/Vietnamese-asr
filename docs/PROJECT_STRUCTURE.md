# Project Structure — VietASR Pro

> Tài liệu mô tả cấu trúc thư mục sau khi tổ chức lại (reorganized).  
> Document describing the directory structure after reorganization.

## Tổng quan / Overview

```
VietASR-Pro/
├── README.md                 # Tài liệu chính (song ngữ VI/EN)
├── PAPER_VietASR_Pro.md      # Bài báo khoa học (tiếng Việt)
├── LICENSE                   # Giấy phép
├── requirements.txt          # Dependencies (hợp nhất)
├── .gitignore                # Git ignore rules
│
├── configs/                  # Cấu hình huấn luyện
├── data/                     # Xử lý & lưu trữ dữ liệu
│   ├── preprocessing/        # Scripts chuẩn bị dữ liệu
│   └── raw/                  # Dữ liệu thô (gitignored)
│
├── src/                      # Mã nguồn chính
│   ├── api/                  # FastAPI server
│   ├── model/                # Định nghĩa model
│   ├── training/             # Pipeline huấn luyện
│   ├── evaluation/           # Đánh giá WER/CER
│   ├── data/                 # Data processing module
│   └── utils/                # Tiện ích (quantization, ONNX, ...)
│
├── notebooks/                # Jupyter notebooks (Colab/Kaggle)
├── app/                      # Flask Web Demo (port 5000)
├── tests/                    # Unit/Integration tests
├── scripts/                  # Utility scripts
├── results/                  # Kết quả, logs, checkpoints cũ
├── docs/                     # Tài liệu bổ sung
├── third_party/              # Công cụ bên thứ ba (PyFlame)
├── final_model/              # Model Kaggle tốt nhất (gitignored)
└── token/                    # HF token (gitignored)
```

## Quy tắc / Conventions

1. **`src/`** chứa tất cả mã nguồn core — training, evaluation, API, utils
2. **`app/`** chứa Flask web demo — frontend + backend
3. **`data/preprocessing/`** chứa scripts xử lý dataset
4. **`scripts/`** chứa utility scripts (check, setup, migration)
5. **`tests/`** chứa test files
6. **`docs/`** chứa tài liệu bổ sung
7. **`third_party/`** chứa công cụ bên ngoài không phải core project
8. **`results/`** chứa outputs — training history, errors, checkpoints cũ

## Files gitignored (không push lên Git)

- `token/` — Hugging Face API token
- `final_model/` — Model weights (~377MB)
- `data/raw/` — Raw audio datasets
- `results/checkpoints/` — Large checkpoint files
- `*.safetensors`, `*.bin`, `*.wav`, `*.mp3` — Large binary files
