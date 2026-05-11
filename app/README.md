# Vietnamese ASR Web Demo

**Web application đầy đủ cho nhận dạng giọng nói tiếng Việt**

---

## Tổng Quan

Web demo này cung cấp giao diện đẹp và đầy đủ tính năng để:
- Nhận dạng giọng nói tiếng Việt real-time
- Ghi âm liên tục hoặc từng đoạn
- Chatbot hỗ trợ phiên âm và chuẩn hóa vùng miền
- Tích hợp với model ASR đã train

## Tính Năng Chính

### 1. Nhận Dạng Giọng Nói
- Hỗ trợ tiếng Việt chính xác (3 vùng miền)
- Ghi âm liên tục hoặc từng đoạn
- Hiển thị kết quả theo thời gian thực
- Sử dụng Web Speech API + Model ASR

### 2. Chatbot Phiên Âm
- Phiên âm tiếng Việt sang IPA
- Chuẩn hóa từ vùng miền sang tiếng Việt chuẩn
- Đếm từ và ký tự tự động
- Hỗ trợ trực tuyến 24/7

### 3. Công Cụ Hữu Ích
- Đếm số từ tự động
- Sao chép văn bản nhanh
- Xóa và bắt đầu lại
- Lịch sử nhận dạng
- Phím tắt keyboard

### 4. Giao Diện
- Thiết kế hiện đại với gradient đẹp mắt
- Responsive trên mọi thiết bị
- Trạng thái rõ ràng (đang nghe, dừng, lỗi)
- Không dùng icon/emoji (tránh bug)

## Cấu Trúc Project

```
web_demo/
├── app.py                      # Flask backend
├── requirements.txt            # Python dependencies
├── templates/
│   └── index.html             # Main HTML page
├── static/
│   ├── css/
│   │   └── style.css         # Styles
│   └── js/
│       └── main.js           # JavaScript logic
└── README.md                  # This file
```

## Cài Đặt

### Bước 1: Clone hoặc vào thư mục

```bash
cd "D:\Projects\do_an_tri_tue_nhan_tao\web_demo"
```

### Bước 2: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 3: Kiểm tra model

Đảm bảo model đã được train ở thư mục:
```
D:\Projects\do_an_tri_tue_nhan_tao\results\final_model\
```

Nếu chưa có, web vẫn chạy được với Web Speech API.

Nếu bạn muốn load model trực tiếp từ Hugging Face, đặt một trong các biến môi trường sau trước khi chạy `app.py`:

```bash
set ASR_MODEL_SOURCE=doof-ferb/vlsp2020_vinai_100h
```

Hoặc dùng URL đầy đủ:

```bash
set ASR_MODEL_SOURCE=https://huggingface.co/datasets/doof-ferb/vlsp2020_vinai_100h
```

Lưu ý: repo này phải chứa đầy đủ file model Wav2Vec2 (`config.json`, tokenizer, weights, processor). Nếu chỉ là dataset, backend sẽ tự fallback sang model local hoặc pretrained khác.

### Bước 4: Chạy server

```bash
python app.py
```

### Bước 5: Mở trình duyệt

Truy cập: **http://localhost:5000**

## Yêu Cầu Kỹ Thuật

### Trình duyệt
- **Chrome** (khuyến nghị)
- **Edge** (Chromium)
- **Safari** (macOS)
- Firefox (limited support)

### Quyền truy cập
- **Microphone access** (bắt buộc)
- **Internet connection** (cho Web Speech API)

### Hệ thống
- **Python:** 3.8+
- **RAM:** 4GB+ (8GB khuyến nghị)
- **GPU:** Optional (nếu dùng model ASR)

## Sử Dụng

### Ghi Âm Đơn Giản

1. Nhấn **"Bắt Đầu Ghi Âm"**
2. Nói vào microphone
3. Nhấn **"Dừng Ghi Âm"** khi xong
4. Xem kết quả hiển thị

### Ghi Âm Liên Tục

1. Bật **"Ghi âm liên tục"** trong Cài Đặt
2. Nhấn **"Bắt Đầu Ghi Âm"**
3. Nói thoải mái, hệ thống tự động nhận dạng
4. Nhấn **"Dừng Ghi Âm"** khi muốn dừng

### Sử Dụng Chatbot

1. Click vào **Trợ Lý Phiên Âm** (góc dưới phải)
2. Nhập câu hỏi hoặc chọn Quick Action:
   - **Phiên Âm** - Chuyển văn bản sang IPA
   - **Chuẩn Hóa** - Chuẩn hóa từ vùng miền
   - **Đếm Từ** - Đếm từ và ký tự
3. Nhấn Enter hoặc **Ctrl+Enter** để gửi

### Phím Tắt

- **Ctrl + Enter** - Gửi tin nhắn chatbot
- **Ctrl + C** - Sao chép văn bản (mặc định)
- **Ctrl + L** - Xóa tất cả

## API Endpoints

### GET /
Trang chủ web application

### GET /api/status
Kiểm tra trạng thái model

**Response:**
```json
{
    "model_loaded": true,
    "device": "cuda",
    "status": "ready"
}
```

### POST /api/transcribe
Nhận dạng audio file

**Request:**
- `audio`: Audio file (WAV format)

**Response:**
```json
{
    "success": true,
    "transcription": "xin chào các bạn",
    "word_count": 4,
    "timestamp": "14:30:25"
}
```

### POST /api/chatbot
Chatbot API

**Request:**
```json
{
    "message": "phiên âm xin chào",
    "action": "phonetic",
    "text": "xin chào"
}
```

**Response:**
```json
{
    "success": true,
    "response": "Phiên âm: /sin cau/",
    "timestamp": "14:30:25"
}
```

### POST /api/normalize
Chuẩn hóa vùng miền

**Request:**
```json
{
    "text": "dạ ui chi zô"
}
```

**Response:**
```json
{
    "success": true,
    "original": "dạ ui chi zô",
    "normalized": "vâng vâng gì vào"
}
```

### POST /api/phonetic
Phiên âm sang IPA

**Request:**
```json
{
    "text": "xin chào"
}
```

**Response:**
```json
{
    "success": true,
    "text": "xin chào",
    "phonetic": "sin cau"
}
```

## Cấu Hình

### Thay đổi cổng server

Trong `app.py`, dòng cuối:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

Đổi `port=5000` thành cổng khác nếu cần.

### Thay đổi model path

Trong `app.py`, hàm `load_model()`:
```python
model_path = "results/final_model"
```

Đổi thành đường dẫn model khác nếu cần.

### Thêm từ điển vùng miền

Trong `app.py`, dict `DIALECT_MAPPING`:
```python
DIALECT_MAPPING = {
    "dạ": "vâng",
    "ui": "vâng",
    # Thêm mapping mới ở đây
    "từ_vùng_miền": "từ_chuẩn"
}
```

## Troubleshooting

### Lỗi: Microphone không hoạt động

**Nguyên nhân:** Trình duyệt chặn quyền truy cập

**Giải pháp:**
1. Click biểu tượng khóa/camera trên address bar
2. Cho phép quyền Microphone
3. Reload trang

### Lỗi: Model not loaded

**Nguyên nhân:** Chưa train model hoặc sai đường dẫn

**Giải pháp:**
1. Train model theo hướng dẫn
2. Hoặc web vẫn hoạt động với Web Speech API only
3. Check console logs để xem chi tiết

### Lỗi: Web Speech API không hoạt động

**Nguyên nhân:** Trình duyệt không hỗ trợ

**Giải pháp:**
- Dùng Chrome/Edge (khuyến nghị)
- Cần kết nối internet
- Check console log

### Lỗi: CORS errors

**Nguyên nhân:** Chạy từ file:// thay vì http://

**Giải pháp:**
- PHẢI chạy qua Flask server: `python app.py`
- KHÔNG mở file HTML trực tiếp

## Phát Triển Thêm

### Tính năng có thể thêm:

1. **Export results**
   - Xuất văn bản ra file TXT
   - Xuất audio ra WAV
   - Xuất lịch sử ra JSON

2. **Upload audio files**
   - Nhận dạng từ file có sẵn
   - Batch processing nhiều files

3. **Real-time waveform**
   - Hiển thị waveform khi ghi âm
   - Volume meter

4. **User accounts**
   - Login/Register
   - Lưu lịch sử cá nhân
   - Settings sync

5. **Advanced settings**
   - Noise reduction
   - Voice activity detection
   - Confidence threshold

## Bảo Mật

- ❌ **KHÔNG** lưu audio files lâu dài (xóa sau xử lý)
- ❌ **KHÔNG** gửi dữ liệu nhạy cảm lên server
- ✅ **CÓ** CORS protection
- ✅ **CÓ** Input validation

## Performance

### Tối ưu hóa:

1. **Lazy loading** các dependencies
2. **Caching** model trong memory
3. **Async processing** cho API calls
4. **Debouncing** cho interim results

### Metrics:

- **Transcription latency:** ~500ms (Web Speech API)
- **Model inference:** ~1-2s (CPU), ~100-200ms (GPU)
- **Chatbot response:** ~200-500ms

## Credits

- **Web Speech API** - Google
- **Wav2Vec2** - Facebook AI
- **Flask** - Pallets Projects
- **Design inspiration** - Modern web practices

## License

Educational/Research use only

---

## Quick Start Commands

```bash
# Install
pip install -r requirements.txt

# Run
python app.py

# Open browser
# http://localhost:5000
```

---

**Developed for:** Vietnamese ASR Project  
**Version:** 1.0  
**Last Updated:** February 2026
