# Hướng dẫn Fix Lỗi Audio Files

## Vấn đề

Lỗi bạn gặp phải:
```
Error opening 'Data\\vivos\\vivos\\test\\waves\\VIVOSDEV18\\VIVOSDEV18_191.wav': System error.
```

## Nguyên nhân

**Đường dẫn relative trong JSONL files** - Khi chạy training từ thư mục khác (ví dụ: `/content/Vietnamese-asr` trên Colab), Python không tìm thấy file vì đường dẫn tương đối `Data\vivos\...` không còn đúng.

## Giải pháp đã áp dụng

### 1. ✅ Chuyển đổi paths sang absolute

Đã chạy script `convert_to_absolute_paths.py` để:
- Chuyển tất cả đường dẫn từ relative → absolute
- Files đã fix:
  - `processed_data_vivos/train.jsonl` - 10,494 samples
  - `processed_data_vivos/validation.jsonl` - 1,166 samples  
  - `processed_data_vivos/test.jsonl` - 760 samples

### 2. ✅ Cải thiện error handling

Đã cập nhật `src/data/preprocessing.py` để:
- Tự động **bỏ qua** các file audio bị lỗi
- In thông báo warning thay vì crash
- Tiếp tục training với các file còn lại

## Upload lại dataset lên Google Drive

### Bước 1: Xóa files cũ trên Drive

Vào Google Drive → `MyDrive/VietnameseASR/data/` → Xóa 3 files cũ:
- ❌ `train.jsonl` (cũ)
- ❌ `validation.jsonl` (cũ)
- ❌ `test.jsonl` (cũ)

### Bước 2: Upload files mới với absolute paths

Upload 3 files đã fix từ folder `processed_data_vivos/`:
- ✅ `train.jsonl` (mới - có absolute paths)
- ✅ `validation.jsonl` (mới - có absolute paths)
- ✅ `test.jsonl` (mới - có absolute paths)

**QUAN TRỌNG**: Đảm bảo upload đúng vào folder:
```
MyDrive/VietnameseASR/data/
```

### Bước 3: Cập nhật code trên Colab

Trong Colab notebook, Cell 5 (Clone repo) sẽ tự động pull code mới:

```python
# Cell 5 đã có logic pull code mới
if os.path.exists('/content/Vietnamese-asr'):
    print("⚠️ Repository already exists, using existing copy...")
    os.chdir('/content/Vietnamese-asr')
    !git pull origin main  # ← Tự động cập nhật
```

**Hoặc restart runtime để clone lại từ đầu:**
- Runtime → Restart runtime
- Chạy lại từ Cell 1

## Tại sao absolute paths tốt hơn?

| Relative Paths | Absolute Paths |
|---------------|----------------|
| `Data\vivos\test\...` | `D:\Projects\Đồ án trí tuệ nhân tạo\Data\vivos\test\...` |
| ❌ Phụ thuộc working directory | ✅ Luôn đúng mọi nơi |
| ❌ Lỗi khi chạy từ folder khác | ✅ Hoạt động mọi context |
| ❌ Yêu cầu chdir() trước | ✅ Không cần setup |

## Lưu ý cho Colab

**KHÔNG cần thay đổi** trong Colab vì:
1. Colab sẽ clone repo mới từ GitHub (có code fix)
2. Files JSONL trên Drive có absolute paths
3. Code preprocessing đã có error handling

Chỉ cần:
1. ✅ Upload 3 files JSONL mới lên Drive (thay thế files cũ)
2. ✅ Restart Colab runtime (nếu cần)
3. ✅ Chạy lại từ Cell 1

## Kiểm tra nhanh

Sau khi upload, chạy Cell 8 (Check Dataset) sẽ thấy:

```
✅ All dataset files found!
   - train.jsonl: 10,494 samples
   - validation.jsonl: 1,166 samples
   - test.jsonl: 760 samples
```

Nếu vẫn lỗi, chạy Cell DEBUG để check paths:

```python
# Test đọc 1 sample
import json
with open(f"{DRIVE_ROOT}/data/train.jsonl") as f:
    sample = json.loads(f.readline())
    print(f"Audio path: {sample['audio_path']}")
    # Nên thấy absolute path bắt đầu bằng D:\Projects\...
```

## Checklist hoàn thành

- [x] Fix code xử lý lỗi (`src/data/preprocessing.py`)
- [x] Chuyển đổi paths sang absolute (`convert_to_absolute_paths.py`)
- [x] Commit và push lên GitHub
- [ ] Upload 3 files JSONL mới lên Google Drive
- [ ] Chạy lại training trên Colab

## Files đã tạo/sửa

1. `convert_to_absolute_paths.py` - Script convert paths
2. `fix_corrupted_audio.py` - Script kiểm tra file lỗi
3. `src/data/preprocessing.py` - Cải thiện error handling
4. `processed_data_vivos/*.jsonl` - Files JSONL với absolute paths

---

**Nếu vẫn gặp lỗi**, gửi cho tôi:
1. Output của Cell 8 (Check Dataset)
2. Đường dẫn 1 file bị lỗi
3. Kết quả chạy: `!ls -la "$DRIVE_ROOT/data/"`
