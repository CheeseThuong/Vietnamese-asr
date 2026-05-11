<![CDATA[# VietASR Pro: Hệ thống Nhận dạng Tiếng nói Tiếng Việt dựa trên Wav2Vec 2.0 Fine-tuning

**Tác giả:** Nguyễn Trí Thượng

**Tóm tắt bài báo:** Đồ án Trí tuệ Nhân tạo — Năm học 2025–2026

---

## Tóm tắt

Nhận dạng tiếng nói tự động (Automatic Speech Recognition — ASR) cho tiếng Việt là một bài toán đầy thách thức do đặc thù ngôn ngữ: hệ thống 6 thanh điệu, nhiều phương ngữ vùng miền, và nguồn dữ liệu huấn luyện còn hạn chế so với các ngôn ngữ phổ biến như tiếng Anh hay tiếng Trung. Trong bài báo này, chúng tôi trình bày **VietASR Pro** — một hệ thống nhận dạng tiếng nói tiếng Việt dựa trên kiến trúc Wav2Vec 2.0, được fine-tune trên hai bộ dữ liệu chính: VIVOS (~15 giờ, 46 người nói) và VLSP 2020 / viet_bud500 (quy mô lớn). Quá trình huấn luyện được thực hiện trên hạ tầng Kaggle GPU (T4/P100) với 15.000 steps, đạt Word Error Rate (WER) tốt nhất là **13,31%** tại checkpoint-15000. Hệ thống bao gồm pipeline huấn luyện hoàn chỉnh, ứng dụng web demo Flask với giao diện hiện đại, và server API FastAPI. Kết quả cho thấy phương pháp transfer learning từ mô hình pre-trained Wav2Vec 2.0 tiếng Việt kết hợp với dữ liệu VLSP 2020 mang lại hiệu quả vượt trội so với việc huấn luyện từ đầu trên dữ liệu nhỏ.

**Từ khóa:** Nhận dạng tiếng nói, Wav2Vec 2.0, ASR, tiếng Việt, CTC, transfer learning, fine-tuning

---

## 1. Giới thiệu

### 1.1 Bối cảnh nghiên cứu

Nhận dạng tiếng nói tự động (ASR) là công nghệ chuyển đổi tín hiệu âm thanh chứa giọng nói con người thành văn bản. Trong những năm gần đây, lĩnh vực ASR đã có những bước tiến vượt bậc nhờ sự phát triển của các mô hình học sâu (deep learning), đặc biệt là các kiến trúc Transformer và phương pháp self-supervised learning. Các hệ thống ASR tiên tiến như Wav2Vec 2.0 [1], Whisper [2], và HuBERT [3] đã đạt được kết quả gần ngang bằng con người trên nhiều ngôn ngữ phổ biến.

Tuy nhiên, đối với tiếng Việt, việc xây dựng hệ thống ASR chất lượng cao vẫn là một bài toán đầy thách thức. Tiếng Việt là ngôn ngữ thanh điệu với 6 thanh (ngang, huyền, sắc, nặng, hỏi, ngã), trong đó sự khác biệt về thanh điệu có thể thay đổi hoàn toàn ý nghĩa của từ (ví dụ: "ma", "má", "mà", "mả", "mã", "mạ" là 6 từ khác nhau). Ngoài ra, tiếng Việt có sự đa dạng phương ngữ đáng kể giữa ba miền Bắc, Trung, Nam, gây thêm khó khăn cho bài toán nhận dạng. Thêm vào đó, nguồn dữ liệu huấn luyện chất lượng cao cho tiếng Việt còn hạn chế so với tiếng Anh — bộ dữ liệu VIVOS chỉ có khoảng 15 giờ audio, trong khi LibriSpeech cho tiếng Anh có tới 960 giờ.

### 1.2 Khoảng trống nghiên cứu

Các hệ thống ASR mã nguồn mở cho tiếng Việt còn khá ít và thường gặp một số hạn chế: (1) thiếu pipeline huấn luyện end-to-end hoàn chỉnh mà người dùng có thể tái tạo, (2) không có giao diện web demo trực quan để kiểm thử nhanh, và (3) thiếu tài liệu hướng dẫn chi tiết bằng tiếng Việt. Đặc biệt, việc fine-tune các mô hình pre-trained lớn như Wav2Vec 2.0 trên dữ liệu tiếng Việt đòi hỏi hiểu biết về kiến trúc model, kỹ thuật preprocessing, và hardware phù hợp — những kiến thức mà sinh viên và nhà nghiên cứu mới bắt đầu thường gặp khó khăn.

### 1.3 Đóng góp của bài báo

Bài báo này trình bày các đóng góp sau:

1. **Pipeline huấn luyện hoàn chỉnh:** Xây dựng quy trình fine-tuning Wav2Vec 2.0 trên Kaggle GPU, từ tiền xử lý dữ liệu đến đánh giá kết quả, có thể tái tạo được.

2. **Kết hợp dữ liệu:** Gộp hai bộ dữ liệu VIVOS và VLSP 2020 (viet_bud500) để tăng lượng dữ liệu huấn luyện, đạt WER 13,31%.

3. **Ứng dụng web demo:** Phát triển ứng dụng Flask với giao diện hiện đại (dark mode, ghi âm trực tiếp, upload file) và server FastAPI RESTful.

4. **Phân tích thất bại:** Trình bày chi tiết một trường hợp huấn luyện thất bại (WER = 100%) và phân tích nguyên nhân, giúp cộng đồng tránh những lỗi tương tự.

5. **Tài liệu song ngữ:** Cung cấp tài liệu đầy đủ bằng tiếng Việt và tiếng Anh cho cộng đồng nghiên cứu.

---

## 2. Các công trình liên quan

### 2.1 Wav2Vec 2.0

Wav2Vec 2.0 [1] là kiến trúc self-supervised learning cho nhận dạng tiếng nói được đề xuất bởi Baevski và cộng sự tại Facebook AI Research vào năm 2020. Mô hình hoạt động theo hai giai đoạn: (1) pre-training trên dữ liệu audio không gán nhãn bằng contrastive learning, và (2) fine-tuning trên dữ liệu có nhãn với CTC loss. Trên benchmark LibriSpeech, Wav2Vec 2.0 chỉ cần 10 phút dữ liệu có nhãn để đạt WER 4,8%, cho thấy khả năng transfer learning mạnh mẽ cho các ngôn ngữ ít tài nguyên.

Kiến trúc Wav2Vec 2.0 gồm ba thành phần chính: (a) Feature Encoder — mạng CNN 7 lớp chuyển raw audio waveform thành biểu diễn latent, (b) Context Network — mạng Transformer mã hóa ngữ cảnh toàn cục, và (c) Quantization Module — lượng tử hóa các biểu diễn latent cho quá trình contrastive learning.

### 2.2 Whisper

Whisper [2] của OpenAI là một hệ thống ASR đa ngôn ngữ được huấn luyện weak-supervised trên 680.000 giờ dữ liệu thu thập từ web. Whisper hỗ trợ nhiều ngôn ngữ bao gồm tiếng Việt, nhưng do tính chất đa ngôn ngữ, chất lượng nhận dạng tiếng Việt có thể không bằng mô hình chuyên biệt. Tuy nhiên, Whisper cung cấp baseline hữu ích cho việc so sánh.

### 2.3 ASR cho tiếng Việt

Một số công trình ASR tiếng Việt đáng chú ý bao gồm:

- **PhoSpeech** [4] — Pipeline pre-training và fine-tuning dựa trên Wav2Vec2 cho tiếng Việt, sử dụng dữ liệu từ nhiều nguồn.
- **nguyenvulebinh/wav2vec2-base-vietnamese-250h** — Mô hình pre-trained trên 250 giờ dữ liệu tiếng Việt, được cung cấp trên Hugging Face Hub. Đây là mô hình base được sử dụng trong nghiên cứu này.
- **VLSP ASR Shared Task** — Cuộc thi VLSP hàng năm thúc đẩy nghiên cứu ASR tiếng Việt, với bộ dữ liệu chuẩn VLSP 2020 được sử dụng rộng rãi.

### 2.4 Transfer learning cho ngôn ngữ tài nguyên thấp

Transfer learning đã chứng minh hiệu quả đặc biệt cho các ngôn ngữ tài nguyên thấp (low-resource languages). Phương pháp phổ biến là pre-train mô hình trên dữ liệu không gán nhãn quy mô lớn, sau đó fine-tune trên dữ liệu có nhãn của ngôn ngữ đích. Conneau và cộng sự [5] đã chỉ ra rằng XLSR — mô hình Wav2Vec 2.0 đa ngôn ngữ — có thể đạt kết quả tốt trên nhiều ngôn ngữ chỉ với vài giờ dữ liệu fine-tuning. Nghiên cứu này áp dụng phương pháp tương tự, sử dụng mô hình Vietnamese pre-trained và fine-tune trên dữ liệu VIVOS + VLSP 2020.

---

## 3. Phương pháp

### 3.1 Kiến trúc mô hình

Mô hình VietASR Pro sử dụng kiến trúc **Wav2Vec 2.0 base** với các thông số sau:

| Thành phần | Thông số |
|---|---|
| Feature Encoder | 7 lớp CNN (kernel sizes: 10, 3, 3, 3, 3, 2, 2; strides: 5, 2, 2, 2, 2, 2, 2) |
| Hidden size | 768 |
| Số lớp Transformer | 12 |
| Số attention heads | 12 |
| Intermediate size | 3072 |
| Vocab size | 110 tokens |
| Pad token ID | 109 |
| CTC loss reduction | mean |

*Bảng 1: Thông số kiến trúc mô hình Wav2Vec 2.0 base sử dụng trong VietASR Pro.*

**Hình 1: Kiến trúc Wav2Vec 2.0 cho VietASR Pro**

Kiến trúc hoạt động như sau: Tín hiệu audio đầu vào (16kHz, mono) được đưa qua Feature Encoder — một mạng CNN 7 lớp — để trích xuất biểu diễn latent với tần suất lấy mẫu khoảng 49 frame/giây. Các biểu diễn này được đưa vào Context Network — một mạng Transformer 12 lớp với multi-head self-attention — để mã hóa thông tin ngữ cảnh toàn cục. Cuối cùng, CTC Head — một lớp Linear — ánh xạ output của Transformer sang không gian 110 token (bao gồm toàn bộ ký tự tiếng Việt có dấu, chữ số, và các token đặc biệt). Quá trình giải mã sử dụng greedy decoding (argmax) hoặc beam search với Language Model.

Trong quá trình fine-tuning, Feature Encoder được đóng băng (freeze) — chỉ huấn luyện Context Network và CTC Head. Chiến lược này giúp giảm số lượng tham số cần cập nhật và tránh overfitting khi lượng dữ liệu có nhãn hạn chế.

### 3.2 Dữ liệu huấn luyện

#### 3.2.1 VIVOS

VIVOS [6] là bộ dữ liệu tiếng nói tiếng Việt miễn phí được xây dựng bởi AILAB, Trường Đại học Khoa học Tự nhiên, TP.HCM. Bộ dữ liệu chứa khoảng **15 giờ** audio ghi âm từ **46 người nói** (giọng miền Nam và miền Bắc), được chia thành tập train và tập test. Chất lượng audio cao, đã được chuẩn hóa 16kHz mono.

#### 3.2.2 VLSP 2020 / viet_bud500

Bộ dữ liệu VLSP 2020 (được phân phối dưới định dạng viet_bud500 trên Hugging Face Hub) là tập dữ liệu tiếng Việt quy mô lớn, bao gồm dữ liệu từ nhiều nguồn khác nhau. Bộ dữ liệu được tải về và sử dụng thông qua thư viện Hugging Face Datasets.

#### 3.2.3 Pipeline tiền xử lý dữ liệu

Quy trình tiền xử lý dữ liệu bao gồm các bước sau:

1. **Resampling:** Tất cả audio được resample về 16kHz nếu sample rate khác.
2. **Chuyển đổi mono:** Audio stereo được chuyển thành mono bằng cách lấy trung bình các kênh.
3. **Chuẩn hóa biên độ:** Normalize biên độ audio về khoảng [-1, 1].
4. **Chuẩn hóa văn bản:** Chuyển transcript sang lowercase, loại bỏ ký tự đặc biệt, chuẩn hóa khoảng trắng, loại bỏ các token đặc biệt (`<unk>`, `<s>`, `</s>`, `<pad>`).
5. **Tokenization:** Sử dụng char-level tokenizer với 110 token, bao gồm đầy đủ ký tự tiếng Việt có dấu (à, á, ạ, ả, ã, â, ầ, ấ, ậ, ẩ, ẫ, ă, ằ, ắ, ặ, ẳ, ẵ, đ, ...), chữ số (0-9), và các token đặc biệt (`<pad>`, `<unk>`, `|`).

Hàm chuẩn hóa văn bản `normalize_text()` thực hiện các bước regex sau:
- Chuyển thành chữ thường: `text.lower()`
- Loại bỏ ký tự không phải chữ cái/khoảng trắng tiếng Việt
- Loại bỏ token đặc biệt: `<unk>`, `<s>`, `</s>`, `<pad>`
- Gộp nhiều khoảng trắng liên tiếp

### 3.3 Quá trình fine-tuning

#### 3.3.1 Cấu hình huấn luyện

Mô hình được fine-tune với các hyperparameters sau:

| Hyperparameter | Giá trị |
|---|---|
| Pre-trained model | `nguyenvulebinh/wav2vec2-base-vietnamese-250h` |
| Optimizer | AdamW |
| Learning rate | 1 × 10⁻⁴ (warmup 500 steps, cosine decay) |
| Batch size | 32 |
| Gradient accumulation steps | 2 |
| Epochs | Chạy tới khi đạt 15.000 steps (~1,26 epochs) |
| FP16 | Có (mixed precision) |
| Max steps | 20.000 (dừng sớm tại 15.000) |
| Eval steps | Mỗi 1.000 steps |
| Save steps | Mỗi 1.000 steps |
| Save total limit | 2 |
| Early stopping patience | 5 |
| Weight decay | 0,005 |
| Gradient checkpointing | Có |
| Loss function | CTC Loss (mean reduction) |

*Bảng 2: Cấu hình hyperparameters cho quá trình fine-tuning.*

#### 3.3.2 Phần cứng

Quá trình huấn luyện được thực hiện trên hạ tầng **Kaggle**, sử dụng GPU Tesla T4 (16GB VRAM) hoặc P100 (16GB VRAM). Batch size 32 với mixed precision (FP16) cho phép tận dụng tối đa bộ nhớ GPU. Tổng thời gian huấn luyện khoảng 11.921 giây (~3,3 giờ) cho 15.000 steps.

#### 3.3.3 Chiến lược fine-tuning

Chiến lược fine-tuning áp dụng các kỹ thuật sau:
- **Feature Encoder freezing:** Đóng băng 7 lớp CNN để giữ nguyên khả năng trích xuất đặc trưng âm thanh đã học trong quá trình pre-training.
- **Attention/Hidden dropout:** 0,1 — chống overfitting.
- **Feat projection dropout:** 0,0 — giữ nguyên đặc trưng input.
- **Mask time probability:** 0,05 — SpecAugment nhẹ.
- **Layerdrop:** 0,0 — không bỏ lớp Transformer ngẫu nhiên trong quá trình này.

---

## 4. Thực nghiệm

### 4.1 Thiết lập thực nghiệm

Thí nghiệm được thực hiện với hai cấu hình:

1. **Cấu hình A (Kaggle GPU):** Fine-tune trên dữ liệu VIVOS + VLSP 2020 (viet_bud500), sử dụng GPU Kaggle, 15.000 steps.

2. **Cấu hình B (Local CPU — baseline thất bại):** Huấn luyện trên dữ liệu VIVOS riêng biệt (~15 giờ) trên CPU, 3.000 steps (~9,15 epochs). Cấu hình này phục vụ so sánh và phân tích lỗi.

Metric đánh giá chính là **Word Error Rate (WER)** — tỷ lệ lỗi tính trên đơn vị từ, bao gồm insertion, deletion, và substitution:

$$WER = \frac{S + D + I}{N}$$

trong đó *S* là số lần thay thế (substitution), *D* là số lần xóa (deletion), *I* là số lần chèn (insertion), và *N* là tổng số từ trong câu tham chiếu.

### 4.2 Kết quả

#### 4.2.1 Cấu hình A — Kaggle Training

| Checkpoint | Steps | Eval Loss | WER (%) | Cải thiện so với step đầu |
|---|---|---|---|---|
| checkpoint-1000 | 1.000 | 0,1363 | 15,57 | — |
| checkpoint-2000 | 2.000 | 0,1207 | 15,07 | -0,50 |
| checkpoint-3000 | 3.000 | 0,1140 | 14,82 | -0,75 |
| checkpoint-4000 | 4.000 | 0,1150 | 14,80 | -0,77 |
| checkpoint-5000 | 5.000 | 0,1071 | 14,59 | -0,98 |
| checkpoint-6000 | 6.000 | 0,0980 | 14,31 | -1,26 |
| checkpoint-7000 | 7.000 | 0,0965 | 14,27 | -1,30 |
| checkpoint-8000 | 8.000 | 0,0914 | 14,18 | -1,39 |
| checkpoint-9000 | 9.000 | 0,0879 | 13,96 | -1,61 |
| checkpoint-10000 | 10.000 | 0,0844 | 13,90 | -1,67 |
| checkpoint-11000 | 11.000 | 0,0818 | 13,72 | -1,85 |
| checkpoint-12000 | 12.000 | 0,0756 | 13,49 | -2,08 |
| checkpoint-13000 | 13.000 | 0,0751 | 13,58 | -1,99 |
| checkpoint-14000 | 14.000 | 0,0732 | 13,41 | -2,16 |
| **checkpoint-15000** | **15.000** | **0,0664** | **13,31** | **-2,26** |

*Bảng 3: Kết quả WER theo checkpoint trên cấu hình Kaggle GPU. Model tốt nhất tại checkpoint-15000 với WER = 13,31%.*

WER giảm đều đặn từ 15,57% tại step 1.000 xuống 13,31% tại step 15.000, tương đương mức cải thiện **2,26 điểm phần trăm** (giảm 14,5% tương đối). Eval loss cũng giảm tương ứng từ 0,1363 xuống 0,0664, cho thấy mô hình vẫn đang học và chưa overfitting.

#### 4.2.2 Cấu hình B — Local Training (FAILED)

| Metric | Step 500 | Step 1.000 | Step 2.000 | Step 3.000 |
|---|---|---|---|---|
| Training Loss | — | — | — | 4,99 |
| Eval Loss | 5,018 | 4,981 | 4,982 | 4,984 |
| **WER** | **100%** | **100%** | **100%** | **100%** |
| CER | 99,93% | 99,93% | 99,93% | 99,93% |

*Bảng 4: Kết quả huấn luyện thất bại trên cấu hình Local CPU.*

**Phân tích nguyên nhân thất bại:**

Mô hình huấn luyện trên CPU với dữ liệu VIVOS riêng (~15 giờ) hoàn toàn không học được (WER cố định 100% suốt 3.000 steps). Mặc dù training loss giảm từ 32,26 xuống 4,99, eval metrics không cải thiện, cho thấy hiện tượng **data mismatch** hoặc **overfitting trên noise**. Nguyên nhân được xác định gồm:

1. **Data preprocessing sai:** Ánh xạ audio-text không chính xác trong quá trình chuẩn bị dữ liệu JSONL.
2. **Dataset quá nhỏ:** VIVOS chỉ có ~15 giờ — không đủ khi kết hợp với cấu hình huấn luyện không tối ưu.
3. **Cấu hình huấn luyện không phù hợp:** Learning rate cao (3 × 10⁻⁴), thiếu warmup, và batch size nhỏ trên CPU.
4. **Không sử dụng GPU:** Training trên CPU dẫn đến convergence chậm và khó phát hiện lỗi sớm.

**Bài học rút ra:** Kết quả này minh chứng tầm quan trọng của (a) kiểm tra kỹ pipeline preprocessing trước khi huấn luyện, (b) sử dụng GPU cho ASR training, và (c) kết hợp dữ liệu đủ lớn.

### 4.3 Phân tích lỗi

#### 4.3.1 Lỗi trong dữ liệu

Quá trình kiểm tra chất lượng dữ liệu (thực hiện bởi script `check_dataset.py`) cho thấy:

- **dataset_errors.csv:** File chứa danh sách các sample có vấn đề, bao gồm transcript rỗng, audio bị hỏng, hoặc audio quá ngắn/dài. Trong quá trình kiểm tra, số lỗi phát hiện rất ít, cho thấy chất lượng dữ liệu tổng thể tốt.

- **unknown_chars.txt:** File ghi nhận các ký tự xuất hiện trong transcript nhưng không có trong vocabulary của tokenizer. Phân tích cho thấy một số ký tự Unicode đặc biệt (dấu kết hợp combining marks như ̀, ́, ̣ — U+0300, U+0301, U+0323) và ký tự Cyrillic lạc (у — U+0443) xuất hiện trong vocab. Đây là artifact từ quá trình tạo vocabulary, nhưng không ảnh hưởng đáng kể đến hiệu suất vì các ký tự này hiếm gặp trong dữ liệu thực.

#### 4.3.2 Các lỗi nhận dạng phổ biến

Dựa trên phân tích kết quả transcription, các lỗi phổ biến bao gồm:

1. **Nhầm thanh điệu:** Các từ chỉ khác nhau về dấu (ví dụ: "là" ↔ "la", "được" ↔ "đước") — đây là lỗi phổ biến nhất do CTC decoding không mô hình hóa rõ ràng thanh điệu.

2. **Nhầm phụ âm đầu tương tự:** "ch" ↔ "tr", "s" ↔ "x", "d" ↔ "gi" — phản ánh sự khác biệt phương ngữ giữa các vùng miền.

3. **Thiếu/thừa từ ngắn:** Các từ chức năng ngắn như "là", "và", "đã", "cũng" đôi khi bị bỏ sót hoặc chèn thêm.

4. **Từ hiếm gặp (OOV):** Tên riêng và thuật ngữ chuyên ngành thường bị nhận dạng sai do ít xuất hiện trong dữ liệu huấn luyện.

#### 4.3.3 Đường cong Training Loss

**Hình 2: Mô tả đường cong training loss**

Trên cấu hình Kaggle, training loss bắt đầu ở mức ~0,89 (step 100) và giảm đều đặn xuống ~0,50 (step 15.000). Quy luật giảm khá ổn định, không có hiện tượng loss tăng đột biến kéo dài. Gradient norm dao động trong khoảng 3-12 với một vài spike cô lập (lên tới 46,5 tại step 9600), nhưng không ảnh hưởng đến tính ổn định tổng thể của quá trình huấn luyện. Learning rate tuân theo lịch trình cosine decay từ 1 × 10⁻⁴ xuống 2,56 × 10⁻⁵.

---

## 5. Ứng dụng Demo (Web Application)

### 5.1 Flask Web Demo

Ứng dụng web demo chính của VietASR Pro được xây dựng trên framework **Flask** (Python), cung cấp giao diện trực quan cho người dùng cuối.

#### 5.1.1 Kiến trúc ứng dụng

Ứng dụng Flask gồm các thành phần:
- **Backend (`app/app.py`):** Xử lý request, load model, tiền xử lý audio, transcription, và các API endpoints.
- **Frontend (`app/templates/index.html`, `app/static/`):** Giao diện HTML/CSS/JavaScript hiện đại với thiết kế responsive.
- **Model loading:** Tự động tìm model local (`final_model/`) hoặc fallback về Hugging Face Hub.

#### 5.1.2 Tính năng giao diện

Giao diện web demo của VietASR Pro bao gồm:

1. **Ghi âm trực tiếp:** Người dùng bấm nút ghi âm, nói vào microphone, và nhận kết quả transcription real-time. Có visualizer sóng âm thanh hiển thị trong quá trình ghi.

2. **Upload file audio:** Hỗ trợ các định dạng WAV, MP3, FLAC, OGG, M4A, AAC, WMA, OPUS. File được tự động chuyển đổi sang WAV 16kHz trước khi transcribe.

3. **Dark mode / Light mode:** Chuyển đổi giao diện tối/sáng theo sở thích người dùng.

4. **Sidebar navigation:** Menu bên trái với các tab: Ghi âm, Upload, Lịch sử, Cài đặt.

5. **Lịch sử nhận dạng:** Lưu trữ các kết quả transcription trước đó, cho phép xem lại và sao chép.

6. **AI Chatbot:** Hỗ trợ phiên âm IPA, chuẩn hóa từ vùng miền (ví dụ: "dạ" → "vâng", "chi" → "gì"), và đếm từ/ký tự.

7. **Device info:** Hiển thị thông tin phần cứng (CPU/GPU), trạng thái model, và nguồn model đang sử dụng.

#### 5.1.3 API Endpoints (Flask)

| Method | Endpoint | Chức năng |
|---|---|---|
| GET | `/` | Giao diện chính |
| POST | `/api/transcribe` | Transcribe audio từ ghi âm |
| POST | `/api/upload` | Upload và transcribe file |
| POST | `/api/chatbot` | Chatbot hỗ trợ |
| POST | `/api/normalize` | Chuẩn hóa từ vùng miền |
| GET | `/api/status` | Trạng thái model |
| GET | `/api/device-info` | Thông tin thiết bị |
| POST | `/api/device-select` | Chọn thiết bị (CPU/GPU) |

*Bảng 5: Các API endpoints của Flask web demo.*

### 5.2 FastAPI Server

Ngoài Flask web demo, hệ thống còn cung cấp server **FastAPI** — một REST API thay thế phù hợp cho tích hợp vào ứng dụng khác hoặc deploy trên production.

#### 5.2.1 Kiến trúc

FastAPI server (`src/api/server.py`) cung cấp:
- **Swagger UI tự động:** Truy cập `/docs` để xem và thử nghiệm tất cả API endpoints.
- **Async support:** Hỗ trợ xử lý bất đồng bộ cho nhiều request đồng thời.
- **Type validation:** Sử dụng Pydantic models cho input/output validation.
- **Language Model support:** Tùy chọn sử dụng KenLM 5-gram cho beam search decoding.

#### 5.2.2 API Endpoints (FastAPI)

| Method | Endpoint | Chức năng |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/transcribe` | Transcribe audio file |
| POST | `/transcribe-stream` | Transcribe audio stream |
| POST | `/load-model` | Load/reload model |
| GET | `/model-info` | Thông tin model |
| GET | `/app` | Serve web UI tĩnh |

*Bảng 6: Các API endpoints của FastAPI server.*

---

## 6. Kết luận

### 6.1 Tóm tắt đóng góp

Bài báo đã trình bày VietASR Pro — hệ thống nhận dạng tiếng nói tiếng Việt dựa trên Wav2Vec 2.0 fine-tuning. Các đóng góp chính bao gồm:

1. Đạt WER **13,31%** trên bộ dữ liệu đánh giá bằng cách fine-tune Wav2Vec 2.0 trên dữ liệu VIVOS + VLSP 2020, sử dụng Kaggle GPU.

2. Xây dựng pipeline huấn luyện hoàn chỉnh và có thể tái tạo, bao gồm tiền xử lý dữ liệu, huấn luyện, đánh giá, và triển khai.

3. Phát triển ứng dụng web demo Flask hiện đại và server FastAPI RESTful, cho phép người dùng dễ dàng tương tác với hệ thống.

4. Phân tích chi tiết một trường hợp huấn luyện thất bại, cung cấp bài học hữu ích cho cộng đồng.

### 6.2 Hạn chế

Hệ thống hiện tại có một số hạn chế:

1. **Thiếu Language Model:** Chưa tích hợp Language Model (KenLM hoặc neural LM) cho quá trình giải mã. Theo các nghiên cứu trước đó, LM có thể cải thiện WER thêm 20-30%.

2. **Hạn chế về phương ngữ:** Model chủ yếu được huấn luyện trên giọng chuẩn, chưa được kiểm thử kỹ trên giọng các vùng miền khác nhau.

3. **Dữ liệu giới hạn:** Mặc dù đã kết hợp VIVOS và VLSP 2020, tổng lượng dữ liệu vẫn nhỏ so với các hệ thống ASR thương mại.

4. **Chỉ hỗ trợ offline:** Chưa có khả năng streaming inference real-time.

5. **Chưa so sánh với Whisper:** Cần thêm thí nghiệm so sánh với Whisper Vietnamese để đánh giá toàn diện hơn.

### 6.3 Hướng phát triển

Các hướng phát triển trong tương lai bao gồm:

1. **Tích hợp Language Model:** Xây dựng KenLM 5-gram từ dữ liệu huấn luyện và tích hợp vào pyctcdecode cho beam search decoding.

2. **Mở rộng dữ liệu:** Thu thập thêm dữ liệu từ nhiều nguồn (Common Voice, YouTube, podcast) và nhiều phương ngữ.

3. **Streaming inference:** Phát triển khả năng nhận dạng real-time cho ứng dụng conversational AI.

4. **Deploy production:** Tối ưu hóa model (quantization, ONNX export) và deploy trên cloud (AWS/GCP).

5. **Mobile deployment:** Phát triển phiên bản cho thiết bị di động (iOS/Android).

---

## Tài liệu tham khảo

[1] Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. *Advances in Neural Information Processing Systems*, 33, 12449-12460. https://arxiv.org/abs/2006.11477

[2] Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). Robust Speech Recognition via Large-Scale Weak Supervision. *arXiv preprint arXiv:2212.04356*. https://arxiv.org/abs/2212.04356

[3] Hsu, W.-N., Bolte, B., Tsai, Y.-H. H., Lakhotia, K., Salakhutdinov, R., & Mohamed, A. (2021). HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 29, 3451-3460.

[4] Pham, N. Q., et al. (2022). PhoSpeech: Automatic Speech Recognition for Vietnamese. *Proceedings of INTERSPEECH 2022*.

[5] Conneau, A., Baevski, A., Collobert, R., Mohamed, A., & Auli, M. (2020). Unsupervised Cross-lingual Representation Learning for Speech Recognition. *arXiv preprint arXiv:2006.13979*. https://arxiv.org/abs/2006.13979

[6] Luong, H.-T., & Vu, H.-Q. (2016). A non-expert Kaldi recipe for Vietnamese Speech Recognition System. *Proceedings of the Third International Workshop on Worldwide Language Service Infrastructure (WLSI)*. https://ailab.hcmus.edu.vn/vivos

[7] Hugging Face Transformers Library. https://huggingface.co/docs/transformers

[8] Flask — Web development, one drop at a time. https://flask.palletsprojects.com/

[9] FastAPI — Modern, fast (high-performance), web framework for building APIs. https://fastapi.tiangolo.com/

[10] Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006). Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks. *Proceedings of the 23rd International Conference on Machine Learning (ICML)*, 369-376.

---

*Bài báo được hoàn thành vào tháng 4 năm 2026.*

*Tác giả: Nguyễn Trí Thượng — Đồ án Trí tuệ Nhân tạo — Năm học 2025–2026*
]]>
