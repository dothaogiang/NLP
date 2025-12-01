# BÁO CÁO NGHIÊN CỨU KHOA HỌC

## CHỦ ĐỀ: TỔNG QUAN VÀ CÁC HƯỚNG TIẾP CẬN TRONG BÀI TOÁN TEXT-TO-SPEECH (TTS)

---

### 1. TỔNG QUAN BÀI TOÁN & TÌNH HÌNH NGHIÊN CỨU

**1.1. Định nghĩa bài toán**
Text-to-Speech (TTS) là quá trình chuyển đổi văn bản đầu vào thành tín hiệu âm thanh lời nói (waveform) mang đặc trưng tự nhiên của con người. Một hệ thống TTS hiện đại thường bao gồm:

- **Frontend (Text Analysis):** Chuẩn hóa văn bản và chuyển đổi sang dạng ngữ âm (Phonemes).
- **Backend (Speech Synthesis):** Tạo ra tín hiệu âm thanh từ dữ liệu ngữ âm.

**1.2. Tình hình nghiên cứu hiện tại**
Ngành xử lý tiếng nói đang chứng kiến sự chuyển dịch mạnh mẽ từ các phương pháp xử lý tín hiệu (Signal Processing) sang Deep Learning và hiện tại là Generative AI (AI tạo sinh).

- **Xu hướng:** Tập trung vào khả năng Zero-shot (sao chép giọng nói chưa từng thấy chỉ với mẫu < 3 giây) và Cross-lingual (giữ chất giọng khi nói ngôn ngữ khác).
- **Thách thức cốt lõi:** Cân bằng bộ ba: Chất lượng tự nhiên (Naturalness) - Tốc độ suy luận (Inference Speed) - Tài nguyên tính toán (Computing Cost).

---

### 2. CÁC PHƯƠNG PHÁP TRIỂN KHAI & ĐÁNH GIÁ (3 LEVEL)

#### Level 1: Phương pháp Truyền thống (Concatenative / Parametric)

- **Mô tả:** Dựa trên việc cắt ghép các từ/âm tiết đã thu âm sẵn (Concatenative) hoặc dùng mô hình thống kê HMM để sinh tham số giọng nói.
- **Ưu điểm:**
  - Tốc độ cực nhanh, chạy tốt trên CPU yếu hoặc vi điều khiển.
  - Tốn rất ít tài nguyên lưu trữ và RAM.
- **Nhược điểm:**
  - Giọng nói thiếu tự nhiên, nghe như máy (robotic), hay bị giật cục.
  - Rất khó để thay đổi giọng đọc hoặc ngữ điệu nếu không thu âm lại.
- **Phù hợp với:** Thiết bị nhúng (IoT), đồ chơi thông minh, hệ thống thông báo công cộng, máy đọc màn hình (Screen Reader) ưu tiên tốc độ.

#### Level 2: Neural TTS (Deep Learning - Speaker Specific)

- **Mô tả:** Sử dụng mạng Nơ-ron (CNN, RNN, Transformer) để học ánh xạ từ Text sang Mel-spectrogram (ví dụ: Tacotron 2, FastSpeech).
- **Ưu điểm:**
  - Độ tự nhiên cao, mượt mà, tiệm cận giọng người thật.
  - Quy trình training ổn định cho một giọng nói cụ thể (Single Speaker).
- **Nhược điểm:**
  - Phụ thuộc nặng nề vào dữ liệu chất lượng cao (Studio quality).
  - Kém linh hoạt: Khó mở rộng ra nhiều giọng nói mới (Multi-speaker) nếu thiếu dữ liệu của người đó.
- **Phù hợp với:** Trợ lý ảo (Virtual Assistants), sách nói (Audiobooks), tổng đài tự động (Call Center).

#### Level 3: Generative / Few-shot TTS (Large Scale Models)

- **Mô tả:** Coi TTS là bài toán mô hình ngôn ngữ (Language Modeling) hoặc sử dụng Diffusion. Huấn luyện trên hàng chục nghìn giờ dữ liệu đa dạng (ví dụ: VALL-E, XTTS).
- **Ưu điểm:**
  - **Zero-shot Cloning:** Sao chép giọng bất kỳ chỉ với 3-10 giây mẫu âm thanh.
  - **Giàu cảm xúc:** Có thể điều khiển tiếng cười, tiếng thở, tiếng khóc qua văn bản (prompt).
- **Nhược điểm:**
  - Tài nguyên tính toán rất lớn, mô hình nặng, khó chạy real-time trên thiết bị yếu.
  - Vấn đề "Ảo giác" (Hallucination): Đôi khi nói sai từ, lặp từ hoặc phát ra âm thanh lạ.
- **Phù hợp với:** Sáng tạo nội dung (Content Creator), lồng tiếng phim tự động (AI Dubbing), NPC trong Game.

---

### 3. CHIẾN LƯỢC TỐI ƯU HÓA PIPELINE (NGHIÊN CỨU)

Để tối thiểu hóa nhược điểm và tối đa hóa ưu điểm, các nghiên cứu hiện đại áp dụng các chiến lược sau:

**3.1. Tối ưu hóa Tốc độ (Giải quyết nhược điểm Level 2 & 3)**

- **Phương pháp:** Chuyển từ mô hình Autoregressive (sinh tuần tự từng từ) sang **Non-autoregressive** (sinh song song toàn bộ câu).
- **Minh chứng:** Mô hình **FastSpeech 2** giúp tăng tốc độ suy luận lên gấp hàng chục lần so với Tacotron 2 mà không giảm chất lượng.

**3.2. Tối đa hóa Độ tự nhiên & Cảm xúc**

- **Phương pháp:** Sử dụng **Reference Encoder** hoặc **Latent Space Control**.
- **Cơ chế:** Trích xuất đặc trưng cảm xúc từ một file âm thanh mẫu (ví dụ: giọng buồn) và "ép" mô hình TTS sinh âm thanh theo phong cách đó (Style Transfer).

**3.3. Tối thiểu hóa lỗi và Đảm bảo an toàn**

- **Phương pháp:** Kết hợp với Large Language Models (LLM) để xử lý văn bản đầu vào tốt hơn, giảm lỗi ngữ pháp.
- **Đạo đức AI:** Tích hợp **Audio Watermarking** (như AudioSeal) để đánh dấu giọng nói do AI tạo ra, ngăn chặn Deepfake.

---

### 4. TÀI LIỆU THAM KHẢO (KEY PAPERS)

1.  **Tacotron 2:** Shen, J., et al. (2018). _"Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions."_ (Google).
2.  **FastSpeech 2:** Ren, Y., et al. (2020). _"FastSpeech 2: Fast and High-Quality End-to-End Text to Speech."_ (Microsoft).
3.  **VALL-E:** Wang, C., et al. (2023). _"Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers."_ (Microsoft).
4.  **HiFi-GAN:** Kong, J., et al. (2020). _"HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis."_

---

_Người thực hiện báo cáo: [Tên của bạn/Nhóm của bạn]_
_Ngày thực hiện: [Ngày/Tháng/Năm]_
