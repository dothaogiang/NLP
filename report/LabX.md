# BÁO CÁO Lab X

## CHỦ ĐỀ: TỔNG QUAN VÀ CÁC HƯỚNG TIẾP CẬN TRONG BÀI TOÁN TEXT-TO-SPEECH (TTS)

### 1. TỔNG QUAN BÀI TOÁN & TÌNH HÌNH NGHIÊN CỨU

**1.1. Định nghĩa bài toán**
Text-to-Speech (TTS) là quá trình chuyển đổi văn bản đầu vào thành tín hiệu âm thanh lời nói (waveform) mang đặc trưng tự nhiên của con người. Một hệ thống TTS hiện đại thường bao gồm:

- **Frontend (Text Analysis):** Chuẩn hóa văn bản và chuyển đổi sang dạng ngữ âm (Phonemes).
- **Backend (Speech Synthesis):** Tạo ra tín hiệu âm thanh từ dữ liệu ngữ âm.

**1.2. Tình hình nghiên cứu hiện tại**
Ngành xử lý tiếng nói đang chứng kiến sự chuyển dịch mạnh mẽ từ các phương pháp xử lý tín hiệu (Signal Processing) sang Deep Learning và hiện tại là Generative AI (AI tạo sinh).

- **Xu hướng:** Tập trung vào khả năng Zero-shot (sao chép giọng nói chưa từng thấy chỉ với mẫu < 3 giây) và Cross-lingual (giữ chất giọng khi nói ngôn ngữ khác).
- **Thách thức cốt lõi:** Cân bằng bộ ba: Chất lượng tự nhiên (Naturalness) - Tốc độ suy luận (Inference Speed) - Tài nguyên tính toán (Computing Cost).

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

### 3. CHIẾN LƯỢC TỐI ƯU HÓA PIPELINE (CHI TIẾT KỸ THUẬT)

Để giải quyết bài toán "Tam giác bất khả thi" trong TTS (Tốc độ - Chất lượng - Tài nguyên), các nghiên cứu hiện đại không chỉ dùng một model đơn lẻ mà thiết kế các pipeline phức hợp. Dưới đây là phân tích sâu:

#### 3.1. Tối ưu hóa Tốc độ: Cuộc cách mạng Non-Autoregressive

Vấn đề lớn nhất của các mô hình Level 2 đời đầu (như Tacotron 2) là cơ chế **Autoregressive (AR)**: Để sinh ra khung âm thanh (frame) tại thời điểm $t$, model buộc phải biết frame tại thời điểm $t-1$. Điều này khiến GPU mạnh đến mấy cũng không thể chạy song song, tạo ra độ trễ lớn.

- **Phương pháp:** Chuyển sang kiến trúc **Non-Autoregressive (NAR)** hay còn gọi là Parallel TTS.
- **Cơ chế hoạt động (Case Study: FastSpeech 2):**
  - Thay vì sinh tuần tự, model sinh tất cả các frame âm thanh cùng một lúc.
  - **Thách thức:** Làm sao model biết một từ (ví dụ: "Hello") sẽ kéo dài bao lâu trong âm thanh nếu không sinh tuần tự?
  - **Giải pháp - Bộ dự đoán thời lượng (Duration Predictor):** Đây là module cốt lõi. Trong quá trình training, model học được mỗi âm vị (phoneme) cần bao nhiêu frame thời gian. Khi suy luận (inference), module này dự đoán trước độ dài của toàn bộ câu, sau đó "kéo giãn" (expand) vector văn bản ra đúng độ dài đó và điền thông tin âm thanh vào song song.
- **Kết quả kỹ thuật:** Độ phức tạp thuật toán giảm từ $O(N)$ (phụ thuộc độ dài câu) xuống $O(1)$ (hằng số), tăng tốc độ suy luận lên gấp 50-100 lần so với Tacotron 2.

#### 3.2. Tối đa hóa Độ tự nhiên & Cảm xúc: Variance Adaptor & Latent Space

Giọng nói tự nhiên không chỉ là phát âm đúng, mà còn nằm ở sự biến thiên của: **Cao độ (Pitch)**, **Năng lượng (Energy/Volume)**, và **Nhịp điệu (Duration)**.

- **Phương pháp 1: Variance Adaptor (Bộ thích nghi biến thiên)**

  - **Cơ chế:** Thay vì để model tự "đoán" các thông số này một cách ngẫu nhiên, pipeline chèn thêm các layer dự đoán riêng biệt:
    - _Pitch Predictor:_ Dự đoán độ trầm bổng của từng từ.
    - _Energy Predictor:_ Dự đoán độ to nhỏ (nhấn trọng âm).
  - **Ưu điểm:** Cho phép người dùng can thiệp thủ công. Ví dụ: Có thể tăng giá trị output của _Pitch Predictor_ lên 1.2 lần để giọng nói nghe vui vẻ, phấn khích hơn mà không cần train lại model.

- **Phương pháp 2: Reference Encoder (Style Transfer)**
  - **Cơ chế:**
    1.  Đưa một file âm thanh mẫu (Reference Audio) có cảm xúc mong muốn (ví dụ: giọng buồn, thì thầm) vào một mạng nơ-ron tích chập (CNN).
    2.  Mạng này nén âm thanh đó thành một vector đặc trưng gọi là **Style Embedding**.
    3.  Vector này được cộng gộp với vector văn bản đầu vào.
  - **Kết quả:** Model TTS sẽ "nhúng" phong cách của file mẫu vào nội dung văn bản mới. Đây là cách tạo ra các giọng đọc truyện diễn cảm.

#### 3.3. Tối thiểu hóa lỗi & Đạo đức AI: Hybrid LLM & Watermarking

Level 3 (Generative TTS) thường gặp lỗi "ảo giác" (hallucination) do bản chất ngẫu nhiên của quá trình lấy mẫu (sampling).

- **Phương pháp 1: Text-Aware Refinement (Kết hợp LLM)**

  - **Vấn đề:** Các từ đồng âm khác nghĩa (homographs). Ví dụ: từ "Live" trong "I live here" (sống) khác với "Live show" (trực tiếp). Model TTS thuần túy thường sai chỗ này.
  - **Cơ chế Pipeline:**
    1.  **Giai đoạn 1 (Text Processing):** Sử dụng một LLM nhỏ hoặc BERT model để phân tích ngữ nghĩa câu văn, gán nhãn từ loại và ngữ điệu cho từng từ.
    2.  **Giai đoạn 2 (Synthesis):** Model TTS nhận đầu vào không chỉ là ký tự, mà là chuỗi [Ký tự + Nhãn ngữ nghĩa], giúp loại bỏ lỗi phát âm sai ngữ cảnh.

- **Phương pháp 2: Audio Watermarking (Bảo vệ bản quyền & Chống Deepfake)**
  - **Cơ chế (AudioSeal):** Không giống như watermark hình ảnh (logo mờ), watermark âm thanh là sự thay đổi cực nhỏ trong phổ tần số (frequency domain) mà tai người không nghe thấy được.
  - **Triển khai:** Một module "Watermarker" được tích hợp ngay sau Vocoder. Tín hiệu này bền vững (robust) ngay cả khi file âm thanh bị nén (MP3), cắt ghép hay pha tạp âm, giúp các công cụ kiểm duyệt phát hiện ra đây là giọng AI tạo ra.

### 4. TÀI LIỆU THAM KHẢO

1.  **Tacotron 2:** Shen, J., et al. (2018). _"Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions."_ (Google).
2.  **FastSpeech 2:** Ren, Y., et al. (2020). _"FastSpeech 2: Fast and High-Quality End-to-End Text to Speech."_ (Microsoft).
3.  **VALL-E:** Wang, C., et al. (2023). _"Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers."_ (Microsoft).
4.  **HiFi-GAN:** Kong, J., et al. (2020). _"HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis."_

---

_Người thực hiện báo cáo: [Đỗ Thảo Giang]_
_Ngày thực hiện: [01/12/2025]_
