# Báo cáo Lab 5: Phân loại Văn bản với RNN/LSTM

## 1. Mục tiêu

Bài thực hành này nhằm mục tiêu so sánh hiệu năng của các phương pháp phân loại văn bản khác nhau trên bộ dữ liệu HWU (phân loại ý định người dùng). Bốn phương pháp đã được xây dựng và đánh giá:

- Baseline 1: TF-IDF + Logistic Regression (Mô hình Machine Learning cổ điển).

- Baseline 2: Word2Vec (Trung bình) + Mạng nơ-ron (Dense).

- Nâng cao 1: LSTM với Embedding "pre-trained" (tự huấn luyện trên tập train và đóng băng).

- Nâng cao 2: LSTM với Embedding được học từ đầu (học trong quá trình huấn luyện).

## 2. Phương pháp & Dữ liệu

- Dữ liệu: Bộ dữ liệu HWU được sử dụng, bao gồm 64 lớp (ý định) khác nhau.

- Phân chia:

        - Train: 8.954 mẫu

        - Validation: 1.076 mẫu

        - Test: 1.076 mẫu

- Tiền xử lý:

        - Nhãn: Các nhãn (intent) dạng text được `LabelEncoder` chuyển đổi thành 64 chỉ số dạng số.

        - Văn bản (cho LSTM): Văn bản được `Tokenizer` (Keras) chuyển thành chuỗi chỉ số (sequences) và được đệm (pad) để có cùng độ dài tối đa là 50 tokens. Từ vựng được giới hạn ở 10.000 từ.

## 3. Kết quả Thực nghiệm

Mô hình 1: TF-IDF + Logistic Regression (Baseline)
Mô hình cổ điển sử dụng TfidfVectorizer (với 5.000 features) và LogisticRegression đã đạt được kết quả rất cao.
Kết quả (Test set): Accuracy = 0.84, Macro F1-score = 0.84 (làm tròn từ 0.835)

Mô hình 2: Word2Vec (Trung bình) + Dense
Mô hình này tự huấn luyện Word2Vec (100 chiều) trên tập train, sau đó lấy vector trung bình của mỗi câu và đưa vào một mạng nơ-ron (Dense 128 -> Dense 64).

Kết quả (Test set): Accuracy = 0.20, Macro F1-score = 0.14 (làm tròn từ 0.136)

Mô hình 3: LSTM (Pre-trained & Đóng băng)
Mô hình này sử dụng Embedding Layer được khởi tạo bằng trọng số Word2Vec từ Mô hình 2. Lớp Embedding này bị đóng băng (trainable=False).

Kết quả (Test set): Accuracy = 0.12, Macro F1-score = 0.06 (làm tròn từ 0.058)

Mô hình 4: LSTM (Học từ đầu)
Mô hình này có kiến trúc tương tự Mô hình 3, nhưng lớp Embedding được học từ đầu (trainable=True).

Kết quả (Test set): Accuracy = 0.02, Macro F1-score = 0.00 (làm tròn từ 0.0005)

Ghi chú: Mô hình này đã sụp đổ hoàn toàn. Quá trình huấn luyện dừng sớm (epoch 5) do val_loss không cải thiện, cho thấy mô hình không học được bất cứ điều gì.

## 4.Phân tích Kết quả

### 4.1. Bảng so sánh định lượng

Bảng so sánh hiệu năng mô hình

| Pipeline                     | Macro F1-score (Test) | Test Loss |
| ---------------------------- | --------------------- | --------- |
| TF-IDF + Logistic Regression | 0.835                 | N/A       |
| Word2Vec (Avg) + Dense       | 0.136                 | 3.141     |
| LSTM (Pre-trained)           | 0.058                 | 3.454     |
| LSTM (Scratch)               | 0.001                 | 4.125     |

Kết quả đáng ngạc nhiên nhất là mô hình TF-IDF (Mô hình 1) hoạt động tốt vượt trội so với tất cả các mô hình nơ-ron. Các mô hình dựa trên LSTM (Mô hình 3 & 4), vốn được kỳ vọng sẽ hiểu ngữ cảnh tốt hơn, lại thất bại thảm hại.
Điều này có thể do nhiều nguyên nhân:

- Dữ liệu hạn chế: Bộ dữ liệu HWU tương đối nhỏ (chỉ ~9.000 mẫu huấn luyện). Mô hình LSTM có thể không đủ dữ liệu để học các biểu diễn hữu ích, trong khi mô hình TF-IDF tận dụng tốt các đặc trưng tần suất từ.
- Quá khớp (Overfitting): Mô hình LSTM có nhiều tham số hơn và dễ bị quá khớp trên tập huấn luyện nhỏ, dẫn đến hiệu năng kém trên tập kiểm tra.
- Cấu trúc mô hình chưa tối ưu: Các kiến trúc LSTM được sử dụng có thể chưa được tối ưu hóa tốt cho nhiệm vụ này. Việc sử dụng các kỹ thuật như dropout, điều chỉnh siêu tham số, hoặc kiến trúc phức tạp hơn có thể cải thiện hiệu năng.

### 4.2. Phân tích định tính (Câu ví dụ)

## Bảng ví dụ dự đoán của các mô hình

| Câu test                                    | Nhãn đúng       | Dự đoán TF-IDF | Dự đoán W2V (Avg) | Dự đoán LSTM (Pre) | Dự đoán LSTM (Scratch) |
| ------------------------------------------- | --------------- | -------------- | ----------------- | ------------------ | ---------------------- |
| 1. "can you remind me to not call my mom"   | reminder_create | calendar_set   | general_explain   | datetime_query     | email_query            |
| 2. "is it going... sunny or rainy tomorrow" | weather_query   | weather_query  | calendar_query    | qa_currency        | email_query            |
| 3. "find a flight... but not through paris" | flight_search   | general_negate | takeaway_order    | email_sendemail    | email_query            |

- Câu 1 (Phụ thuộc xa & Phủ định):

Lý thuyết: Đây là câu mà LSTM được kỳ vọng. Một mô hình túi từ (bag-of-words) sẽ thấy "remind", "call", "mom" và có thể dự đoán calendar_set (như TF-IDF đã làm). Yếu tố "not" nằm xa từ "call", đòi hỏi mô hình phải có "trí nhớ" để hiểu rằng đây là một lời nhắc không gọi.

Thực tế: Tất cả các mô hình đều dự đoán sai. Các mô hình LSTM (3 & 4) thất bại hoàn toàn, chúng không học được bất kỳ ngữ cảnh nào. Mô hình 4 (LSTM Scratch) sụp đổ và chỉ dự đoán email_query cho mọi thứ.

- Câu 2 (Từ khóa mạnh):

Lý thuyết: Câu này chứa các từ khóa rất mạnh là "sunny", "rainy", "tomorrow".

Thực tế: TF-IDF dự đoán đúng. Điều này cho thấy sức mạnh của nó khi các từ khóa là tín hiệu dự đoán chính. Các mô hình nơ-ron, do không được huấn luyện hiệu quả, đã thất bại trong việc liên kết các từ khóa này với ý định đúng.

- Câu 3 (Điều kiện loại trừ):

Lý thuyết: Tương tự câu 1, câu này có yếu tố ngữ cảnh "but not through paris". Một mô hình túi từ có thể bị nhầm lẫn bởi "not" và dự đoán general_negate (chính xác như TF-IDF đã làm). LSTM lẽ ra phải hiểu ý định chính là flight_search.

Thực tế: Tất cả các mô hình đều sai. TF-IDF bị từ "not" đánh lừa. Các mô hình LSTM một lần nữa cho thấy chúng không học được gì.

### 4.3. Nhận xét chung: Ưu và Nhược điểm

- TF-IDF + Logistic Regression:

Ưu điểm: Cực kỳ nhanh, hiệu quả, và là một baseline mạnh mẽ. Hoạt động xuất sắc khi các từ khóa (n-grams) là yếu tố dự đoán chính (như trong bài toán 64 ý định này).

Nhược điểm: Hoàn toàn "mù" về ngữ nghĩa và thứ tự từ. Nó bị đánh lừa bởi các từ phủ định hoặc cấu trúc phức tạp (như ở Câu 3).

- Word2Vec (Trung bình) + Dense:

Ưu điểm: Nắm bắt được một phần ngữ nghĩa của từ (tốt hơn TF-IDF).

Nhược điểm: Phá hủy hoàn toàn thông tin về thứ tự khi lấy trung bình. Kết quả (F1=0.14) cho thấy đây là một chiến lược rất kém hiệu quả cho bài toán này.

- LSTM (Cả 2 loại):

Ưu điểm (Lý thuyết): Có khả năng hiểu được ngữ cảnh, thứ tự từ và các phụ thuộc xa (như từ "not").

Nhược điểm (Thực tế):

Thất bại hoàn toàn (F1 < 0.1): Các mô hình này đòi hỏi lượng dữ liệu rất lớn và tinh chỉnh kỹ lưỡng. Với 8.954 câu chia cho 64 lớp (trung bình chỉ ~140 mẫu/lớp), dữ liệu là quá ít để LSTM học hiệu quả.

Chiến lược Embedding sai (Mô hình 3): Việc "đóng băng" (trainable=False) một lớp embedding vốn chỉ được huấn luyện trên 8.954 câu là một ý tưởng tồi. Các vector embedding này không đủ tốt, và việc đóng băng chúng khiến mô hình không thể cải thiện.

Mô hình 4: Mô hình học từ đầu không hội tụ, có thể do vanishing gradients, dữ liệu không đủ, hoặc cần nhiều epoch và tinh chỉnh hơn.

## 5. Kết luận

Không như kỳ vọng ban đầu, các mô hình RNN/LSTM phức tạp đã thất bại hoàn toàn trên bộ dữ liệu này. Nguyên nhân chính dường như là do không đủ dữ liệu (data sparsity) để huấn luyện hiệu quả một mạng nơ-ron sâu cho bài toán 64 lớp.

Ngược lại, mô hình TF-IDF + Logistic Regression cổ điển lại là người chiến thắng rõ ràng (Macro F1 = 0.84). Điều này chứng minh rằng đối với các bài toán phân loại ý định (nơi các từ khóa rất quan trọng) và khi dữ liệu bị hạn chế, một baseline mạnh mẽ, đơn giản thường là lựa chọn hiệu quả nhất.
