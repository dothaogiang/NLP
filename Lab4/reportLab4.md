# Báo cáo Lab 4: Phân loại Văn bản

## 1. Giải thích các bước thực thi

### Task 1 & 2 (Baseline): Xây dựng lớp TextClassifier dùng LogisticRegression và TfidfVectorizer. Chạy test/lab5_test.py để huấn luyện trên bộ data tài chính (đã lọc 3365 mẫu nhị phân) và ghi nhận kết quả baseline.

### Task 3 (Spark): Cập nhật file test/lab5_spark_sentiment_analysis.py để tải dữ liệu từ Hugging Face (thay vì file .csv bị thiếu). Chạy thành công pipeline của Spark để xử lý dữ liệu lớn.

### Task 4 (Improvement - LSTM): Xây dựng và huấn luyện mô hình mạng nơ-ron (LSTM) bằng TensorFlow/Keras trên cùng bộ dữ liệu để so sánh hiệu suất với mô hình baseline.

## 2. Hướng dẫn chạy code

1.  **Cài đặt thư viện:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Chạy test baseline (Logistic Regression):**

    ```bash
    python test/lab5_test.py
    ```

3.  **Chạy thí nghiệm cải tiến (LSTM):**
    _(Lưu ý: Cần có kết nối mạng để tải dataset và có thể mất vài phút để huấn luyện)_
    ```bash
    python test/lab5_improvement_test.py
    ```
4.  **Chạy ví dụ Spark:**
    ```bash
    python test/lab5_spark_sentiment_analysis.py
    ```

## 3. Phân tích kết quả

Phần này so sánh hiệu suất của các mô hình trên bộ dataset Twitter Financial News (đã lọc 3365 mẫu nhị phân).

Model

Accuracy

F1-score (Binary)

Task 2 (Baseline)

(Logistic Regression + TF-IDF)

0.7964

0.8375

Task 3 (Spark)

(Logistic Regression + HashingTF)

0.8208

0.8138

Task 4 (Improved)

(LSTM + Embedding)
accuracy: 0.796
f1-score: 0.8375

Phân tích

(Bạn hãy viết phân tích của mình ở đây sau khi có kết quả Task 4. Đây là một ví dụ):

Kết quả cho thấy cả 3 mô hình đều cho hiệu suất tốt trên bộ dữ liệu này.

Baseline (Logistic Regression + TF-IDF) đạt F1-score là 0.8375. Đây là một điểm chuẩn rất mạnh, cho thấy các đặc trưng TF-IDF hoạt động hiệu quả.

Spark (Logistic Regression + HashingTF) cho kết quả Accuracy cao nhất (0.8208) nhưng F1-score thấp hơn một chút (0.8138). Điều này cho thấy HashingTF là một kỹ thuật vector hóa hiệu quả và có thể mở rộng cho dữ liệu lớn, dù có thể mất một chút độ chính xác (do xung đột hash) so với TfidfVectorizer truyền thống.
