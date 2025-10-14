# BÁO CÁO LAB 2: VECTOR HÓA ĐẾM (COUNT VECTORIZATION)

## 1. Mục tiêu

Chuyển đổi tập văn bản thành biểu diễn số (numerical representation) để phục vụ cho mô hình học máy.

Hiểu nguyên lý hoạt động của Count Vectorizer.

Thực hành xây dựng ma trận tài liệu - thuật ngữ (Document-Term Matrix).

## 2. Mô tả phương pháp

Sử dụng lớp CountVectorizer trong src/representations/count_vectorizer.py.

Tích hợp RegexTokenizer để đảm bảo phân đoạn thống nhất.

Các bước chính:

Token hóa từng tài liệu.

Xây dựng từ điển (vocabulary) — ánh xạ mỗi từ thành chỉ số.

Tạo ma trận đếm (count matrix) — mỗi hàng là tài liệu, mỗi cột là số lần xuất hiện của một từ.

## 3. Triển khai

fit(corpus): Học từ điển từ corpus.

transform(corpus): Chuyển văn bản thành ma trận đếm.

fit_transform(corpus): Kết hợp hai bước trên.

Kiểm thử bằng test/lab2_test.py.

## 4. Kết quả

Corpus mẫu

1. I love NLP.
2. NLP is a subfield of AI.
3. AI is a part of programming.

Từ điển học được:
{'a': 1, 'ai': 2, 'i': 3, 'is': 4, 'love': 5, 'nlp': 6, 'of': 7, 'programming': 8, 'subfield': 9, '.': 0}

Ma trận tài liệu-thuật ngữ:

[[1, 0, 0, 1, 0, 1, 1, 0, 0, 0],
 [1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
 [1, 1, 1, 0, 1, 0, 1, 1, 0, 1]]

Kết quả trên dữ liệu UD (5 câu):

Từ điển và ma trận thay đổi tùy nội dung, nhưng biểu diễn đúng cấu trúc document-term.

## 5. Kết luận

Lab 2 giúp củng cố kỹ năng biểu diễn văn bản bằng vector đếm, là nền tảng cho các phương pháp biểu diễn cao cấp hơn như TF-IDF, Word2Vec, BERT.
Sự tích hợp với RegexTokenizer đảm bảo pipeline xử lý thống nhất và chính xác hơn.
