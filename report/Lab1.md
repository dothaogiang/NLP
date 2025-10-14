# BÁO CÁO LAB 1: PHÂN ĐOẠN VĂN BẢN (TEXT TOKENIZATION)

## 1. Mục tiêu

Hiểu và triển khai các kỹ thuật phân đoạn văn bản (tokenization).

So sánh hai phương pháp phân đoạn: dựa vào khoảng trắng (SimpleTokenizer) và biểu thức chính quy (RegexTokenizer).

Ứng dụng kiểm thử trên dữ liệu thực tế từ tập UD_English-EWT.

## 2. Mô tả phương pháp

### a. SimpleTokenizer

Chia văn bản bằng dấu cách.

Thêm khoảng trắng quanh dấu câu (.,?!) để tách riêng.

Chuyển văn bản thành chữ thường (lowercase).

### b. RegexTokenizer

Sử dụng mẫu regex \w+|[^\w\s] để tách từ và dấu câu riêng biệt.

Loại bỏ khoảng trắng thừa và chuyển toàn bộ sang chữ thường.

## 3. Triển khai

Viết trong src/preprocessing/simple_tokenizer.py và regex_tokenizer.py.

Giao diện được định nghĩa trong interfaces.py.

Thực thi và kiểm thử bằng main.py.

## 4. Kết quả

Ví dụ đầu vào:
"Hello, world! This is a test."

SimpleTokenizer →
['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']

RegexTokenizer →
['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']

Kết quả trên tập UD_English-EWT (mẫu 500 ký tự):
['the', 'dt', 'old', 'jj', 'granny', 'nns', ...]

## 6. Kết luận

Lab 1 giúp hiểu cơ bản về việc chuyển đổi văn bản thô thành chuỗi token, bước đầu tiên trong pipeline NLP.
Phương pháp Regex hoạt động hiệu quả hơn, và có thể mở rộng cho các ngôn ngữ khác.
