import spacy
from spacy import displacy

# 1. Tải mô hình
print("Đang tải mô hình...")
nlp = spacy.load("en_core_web_md")

# 2. Phân tích câu
text = "The quick brown fox jumps over the lazy dog."
doc = nlp(text)

# 3. Khởi chạy server hiển thị
print("Đang khởi chạy server...")
print("Hãy mở trình duyệt và truy cập: http://127.0.0.1:5000")
# serve() sẽ giữ chương trình chạy mãi cho đến khi bạn tắt nó
displacy.serve(doc, style="dep")