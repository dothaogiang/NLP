import gensim
from gensim.models import Word2Vec
import os

def read_corpus(file_path):
    """
    Đọc dữ liệu từ file và tách thành các câu (list of lists of words).
    Đây là định dạng đầu vào mà Gensim yêu cầu.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Tách câu thành các từ
            yield gensim.utils.simple_preprocess(line)

def main():
    # Đường dẫn file dữ liệu và model
    data_file = '../data/UD_English-EWT/en_ewt-ud-train.txt'
    model_path = 'results/word2vec_ewt.model'

    print("--- Bắt đầu huấn luyện model Word2Vec với Gensim ---")

    # 1. Đọc dữ liệu
    documents = list(read_corpus(data_file))
    print(f"Đã đọc {len(documents)} câu từ file.")

    # 2. Huấn luyện model
    # vector_size: Số chiều của vector từ
    # window: Kích thước cửa sổ ngữ cảnh (số từ xung quanh từ trung tâm)
    # min_count: Ngưỡng tần suất, bỏ qua các từ xuất hiện ít hơn giá trị này
    # workers: Số luồng CPU để sử dụng
    print("Bắt đầu training...")
    model = Word2Vec(sentences=documents, vector_size=100, window=5, min_count=1, workers=4)
    print("Training hoàn tất!")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # 3. Lưu model
    model.save(model_path)
    print(f"Model đã được lưu tại: {model_path}")

    # 4. Tải lại model và sử dụng
    loaded_model = Word2Vec.load(model_path)
    print("\n--- Thử nghiệm model vừa huấn luyện ---")
    
    # Tìm từ tương đồng với 'cat'
    try:
        similar_to_cat = loaded_model.wv.most_similar('cat')
        print("\nTừ tương đồng với 'cat':")
        print(similar_to_cat)
    except KeyError:
        print("'cat' không có trong từ điển.")
    
    # Giải bài toán analogy: king - man + woman = queen
    try:
        result = loaded_model.wv.most_similar(positive=['king', 'woman'], negative=['man'])
        print("\nAnalogy: king - man + woman = ?")
        print(result)
    except Exception as e:
        print(f"Không thể giải analogy: {e}")

if __name__ == "__main__":
    main()