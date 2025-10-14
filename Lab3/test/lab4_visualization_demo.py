import sys
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# ----------------------------------------------------

from src.representations.word_embedder import WordEmbedder

def visualize_embeddings(model, words):
    """
    Hàm để giảm chiều và trực quan hóa các word vector.
    """
    print("Đang lấy vector cho các từ...")
    word_vectors = np.array([model.get_vector(word) for word in words])
    
    # Lọc ra các từ không có trong từ điển (vector toàn số 0)
    valid_indices = [i for i, v in enumerate(word_vectors) if v.any()]
    if not valid_indices:
        print("Không có từ nào trong danh sách được tìm thấy trong từ điển.")
        return
        
    word_vectors = word_vectors[valid_indices]
    words = [words[i] for i in valid_indices]

    print("Đang giảm chiều bằng t-SNE...")
    # Sử dụng t-SNE để giảm chiều xuống 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(words)-1), init='pca', learning_rate='auto')
    vectors_2d = tsne.fit_transform(word_vectors)
    
    print("Đang vẽ biểu đồ...")
    # Vẽ biểu đồ scatter plot
    plt.figure(figsize=(14, 10))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])
    
    # Gắn nhãn cho các điểm
    for i, word in enumerate(words):
        plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]))
        
    plt.title("Trực quan hóa Word Embeddings bằng t-SNE")
    plt.xlabel("Chiều thứ nhất (t-SNE)")
    plt.ylabel("Chiều thứ hai (t-SNE)")
    plt.grid(True)
    plt.show()

def main():
    # Khởi tạo embedder
    print("Đang tải model 'glove-wiki-gigaword-50'...")
    embedder = WordEmbedder(model_name='glove-wiki-gigaword-50')

    # Chọn một danh sách các từ để trực quan hóa
    words_to_plot = [
        'king', 'queen', 'prince', 'princess',
        'cat', 'dog', 'lion', 'tiger',
        'vietnam', 'japan', 'france', 'italy',
        'computer', 'software', 'internet', 'phone'
    ]

    # Gọi hàm trực quan hóa
    if embedder.model:
        visualize_embeddings(embedder, words_to_plot)
    else:
        print("Không thể tải model. Tác vụ trực quan hóa bị hủy.")

if __name__ == "__main__":
    main()