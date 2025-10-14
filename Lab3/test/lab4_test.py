import sys
import os


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.representations.word_embedder import WordEmbedder

def main():
    # Khởi tạo WordEmbedder với model glove-wiki-gigaword-50
    # Lần đầu chạy sẽ mất thời gian để tải model (khoảng 65MB)
    embedder = WordEmbedder(model_name='glove-wiki-gigaword-50')
    
    if embedder.model is None:
        print("Không thể thực hiện đánh giá do model chưa được tải.")
        return

    print("\n--- Bắt đầu đánh giá WordEmbedder ---")

    # 1. Lấy vector cho từ 'king'
    print("\n1. Vector của từ 'king':")
    king_vector = embedder.get_vector('king')
    print(king_vector)
    print(f"   (Kích thước: {king_vector.shape})")

    # 2. Tính độ tương đồng
    print("\n2. Độ tương đồng ngữ nghĩa:")
    similarity_king_queen = embedder.get_similarity('king', 'queen')
    print(f"   - Độ tương đồng giữa 'king' và 'queen': {similarity_king_queen:.4f}")
    
    similarity_king_man = embedder.get_similarity('king', 'man')
    print(f"   - Độ tương đồng giữa 'king' và 'man': {similarity_king_man:.4f}")

    # 3. Tìm các từ đồng nghĩa nhất với 'computer'
    print("\n3. Top 10 từ đồng nghĩa nhất với 'computer':")
    similar_words = embedder.get_most_similar('computer', top_n=10)
    for word, score in similar_words:
        print(f"   - {word}: {score:.4f}")

    # 4. Nhúng câu "The queen rules the country."
    sentence = "The queen rules the country."
    print(f"\n4. Nhúng câu: '{sentence}'")
    document_vector = embedder.embed_document(sentence)
    print(document_vector)
    print(f"   (Kích thước: {document_vector.shape})")

    print("\n--- Đánh giá hoàn tất ---")

if __name__ == "__main__":
    main()