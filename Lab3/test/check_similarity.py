import sys
sys.path.append('src')
from representations.word_embedder import WordEmbedder

def main():
    # Khởi tạo embedder. Lần đầu chạy sẽ tải model về.
    print("Đang tải model...")
    embedder = WordEmbedder(model_name='glove-wiki-gigaword-50')

    if embedder.model is None:
        print("Lỗi: Không thể tải model.")
        return

    print("\n--- Bắt đầu tính toán độ tương đồng ---")

    # Tính toán độ tương đồng cho cặp từ thứ nhất
    sim_king_queen = embedder.get_similarity('king', 'queen')

    # Tính toán độ tương đồng cho cặp từ thứ hai
    sim_king_man = embedder.get_similarity('king', 'man')

    # In kết quả đã được định dạng
    print(f"\nKết quả:")
    print(f"  - Độ tương đồng giữa 'king' và 'queen': {sim_king_queen:.4f}")
    print(f"  - Độ tương đồng giữa 'king' và 'man': {sim_king_man:.4f}")

if __name__ == "__main__":
    main()