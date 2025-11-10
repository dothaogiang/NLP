import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset  # Thêm thư viện này

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.text_classifier import TextClassifier

def load_financial_dataset():
    """
    Tải và lọc bộ dataset 'twitter-financial-news-sentiment'.
    Chỉ giữ lại nhãn 0 (Bearish) và 1 (Bullish) để 
    giữ bài toán ở dạng phân loại nhị phân.
    """
    print("Đang tải dataset từ Hugging Face")
    # Tải dataset (chỉ cần split 'train' vì nó chứa tất cả dữ liệu)
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment", "default", split="train")
    
    # Lọc để chỉ giữ nhãn 0 và 1
    filtered_dataset = dataset.filter(lambda example: example['label'] in [0, 1])
    
    print(f"Đã lọc dataset, còn lại {len(filtered_dataset)} mẫu (từ {len(dataset)} mẫu).")
    
    texts = filtered_dataset['text']
    labels = filtered_dataset['label']
    
    return texts, labels

def run_baseline_test():
    print(" Chạy Baseline Test (Task 2) với Dataset lớn")
    
    # 1. Tải và chuẩn bị dữ liệu
    texts, labels = load_financial_dataset()

    # 2. Chia dữ liệu 
    X_train, X_test, y_train, y_test = train_test_split(
        texts, 
        labels, 
        test_size=0.2,       # 20% cho test
        random_state=42,     # Để đảm bảo kết quả nhất quán
        stratify=labels      # Giữ tỷ lệ nhãn ở cả 2 tập train/test
    )
    
    print(f"Tổng số mẫu: {len(texts)}")
    print(f"Tập huấn luyện: {len(X_train)} mẫu")
    print(f"Tập kiểm tra: {len(X_test)} mẫu")

    # 3. Khởi tạo Vectorizer
    vectorizer = TfidfVectorizer(
        lowercase=True, 
        stop_words='english', 
        max_features=5000  # Giới hạn số lượng features khi dùng dataset lớn
    )

    # 4. Khởi tạo TextClassifier
    classifier = TextClassifier(vectorizer=vectorizer)

    # 5. Huấn luyện model
    print("\n Huấn luyện model baseline (LogisticRegression) ")
    classifier.fit(X_train, y_train)

    # 6. Tạo dự đoán
    print("Tạo dự đoán trên tập test")
    y_pred = classifier.predict(X_test)

    # 7. Đánh giá
    metrics = classifier.evaluate(y_test, y_pred)
    
    print("\n Kết quả Model Baseline (LogisticRegression)")
    print(f"Metrics: {metrics}")
    
    print(f"Baseline Accuracy: {metrics['accuracy']:.4f}")
    print(f"Baseline F1-score: {metrics['f1_score']:.4f}")

if __name__ == "__main__":
    run_baseline_test()