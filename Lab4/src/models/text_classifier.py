from typing import List, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TextClassifier:
    """
    Một lớp bao bọc (wrapper) cho pipeline phân loại văn bản,
    bao gồm vectorization và mô hình Logistic Regression.
    """
    def __init__(self, vectorizer):
        """
        Khởi tạo classifier.
        [cite: 73]
        :param vectorizer: Một instance của Vectorizer (ví dụ: TfidfVectorizer)
        """
        self.vectorizer = vectorizer
        # Khởi tạo model
        self.model = None

    def fit(self, texts: List[str], labels: List[int]):
        """
        Huấn luyện model.
        [cite: 8, 75]
        """
        # Dùng vectorizer để 'fit_transform' dữ liệu văn bản huấn luyện
        X = self.vectorizer.fit_transform(texts)
        
        # Khởi tạo LogisticRegression. 'solver="liblinear"' tốt cho dataset nhỏ.
        self.model = LogisticRegression(solver='liblinear')
        
        # Huấn luyện model
        self.model.fit(X, labels)
        print("Model đã được huấn luyện.")

    def predict(self, texts: List[str]) -> List[int]:
        """
        Dự đoán nhãn cho văn bản mới.
        [cite: 9, 80]
        """
        if self.model is None:
            raise ValueError("Model chưa được huấn luyện.")
        
        # Dùng vectorizer để 'transform' dữ liệu mới.
        # sử dụng cùng một bộ từ vựng đã học từ bước 'fit'.
        X = self.vectorizer.transform(texts)
        
        # Trả về dự đoán của model
        return self.model.predict(X)

    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """
        Đánh giá hiệu suất của model.
        [cite: 10, 84]
        """
        # Tính toán các metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        
        return metrics