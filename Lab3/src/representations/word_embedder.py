import gensim.downloader as api
import numpy as np
import re

class WordEmbedder:
    """
    Lớp này dùng để tải và sử dụng các mô hình word embedding có sẵn từ thư viện gensim.
    """
    def __init__(self, model_name: str = 'glove-wiki-gigaword-50'):
        """
        Hàm khởi tạo, tải mô hình embedding.

        Args:
            model_name (str): Tên của mô hình pre-trained cần tải từ gensim.
                              Ví dụ: 'glove-wiki-gigaword-50'.
        """
        print(f"Đang tải model '{model_name}'... Vui lòng chờ.")
        try:
            self.model = api.load(model_name)
            self.vector_size = self.model.vector_size
            print("Tải model thành công!")
        except Exception as e:
            print(f"Lỗi khi tải model: {e}")
            self.model = None

    def get_vector(self, word: str) -> np.ndarray:
        """
        Lấy vector biểu diễn cho một từ.

        Args:
            word (str): Từ cần lấy vector.

        Returns:
            np.ndarray: Vector của từ. Trả về vector không nếu từ không có trong từ điển.
        """
        if self.model is None:
            print("Model chưa được tải.")
            return np.zeros(self.vector_size)
            
        try:
            return self.model[word]
        except KeyError:
            # Xử lý trường hợp từ không có trong từ điển (Out-of-Vocabulary - OOV)
            # print(f"Cảnh báo: Từ '{word}' không có trong từ điển.")
            return np.zeros(self.vector_size)

    def get_similarity(self, word1: str, word2: str) -> float:
        """
        Tính độ tương đồng cosine giữa hai từ.

        Args:
            word1 (str): Từ thứ nhất.
            word2 (str): Từ thứ hai.

        Returns:
            float: Giá trị độ tương đồng, trong khoảng [-1, 1].
        """
        if self.model is None:
            print("Model chưa được tải.")
            return 0.0

        if word1 not in self.model.key_to_index or word2 not in self.model.key_to_index:
            print(f"Một trong hai từ '{word1}' hoặc '{word2}' không có trong từ điển.")
            return 0.0
            
        return self.model.similarity(word1, word2)

    def get_most_similar(self, word: str, top_n: int = 10):
        """
        Tìm top N từ tương đồng nhất với một từ cho trước.

        Args:
            word (str): Từ gốc.
            top_n (int): Số lượng từ tương đồng cần tìm.

        Returns:
            list: Danh sách các tuple (từ, độ tương đồng).
        """
        if self.model is None:
            print("Model chưa được tải.")
            return []
            
        try:
            return self.model.most_similar(word, topn=top_n)
        except KeyError:
            print(f"Từ '{word}' không có trong từ điển.")
            return []
    
    def tokenize(self, text: str):
        """
        Hàm tokenizer đơn giản để tách câu thành các từ.
        Chuyển về chữ thường và loại bỏ các ký tự không phải chữ cái.
        """
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def embed_document(self, document: str) -> np.ndarray:
        """
        Nhúng một văn bản bằng cách lấy trung bình vector của các từ trong văn bản đó.

        Args:
            document (str): Văn bản đầu vào.

        Returns:
            np.ndarray: Vector biểu diễn cho toàn bộ văn bản.
        """
        if self.model is None:
            print("Model chưa được tải.")
            return np.zeros(self.vector_size)

        tokens = self.tokenize(document)
        word_vectors = []
        
        for token in tokens:
            # Chỉ lấy vector của những từ có trong từ điển
            if token in self.model.key_to_index:
                word_vectors.append(self.get_vector(token))

        if not word_vectors:
            # Nếu văn bản không chứa từ nào trong từ điển, trả về vector không
            return np.zeros(self.vector_size)
        
        # Tính trung bình cộng các vector từ
        document_vector = np.mean(word_vectors, axis=0)
        return document_vector