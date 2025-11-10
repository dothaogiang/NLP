import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 1. Tải Dữ Liệu
def load_financial_dataset():
    print("Đang tải dataset từ Hugging Face...")
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment", "default", split="train")
    filtered_dataset = dataset.filter(lambda example: example['label'] in [0, 1])
    print(f"Đã lọc dataset, còn lại {len(filtered_dataset)} mẫu.")
    return filtered_dataset['text'], filtered_dataset['label']

#2. Tiền xử lý cho LSTM 
def preprocess_for_lstm(X_train, X_test):
    """
    Chuyển đổi văn bản thô thành các chuỗi số (sequences)
    đã được đệm (padding) cho LSTM.
    """
    print("Đang tiền xử lý cho LSTM...")
    VOCAB_SIZE = 10000
    # Độ dài tối đa của 1 câu
    MAX_LEN = 100
    
    # 1. Tạo Tokenizer
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train) # Chỉ fit trên tập train
    
    # 2. Chuyển văn bản thành chuỗi số
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # 3. Đệm (Padding)
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    
    return X_train_pad, X_test_pad, VOCAB_SIZE, MAX_LEN

# 3. Xây dựng mô hình LSTM
def build_lstm_model(VOCAB_SIZE, MAX_LEN):
    """
    Xây dựng kiến trúc mô hình LSTM.
    """
    EMBED_DIM = 128 # Kích thước vector nhúng
    LSTM_UNITS = 64
    
    model = Sequential()
    # 1. Lớp Nhúng (Embedding): Biến số nguyên thành vector
    model.add(Embedding(input_dim=VOCAB_SIZE, 
                        output_dim=EMBED_DIM, 
                        input_length=MAX_LEN))
    # Chống overfitting
    model.add(SpatialDropout1D(0.2))
    
    # 2. Lớp LSTM: Học trình tự
    model.add(LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2))
    
    # 3. Lớp Output: Phân loại nhị phân
    model.add(Dense(1, activation='sigmoid')) 
    
    # Biên dịch model
    model.compile(loss='binary_crossentropy', 
                  optimizer='adam',
                  metrics=['accuracy'])
    
    print(model.summary())
    return model

# 4. Hàm chạy chính 
def run_improvement_test():
    # 1. Tải và chia dữ liệu
    texts, labels = load_financial_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels)

    # 2. Tiền xử lý cho LSTM
    X_train_pad, X_test_pad, VOCAB_SIZE, MAX_LEN = preprocess_for_lstm(X_train, X_test)
    
    y_train_np = np.array(y_train)
    y_test_np = np.array(y_test)

    # 3. Xây dựng model
    model = build_lstm_model(VOCAB_SIZE, MAX_LEN)

    # 4. Huấn luyện model
    print("\n huấn luyện model LSTM")
    # Dừng sớm nếu model không cải thiện
    callbacks = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
    
    history = model.fit(
        X_train_pad, y_train_np,
        epochs=10, # Huấn luyện tối đa 10 epochs
        batch_size=64,
        validation_split=0.1, # Dùng 10% tập train để validation
        callbacks=callbacks
    )
    
    print(" hoàn tất ")

    # 5. Đánh giá
    print(" đánh giá trên tập test ")
    # Đánh giá Accuracy
    loss, accuracy = model.evaluate(X_test_pad, y_test_np)
    
    # Đánh giá F1-score
    y_pred_probs = model.predict(X_test_pad)
    y_pred = (y_pred_probs > 0.5).astype(int) # Chuyển xác suất > 0.5 thành 1, còn lại 0
    
    f1 = f1_score(y_test_np, y_pred, average='binary', zero_division=0)
    
    print("\n Kết quả Model Cải tiến (LSTM) ")
    print(f"Improved Accuracy: {accuracy:.4f}")
    print(f"Improved F1-score: {f1:.4f}")

if __name__ == "__main__":
    run_improvement_test()