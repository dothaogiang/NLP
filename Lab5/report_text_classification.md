# Báo cáo Lab 5: Xây dựng mô hình RNN cho Part-of-Speech Tagging

## 1. Mục tiêu

Bài thực hành này nhằm mục đích xây dựng một mô hình mạng nơ-ron hồi quy (RNN) hoàn chỉnh để giải quyết bài toán gán nhãn từ loại (Part-of-Speech Tagging). Mục tiêu là áp dụng các kiến thức lý thuyết về PyTorch để xử lý dữ liệu chuỗi, bao gồm:

- Tải và tiền xử lý dữ liệu văn bản từ định dạng CoNLL-U.

- Xây dựng từ điển (vocabulary) cho từ và nhãn, bao gồm cả các token đặc biệt.

- Tạo một lớp `Dataset` và `DataLoader` tùy chỉnh, có khả năng xử lý các câu có độ dài khác nhau bằng kỹ thuật đệm (padding).
  Xây dựng mô hình RNN đơn giản bằng các khối `nn.Embedding`, `nn.RNN`, và `nn.Linear`.

- Huấn luyện và đánh giá hiệu năng của mô hình trên tập dữ liệu thực tế.

## 2. Phương pháp Thực hiện

### 2.1 Task 1: Tải và Tiền xử lý Dữ liệu

- Tải dữ liệu: Một hàm `load_conllu` đã được viết để đọc các file `.conllu.` Hàm này duyệt qua file, bỏ qua các dòng chú thích và các token đa từ (1-2), đồng thời trích xuất cặp (từ, nhãn) từ cột 2 (FORM) và 4 (UPOS).

- Kết quả tải:

  - Tải thành công 12.544 câu từ tập `en_ewt-ud-train.conllu`.

  - Tải thành công 2.001 câu từ tập `en_ewt-ud-dev.conllu`.

- Xây dựng Từ điển

- Từ điển được xây dựng chỉ từ tập huấn luyện.

- Hai token đặc biệt đã được thêm vào: <PAD> (index 0) và <UNK> (index 1).

- Kích thước từ điển từ (word_to_ix): 19.675

- Kích thước từ điển nhãn (tag_to_ix): 18

### 2.2 Task 2: Tạo PyTorch Dataset và DataLoader

- `POSDataset`: Một lớp `POSDataset` kế thừa từ `torch.utils.data.Dataset` đã được tạo.

  Hàm **getitem** nhận vào một chỉ số idx, lấy câu tương ứng và chuyển đổi các từ/nhãn thành các chỉ số (index) bằng cách tra cứu từ điển `word_to_ix` và `tag_to_ix`.

- Collate Function: Một hàm `collate_fn` đã được định nghĩa để xử lý các câu có độ dài không đồng đều trong một batch.

  Hàm này sử dụng `torch.nn.utils.rnn.pad_sequence` để đệm (pad) cả tensor câu và tensor nhãn về cùng một độ dài (bằng câu dài nhất trong batch), sử dụng giá trị `padding_value=0`.

- DataLoader: Các DataLoader cho `train_loader` và `dev_loader` đã được khởi tạo với `BATCH_SIZE = 32`. Kết quả kiểm tra một batch cho thấy shape [32, 33], xác nhận `batch_first=True` và cơ chế padding hoạt động chính xác.

### 2.3 Task 3: Xây dựng Mô hình RNN

Một lớp SimpleRNNForTokenClassification đã được định nghĩa với kiến trúc 3 lớp:

1. `nn.Embedding`: Chuyển đổi các chỉ số từ (vocab_size: 19675) thành vector dày đặc (embedding_dim: 100). Lớp này cũng được thông báo `padding_idx=0` để bỏ qua việc học cho token <PAD>.

2. `nn.RNN`: Nhận chuỗi vector embedding và xử lý chúng với 128 đơn vị ẩn (hidden_dim: 128). `batch_first=True` được sử dụng để khớp với đầu ra của DataLoader.

3. `nn.Linear`: Nhận đầu ra của RNN (shape: `hidden_dim`) và ánh xạ nó sang không gian nhãn (tagset_size: 18) để đưa ra dự đoán (logits) cho mỗi token.

### 2.4 Task 4: Khởi tạo và Huấn luyện

- Thiết lập: Mô hình được huấn luyện trong 5 epochs.

  - Loss Function: nn.CrossEntropyLoss được sử dụng, với ignore_index=PAD_INDEX (giá trị 0). Điều này cực kỳ quan trọng, giúp hàm loss bỏ qua các vị trí đệm (<PAD>) khi tính toán, đảm bảo mô hình chỉ học trên các token thực.

  - Optimizer: torch.optim.Adam với learning rate lr=0.001.

- Vòng lặp huấn luyện: Mô hình được huấn luyện trên thiết bị CPU. Trong mỗi epoch, mô hình thực hiện 5 bước: zero_grad(), forward pass, loss.backward(), và optimizer.step(). Hàm loss được reshape (.view(-1)) để phù hợp với yêu cầu của CrossEntropyLoss.
- Kết quả huấn luyện: Sau 5 epochs, loss giảm từ khoảng 2.89 xuống còn khoảng 0.45, cho thấy mô hình đã học được các mẫu từ dữ liệu.

## 3. Kết quả và Phân tích

Bảng dưới đây tóm tắt kết quả huấn luyện qua 5 epochs:

| Epoch | Train Loss (avg) | Train Accuracy | Development Accuracy |
| :---: | ---------------: | -------------: | -------------------: |
|   1   |           1.1309 |         0.7687 |               0.7418 |
|   2   |           0.6289 |         0.8360 |               0.7966 |
|   3   |           0.4714 |         0.8754 |               0.8279 |
|   4   |           0.3731 |         0.9006 |               0.8388 |
|   5   |           0.3047 |         0.9194 |               0.8473 |

- Độ chính xác trên tập huấn luyện và tập phát triển đều tăng qua các epoch, với độ chính xác cuối cùng trên tập phát triển đạt khoảng 84.73%.

- Mô hình học hiệu quả: Giá trị Train Loss giảm đều đặn và mạnh mẽ qua 5 epochs (từ 1.1309 xuống 0.3047). Đồng thời, Train Accuracy tăng từ 76.8% lên 91.9%. Điều này cho thấy mô hình RNN đơn giản có khả năng học được các mẫu (pattern) trong dữ liệu huấn luyện.

- Khả năng Tổng quát hóa: Development Accuracy (độ chính xác trên tập dữ liệu chưa thấy) cũng tăng đều đặn, từ 74.1% lên 84.73%. Điều này chứng tỏ mô hình không chỉ học vẹt mà còn có khả năng tổng quát hóa tốt.

- Hiện tượng Overfitting: Vào Epoch 5, khoảng cách giữa Train Acc (91.9%) và Dev Acc (84.7%) là khá đáng kể (khoảng 7.2%). Điều này cho thấy dấu hiệu của overfitting (mô hình bắt đầu học quá khớp với dữ liệu train). Nếu tiếp tục huấn luyện, Dev Acc có thể bắt đầu giảm.

### Đánh giá:

Hàm `predict_sentence` được sử dụng để kiểm tra mô hình trên các câu mới (chưa có trong từ điển). Kết quả rất khả quan:

        Câu: 'From the AP comes this story'

        - Dự đoán: `[('From', 'ADP'), ('the', 'DET'), ('AP', 'PROPN'), ('comes', 'VERB'), ('this', 'DET'), ('story', 'NOUN')]`

        - Nhận xét: Dự đoán chính xác.

        Câu: 'This model is pretty good'
        - Dự đoán: `[('This', 'PRON'), ('model', 'NOUN'), ('is', 'AUX'), ('pretty', 'ADV'), ('good', 'ADJ')]`

        - Nhận xét: Dự đoán chính xác.

        Câu: 'The old man is running'

        - Dự đoán: `[('The', 'DET'), ('old', 'ADJ'), ('man', 'NOUN'), ('is', 'AUX'), ('running', 'VERB')]`

        Nhận xét: Dự đoán chính xác, mô hình phân biệt được is (AUX) và running (VERB).

## 4. Kết luận

- Bài thực hành đã hoàn thành thành công. Chúng ta đã xây dựng một pipeline PyTorch hoàn chỉnh cho bài toán POS tagging, từ việc đọc file CoNLL-U, tạo `Dataset` với `collate_fn` tùy chỉnh, đến việc xây dựng và huấn luyện mô hình RNN.

- Mô hình RNN đơn giản đạt được độ chính xác 84.73% trên tập development sau 5 epochs, cho thấy khả năng nắm bắt thông tin chuỗi hiệu quả. Các thử nghiệm dự đoán trên câu mới cũng cho kết quả chính xác, chứng minh tính thực tiễn của mô hình.
