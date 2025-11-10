# Báo cáo tóm tắt Lab 4: Word Embeddings

## 1. Mục tiêu phân tích

Bài lab này không chỉ là một bài tập triển khai code, mà là một cuộc khảo sát thực tiễn về sự chuyển đổi mô hình trong NLP: từ biểu diễn từ vựng rời rạc, tần suất (TF-IDF) sang không gian vector ngữ nghĩa dày đặc. Mục tiêu của báo cáo này là phân tích sâu sắc các khía cạnh sau:

Bản chất của không gian ngữ nghĩa: Phân tích cách các mô hình như GloVe mã hóa các mối quan hệ trừu tượng (như giới tính, chức vụ) vào cấu trúc hình học của không gian vector.

Sự đánh đổi giữa các phương pháp: Đánh giá ưu và nhược điểm của việc sử dụng model có sẵn, tự huấn luyện trên dữ liệu nhỏ, và escalability (khả năng mở rộng) với các công cụ Big Data như Spark.

Ý nghĩa của trực quan hóa: Diễn giải biểu đồ t-SNE không chỉ như một hình ảnh đẹp, mà như một bản đồ tri thức, tiết lộ cấu trúc ngữ nghĩa tiềm ẩn trong dữ liệu.

## 2. Phân tích Phương pháp luận và Lựa chọn Kỹ thuật

### 2.1. Lớp WordEmbedder Transfer Learning

Bắt đầu với một model pre-trained như glove-wiki-gigaword-50.

Phân tích: Model này không chỉ học được rằng "vua" và "nữ hoàng" có liên quan, mà nó còn học được từ vô số ngữ cảnh khác nhau, cho phép nó định vị các vector này trong một không gian đa chiều phức tạp, nắm bắt được các sắc thái.

Phương pháp embed_document bằng cách lấy trung bình cộng các vector từ là một kỹ thuật baseline hiệu quả nhưng cũng có hạn chế:

- Ưu điểm: Nhanh, đơn giản, và thường cho kết quả tốt một cách đáng ngạc nhiên vì nó giữ lại "tín hiệu ngữ nghĩa" trung tâm của câu.

- Nhược điểm & Phân tích: bỏ qua thứ tự từ, làm cho hai câu "chó cắn người" và "người cắn chó" có thể có vector biểu diễn gần giống nhau. Nó cũng dễ bị ảnh hưởng bởi các từ dừng (stop words) nếu không được lọc kỹ, làm "pha loãng" vector ngữ nghĩa.

### 2.2. Huấn luyện với Gensim

Việc tự huấn luyện model trên một tập dữ liệu nhỏ cho thấy một thực tế quan trọng: chất lượng của embedding phụ thuộc hoàn toàn vào chất lượng và quy mô của dữ liệu đầu vào.

Phân tích: Model tự huấn luyện có thể rất tốt trong việc nắm bắt các mối quan hệ đặc thù chỉ có trong tập dữ liệu nhỏ đó. Tuy nhiên, nó sẽ có một "từ vựng" rất hạn chế và hoàn toàn "mù" về thế giới bên ngoài. Nó không thể biết mối quan hệ giữa "vua" và "tổng thống" nếu khái niệm "tổng thống" không xuất hiện. Đây là một minh chứng cho bài toán "khởi động lạnh" (cold start) trong machine learning.

## 2.3. Mở rộng với PySpark

Task này chuyển trọng tâm từ "thuật toán" sang "hệ thống". Gensim rất tốt, nhưng nó hoạt động trong bộ nhớ RAM của một máy duy nhất.

Phân tích: Khi dữ liệu lên đến hàng Terabyte, không một máy đơn lẻ nào có thể chứa nổi. Spark giải quyết vấn đề này bằng cách phân tán cả dữ liệu và tính toán trên một cụm máy tính. Lỗi OutOfMemoryError gặp phải không phải là một bug, mà là một bài học thực tế về giới hạn của tính toán đơn luồng. Việc cấu hình spark.driver.memory cho thấy rằng các bài toán NLP quy mô lớn đòi hỏi kiến thức về kỹ thuật hệ thống (systems engineering) cũng nhiều như kiến thức về khoa học dữ liệu.

## 3. Kết quả

### 3.1. Phép toán trên Ngữ nghĩa

Kết quả similarity('king', 'queen') ≈ 0.7839 và similarity('king', 'man') ≈ 0.5309 không chỉ là những con số. Chúng là bằng chứng định lượng cho thấy không gian vector đã học được các thuộc tính ngữ nghĩa trừu tượng:

Phân tích: Khoảng cách gần giữa king và queen phản ánh chúng cùng chia sẻ nhiều thuộc tính: [hoàng gia], [cai trị], [quyền lực]. Khoảng cách xa hơn giữa king và man cho thấy king là một tập hợp con chuyên biệt hơn của man, với thuộc tính [hoàng gia] được thêm vào. Điều này mở ra khả năng thực hiện các phép toán vector nổi tiếng, ví dụ:
vector('king') - vector('man') + vector('woman') ≈ vector('queen')
Phép toán này cho thấy model đã học được một "vector giới tính" và một "vector hoàng gia" một cách độc lập.

### 3.2. Biểu đồ t-SNE

Biểu đồ t-SNE là phần phân tích định tính mạnh mẽ nhất.

Sự hội tụ trong cụm (Intra-cluster Cohesion): Việc các từ như vietnam, japan, france, italy gom lại thành một cụm không phải là ngẫu nhiên. Nó xảy ra vì trong kho dữ liệu hàng tỷ từ, chúng thường xuyên xuất hiện trong các ngữ cảnh tương tự: "...thủ đô của [tên nước]...", "...du lịch đến [tên nước]...", "...kinh tế của [tên nước]...". t-SNE đã trực quan hóa các ngữ cảnh chung này.

Sự cách biệt giữa các cụm (Inter-cluster Separation): Khoảng cách lớn giữa cụm "công nghệ" và cụm "động vật" cho thấy sự khác biệt về ngữ cảnh sử dụng. Rất hiếm khi một tài liệu nói về software lại đồng thời nói về lion trong cùng một câu. Khoảng cách trong biểu đồ chính là thước đo cho sự khác biệt về ngữ cảnh.

Cấu trúc: Nhìn kỹ hơn vào cụm công nghệ, Có thể thấy software sẽ gần với computer hơn là phone, phản ánh mối quan hệ "phần mềm chạy trên phần cứng". internet có thể nằm ở một vị trí trung gian, kết nối tất cả các thiết bị. Những cấu trúc tinh vi này cho thấy word embedding không chỉ nhóm các từ lại, mà còn sắp xếp chúng theo một logic có trật tự.

## 4. Khó khăn và Giải pháp

Trong quá trình thực hiện, một số thách thức kỹ thuật đã phát sinh và được giải quyết thành công:

- Lỗi đường dẫn (FileNotFoundError):

  - Vấn đề: Chương trình không tìm thấy file dữ liệu hoặc không thể lưu model do cấu trúc thư mục phức tạp.

  - Giải pháp: Chạy tất cả các script từ thư mục gốc của project (Lab3) và sử dụng đường dẫn tương đối (../data/) để truy cập các file ở thư mục ngang hàng. Tự động tạo thư mục results bằng os.makedirs trước khi lưu.

- Lỗi môi trường PySpark trên Windows:

  - Vấn đề: Spark không thể khởi chạy do thiếu file winutils.exe và gây ra lỗi java.lang.UnsatisfiedLinkError.

  - Giải pháp: Tải bộ winutils tương thích với Hadoop, tạo thư mục C:\hadoop\bin và thiết lập biến môi trường HADOOP_HOME.

- Lỗi tràn bộ nhớ của Spark (OutOfMemoryError):

  - Vấn đề: Khi xử lý file dữ liệu lớn, Spark bị hết RAM mặc định được cấp phát cho Java Virtual Machine (JVM).

  - Giải pháp: Tăng bộ nhớ cho Spark driver bằng cách thêm cấu hình .config("spark.driver.memory", "4g") khi khởi tạo SparkSession.

5. Kết luận và Hướng phát triển
   Bài lab đã chứng minh một cách thuyết phục rằng có thể biểu diễn ngữ nghĩa của từ trong một không gian vector toán học. Các vector này không chỉ nắm bắt sự tương đồng bề mặt mà còn cả các mối quan hệ cấu trúc phức tạp, cho phép thực hiện các phép toán ngữ nghĩa.

Hướng phát triển:

Vượt qua giới hạn của "Túi từ": Kỹ thuật trung bình cộng vector đã bỏ qua thứ tự từ. Bước tiếp theo tự nhiên sẽ là khám phá các mô hình nhận biết ngữ cảnh và thứ tự như Sentence-BERT hoặc các kiến trúc Transformer, để có được các vector biểu diễn câu/văn bản chính xác hơn.

Đánh giá trên Tác vụ Cụ thể (Downstream Task): Giá trị thực sự của các word embedding nằm ở việc chúng cải thiện hiệu suất của các mô hình khác. Một bước phát triển hợp lý là áp dụng các vector đã huấn luyện này làm đầu vào cho một bài toán phân loại văn bản (ví dụ: phân tích cảm xúc) và đo lường sự cải thiện.

Khám phá các loại Embedding khác: So sánh hiệu quả giữa Word2Vec (CBOW/Skip-gram), GloVe, và fastText (với khả năng xử lý từ ngoài từ điển - OOV) trên cùng một bộ dữ liệu để hiểu rõ hơn về ưu và nhược điểm của từng loại.
