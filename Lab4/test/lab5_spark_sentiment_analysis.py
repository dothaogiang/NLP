# test/lab5_spark_sentiment_analysis.py
# PHIÊN BẢN CẬP NHẬT (FIX LỖI EOFException)

import os
import sys

# --- FIX cho lỗi Python Version (Giữ nguyên) ---
current_python_executable = sys.executable
os.environ['PYSPARK_PYTHON'] = current_python_executable
os.environ['PYSPARK_DRIVER_PYTHON'] = current_python_executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from datasets import load_dataset

def load_hf_dataset_to_spark(spark):
    """
    Tải dataset từ Hugging Face, lọc, và chuyển sang Spark DataFrame.
    """
    print("Đang tải dataset 'twitter-financial-news-sentiment' từ Hugging Face...")
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment", "default", split="train")
    
    filtered_dataset = dataset.filter(lambda example: example['label'] in [0, 1])
    
    print("Đang chuyển đổi sang Pandas...")
    pandas_df = filtered_dataset.to_pandas()
    
    # --- FIX MỚI CHO LỖI EOFException ---
    # Tắt "cầu nối" Arrow. Spark sẽ dùng một cách khác, chậm hơn nhưng ổn định hơn.
    print("Đang tắt Arrow và chuyển đổi sang Spark DataFrame (Chế độ an toàn)...")
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
    
    # 4. Chuyển từ Pandas sang Spark DataFrame
    df = spark.createDataFrame(pandas_df)
    
    print("Tải dữ liệu thành công.")
    return df

def run_spark_analysis():
    # 1. Khởi tạo Spark Session
    spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR") 

    print("Spark Session đã được khởi tạo.")

    # 2. Tải dữ liệu
    df = load_hf_dataset_to_spark(spark)
    print("DataFrame đã tải, bắt đầu hiển thị 5 dòng đầu:")
    df.show(5) # Lệnh này sẽ kích hoạt Spark Job
    print("Hiển thị 5 dòng đầu thành công.")

    # 3. Chia dữ liệu
    (trainingData, testData) = df.randomSplit([0.8, 0.2], seed=42)
    print(f"Tổng số mẫu đã lọc: {df.count()}")
    print(f"Tập huấn luyện: {trainingData.count()} mẫu")
    print(f"Tập kiểm tra: {testData.count()} mẫu")

    # 4. Xây dựng Pipeline
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
    idf = IDF(inputCol="raw_features", outputCol="features")
    lr = LogisticRegression(maxIter=10, regParam=0.001, featuresCol="features", labelCol="label")
    pipeline = Pipeline(stages=[tokenizer, stopwordsRemover, hashingTF, idf, lr])

    # 5. Huấn luyện Model
    print("\nĐang huấn luyện pipeline Spark...")
    model = pipeline.fit(trainingData)
    print("Huấn luyện hoàn tất.")

    # 6. Đánh giá Model
    print("Đang đánh giá trên tập test...")
    predictions = model.transform(testData)
    
    evaluator_acc = MulticlassClassificationEvaluator(metricName="accuracy", labelCol="label", predictionCol="prediction")
    accuracy = evaluator_acc.evaluate(predictions)
    
    evaluator_f1 = MulticlassClassificationEvaluator(metricName="f1", labelCol="label", predictionCol="prediction")
    f1 = evaluator_f1.evaluate(predictions)

    print("\n--- Kết quả Spark ML Pipeline (trên data Hugging Face) ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")

    # 7. Dừng Spark Session
    spark.stop()

if __name__ == "__main__":
    run_spark_analysis()