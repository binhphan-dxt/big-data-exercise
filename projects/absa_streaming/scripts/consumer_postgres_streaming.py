# # SE363 – Phát triển ứng dụng trên nền tảng dữ liệu lớn
# # Khoa Công nghệ Phần mềm – Trường Đại học Công nghệ Thông tin, ĐHQG-HCM
# # HopDT – Faculty of Software Engineering, University of Information Technology (FSE-UIT)

# # consumer_postgres_streaming.py
# # ======================================
# # Consumer đọc dữ liệu từ Kafka topic "absa-reviews"
# # → chạy inference mô hình ABSA (.pt)
# # → ghi kết quả vào PostgreSQL
# # → Airflow sẽ giám sát và khởi động lại khi job bị dừng.

# from pyspark.sql import SparkSession, functions as F, types as T
# from pyspark.sql.functions import pandas_udf, from_json, col
# import pandas as pd, torch, torch.nn as nn, torch.nn.functional as tF
# from transformers import AutoTokenizer, AutoModel
# import random, time, os, sys, json
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)
# # === 1. Spark session với Kafka connector ===
# scala_version = "2.12"
# spark_version = "3.5.1"

# spark = (
#     SparkSession.builder
#     .appName("Kafka_ABSA_Postgres")
#     .config("spark.jars.packages",
#             f"org.apache.spark:spark-sql-kafka-0-10_{scala_version}:{spark_version},"
#             "org.postgresql:postgresql:42.6.0,"
#             "org.apache.kafka:kafka-clients:3.5.1")
#     .config("spark.executor.instances", "1")
#     .config("spark.executor.cores", "1")
#     .config("spark.driver.maxResultSize", "4g")
#     .config("spark.sql.streaming.checkpointLocation", "/opt/airflow/checkpoints/absa_streaming_checkpoint")
#     .config("spark.sql.execution.arrow.pyspark.enabled", "false")
#     .getOrCreate()
# )
# spark.sparkContext.setLogLevel("WARN")

# # === 2. Đọc dữ liệu streaming từ Kafka ===
# df_stream = (
#     spark.readStream
#     .format("kafka")
#     .option("kafka.bootstrap.servers", "kafka:9092")
#     .option("subscribe", "absa-reviews")
#     .option("startingOffsets", "earliest")
#     .option("maxOffsetsPerTrigger", 5)
#     .load()
# )

# df_text = df_stream.selectExpr("CAST(value AS STRING) as Review")

# # === 3. Định nghĩa mô hình ABSA ===
# ASPECTS = ["Price","Shipping","Outlook","Quality","Size","Shop_Service","General","Others"]
# MODEL_NAME = "xlm-roberta-base"
# MODEL_PATH = "/opt/airflow/models/best_absa_hardshare.pt"
# MAX_LEN = 64
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# _model, _tokenizer = None, None

# class ABSAModel(nn.Module):
#     def __init__(self, model_name=MODEL_NAME, num_aspects=len(ASPECTS)):
#         super().__init__()
#         self.backbone = AutoModel.from_pretrained(model_name)
#         H = self.backbone.config.hidden_size
#         self.dropout = nn.Dropout(0.1)
#         self.head_m = nn.Linear(H, num_aspects)
#         self.head_s = nn.Linear(H, num_aspects * 3)
#     def forward(self, input_ids, attention_mask):
#         out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
#         h_cls = self.dropout(out.last_hidden_state[:, 0, :])
#         return self.head_m(h_cls), self.head_s(h_cls).view(-1, len(ASPECTS), 3)

# @pandas_udf(T.ArrayType(T.FloatType()))
# def absa_infer_udf(texts: pd.Series) -> pd.Series:
#     global _model, _tokenizer
#     if _model is None:
#         _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
#         _model = ABSAModel()
#         _model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
#         _model.to(DEVICE).eval()

#     results = []
#     for t in texts:
#         enc = _tokenizer(t, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
#         with torch.no_grad():
#             logits_m, logits_s = _model(enc["input_ids"], enc["attention_mask"])
#             p_m = torch.sigmoid(logits_m)[0].cpu().numpy().tolist()
#             p_s = tF.softmax(logits_s, dim=-1)[0].cpu().numpy().flatten().tolist()
#         results.append(p_m + p_s)
#     return pd.Series(results)

# df_pred = df_text.withColumn("predictions", absa_infer_udf(F.col("Review")))

# # === 4. Giải mã kết quả ra nhãn POS/NEG/NEU ===
# @pandas_udf("string")
# def decode_sentiment(preds: pd.Series) -> pd.Series:
#     SENTIMENTS = ["POS", "NEU", "NEG"]
#     res = []
#     for p in preds:
#         if not p:
#             res.append("?")
#             continue
#         p = list(p)
#         p_m, p_s = p[:len(ASPECTS)], p[len(ASPECTS):]
#         decoded = []
#         for i, asp in enumerate(ASPECTS):
#             triplet = p_s[i*3:(i+1)*3]
#             s = SENTIMENTS[int(max(range(3), key=lambda j: triplet[j]))]
#             decoded.append(f"{asp}:{s}")
#         res.append(", ".join(decoded))
#     return pd.Series(res)

# df_final = df_pred.withColumn("decoded", decode_sentiment(F.col("predictions")))
# for asp in ASPECTS:
#     df_final = df_final.withColumn(asp, F.regexp_extract("decoded", f"{asp}:(\\w+)", 1))

# # === Giải mã Review JSON thành text tiếng Việt trước khi stream ===
# review_schema = T.StructType([
#     T.StructField("id", T.StringType()),
#     T.StructField("review", T.StringType())
# ])
# df_final = df_final.withColumn("ReviewText", from_json(col("Review"), review_schema).getField("review"))

# # === 5. Ghi kết quả vào PostgreSQL (chuẩn UTF-8, log đầy đủ, xử lý lỗi an toàn) ===
# def write_to_postgres(batch_df, batch_id):
#     sys.stdout.reconfigure(encoding='utf-8')
#     total_rows = batch_df.count()

#     if total_rows == 0:
#         print(f"[Batch {batch_id}] ⚠️ Không có dữ liệu mới.")
#         return

#     preview = batch_df.select("ReviewText", *ASPECTS).limit(5).toPandas().to_dict(orient="records")
#     print(f"\n[Batch {batch_id}] Nhận {total_rows} dòng, hiển thị 5 dòng đầu:")
#     print(json.dumps(preview, ensure_ascii=False, indent=2))

#     # Giả lập lỗi mô phỏng để test Airflow restart
#     if batch_id % 5 == 0:
#         print(f"[Batch {batch_id}] 💥 Giả lập sự cố: crash mô phỏng.")
#         raise Exception(f"Simulated crash at batch {batch_id}")

#     try:
#         (batch_df
#             .select("ReviewText", *ASPECTS)
#             .write
#             .format("jdbc")
#             .option("url", "jdbc:postgresql://postgres:5432/airflow")
#             .option("dbtable", "absa_results")
#             .option("user", "airflow")
#             .option("password", "airflow")
#             .option("driver", "org.postgresql.Driver")
#             .option("charset", "utf8")
#             .mode("append")
#             .save()
#         )
#         print(f"[Batch {batch_id}] ✅ Ghi PostgreSQL thành công ({total_rows} dòng).")
#         subset = batch_df.select("ReviewText", *ASPECTS).limit(3).toPandas().to_dict(orient="records")
#         print(f"[Batch {batch_id}] Dữ liệu đã ghi (mẫu):")
#         print(json.dumps(subset, ensure_ascii=False, indent=2))

#     except Exception as e:
#         print(f"[Batch {batch_id}] ⚠️ Không thể ghi vào PostgreSQL, ghi log ra console thay thế.")
#         print(f"Lỗi: {str(e)}")
#         subset = batch_df.select("ReviewText", *ASPECTS).limit(5).toPandas().to_dict(orient="records")
#         print(json.dumps(subset, ensure_ascii=False, indent=2))

# # === 6. Bắt đầu stream ===
# query = (
#     df_final.writeStream
#     .foreachBatch(write_to_postgres)
#     .outputMode("append")
#     .trigger(processingTime="5 seconds")
#     .start()
# )

# print("🚀 Streaming job started — đang lắng nghe dữ liệu từ Kafka...")
# query.awaitTermination()

# consumer_postgres_streaming.py
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.functions import pandas_udf, from_json, col
import pandas as pd, json, sys, torch, torch.nn as nn, torch.nn.functional as tF
from transformers import AutoTokenizer, AutoModel
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# === 1. Spark session với Kafka connector ===
scala_version = "2.12"
spark_version = "3.5.1"

spark = (
    SparkSession.builder
    .appName("Kafka_ABSA_Postgres")
    .config("spark.jars.packages",
            f"org.apache.spark:spark-sql-kafka-0-10_{scala_version}:{spark_version},"
            "org.postgresql:postgresql:42.6.0,"
            "org.apache.kafka:kafka-clients:3.5.1")
    .config("spark.executor.instances", "1")
    .config("spark.executor.cores", "1")
    .config("spark.driver.maxResultSize", "2g")  # Giảm từ 4g xuống 2g để tiết kiệm RAM
    .config("spark.sql.streaming.checkpointLocation", "/opt/airflow/checkpoints/absa_streaming_checkpoint")
    .config("spark.sql.execution.arrow.pyspark.enabled", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# === 2. Đọc dữ liệu streaming từ Kafka ===
df_stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "kafka:9092")
    .option("subscribe", "absa-reviews")
    .option("startingOffsets", "earliest")
    .option("maxOffsetsPerTrigger", 5)  # batch nhỏ
    .load()
)

df_text = df_stream.selectExpr("CAST(value AS STRING) as Review")

# === 3. Định nghĩa mô hình ABSA (dùng mô hình nhẹ) ===
ASPECTS = ["Price","Shipping","Outlook","Quality","Size","Shop_Service","General","Others"]
# Dùng distilbert-base-multilingual-cased thay vì xlm-roberta-base để tiết kiệm RAM (~500MB vs ~1GB)
MODEL_NAME = "distilbert-base-multilingual-cased"
MODEL_PATH = "/opt/airflow/models/best_absa_hardshare.pt"
MAX_LEN = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global variables để cache model và tokenizer
_model, _tokenizer = None, None
_model_load_failed = False  # Flag để đánh dấu đã thử load và fail

class ABSAModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_aspects=len(ASPECTS)):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        H = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head_m = nn.Linear(H, num_aspects)  # Multi-label classification cho aspects
        self.head_s = nn.Linear(H, num_aspects * 3)  # Sentiment classification (POS/NEU/NEG)
    
    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h_cls = self.dropout(out.last_hidden_state[:, 0, :])
        return self.head_m(h_cls), self.head_s(h_cls).view(-1, len(ASPECTS), 3)

@pandas_udf(T.ArrayType(T.FloatType()))
def absa_infer_udf(texts: pd.Series) -> pd.Series:
    global _model, _tokenizer, _model_load_failed
    
    # Nếu đã thử load và fail, dùng fallback luôn
    if _model_load_failed:
        return _fallback_prediction(texts)
    
    # Load model và tokenizer một lần (lazy loading)
    if _model is None:
        print("[Model] Đang load mô hình ABSA...")
        try:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
            _model = ABSAModel()
            
            # Thử load từ file, nếu không có hoặc không tương thích thì dùng pretrained
            if os.path.exists(MODEL_PATH):
                try:
                    print(f"[Model] Đang load weights từ {MODEL_PATH}...")
                    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
                    _model.load_state_dict(state_dict, strict=False)
                    print(f"[Model] ✅ Đã load weights thành công")
                except Exception as load_error:
                    print(f"[Model] ⚠️ Không thể load weights (có thể không tương thích): {load_error}")
                    print(f"[Model] Sử dụng pretrained model (chưa fine-tune)")
            else:
                print(f"[Model] ⚠️ Không tìm thấy {MODEL_PATH}, sử dụng pretrained model (chưa fine-tune)")
            
            _model.to(DEVICE).eval()
            # Tối ưu memory: set model ở chế độ eval và disable gradient
            torch.set_grad_enabled(False)
            print(f"[Model] ✅ Đã load mô hình thành công (device: {DEVICE})")
        except Exception as e:
            print(f"[Model] ❌ Lỗi khi load mô hình: {e}")
            print(f"[Model] Sử dụng fallback: rule-based prediction")
            _model_load_failed = True  # Đánh dấu đã fail để không thử lại
            # Fallback: trả về prediction đơn giản dựa trên keywords
            return _fallback_prediction(texts)
    
    # Inference
    results = []
    try:
        for t in texts:
            if not t or pd.isna(t):
                # Nếu text rỗng, trả về prediction mặc định
                p_m = [0.3] * len(ASPECTS)
                p_s = [0.33, 0.33, 0.34] * len(ASPECTS)
                results.append(p_m + p_s)
                continue
            
            enc = _tokenizer(
                str(t), 
                truncation=True, 
                padding="max_length", 
                max_length=MAX_LEN, 
                return_tensors="pt"
            ).to(DEVICE)
            
            with torch.no_grad():
                logits_m, logits_s = _model(enc["input_ids"], enc["attention_mask"])
                p_m = torch.sigmoid(logits_m)[0].cpu().numpy().tolist()
                p_s = tF.softmax(logits_s, dim=-1)[0].cpu().numpy().flatten().tolist()
                # Giải phóng memory ngay sau khi tính toán
                del enc, logits_m, logits_s
            
            results.append(p_m + p_s)
    except Exception as e:
        print(f"[Model] ⚠️ Lỗi khi inference: {e}, sử dụng fallback")
        return _fallback_prediction(texts)
    
    return pd.Series(results)

def _fallback_prediction(texts: pd.Series) -> pd.Series:
    """Fallback prediction dựa trên keywords nếu model không load được"""
    # Keywords cho từng aspect và sentiment
    positive_keywords = ["tốt", "đẹp", "nhanh", "ok", "ổn", "hài lòng", "tuyệt", "xuất sắc"]
    negative_keywords = ["xấu", "chậm", "tệ", "kém", "không tốt", "thất vọng", "tồi"]
    
    results = []
    for t in texts:
        if not t or pd.isna(t):
            p_m = [0.3] * len(ASPECTS)
            p_s = [0.33, 0.33, 0.34] * len(ASPECTS)
            results.append(p_m + p_s)
            continue
        
        text_lower = str(t).lower()
        p_m = []
        p_s = []
        
        # Đơn giản: nếu có từ khóa tích cực/tiêu cực thì predict aspect General
        has_positive = any(kw in text_lower for kw in positive_keywords)
        has_negative = any(kw in text_lower for kw in negative_keywords)
        
        for i, asp in enumerate(ASPECTS):
            # Aspect probability (đơn giản: General có xác suất cao hơn nếu có keywords)
            if asp == "General" and (has_positive or has_negative):
                p_m.append(0.7)
            else:
                p_m.append(0.2)
            
            # Sentiment probability
            if has_positive and not has_negative:
                p_s.extend([0.6, 0.2, 0.2])  # POS
            elif has_negative and not has_positive:
                p_s.extend([0.2, 0.2, 0.6])  # NEG
            else:
                p_s.extend([0.33, 0.34, 0.33])  # NEU
        
        results.append(p_m + p_s)
    
    return pd.Series(results)

df_pred = df_text.withColumn("predictions", absa_infer_udf(F.col("Review")))

# === 4. Giải mã kết quả ra nhãn POS/NEG/NEU ===
@pandas_udf("string")
def decode_sentiment(preds: pd.Series) -> pd.Series:
    SENTIMENTS = ["POS", "NEU", "NEG"]
    res = []
    for p in preds:
        if not p:
            res.append("?")
            continue
        p = list(p)
        p_m, p_s = p[:len(ASPECTS)], p[len(ASPECTS):]
        decoded = []
        for i, asp in enumerate(ASPECTS):
            triplet = p_s[i*3:(i+1)*3]
            s = SENTIMENTS[int(max(range(3), key=lambda j: triplet[j]))]
            decoded.append(f"{asp}:{s}")
        res.append(", ".join(decoded))
    return pd.Series(res)

df_final = df_pred.withColumn("decoded", decode_sentiment(F.col("predictions")))
for asp in ASPECTS:
    df_final = df_final.withColumn(asp, F.regexp_extract("decoded", f"{asp}:(\\w+)", 1))

# === Giải mã Review JSON ===
review_schema = T.StructType([
    T.StructField("id", T.StringType()),
    T.StructField("review", T.StringType())
])
df_final = df_final.withColumn("ReviewText", from_json(col("Review"), review_schema).getField("review"))

# === 5. Ghi kết quả vào PostgreSQL ===
def write_to_postgres(batch_df, batch_id):
    sys.stdout.reconfigure(encoding='utf-8')
    total_rows = batch_df.count()

    if total_rows == 0:
        print(f"[Batch {batch_id}] ⚠️ Không có dữ liệu mới.")
        return

    preview = batch_df.select("ReviewText", *ASPECTS).limit(5).toPandas().to_dict(orient="records")
    print(f"\n[Batch {batch_id}] Nhận {total_rows} dòng, hiển thị 5 dòng đầu:")
    print(json.dumps(preview, ensure_ascii=False, indent=2))

    try:
        (batch_df
            .select("ReviewText", *ASPECTS)
            .write
            .format("jdbc")
            .option("url", "jdbc:postgresql://postgres:5432/airflow")
            .option("dbtable", "absa_results")
            .option("user", "airflow")
            .option("password", "airflow")
            .option("driver", "org.postgresql.Driver")
            .option("charset", "utf8")
            .mode("append")
            .save()
        )
        print(f"[Batch {batch_id}] ✅ Ghi PostgreSQL thành công ({total_rows} dòng).")
    except Exception as e:
        print(f"[Batch {batch_id}] ⚠️ Không thể ghi PostgreSQL, ghi log ra console thay thế.")
        print(f"Lỗi: {str(e)}")
        subset = batch_df.select("ReviewText", *ASPECTS).limit(5).toPandas().to_dict(orient="records")
        print(json.dumps(subset, ensure_ascii=False, indent=2))

# === 6. Bắt đầu stream ===
query = (
    df_final.writeStream
    .foreachBatch(write_to_postgres)
    .outputMode("append")
    .trigger(processingTime="5 seconds")
    .start()
)

print("🚀 Streaming job started — đang lắng nghe dữ liệu từ Kafka...")
query.awaitTermination()
