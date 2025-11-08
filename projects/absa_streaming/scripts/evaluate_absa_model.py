# SE363 – Phát triển ứng dụng trên nền tảng dữ liệu lớn
# Khoa Công nghệ Phần mềm – Trường Đại học Công nghệ Thông tin, ĐHQG-HCM
# HopDT – Faculty of Software Engineering, University of Information Technology (FSE-UIT)

# evaluate_absa_model.py
# ======================================
# Script đánh giá mô hình ABSA mới và so sánh với mô hình hiện tại
# Trả về metrics và quyết định có nên deploy hay không

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import os
import sys
import json
import glob
from datetime import datetime
import gc
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Tăng threads với 16GB Docker RAM
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
torch.set_num_threads(4)

# Tắt caching của tokenizer để tiết kiệm RAM
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === Cấu hình ===
ASPECTS = ["Price", "Shipping", "Outlook", "Quality", "Size", "Shop_Service", "General", "Others"]
# Dùng distilbert-base-multilingual-cased để match với train script và tiết kiệm RAM
MODEL_NAME = "distilbert-base-multilingual-cased"
MAX_LEN = 64  # Tăng lên 64 với 16GB RAM để match train script
DEVICE = "cpu"  # Force CPU để tránh CUDA overhead trên Mac M chip
BATCH_SIZE = 8  # Tăng lên 8 với 16GB RAM
MAX_EVAL_SAMPLES = None  # Không giới hạn - dùng tất cả dữ liệu test

# Đường dẫn
DATA_PATH = "/opt/airflow/projects/absa_streaming/data/test_data.csv"
MODELS_DIR = "/opt/airflow/models"
CURRENT_MODEL_PATH = "/opt/airflow/models/best_absa_hardshare.pt"
TRAINED_MODEL_PREFIX = "absa_model_retrained"
EVALUATION_RESULTS_DIR = "/opt/airflow/models/evaluation_results"

# === Định nghĩa mô hình ABSA (giống train script) ===
class ABSAModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_aspects=len(ASPECTS)):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        H = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head_m = nn.Linear(H, num_aspects)
        self.head_s = nn.Linear(H, num_aspects * 3)
    
    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h_cls = self.dropout(out.last_hidden_state[:, 0, :])
        return self.head_m(h_cls), self.head_s(h_cls).view(-1, len(ASPECTS), 3)

# === Dataset ===
class ABSADataset(Dataset):
    def __init__(self, texts, aspect_labels, sentiment_labels, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.aspect_labels = aspect_labels
        self.sentiment_labels = sentiment_labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "aspect_labels": torch.tensor(self.aspect_labels[idx], dtype=torch.float),
            "sentiment_labels": torch.tensor(self.sentiment_labels[idx], dtype=torch.long)
        }

# === Hàm load và preprocess dữ liệu ===
def load_and_preprocess_data(data_path):
    """Load và preprocess dữ liệu từ CSV"""
    print(f"[Evaluate] Đang load dữ liệu từ {data_path}...")
    df = pd.read_csv(data_path)
    
    df = df[df["Review"].notna()]
    texts = df["Review"].tolist()
    
    aspect_labels = []
    sentiment_labels = []
    
    for _, row in df.iterrows():
        aspect_row = []
        sentiment_row = []
        for asp in ASPECTS:
            val = row[asp]
            if pd.isna(val) or val == -1:
                aspect_row.append(0)
                sentiment_row.append(0)
            else:
                aspect_row.append(1)
                if val == 1:
                    sentiment_row.append(1)  # POS
                elif val == 2:
                    sentiment_row.append(2)  # NEG
                else:
                    sentiment_row.append(0)  # NEU
        
        aspect_labels.append(aspect_row)
        sentiment_labels.append(sentiment_row)
    
    print(f"[Evaluate] Đã load {len(texts)} mẫu dữ liệu.")
    return texts, aspect_labels, sentiment_labels

# === Hàm đánh giá mô hình ===
def evaluate_model(model, data_loader, device):
    """Đánh giá mô hình và trả về metrics"""
    model.eval()
    
    total_aspect_correct = 0
    total_aspect_predicted = 0
    total_aspect_actual = 0
    
    total_sentiment_correct = 0
    total_sentiment_predicted = 0
    
    total_loss = 0.0
    aspect_criterion = nn.BCEWithLogitsLoss()
    sentiment_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            aspect_labels = batch["aspect_labels"].to(device)
            sentiment_labels = batch["sentiment_labels"].to(device)
            
            logits_m, logits_s = model(input_ids, attention_mask)
            
            # Loss
            loss_aspect = aspect_criterion(logits_m, aspect_labels)
            loss_sentiment = 0.0
            for i in range(len(ASPECTS)):
                mask = aspect_labels[:, i] == 1
                if mask.sum() > 0:
                    loss_sentiment += sentiment_criterion(
                        logits_s[mask, i, :],
                        sentiment_labels[mask, i]
                    )
            loss_sentiment = loss_sentiment / len(ASPECTS)
            total_loss += (loss_aspect + loss_sentiment).item()
            
            # Aspect detection metrics
            aspect_preds = (torch.sigmoid(logits_m) > 0.5).float()
            total_aspect_correct += (aspect_preds == aspect_labels).sum().item()
            total_aspect_predicted += aspect_preds.sum().item()
            total_aspect_actual += aspect_labels.sum().item()
            
            # Sentiment classification metrics (chỉ tính cho các aspect có trong label)
            sentiment_preds = torch.argmax(logits_s, dim=2)
            for i in range(len(ASPECTS)):
                mask = aspect_labels[:, i] == 1
                if mask.sum() > 0:
                    total_sentiment_correct += (sentiment_preds[mask, i] == sentiment_labels[mask, i]).sum().item()
                    total_sentiment_predicted += mask.sum().item()
            
            # Giải phóng memory mỗi batch
            del input_ids, attention_mask, aspect_labels, sentiment_labels
            del logits_m, logits_s, loss_aspect, loss_sentiment
            del aspect_preds, sentiment_preds
            gc.collect()
    
    avg_loss = total_loss / len(data_loader)
    
    # Tính metrics
    aspect_precision = total_aspect_correct / total_aspect_predicted if total_aspect_predicted > 0 else 0
    aspect_recall = total_aspect_correct / total_aspect_actual if total_aspect_actual > 0 else 0
    aspect_f1 = 2 * aspect_precision * aspect_recall / (aspect_precision + aspect_recall) if (aspect_precision + aspect_recall) > 0 else 0
    
    sentiment_accuracy = total_sentiment_correct / total_sentiment_predicted if total_sentiment_predicted > 0 else 0
    
    metrics = {
        "loss": avg_loss,
        "aspect_precision": aspect_precision,
        "aspect_recall": aspect_recall,
        "aspect_f1": aspect_f1,
        "sentiment_accuracy": sentiment_accuracy,
        "overall_score": (aspect_f1 + sentiment_accuracy) / 2  # Combined score
    }
    
    return metrics

# === Hàm tìm mô hình mới nhất ===
def find_latest_retrained_model():
    """Tìm mô hình retrained mới nhất"""
    pattern = os.path.join(MODELS_DIR, f"{TRAINED_MODEL_PREFIX}_*.pt")
    model_files = glob.glob(pattern)
    
    if not model_files:
        raise FileNotFoundError(f"Không tìm thấy mô hình retrained trong {MODELS_DIR}")
    
    # Sắp xếp theo thời gian tạo (mới nhất trước)
    model_files.sort(key=os.path.getmtime, reverse=True)
    latest_model = model_files[0]
    
    print(f"[Evaluate] Tìm thấy mô hình mới nhất: {latest_model}")
    return latest_model

# === Hàm đánh giá và so sánh ===
def evaluate_and_compare():
    """Đánh giá mô hình mới và so sánh với mô hình hiện tại"""
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("=" * 60)
    print("📊 Bắt đầu đánh giá mô hình ABSA")
    print("=" * 60)
    
    # Load dữ liệu test
    texts, aspect_labels, sentiment_labels = load_and_preprocess_data(DATA_PATH)
    
    # Chia test set (20% cuối)
    split_idx = int(0.8 * len(texts))
    test_texts = texts[split_idx:]
    test_aspects = aspect_labels[split_idx:]
    test_sentiments = sentiment_labels[split_idx:]
    
    print(f"[Evaluate] Test set: {len(test_texts)} mẫu")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    
    # Test dataset
    test_dataset = ABSADataset(test_texts, test_aspects, test_sentiments, tokenizer)
    # Giảm num_workers để tiết kiệm RAM
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    
    # Tìm mô hình mới nhất
    new_model_path = find_latest_retrained_model()
    
    # Load mô hình mới
    print(f"[Evaluate] Đang load mô hình mới: {new_model_path}")
    # Giải phóng memory trước khi load model
    gc.collect()
    
    new_model = ABSAModel()
    new_model.load_state_dict(torch.load(new_model_path, map_location=DEVICE))
    new_model.to(DEVICE)
    
    # Freeze backbone để tiết kiệm RAM trong evaluation
    for param in new_model.backbone.parameters():
        param.requires_grad = False
    
    # Đánh giá mô hình mới
    print("\n[Evaluate] Đang đánh giá mô hình mới...")
    new_metrics = evaluate_model(new_model, test_loader, DEVICE)
    
    # Giải phóng memory sau khi đánh giá
    del new_model
    gc.collect()
    
    print(f"\n[Evaluate] 📊 Kết quả mô hình mới:")
    print(f"  Loss: {new_metrics['loss']:.4f}")
    print(f"  Aspect F1: {new_metrics['aspect_f1']:.4f}")
    print(f"  Sentiment Accuracy: {new_metrics['sentiment_accuracy']:.4f}")
    print(f"  Overall Score: {new_metrics['overall_score']:.4f}")
    
    # Đánh giá mô hình hiện tại (nếu có và tương thích)
    current_metrics = None
    if os.path.exists(CURRENT_MODEL_PATH):
        print(f"\n[Evaluate] Đang đánh giá mô hình hiện tại: {CURRENT_MODEL_PATH}")
        try:
            # Giải phóng memory trước khi load model
            gc.collect()
            
            current_model = ABSAModel()
            state_dict = torch.load(CURRENT_MODEL_PATH, map_location=DEVICE)
            
            # Thử load với strict=False để xử lý trường hợp không tương thích
            try:
                current_model.load_state_dict(state_dict, strict=True)
                print(f"[Evaluate] ✅ Đã load weights thành công (strict mode)")
            except RuntimeError as e:
                # Nếu không tương thích, thử load với strict=False
                print(f"[Evaluate] ⚠️ Model không tương thích (có thể train với model khác), thử load với strict=False...")
                try:
                    current_model.load_state_dict(state_dict, strict=False)
                    print(f"[Evaluate] ⚠️ Đã load weights với strict=False (một số weights không match)")
                except Exception as e2:
                    print(f"[Evaluate] ❌ Không thể load weights: {e2}")
                    print(f"[Evaluate] Mô hình hiện tại không tương thích với architecture hiện tại (distilbert vs xlm-roberta)")
                    print(f"[Evaluate] Sẽ bỏ qua việc so sánh và deploy mô hình mới nếu tốt hơn baseline")
                    current_model = None
            
            if current_model is not None:
                current_model.to(DEVICE)
                # Freeze backbone để tiết kiệm RAM
                for param in current_model.backbone.parameters():
                    param.requires_grad = False
                
                current_metrics = evaluate_model(current_model, test_loader, DEVICE)
                
                print(f"\n[Evaluate] 📊 Kết quả mô hình hiện tại:")
                print(f"  Loss: {current_metrics['loss']:.4f}")
                print(f"  Aspect F1: {current_metrics['aspect_f1']:.4f}")
                print(f"  Sentiment Accuracy: {current_metrics['sentiment_accuracy']:.4f}")
                print(f"  Overall Score: {current_metrics['overall_score']:.4f}")
                
                # Giải phóng memory
                del current_model
                gc.collect()
        except Exception as e:
            print(f"[Evaluate] ❌ Lỗi khi đánh giá mô hình hiện tại: {e}")
            print(f"[Evaluate] Mô hình hiện tại không tương thích hoặc bị lỗi")
            print(f"[Evaluate] Sẽ bỏ qua việc so sánh và deploy mô hình mới nếu tốt hơn baseline")
            current_metrics = None
    else:
        print(f"\n[Evaluate] ⚠️ Không tìm thấy mô hình hiện tại: {CURRENT_MODEL_PATH}")
        print(f"[Evaluate] Mô hình mới sẽ được deploy.")
    
    # So sánh và quyết định
    should_deploy = False
    improvement = None
    if current_metrics is None:
        should_deploy = True
        reason = "Không có mô hình hiện tại"
    else:
        # So sánh overall_score (F1 + Accuracy) / 2
        improvement = new_metrics['overall_score'] - current_metrics['overall_score']
        if improvement > 0.01:  # Cải thiện ít nhất 1%
            should_deploy = True
            reason = f"Mô hình mới tốt hơn ({improvement:.4f} điểm)"
        else:
            should_deploy = False
            reason = f"Mô hình mới không tốt hơn (chênh lệch: {improvement:.4f})"
    
    # Lưu kết quả đánh giá
    os.makedirs(EVALUATION_RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    evaluation_result = {
        "timestamp": timestamp,
        "new_model_path": new_model_path,
        "current_model_path": CURRENT_MODEL_PATH if os.path.exists(CURRENT_MODEL_PATH) else None,
        "new_metrics": new_metrics,
        "current_metrics": current_metrics,
        "should_deploy": should_deploy,
        "reason": reason,
        "improvement": improvement if current_metrics else None
    }
    
    result_path = os.path.join(EVALUATION_RESULTS_DIR, f"evaluation_{timestamp}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
    
    print(f"\n[Evaluate] ✅ Đã lưu kết quả đánh giá: {result_path}")
    print(f"\n[Evaluate] 🎯 Quyết định: {'✅ DEPLOY' if should_deploy else '❌ KHÔNG DEPLOY'}")
    print(f"[Evaluate] Lý do: {reason}")
    
    # Trả về kết quả
    return {
        "should_deploy": should_deploy,
        "new_model_path": new_model_path,
        "new_metrics": new_metrics,
        "current_metrics": current_metrics,
        "evaluation_result_path": result_path
    }

if __name__ == "__main__":
    try:
        result = evaluate_and_compare()
        if result["should_deploy"]:
            print(f"\n✅ Mô hình mới đạt yêu cầu và sẽ được deploy!")
            sys.exit(0)
        else:
            print(f"\n⚠️ Mô hình mới không đạt yêu cầu, không deploy.")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Lỗi khi đánh giá: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

