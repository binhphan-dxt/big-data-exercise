# SE363 – Phát triển ứng dụng trên nền tảng dữ liệu lớn
# Khoa Công nghệ Phần mềm – Trường Đại học Công nghệ Thông tin, ĐHQG-HCM
# HopDT – Faculty of Software Engineering, University of Information Technology (FSE-UIT)

# train_absa_model.py
# ======================================
# Script huấn luyện mô hình ABSA mới từ dữ liệu training
# Lưu mô hình mới vào thư mục models với timestamp

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import os
import sys
import json
import gc  # Garbage collection để giải phóng memory
from datetime import datetime
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
# Dùng distilbert-base-multilingual-cased thay vì xlm-roberta-base để tiết kiệm RAM (~500MB vs ~1GB)
MODEL_NAME = "distilbert-base-multilingual-cased"
MAX_LEN = 64  # Tăng lên 64 với 16GB RAM
DEVICE = "cpu"  # Force CPU để tránh CUDA overhead trên Mac M chip
BATCH_SIZE = 8  # Tăng lên 8 với 16GB RAM
EPOCHS = 3  # Tăng lên 3 epochs để model tốt hơn
LEARNING_RATE = 2e-5
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = 32
MAX_TRAIN_SAMPLES = None  # Không giới hạn - dùng tất cả dữ liệu
MAX_VAL_SAMPLES = None  # Không giới hạn - dùng tất cả dữ liệu

# Đường dẫn
DATA_PATH = "/opt/airflow/projects/absa_streaming/data/test_data.csv"
MODELS_DIR = "/opt/airflow/models"
TRAINED_MODEL_PREFIX = "absa_model_retrained"

# === Định nghĩa mô hình ABSA ===
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
    print(f"[Train] Đang load dữ liệu từ {data_path}...")
    df = pd.read_csv(data_path)
    
    # Lọc các dòng có review hợp lệ
    df = df[df["Review"].notna()]
    texts = df["Review"].tolist()
    
    # Chuyển đổi labels: -1 -> 0 (không có aspect), 0->0 (NEU), 1->1 (POS), 2->2 (NEG)
    # Aspect labels: binary (0 hoặc 1) - có aspect hay không
    # Sentiment labels: 0 (NEU), 1 (POS), 2 (NEG) cho mỗi aspect
    aspect_labels = []
    sentiment_labels = []
    
    for _, row in df.iterrows():
        aspect_row = []
        sentiment_row = []
        for asp in ASPECTS:
            val = row[asp]
            if pd.isna(val) or val == -1:
                aspect_row.append(0)  # Không có aspect
                sentiment_row.append(0)  # NEU (default)
            else:
                aspect_row.append(1)  # Có aspect
                if val == 1:
                    sentiment_row.append(1)  # POS
                elif val == 2:
                    sentiment_row.append(2)  # NEG
                else:
                    sentiment_row.append(0)  # NEU
        
        aspect_labels.append(aspect_row)
        sentiment_labels.append(sentiment_row)
    
    print(f"[Train] Đã load {len(texts)} mẫu dữ liệu.")
    return texts, aspect_labels, sentiment_labels

# === Hàm train ===
def train_model():
    """Huấn luyện mô hình ABSA"""
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("=" * 60)
    print("🚀 Bắt đầu huấn luyện mô hình ABSA")
    print("=" * 60)
    
    # Load dữ liệu
    texts, aspect_labels, sentiment_labels = load_and_preprocess_data(DATA_PATH)
    
    # Chia train/val (80/20)
    split_idx = int(0.8 * len(texts))
    train_texts = texts[:split_idx]
    train_aspects = aspect_labels[:split_idx]
    train_sentiments = sentiment_labels[:split_idx]
    
    val_texts = texts[split_idx:]
    val_aspects = aspect_labels[split_idx:]
    val_sentiments = sentiment_labels[split_idx:]
    
    print(f"[Train] Train: {len(train_texts)} mẫu, Val: {len(val_texts)} mẫu")
    
    # Tokenizer - không cache để tiết kiệm RAM
    print("[Train] Đang load tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, local_files_only=False)
    
    # Dataset và DataLoader
    train_dataset = ABSADataset(train_texts, train_aspects, train_sentiments, tokenizer)
    val_dataset = ABSADataset(val_texts, val_aspects, val_sentiments, tokenizer)
    
    # Giảm num_workers để tiết kiệm RAM (0 = main process)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    
    # Model - tối ưu memory
    print("[Train] Đang khởi tạo mô hình...")
    # Giải phóng memory trước khi load model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    model = ABSAModel()
    model.to(DEVICE)
    
    # Tắt gradient cho backbone để tiết kiệm RAM (chỉ train classification head)
    print("[Train] Freezing backbone để tiết kiệm RAM...")
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Loss và Optimizer
    aspect_criterion = nn.BCEWithLogitsLoss()
    sentiment_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_val_loss = float('inf')
    training_history = []
    
    for epoch in range(EPOCHS):
        print(f"\n[Epoch {epoch+1}/{EPOCHS}]")
        
        # Train phase
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()  # Zero grad ở đầu epoch
        batch_idx = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            aspect_labels = batch["aspect_labels"].to(DEVICE)
            sentiment_labels = batch["sentiment_labels"].to(DEVICE)
            
            logits_m, logits_s = model(input_ids, attention_mask)
            
            # Loss cho aspect detection
            loss_aspect = aspect_criterion(logits_m, aspect_labels)
            
            # Loss cho sentiment classification
            loss_sentiment = 0.0
            for i in range(len(ASPECTS)):
                mask = aspect_labels[:, i] == 1
                if mask.sum() > 0:
                    loss_sentiment += sentiment_criterion(
                        logits_s[mask, i, :],
                        sentiment_labels[mask, i]
                    )
            loss_sentiment = loss_sentiment / len(ASPECTS)
            
            total_loss = loss_aspect + loss_sentiment
            # Scale loss cho gradient accumulation
            total_loss = total_loss / GRADIENT_ACCUMULATION_STEPS
            total_loss.backward()
            
            train_loss += total_loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            # Gradient accumulation: chỉ update weights sau mỗi N steps
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                # Giải phóng memory ngay lập tức
                del input_ids, attention_mask, aspect_labels, sentiment_labels
                del logits_m, logits_s, loss_aspect, loss_sentiment, total_loss
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                else:
                    gc.collect()  # Force garbage collection cho CPU
        
        # Update weights cho batch cuối nếu chưa update
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                aspect_labels = batch["aspect_labels"].to(DEVICE)
                sentiment_labels = batch["sentiment_labels"].to(DEVICE)
                
                logits_m, logits_s = model(input_ids, attention_mask)
                
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
                
                total_loss = loss_aspect + loss_sentiment
                val_loss += total_loss.item()
                
                # Giải phóng memory
                del input_ids, attention_mask, aspect_labels, sentiment_labels
                del logits_m, logits_s, loss_aspect, loss_sentiment, total_loss
                torch.cuda.empty_cache() if DEVICE == "cuda" else None
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        training_history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })
        
        # Lưu best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  ✅ Best model (Val Loss: {best_val_loss:.4f})")
    
    # Lưu mô hình với timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{TRAINED_MODEL_PREFIX}_{timestamp}.pt"
    model_path = os.path.join(MODELS_DIR, model_filename)
    
    # Đảm bảo thư mục tồn tại
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Lưu state dict
    torch.save(model.state_dict(), model_path)
    print(f"\n[Train] ✅ Đã lưu mô hình: {model_path}")
    
    # Lưu metadata
    metadata = {
        "model_path": model_path,
        "model_filename": model_filename,
        "timestamp": timestamp,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "best_val_loss": best_val_loss,
        "training_history": training_history,
        "device": DEVICE
    }
    
    metadata_path = os.path.join(MODELS_DIR, f"{TRAINED_MODEL_PREFIX}_{timestamp}_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"[Train] ✅ Đã lưu metadata: {metadata_path}")
    print(f"[Train] 📊 Best Validation Loss: {best_val_loss:.4f}")
    
    # Trả về đường dẫn mô hình để task tiếp theo sử dụng
    return model_path, best_val_loss

if __name__ == "__main__":
    try:
        model_path, val_loss = train_model()
        print(f"\n✅ Huấn luyện hoàn tất!")
        print(f"   Model: {model_path}")
        print(f"   Validation Loss: {val_loss:.4f}")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Lỗi khi huấn luyện: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

