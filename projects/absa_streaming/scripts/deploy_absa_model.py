# SE363 – Phát triển ứng dụng trên nền tảng dữ liệu lớn
# Khoa Công nghệ Phần mềm – Trường Đại học Công nghệ Thông tin, ĐHQG-HCM
# HopDT – Faculty of Software Engineering, University of Information Technology (FSE-UIT)

# deploy_absa_model.py
# ======================================
# Script deploy mô hình ABSA mới nếu đã được đánh giá là tốt hơn
# Backup mô hình cũ và thay thế bằng mô hình mới

import os
import sys
import json
import shutil
import glob
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Tăng threads với 16GB Docker RAM
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# === Cấu hình ===
MODELS_DIR = "/opt/airflow/models"
CURRENT_MODEL_PATH = "/opt/airflow/models/best_absa_hardshare.pt"
TRAINED_MODEL_PREFIX = "absa_model_retrained"
EVALUATION_RESULTS_DIR = "/opt/airflow/models/evaluation_results"
BACKUP_DIR = "/opt/airflow/models/backups"

# === Hàm tìm mô hình mới nhất ===
def find_latest_retrained_model():
    """Tìm mô hình retrained mới nhất"""
    pattern = os.path.join(MODELS_DIR, f"{TRAINED_MODEL_PREFIX}_*.pt")
    model_files = glob.glob(pattern)
    
    if not model_files:
        raise FileNotFoundError(f"Không tìm thấy mô hình retrained trong {MODELS_DIR}")
    
    model_files.sort(key=os.path.getmtime, reverse=True)
    latest_model = model_files[0]
    
    print(f"[Deploy] Tìm thấy mô hình mới nhất: {latest_model}")
    return latest_model

# === Hàm tìm kết quả đánh giá mới nhất ===
def find_latest_evaluation_result():
    """Tìm kết quả đánh giá mới nhất"""
    pattern = os.path.join(EVALUATION_RESULTS_DIR, "evaluation_*.json")
    result_files = glob.glob(pattern)
    
    if not result_files:
        raise FileNotFoundError(f"Không tìm thấy kết quả đánh giá trong {EVALUATION_RESULTS_DIR}")
    
    result_files.sort(key=os.path.getmtime, reverse=True)
    latest_result = result_files[0]
    
    print(f"[Deploy] Tìm thấy kết quả đánh giá mới nhất: {latest_result}")
    return latest_result

# === Hàm backup mô hình cũ ===
def backup_current_model():
    """Backup mô hình hiện tại"""
    if not os.path.exists(CURRENT_MODEL_PATH):
        print(f"[Deploy] ⚠️ Không tìm thấy mô hình hiện tại: {CURRENT_MODEL_PATH}")
        return None
    
    os.makedirs(BACKUP_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"best_absa_hardshare_backup_{timestamp}.pt"
    backup_path = os.path.join(BACKUP_DIR, backup_filename)
    
    print(f"[Deploy] Đang backup mô hình cũ: {CURRENT_MODEL_PATH} → {backup_path}")
    shutil.copy2(CURRENT_MODEL_PATH, backup_path)
    
    print(f"[Deploy] ✅ Đã backup mô hình cũ: {backup_path}")
    return backup_path

# === Hàm deploy mô hình mới ===
def deploy_model():
    """Deploy mô hình mới nếu đã được đánh giá là tốt hơn"""
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("=" * 60)
    print("🚀 Bắt đầu deploy mô hình ABSA")
    print("=" * 60)
    
    # Kiểm tra kết quả đánh giá
    try:
        eval_result_path = find_latest_evaluation_result()
        with open(eval_result_path, "r", encoding="utf-8") as f:
            eval_result = json.load(f)
        
        should_deploy = eval_result.get("should_deploy", False)
        reason = eval_result.get("reason", "Không rõ lý do")
        
        print(f"\n[Deploy] Kết quả đánh giá:")
        print(f"  Should Deploy: {should_deploy}")
        print(f"  Lý do: {reason}")
        
        if not should_deploy:
            print(f"\n[Deploy] ❌ Mô hình mới không đạt yêu cầu, không deploy.")
            print(f"[Deploy] Lý do: {reason}")
            return False
        
    except FileNotFoundError as e:
        print(f"\n[Deploy] ⚠️ Không tìm thấy kết quả đánh giá: {e}")
        print(f"[Deploy] ⚠️ Bỏ qua bước deploy.")
        return False
    
    # Tìm mô hình mới nhất
    try:
        new_model_path = find_latest_retrained_model()
    except FileNotFoundError as e:
        print(f"\n[Deploy] ❌ Lỗi: {e}")
        return False
    
    # Backup mô hình cũ (nếu có)
    backup_path = backup_current_model()
    
    # Deploy mô hình mới
    print(f"\n[Deploy] Đang deploy mô hình mới: {new_model_path} → {CURRENT_MODEL_PATH}")
    
    try:
        # Xóa file cũ trước (nếu có) để tránh lỗi permission
        if os.path.exists(CURRENT_MODEL_PATH):
            try:
                # Thử xóa file cũ
                os.remove(CURRENT_MODEL_PATH)
                print(f"[Deploy] ✅ Đã xóa file cũ: {CURRENT_MODEL_PATH}")
            except PermissionError as pe:
                print(f"[Deploy] ⚠️ Không thể xóa file cũ (có thể đang được sử dụng): {pe}")
                # Thử rename thay vì xóa
                old_backup_path = CURRENT_MODEL_PATH + ".old"
                try:
                    os.rename(CURRENT_MODEL_PATH, old_backup_path)
                    print(f"[Deploy] ✅ Đã rename file cũ: {CURRENT_MODEL_PATH} → {old_backup_path}")
                except Exception as re:
                    print(f"[Deploy] ⚠️ Không thể rename file cũ: {re}")
                    # Tiếp tục thử copy (có thể ghi đè được)
            except Exception as e:
                print(f"[Deploy] ⚠️ Lỗi khi xóa file cũ: {e}")
        
        # Copy mô hình mới vào vị trí production
        # Dùng atomic operation: copy vào file tạm rồi rename
        temp_path = CURRENT_MODEL_PATH + ".tmp"
        try:
            shutil.copy2(new_model_path, temp_path)
            # Atomic rename
            os.rename(temp_path, CURRENT_MODEL_PATH)
            print(f"[Deploy] ✅ Đã deploy mô hình mới thành công!")
        except Exception as copy_error:
            # Nếu atomic operation fail, thử copy trực tiếp
            print(f"[Deploy] ⚠️ Atomic operation failed, thử copy trực tiếp: {copy_error}")
            try:
                shutil.copy2(new_model_path, CURRENT_MODEL_PATH)
                print(f"[Deploy] ✅ Đã deploy mô hình mới thành công!")
            except Exception as direct_copy_error:
                raise direct_copy_error
            finally:
                # Xóa file temp nếu còn
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
        
        print(f"[Deploy]   Model mới: {new_model_path}")
        print(f"[Deploy]   Production path: {CURRENT_MODEL_PATH}")
        if backup_path:
            print(f"[Deploy]   Backup mô hình cũ: {backup_path}")
        
        # Lưu thông tin deploy
        deploy_info = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "new_model_path": new_model_path,
            "production_path": CURRENT_MODEL_PATH,
            "backup_path": backup_path,
            "evaluation_result_path": eval_result_path,
            "reason": reason,
            "metrics": eval_result.get("new_metrics", {})
        }
        
        deploy_info_path = os.path.join(MODELS_DIR, "deploy_info.json")
        with open(deploy_info_path, "w", encoding="utf-8") as f:
            json.dump(deploy_info, f, ensure_ascii=False, indent=2)
        
        print(f"[Deploy] ✅ Đã lưu thông tin deploy: {deploy_info_path}")
        
        return True
        
    except Exception as e:
        print(f"\n[Deploy] ❌ Lỗi khi deploy mô hình: {str(e)}")
        
        # Khôi phục mô hình cũ nếu có backup
        if backup_path and os.path.exists(backup_path):
            print(f"[Deploy] Đang khôi phục mô hình cũ từ backup...")
            try:
                # Xóa file hiện tại trước (nếu có)
                if os.path.exists(CURRENT_MODEL_PATH):
                    try:
                        os.remove(CURRENT_MODEL_PATH)
                    except:
                        pass
                
                # Copy từ backup
                shutil.copy2(backup_path, CURRENT_MODEL_PATH)
                print(f"[Deploy] ✅ Đã khôi phục mô hình cũ.")
            except Exception as restore_error:
                print(f"[Deploy] ❌ Lỗi khi khôi phục: {str(restore_error)}")
                print(f"[Deploy] ⚠️ Có thể cần khôi phục thủ công từ: {backup_path}")
        
        return False

if __name__ == "__main__":
    try:
        success = deploy_model()
        if success:
            print(f"\n✅ Deploy hoàn tất!")
            sys.exit(0)
        else:
            print(f"\n⚠️ Deploy không thành công hoặc không cần deploy.")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Lỗi khi deploy: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

