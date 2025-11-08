# SE363 – Phát triển ứng dụng trên nền tảng dữ liệu lớn
# Khoa Công nghệ Phần mềm – Trường Đại học Công nghệ Thông tin, ĐHQG-HCM
# HopDT – Faculty of Software Engineering, University of Information Technology (FSE-UIT)

# absa_model_retrain_dag.py
# ======================================
# DAG: ABSA Model Retraining Pipeline
# Pipeline tự động huấn luyện, đánh giá và deploy mô hình ABSA định kỳ
# Các task được tách biệt: retrain → evaluate → deploy

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import os
import sys

# === Default parameters ===
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,                          # Thử lại tối đa 1 lần nếu lỗi
    "retry_delay": timedelta(minutes=5),   # Mỗi lần retry cách nhau 5 phút
}

# === DAG definition ===
with DAG(
        dag_id="absa_model_retrain",
        default_args=default_args,
        description="Pipeline tự động retrain, evaluate và deploy mô hình ABSA định kỳ",
        schedule_interval=timedelta(days=1),            # Chạy mỗi ngày
        start_date=days_ago(1),
        catchup=False,
        dagrun_timeout=timedelta(hours=2),            # Giới hạn thời gian chạy DAG
        tags=["absa", "retrain", "ml", "model"],
) as dag:

    # === 1️⃣ Task: Retrain Model ===
    # Huấn luyện mô hình ABSA mới từ dữ liệu training
    retrain_model = BashOperator(
        task_id="retrain_model",
        bash_command=(
            "cd /opt/airflow && "
            "python /opt/airflow/projects/absa_streaming/scripts/train_absa_model.py"
        ),
        retries=1,
        retry_delay=timedelta(minutes=5),
        execution_timeout=timedelta(hours=1),         # Timeout 1 giờ cho training
        trigger_rule="all_success",
    )

    # === 2️⃣ Task: Evaluate Model ===
    # Đánh giá mô hình mới và so sánh với mô hình hiện tại
    # Chỉ deploy nếu mô hình mới tốt hơn
    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command=(
            "cd /opt/airflow && "
            "python /opt/airflow/projects/absa_streaming/scripts/evaluate_absa_model.py"
        ),
        retries=1,
        retry_delay=timedelta(minutes=2),
        execution_timeout=timedelta(minutes=30),      # Timeout 30 phút cho evaluation
        trigger_rule="all_success",
    )

    # === 3️⃣ Task: Deploy Model ===
    # Deploy mô hình mới nếu đã được đánh giá là tốt hơn
    # Backup mô hình cũ trước khi deploy
    deploy_model = BashOperator(
        task_id="deploy_model",
        bash_command=(
            "cd /opt/airflow && "
            "python /opt/airflow/projects/absa_streaming/scripts/deploy_absa_model.py"
        ),
        retries=1,
        retry_delay=timedelta(minutes=2),
        execution_timeout=timedelta(minutes=10),      # Timeout 10 phút cho deploy
        trigger_rule="all_success",
    )

    # === 4️⃣ Task: Notify Completion (Optional) ===
    # Thông báo hoàn tất pipeline
    def notify_completion():
        print("=" * 60)
        print("✅ Pipeline retrain mô hình ABSA đã hoàn tất!")
        print("=" * 60)
        print("\n📊 Tóm tắt:")
        print("  1. ✅ Retrain: Đã huấn luyện mô hình mới")
        print("  2. ✅ Evaluate: Đã đánh giá và so sánh mô hình")
        print("  3. ✅ Deploy: Đã deploy mô hình (nếu tốt hơn)")
        print("\n💡 Mô hình mới sẽ được sử dụng trong pipeline streaming ở lần chạy tiếp theo.")
        print("=" * 60)

    notify = PythonOperator(
        task_id="notify_completion",
        python_callable=notify_completion,
        trigger_rule="all_done",  # Chạy dù có task nào fail
    )

    # === Task dependencies ===
    # Pipeline: retrain → evaluate → deploy → notify
    retrain_model >> evaluate_model >> deploy_model >> notify

