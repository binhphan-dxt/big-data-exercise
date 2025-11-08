# ===========================================
# DAG: Vehicle Counting Streaming Lifecycle (1-Hour Demo)
# ===========================================
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import os, subprocess

# === Default parameters ===
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,                          # Retry up to 2 times on failure
    "retry_delay": timedelta(minutes=2),   # Wait 2 minutes between retries
}

# === DAG definition ===
with DAG(
        dag_id="vehicle_counting_lifecycle_demo",
        default_args=default_args,
        description="Orchestrate multi-camera vehicle counting with Kafka–Spark–PostgreSQL streaming (1-Hour Demo)",
        schedule_interval=timedelta(hours=1),            # Run every hour
        start_date=days_ago(1),
        catchup=False,
        dagrun_timeout=timedelta(minutes=55),            # DAG run timeout (< 1h)
        tags=["vehicle-counting", "streaming", "kafka", "spark", "yolo"],
) as dag:

    # === 1️⃣ Deploy Video Producer (Camera 1 & 2) ===
    deploy_video_producer = BashOperator(
        task_id="deploy_video_producer",
        bash_command='bash -c "timeout 45m /opt/airflow/projects/vehicle_counting/scripts/run_producer.sh"',
        retries=3,
        retry_delay=timedelta(minutes=2),
        execution_timeout=timedelta(minutes=50),         # Task-level timeout
        trigger_rule="all_done",
    )

    # === 2️⃣ Deploy Vehicle Consumer (Spark Streaming) ===
    deploy_vehicle_consumer = BashOperator(
        task_id="deploy_vehicle_consumer",
        bash_command='bash -c "timeout 45m /opt/airflow/projects/vehicle_counting/scripts/run_consumer.sh"',
        retries=5,
        retry_delay=timedelta(minutes=2),
        execution_timeout=timedelta(minutes=50),         # Task-level timeout
        trigger_rule="all_done",
    )

    # === 3️⃣ Monitor checkpoint ===
    def monitor_job():
        print("[Monitor] Checking vehicle counting job checkpoint...")
        path = "/opt/airflow/checkpoints/vehicle_counting_checkpoint"
        if os.path.exists(path):
            size = subprocess.check_output(["du", "-sh", path]).decode().split()[0]
            print(f"[Monitor] Checkpoint exists ({size}) → job running normally.")
        else:
            print("[Monitor] No checkpoint found. Possibly failed or cleaned.")

    monitor_stream = PythonOperator(
        task_id="monitor_stream",
        python_callable=monitor_job,
        trigger_rule="all_done",
    )

    # === 4️⃣ Cleanup checkpoint ===
    cleanup_checkpoints = BashOperator(
        task_id="cleanup_checkpoints",
        bash_command=(
            "echo '[Cleanup] Removing old checkpoint...'; "
            "rm -rf /opt/airflow/checkpoints/vehicle_counting_checkpoint || true; "
            "echo '[Cleanup] Done.'"
        ),
        trigger_rule="all_done",
    )

    # === Task dependency ===
    [deploy_video_producer, deploy_vehicle_consumer] >> monitor_stream >> cleanup_checkpoints
