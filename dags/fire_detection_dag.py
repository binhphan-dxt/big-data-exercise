from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import os, subprocess

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
        dag_id="fire_detection_lifecycle",
        default_args=default_args,
        description="Real-time fire detection with U-Net segmentation",
        schedule_interval=timedelta(hours=1),
        start_date=days_ago(1),
        catchup=False,
        dagrun_timeout=timedelta(minutes=55),
        tags=["fire-detection", "streaming", "unet"],
) as dag:

    deploy_fire_producer = BashOperator(
        task_id="deploy_fire_producer",
        bash_command='bash -c "timeout 45m /opt/airflow/projects/fire_segmentation/scripts/run_producer.sh"',
        retries=3,
        retry_delay=timedelta(minutes=2),
        execution_timeout=timedelta(minutes=50),
        trigger_rule="all_done",
    )

    deploy_fire_consumer = BashOperator(
        task_id="deploy_fire_consumer",
        bash_command='bash -c "timeout 45m /opt/airflow/projects/fire_segmentation/scripts/run_consumer.sh"',
        retries=5,
        retry_delay=timedelta(minutes=2),
        execution_timeout=timedelta(minutes=50),
        trigger_rule="all_done",
    )

    def monitor_job():
        print("[Monitor] Checking fire detection checkpoint...")
        path = "/opt/airflow/checkpoints/fire_detection_checkpoint"
        if os.path.exists(path):
            size = subprocess.check_output(["du", "-sh", path]).decode().split()[0]
            print(f"[Monitor] Checkpoint exists ({size})")
        else:
            print("[Monitor] No checkpoint found")
    
    monitor_stream = PythonOperator(
        task_id="monitor_stream",
        python_callable=monitor_job,
        trigger_rule="all_done",
    )

    cleanup_checkpoints = BashOperator(
        task_id="cleanup_checkpoints",
        bash_command=(
            "echo '[Cleanup] Removing checkpoint...'; "
            "rm -rf /opt/airflow/checkpoints/fire_detection_checkpoint || true; "
            "echo '[Cleanup] Done.'"
        ),
        trigger_rule="all_done",
    )

    [deploy_fire_producer, deploy_fire_consumer] >> monitor_stream >> cleanup_checkpoints
