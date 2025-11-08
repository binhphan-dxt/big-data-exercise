from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id='demo_airflow_basic',
    description='DAG minh họa đầu tiên',
    start_date=datetime(2024, 1, 1),
    schedule_interval=timedelta(minutes=1),
    catchup=False
) as dag:

    task1 = BashOperator(
        task_id='print_hello',
        bash_command='echo "Xin chào Airflow đang chạy trong Docker!"'
    )

    task2 = BashOperator(
        task_id='print_time',
        bash_command='date'
    )

    task1 >> task2