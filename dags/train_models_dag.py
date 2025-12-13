"""
Train Models DAG
================
Runs model training pipeline:
1. Train completion probability model
2. Train YAC EPA prediction model

Manual trigger only - no scheduling.
"""

from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,  # No retries - if training fails, fix and re-run manually
    'retry_delay': timedelta(minutes=5),
}

# Define DAG
dag = DAG(
    'train_models',
    default_args=default_args,
    description='Train completion and YAC EPA models',
    schedule_interval=None,  # Manual trigger only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'training', 'nfl'],
)

# Task 1: Train Completion Model
train_completion = BashOperator(
    task_id='train_completion_model',
    bash_command='cd /opt/airflow/project && python scripts/train_completion.py',
    dag=dag,
)

# Task 2: Train YAC EPA Model
train_yac_epa = BashOperator(
    task_id='train_yac_epa_model',
    bash_command='cd /opt/airflow/project && python scripts/train_yac_epa.py',
    dag=dag,
)

# Set task dependencies
train_completion >> train_yac_epa