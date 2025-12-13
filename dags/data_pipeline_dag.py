"""
Data Pipeline DAG
=================
Runs data processing pipeline:
1. Dataframe A: Node-level player features
2. Dataframe B: Play-level context and outcomes
3. Dataframe C: Edge-level pairwise features (v2 - structured edges)
4. Dataframe D: Temporal frame-level aggregates

Manual trigger only - no scheduling.
Each task can be run independently or as part of the full pipeline.
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
    'retries': 0,  # No retries - if processing fails, fix and re-run manually
    'retry_delay': timedelta(minutes=5),
}

# Define DAG
dag = DAG(
    'data_pipeline',
    default_args=default_args,
    description='Process NFL tracking data into graph-structured dataframes',
    schedule_interval=None,  # Manual trigger only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['data', 'processing', 'nfl'],
)

# Task 1: Create Dataframe A (Node-level features)
create_dataframe_a = BashOperator(
    task_id='create_dataframe_a',
    bash_command='cd /opt/airflow/project && python scripts/dataframe_a.py',
    dag=dag,
)

# Task 2: Create Dataframe B (Play-level features)
create_dataframe_b = BashOperator(
    task_id='create_dataframe_b',
    bash_command='cd /opt/airflow/project && python scripts/dataframe_b.py',
    dag=dag,
)

# Task 3: Create Dataframe C v2 (Edge-level features - structured)
create_dataframe_c = BashOperator(
    task_id='create_dataframe_c_v2',
    bash_command='cd /opt/airflow/project && python scripts/dataframe_c_v2.py',
    dag=dag,
)

# Task 4: Create Dataframe D (Temporal aggregates)
create_dataframe_d = BashOperator(
    task_id='create_dataframe_d',
    bash_command='cd /opt/airflow/project && python scripts/dataframe_d.py',
    dag=dag,
)

# Set task dependencies
# A and B can run in parallel (both read raw data)
# C requires both A and B (needs node features + play context)
# D requires A (needs node data for aggregation)

create_dataframe_a >> create_dataframe_c
create_dataframe_b >> create_dataframe_c
create_dataframe_a >> create_dataframe_d

# Pipeline structure:
#     A -----> C
#    / \
#   /   \
#  B --> (C)
#   \
#    \-> D