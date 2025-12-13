from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess

def run_viz_script():
    """Execute the visualization script"""
    result = subprocess.run(
        ['python', '/opt/airflow/scripts/viz_play_test2.py'],
        capture_output=True,
        text=True,
        cwd='/opt/airflow'
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    if result.returncode != 0:
        raise Exception(f"Script failed with return code {result.returncode}")

with DAG(
    'viz_play_oneoff',
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['visualization', 'oneoff']
) as dag:
    
    run_viz = PythonOperator(
        task_id='run_visualization',
        python_callable=run_viz_script
    )