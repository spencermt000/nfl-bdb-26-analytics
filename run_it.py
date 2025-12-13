"""
run_it.py - NFL Big Data Bowl Data Pipeline Orchestrator
=========================================================
Orchestrates the data processing pipeline using Apache Airflow.

PIPELINE STAGES:
1. Dataframe A (v2): Node-level features per frame
2. Dataframe B (v3): Play-level features + ball trajectory  
3. Dataframe C (v3): Edge-level features + ball trajectory
4. Dataframe D (v2): Frame-level player counts

USAGE:
    # Start Airflow (Docker Compose)
    docker-compose up -d
    
    # Or run directly with Python
    python run_it.py
    
    # Or trigger via Airflow UI
    http://localhost:8080
    Username: airflow
    Password: airflow

CONFIGURATION:
    Set PILOT_MODE in each script before running.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import os
import sys

# ============================================================================
# Configuration
# ============================================================================

# Default arguments for all tasks
default_args = {
    'owner': 'spencer',
    'depends_on_past': False,
    'email': ['your_email@example.com'],  # Update with your email
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# ============================================================================
# Helper Functions
# ============================================================================

def check_prerequisites():
    """Check that all required input files exist."""
    print("=" * 80)
    print("CHECKING PREREQUISITES")
    print("=" * 80)
    
    required_files = [
        'data/train/2023_input_all.parquet',
        'data/supplementary_data.csv',
        'data/sumer_bdb/sumer_coverages_player_play.parquet',
        'data/sumer_bdb/sumer_coverages_frame.parquet',
    ]
    
    missing_files = []
    for filepath in required_files:
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"✓ Found: {filepath} ({size_mb:.1f} MB)")
        else:
            print(f"✗ Missing: {filepath}")
            missing_files.append(filepath)
    
    if missing_files:
        raise FileNotFoundError(
            f"Missing {len(missing_files)} required files. "
            f"Please ensure all data files are in place."
        )
    
    print("\n✓ All prerequisites satisfied!")
    return True

def run_dataframe_a():
    """Execute dataframe_a_v2.py"""
    print("\n" + "=" * 80)
    print("RUNNING DATAFRAME A (v2) - NODE-LEVEL FEATURES")
    print("=" * 80 + "\n")
    
    # Import and run the script
    import subprocess
    result = subprocess.run(
        ['python', 'dataframe_a_v2.py'],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR:", result.stderr)
        raise Exception(f"Dataframe A failed with return code {result.returncode}")
    
    # Verify output
    output_file = 'outputs/dataframe_a/v2.parquet'
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Expected output not found: {output_file}")
    
    print(f"\n✓ Dataframe A complete: {output_file}")
    return output_file

def run_dataframe_b():
    """Execute dataframe_b_v3.py"""
    print("\n" + "=" * 80)
    print("RUNNING DATAFRAME B (v3) - PLAY-LEVEL FEATURES + BALL TRAJECTORY")
    print("=" * 80 + "\n")
    
    import subprocess
    result = subprocess.run(
        ['python', 'dataframe_b_v3.py'],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR:", result.stderr)
        raise Exception(f"Dataframe B failed with return code {result.returncode}")
    
    # Check for both pilot and full outputs
    pilot_output = 'outputs/dataframe_b/v3_pilot_3games.parquet'
    full_output = 'outputs/dataframe_b/v3.parquet'
    
    output_file = pilot_output if os.path.exists(pilot_output) else full_output
    
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Expected output not found: {output_file}")
    
    print(f"\n✓ Dataframe B complete: {output_file}")
    return output_file

def run_dataframe_c():
    """Execute dataframe_c_v3.py"""
    print("\n" + "=" * 80)
    print("RUNNING DATAFRAME C (v3) - EDGE-LEVEL FEATURES + BALL TRAJECTORY")
    print("=" * 80 + "\n")
    
    import subprocess
    result = subprocess.run(
        ['python', 'dataframe_c_v3.py'],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR:", result.stderr)
        raise Exception(f"Dataframe C failed with return code {result.returncode}")
    
    # Check for both pilot and full outputs
    pilot_output = 'outputs/dataframe_c/v3_pilot_3games.parquet'
    full_output = 'outputs/dataframe_c/v3.parquet'
    
    output_file = pilot_output if os.path.exists(pilot_output) else full_output
    
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Expected output not found: {output_file}")
    
    print(f"\n✓ Dataframe C complete: {output_file}")
    return output_file

def run_dataframe_d():
    """Execute dataframe_d.py"""
    print("\n" + "=" * 80)
    print("RUNNING DATAFRAME D (v2) - FRAME-LEVEL PLAYER COUNTS")
    print("=" * 80 + "\n")
    
    import subprocess
    result = subprocess.run(
        ['python', 'dataframe_d.py'],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR:", result.stderr)
        raise Exception(f"Dataframe D failed with return code {result.returncode}")
    
    output_file = 'outputs/dataframe_d/v1.parquet'
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Expected output not found: {output_file}")
    
    print(f"\n✓ Dataframe D complete: {output_file}")
    return output_file

def generate_summary_report(**context):
    """Generate a summary report of all pipeline outputs."""
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY REPORT")
    print("=" * 80 + "\n")
    
    import pandas as pd
    
    outputs = {
        'Dataframe A': 'outputs/dataframe_a/v2.parquet',
        'Dataframe B': 'outputs/dataframe_b/v3.parquet',
        'Dataframe B (Pilot)': 'outputs/dataframe_b/v3_pilot_3games.parquet',
        'Dataframe C': 'outputs/dataframe_c/v3.parquet',
        'Dataframe C (Pilot)': 'outputs/dataframe_c/v3_pilot_3games.parquet',
        'Dataframe D': 'outputs/dataframe_d/v1.parquet',
    }
    
    print(f"{'Output':<25} {'Exists':<10} {'Size (MB)':<12} {'Rows':<12} {'Cols':<8}")
    print("-" * 80)
    
    for name, filepath in outputs.items():
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            try:
                df = pd.read_parquet(filepath)
                rows = len(df)
                cols = len(df.columns)
                print(f"{name:<25} {'✓':<10} {size_mb:<12.1f} {rows:<12,} {cols:<8}")
            except Exception as e:
                print(f"{name:<25} {'✓':<10} {size_mb:<12.1f} {'ERROR':<12} {'ERROR':<8}")
        else:
            print(f"{name:<25} {'✗':<10} {'-':<12} {'-':<12} {'-':<8}")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)

# ============================================================================
# Airflow DAG Definition
# ============================================================================

dag = DAG(
    'nfl_bdb_data_pipeline',
    default_args=default_args,
    description='NFL Big Data Bowl 2026 - Data Processing Pipeline',
    schedule_interval=None,  # Manual trigger only
    start_date=days_ago(1),
    catchup=False,
    tags=['nfl', 'bdb', 'analytics', 'data-processing'],
)

# Task 1: Check Prerequisites
check_task = PythonOperator(
    task_id='check_prerequisites',
    python_callable=check_prerequisites,
    dag=dag,
)

# Task 2: Run Dataframe A
df_a_task = PythonOperator(
    task_id='run_dataframe_a',
    python_callable=run_dataframe_a,
    dag=dag,
)

# Task 3: Run Dataframe B (depends on A)
df_b_task = PythonOperator(
    task_id='run_dataframe_b',
    python_callable=run_dataframe_b,
    dag=dag,
)

# Task 4: Run Dataframe D (can run parallel to B)
df_d_task = PythonOperator(
    task_id='run_dataframe_d',
    python_callable=run_dataframe_d,
    dag=dag,
)

# Task 5: Run Dataframe C (depends on both A and B)
df_c_task = PythonOperator(
    task_id='run_dataframe_c',
    python_callable=run_dataframe_c,
    dag=dag,
)

# Task 6: Generate Summary Report
summary_task = PythonOperator(
    task_id='generate_summary_report',
    python_callable=generate_summary_report,
    provide_context=True,
    dag=dag,
)

# ============================================================================
# Task Dependencies
# ============================================================================

# Define the pipeline flow:
# 1. Check prerequisites first
# 2. Run A
# 3. B and D can run in parallel after A
# 4. C runs after both A and B complete
# 5. Summary report at the end

check_task >> df_a_task
df_a_task >> [df_b_task, df_d_task]
[df_a_task, df_b_task] >> df_c_task
[df_c_task, df_d_task] >> summary_task

# ============================================================================
# Standalone Execution (No Airflow)
# ============================================================================

def run_pipeline_standalone():
    """Run the entire pipeline without Airflow."""
    print("=" * 80)
    print("NFL BIG DATA BOWL 2026 - DATA PIPELINE")
    print("Running in STANDALONE mode (no Airflow)")
    print("=" * 80)
    
    start_time = datetime.now()
    
    try:
        # Step 1: Check prerequisites
        check_prerequisites()
        
        # Step 2: Run Dataframe A
        run_dataframe_a()
        
        # Step 3: Run Dataframe B (depends on A)
        run_dataframe_b()
        
        # Step 4: Run Dataframe D (independent)
        run_dataframe_d()
        
        # Step 5: Run Dataframe C (depends on A and B)
        run_dataframe_c()
        
        # Step 6: Generate summary
        generate_summary_report()
        
        # Calculate total time
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "=" * 80)
        print(f"PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Total time: {duration}")
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"PIPELINE FAILED!")
        print(f"Error: {str(e)}")
        print("=" * 80)
        raise

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='NFL Big Data Bowl Data Pipeline Orchestrator'
    )
    parser.add_argument(
        '--mode',
        choices=['standalone', 'airflow'],
        default='standalone',
        help='Execution mode (default: standalone)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'standalone':
        run_pipeline_standalone()
    else:
        print("Airflow mode: Please use 'docker-compose up' and access http://localhost:8080")
        print("Or place this file in your Airflow DAGs folder.")
