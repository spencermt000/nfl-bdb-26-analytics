"""
Generate Play-Level Coverage Data
==================================
Converts frame-level SumerSports coverage data into play-level data
by extracting coverage at ball release (start) and ball landing (end).

INPUTS:
  - data/train/2023_input_all.parquet (input tracking - ball in air)
  - data/train/2023_output_all.parquet (output tracking - post-catch)
  - data/sumer_bdb/sumer_coverages_frame.parquet (frame-level coverage)

OUTPUTS:
  - data/sumer_bdb/sumer_coverages_play.parquet (play-level coverage)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from utils import create_play_level_coverage

print("=" * 80)
print("GENERATE PLAY-LEVEL COVERAGE DATA")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# 1. Load Data
# ============================================================================

print("STEP 1: LOADING DATA")
print("-" * 80)

print("Loading input tracking data (ball in air)...")
input_df = pd.read_parquet('data/train/2023_input_all.parquet')
print(f"  ✓ Loaded {len(input_df):,} rows")
print(f"  Unique plays: {input_df.groupby(['game_id', 'play_id']).ngroups:,}")

print("\nLoading output tracking data (post-catch)...")
# First, check if merged output file exists
if os.path.exists('data/train/2023_output_all.parquet'):
    output_df = pd.read_parquet('data/train/2023_output_all.parquet')
else:
    # If not, merge individual week files
    print("  Merging weekly output files...")
    output_files = []
    for week in range(1, 19):
        week_file = f'data/train/output_2023_w{week:02d}.csv'
        if os.path.exists(week_file):
            df_week = pd.read_csv(week_file)
            output_files.append(df_week)
            print(f"    Loaded week {week:02d}")
    
    output_df = pd.concat(output_files, ignore_index=True)
    
    # Save merged file for future use
    output_df.to_parquet('data/train/2023_output_all.parquet', index=False)
    print(f"    Saved merged output to data/train/2023_output_all.parquet")

print(f"  ✓ Loaded {len(output_df):,} rows")
print(f"  Unique plays: {output_df.groupby(['game_id', 'play_id']).ngroups:,}")

print("\nLoading SumerSports frame-level coverage data...")
sumer_df = pd.read_parquet('data/sumer_bdb/sumer_coverages_frame.parquet')
print(f"  ✓ Loaded {len(sumer_df):,} rows")
print(f"  Unique plays: {sumer_df.groupby(['game_id', 'play_id']).ngroups:,}")

# ============================================================================
# 2. Create Play-Level Coverage
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: CREATING PLAY-LEVEL COVERAGE")
print("-" * 80)

play_coverage = create_play_level_coverage(input_df, output_df, sumer_df)

# ============================================================================
# 3. Save Output
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: SAVING OUTPUT")
print("-" * 80)

output_file = 'data/sumer_bdb/sumer_coverages_play.parquet'
play_coverage.to_parquet(output_file, index=False)

print(f"✓ Saved to: {output_file}")
print(f"  Rows: {len(play_coverage):,}")
print(f"  Columns: {len(play_coverage.columns)}")
print(f"  File size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")

# ============================================================================
# 4. Summary Statistics
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("-" * 80)

print("\nColumn list:")
for i, col in enumerate(play_coverage.columns, 1):
    print(f"  {i:2d}. {col}")

print(f"\nData availability:")
print(f"  Plays with start coverage: {play_coverage['start_scheme'].notna().sum():,} / {len(play_coverage):,}")
print(f"  Plays with land coverage: {play_coverage['land_scheme'].notna().sum():,} / {len(play_coverage):,}")
print(f"  Plays with both: {(play_coverage['start_scheme'].notna() & play_coverage['land_scheme'].notna()).sum():,}")

print(f"\nCoverage changes:")
print(f"  Coverage changed: {play_coverage['coverage_changed'].sum():,} plays")
print(f"  Coverage stable: {(play_coverage['coverage_changed'] == 0).sum():,} plays")

print("\nSample output (first 3 rows):")
print(play_coverage.head(3).to_string())

print("\n" + "=" * 80)
print("COMPLETE")
print("=" * 80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nOutput: {output_file}")
print("\n" + "=" * 80)