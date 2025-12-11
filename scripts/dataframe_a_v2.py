"""
Dataframe A: Node-Level Features (v2)
======================================
Creates player/node-level features for each frame of each play.

INPUTS:
  - data/train/2023_input_all.parquet (merged weekly tracking data)
  - data/supplementary_data.csv (BDB supplementary data)
  - data/sumer_coverages_player_play.parquet (SumerSports player coverage data)

OUTPUTS:
  - outputs/dataframe_a/v2.parquet

FEATURES ADDED:
  - Standardized coordinates (x, y, dir, o)
  - Distance metrics (to ball landing, to LOS)
  - Role indicators (isTargeted, isPasser, isRouteRunner)
  - SumerSports coverage data (coverage_responsibility, targeted_defender, alignment)
  - Velocity/acceleration vectors
  - Player age at game time

CHANGELOG v1 -> v2:
  - Added SumerSports coverage responsibility data
  - Added velocity/acceleration vector components
  - Added player age calculation
  - Improved documentation and progress tracking
  - Better null handling
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from utils import add_play_direction, standardize_play_direction

# ============================================================================
# Configuration
# ============================================================================

print("=" * 80)
print("DATAFRAME A (v2): NODE-LEVEL FEATURES")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# File paths
INPUT_TRACKING = 'data/train/2023_input_all.parquet'
INPUT_SUPPLEMENTARY = 'data/supplementary_data.csv'
INPUT_SUMER_COVERAGE = 'data/sumer_coverages_player_play.parquet'
OUTPUT_DIR = 'outputs/dataframe_a'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'v2.parquet')

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 1. Load Data
# ============================================================================

print("STEP 1: LOADING DATA")
print("-" * 80)

print("Loading player tracking data...")
input_df = pd.read_parquet(INPUT_TRACKING)
print(f"  ✓ Loaded {len(input_df):,} rows")
print(f"  Columns: {list(input_df.columns)}")

print("\nLoading supplementary data...")
supplementary_data = pd.read_csv(INPUT_SUPPLEMENTARY, low_memory=False)
print(f"  ✓ Loaded {len(supplementary_data):,} rows")

print("\nLoading SumerSports coverage data...")
try:
    sumer_coverage = pd.read_parquet(INPUT_SUMER_COVERAGE)
    print(f"  ✓ Loaded {len(sumer_coverage):,} rows")
    print(f"  Columns: {list(sumer_coverage.columns)}")
    has_sumer = True
except FileNotFoundError:
    print("  ⚠ SumerSports coverage data not found - will skip coverage features")
    sumer_coverage = None
    has_sumer = False

# ============================================================================
# 2. Merge with Supplementary Data
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: MERGING SUPPLEMENTARY DATA")
print("-" * 80)

initial_len = len(input_df)
input_df = input_df.merge(
    supplementary_data,
    on=['game_id', 'play_id'],
    how='left'
)
print(f"Merged supplementary data: {len(input_df):,} rows (same: {len(input_df) == initial_len})")

# Check for missing supplementary data
missing_supp = input_df['possession_team'].isna().sum()
if missing_supp > 0:
    print(f"  ⚠ Warning: {missing_supp:,} rows missing supplementary data")

# ============================================================================
# 3. Standardize Coordinates
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: STANDARDIZING COORDINATES")
print("-" * 80)

print("Adding play direction...")
input_df = add_play_direction(input_df)
direction_counts = input_df['direction'].value_counts()
print(f"  Play directions:")
for direction, count in direction_counts.items():
    print(f"    {direction}: {count:,}")

print("\nStandardizing x, y, dir, o coordinates...")
input_df = standardize_play_direction(input_df)
print("  ✓ Coordinates standardized (all plays now going RIGHT)")

# ============================================================================
# 4. Calculate Derived Features
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: CALCULATING DERIVED FEATURES")
print("-" * 80)

# Distance to ball landing spot
print("Calculating distance to ball landing spot...")
input_df['e_dist_ball_land'] = np.sqrt(
    (input_df['x'] - input_df['ball_land_x'])**2 + 
    (input_df['y'] - input_df['ball_land_y'])**2
)
print(f"  ✓ e_dist_ball_land: mean={input_df['e_dist_ball_land'].mean():.2f} yards")

# Distance to line of scrimmage
print("\nCalculating distance to line of scrimmage...")
input_df['los_dist'] = np.abs(input_df['x'] - input_df['absolute_yardline_number'])
print(f"  ✓ los_dist: mean={input_df['los_dist'].mean():.2f} yards")

# Role indicators
print("\nCreating role indicator flags...")
input_df['isTargeted'] = (input_df['player_role'] == 'Targeted Receiver').astype(int)
input_df['isPasser'] = (input_df['player_role'] == 'Passer').astype(int)
input_df['isRouteRunner'] = (
    (input_df['player_role'] == 'Targeted Receiver') | 
    (input_df['player_role'] == 'Other Route Runner')
).astype(int)

print(f"  Targeted receivers: {input_df['isTargeted'].sum():,} player-frames")
print(f"  Passers: {input_df['isPasser'].sum():,} player-frames")
print(f"  Route runners: {input_df['isRouteRunner'].sum():,} player-frames")

# Velocity and acceleration components
print("\nCalculating velocity and acceleration vectors...")
# Convert dir and o from degrees to radians
input_df['dir_rad'] = np.deg2rad(input_df['dir'])
input_df['o_rad'] = np.deg2rad(input_df['o'])

# Velocity components (s is speed in yards/sec, dir is direction of movement)
input_df['v_x'] = input_df['s'] * np.cos(input_df['dir_rad'])
input_df['v_y'] = input_df['s'] * np.sin(input_df['dir_rad'])

# Acceleration components (a is acceleration magnitude)
input_df['a_x'] = input_df['a'] * np.cos(input_df['dir_rad'])
input_df['a_y'] = input_df['a'] * np.sin(input_df['dir_rad'])

print(f"  ✓ Velocity components: v_x, v_y")
print(f"  ✓ Acceleration components: a_x, a_y")

# Orientation components (for body facing direction)
input_df['o_x'] = np.cos(input_df['o_rad'])
input_df['o_y'] = np.sin(input_df['o_rad'])
print(f"  ✓ Orientation components: o_x, o_y")

# Player age at game time
print("\nCalculating player age...")
if 'player_birth_date' in input_df.columns and 'game_date' in input_df.columns:
    input_df['player_birth_date'] = pd.to_datetime(input_df['player_birth_date'], errors='coerce')
    input_df['game_date'] = pd.to_datetime(input_df['game_date'], errors='coerce')
    
    # Calculate age in years
    input_df['player_age'] = (
        (input_df['game_date'] - input_df['player_birth_date']).dt.days / 365.25
    )
    
    age_stats = input_df.groupby('nfl_id')['player_age'].first()
    print(f"  ✓ Player ages: mean={age_stats.mean():.1f}, min={age_stats.min():.1f}, max={age_stats.max():.1f}")
else:
    print("  ⚠ Warning: Missing birth_date or game_date, skipping age calculation")
    input_df['player_age'] = np.nan

# Drop temporary columns
input_df = input_df.drop(columns=['dir_rad', 'o_rad'], errors='ignore')

# ============================================================================
# 5. Merge SumerSports Coverage Data
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: MERGING SUMERSPORTS COVERAGE DATA")
print("-" * 80)

if has_sumer:
    print("Merging player-level coverage data...")
    
    # Ensure consistent types
    input_df['game_id'] = input_df['game_id'].astype(str)
    input_df['play_id'] = input_df['play_id'].astype(float)
    input_df['nfl_id'] = input_df['nfl_id'].astype(float)
    
    sumer_coverage['game_id'] = sumer_coverage['game_id'].astype(str)
    sumer_coverage['play_id'] = sumer_coverage['play_id'].astype(float)
    sumer_coverage['nfl_id'] = sumer_coverage['nfl_id'].astype(float)
    
    # Merge on game_id, play_id, nfl_id
    # Note: SumerSports data is per-play, not per-frame, so same values across all frames
    initial_len = len(input_df)
    input_df = input_df.merge(
        sumer_coverage[['game_id', 'play_id', 'nfl_id', 'coverage_responsibility', 
                        'targeted_defender', 'coverage_responsibility_side', 'alignment']],
        on=['game_id', 'play_id', 'nfl_id'],
        how='left'
    )
    
    print(f"  ✓ Merged coverage data: {len(input_df):,} rows (same: {len(input_df) == initial_len})")
    
    # Check coverage stats
    if 'coverage_responsibility' in input_df.columns:
        cov_counts = input_df.groupby('nfl_id')['coverage_responsibility'].first().value_counts()
        print(f"\n  Coverage responsibility distribution:")
        for cov_type, count in cov_counts.head(10).items():
            print(f"    {cov_type}: {count:,} players")
    
    if 'targeted_defender' in input_df.columns:
        targeted = input_df.groupby('nfl_id')['targeted_defender'].first().sum()
        total = input_df.groupby('nfl_id')['targeted_defender'].first().count()
        print(f"\n  Targeted defenders: {targeted:,} / {total:,} ({100*targeted/total:.1f}%)")
else:
    print("Skipping SumerSports coverage data (file not found)")
    # Create placeholder columns
    input_df['coverage_responsibility'] = None
    input_df['targeted_defender'] = None
    input_df['coverage_responsibility_side'] = None
    input_df['alignment'] = None

# ============================================================================
# 6. Select Final Columns
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: SELECTING FINAL COLUMNS")
print("-" * 80)

# Define columns to keep
df_a_cols = [
    # Identifiers
    'game_id', 'play_id', 'frame_id', 'nfl_id',
    
    # Player attributes
    'player_name', 'player_height', 'player_weight', 'player_birth_date', 'player_age',
    'player_position', 'player_side', 'player_role',
    
    # Standardized tracking data
    'x', 'y', 's', 'a', 'dir', 'o',
    
    # Velocity and acceleration vectors
    'v_x', 'v_y', 'a_x', 'a_y',
    
    # Orientation vectors
    'o_x', 'o_y',
    
    # Distance metrics
    'e_dist_ball_land', 'los_dist',
    
    # Role indicators
    'isTargeted', 'isPasser', 'isRouteRunner',
    
    # SumerSports coverage data
    'coverage_responsibility', 'targeted_defender', 
    'coverage_responsibility_side', 'alignment'
]

# Filter to only columns that exist
df_a_cols = [col for col in df_a_cols if col in input_df.columns]

output_df = input_df[df_a_cols].copy()

print(f"Final columns ({len(df_a_cols)}):")
for i, col in enumerate(df_a_cols, 1):
    print(f"  {i:2d}. {col}")

# ============================================================================
# 7. Data Quality Checks
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: DATA QUALITY CHECKS")
print("-" * 80)

# Check for nulls
print("Null value counts:")
null_counts = output_df.isnull().sum()
null_counts = null_counts[null_counts > 0].sort_values(ascending=False)
if len(null_counts) > 0:
    for col, count in null_counts.items():
        pct = 100 * count / len(output_df)
        print(f"  {col}: {count:,} ({pct:.2f}%)")
else:
    print("  ✓ No null values!")

# Check for duplicates
print("\nChecking for duplicate rows...")
dup_count = output_df.duplicated(subset=['game_id', 'play_id', 'frame_id', 'nfl_id']).sum()
if dup_count > 0:
    print(f"  ⚠ Warning: {dup_count:,} duplicate player-frames found")
else:
    print("  ✓ No duplicates!")

# Summary statistics
print("\nSummary statistics:")
print(f"  Total rows: {len(output_df):,}")
print(f"  Unique games: {output_df['game_id'].nunique():,}")
print(f"  Unique plays: {output_df['play_id'].nunique():,}")
print(f"  Unique players: {output_df['nfl_id'].nunique():,}")
print(f"  Frames per play (avg): {len(output_df) / output_df.groupby(['game_id', 'play_id']).ngroups:.1f}")

# ============================================================================
# 8. Save Output
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: SAVING OUTPUT")
print("-" * 80)

output_df.to_parquet(OUTPUT_FILE, engine='pyarrow', index=False)
print(f"✓ Saved to: {OUTPUT_FILE}")
print(f"  File size: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.1f} MB")

# Sample output
print("\nSample output (first 3 rows):")
print(output_df.head(3).to_string())

# ============================================================================
# Complete
# ============================================================================

print("\n" + "=" * 80)
print("DATAFRAME A (v2): COMPLETE")
print("=" * 80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nOutput: {OUTPUT_FILE}")
print(f"Rows: {len(output_df):,}")
print(f"Columns: {len(output_df.columns)}")
print("\n" + "=" * 80)