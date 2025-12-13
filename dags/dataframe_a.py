"""
Dataframe A: Node-Level Features (v2 - Memory Optimized)
=========================================================
Creates player/node-level features for each frame of each play.
MEMORY OPTIMIZED: Processes data in chunks to handle large datasets.

INPUTS:
  - data/train/2023_input_all.parquet (merged weekly tracking data)
  - data/supplementary_data.csv (BDB supplementary data)
  - data/sumer_coverages_player_play.parquet (SumerSports player coverage data)

OUTPUTS:
  - outputs/dataframe_a/v1.parquet

FEATURES ADDED:
  - Standardized coordinates (x, y, dir, o, ball_land_x, ball_land_y)
  - Distance metrics (to ball landing, to LOS)
  - Role indicators (isTargeted, isPasser, isRouteRunner)
  - SumerSports coverage data (coverage_responsibility, targeted_defender, alignment)
  - Velocity/acceleration vectors
  - Player age at game time

CHANGELOG v1 -> v2:
  - Added SumerSports coverage responsibility data
  - Added velocity/acceleration vector components
  - Added player age calculation
  - Standardized ball landing coordinates (ball_land_x, ball_land_y)
  - Included ball landing coordinates in output
  - MEMORY OPTIMIZED: Process in chunks to handle large datasets
  - Improved documentation and progress tracking
  - Better null handling
"""

import pandas as pd
import numpy as np
import os
import gc
from datetime import datetime
from utils import add_play_direction, standardize_play_direction

# ============================================================================
# Configuration
# ============================================================================

print("=" * 80)
print("DATAFRAME A (v2): NODE-LEVEL FEATURES [MEMORY OPTIMIZED]")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# File paths
INPUT_TRACKING = 'data/train/2023_input_all.parquet'
INPUT_SUPPLEMENTARY = 'data/supplementary_data.csv'
INPUT_SUMER_COVERAGE_PLAYER = 'data/sumer_bdb/sumer_coverages_player_play.parquet'
INPUT_SUMER_COVERAGE_FRAME = 'data/sumer_bdb/sumer_coverages_frame.parquet'
OUTPUT_DIR = 'outputs/dataframe_a'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'v1.parquet')

# Memory optimization settings
CHUNK_SIZE = 500000  # Process 500k rows at a time
USE_DTYPES_OPTIMIZATION = True

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 1. Load Data with Memory Optimization
# ============================================================================

print("STEP 1: LOADING DATA (MEMORY OPTIMIZED)")
print("-" * 80)

print("Loading player tracking data...")
# Read with optimized dtypes
dtype_dict = {
    'game_id': 'int32',
    'play_id': 'int32', 
    'nfl_id': 'int32',
    'frame_id': 'int16',
    'player_position': 'category',
    'player_side': 'category',
    'player_role': 'category',
    'player_to_predict': 'int32'
}

input_df = pd.read_parquet(INPUT_TRACKING)

# Optimize dtypes
if USE_DTYPES_OPTIMIZATION:
    print("  Optimizing data types...")
    for col, dtype in dtype_dict.items():
        if col in input_df.columns:
            try:
                input_df[col] = input_df[col].astype(dtype)
            except:
                pass

print(f"  ✓ Loaded {len(input_df):,} rows")
print(f"  Memory usage: {input_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

print("\nLoading supplementary data...")
supplementary_data = pd.read_csv(INPUT_SUPPLEMENTARY, low_memory=False)
print(f"  ✓ Loaded {len(supplementary_data):,} rows")

print("\nLoading SumerSports coverage data...")

# Player-level coverage (per play)
try:
    sumer_coverage_player = pd.read_parquet(INPUT_SUMER_COVERAGE_PLAYER)
    print(f"  ✓ Loaded player-level coverage: {len(sumer_coverage_player):,} rows")
    has_sumer_player = True
except FileNotFoundError:
    print("  ⚠ SumerSports player coverage data not found")
    sumer_coverage_player = None
    has_sumer_player = False

# Frame-level coverage (per frame)
try:
    sumer_coverage_frame = pd.read_parquet(INPUT_SUMER_COVERAGE_FRAME)
    print(f"  ✓ Loaded frame-level coverage: {len(sumer_coverage_frame):,} rows")
    has_sumer_frame = True
except FileNotFoundError:
    print("  ⚠ SumerSports frame coverage data not found")
    sumer_coverage_frame = None
    has_sumer_frame = False

# ============================================================================
# 2. Merge with Supplementary Data
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: MERGING SUPPLEMENTARY DATA")
print("-" * 80)

# Select only needed columns from supplementary data to save memory
supp_cols = ['game_id', 'play_id', 'possession_team', 'defensive_team', 
             'yardline_side', 'game_date', 'season', 'week']
supp_cols = [c for c in supp_cols if c in supplementary_data.columns]
supplementary_data = supplementary_data[supp_cols]

initial_len = len(input_df)
input_df = input_df.merge(
    supplementary_data,
    on=['game_id', 'play_id'],
    how='left'
)
print(f"Merged supplementary data: {len(input_df):,} rows (same: {len(input_df) == initial_len})")

# Free memory
del supplementary_data
gc.collect()

# ============================================================================
# 3. Standardize Coordinates (CHUNKED PROCESSING)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: STANDARDIZING COORDINATES (CHUNKED)")
print("-" * 80)

print(f"Processing in chunks of {CHUNK_SIZE:,} rows...")

# Get unique plays
unique_plays = input_df[['game_id', 'play_id']].drop_duplicates()
n_plays = len(unique_plays)
print(f"  Total plays to process: {n_plays:,}")

# Process in chunks by play to avoid splitting plays across chunks
chunk_results = []
chunk_num = 0

for i in range(0, n_plays, CHUNK_SIZE // 100):  # Approx 100 rows per play
    chunk_num += 1
    start_idx = i
    end_idx = min(i + CHUNK_SIZE // 100, n_plays)
    
    # Get plays for this chunk
    chunk_plays = unique_plays.iloc[start_idx:end_idx]
    
    # Filter data for these plays
    chunk_df = input_df.merge(
        chunk_plays,
        on=['game_id', 'play_id'],
        how='inner'
    )
    
    if len(chunk_df) == 0:
        continue
    
    print(f"  Chunk {chunk_num}: Processing {len(chunk_df):,} rows ({start_idx:,}-{end_idx:,} plays)...")
    
    # Add play direction
    chunk_df = add_play_direction(chunk_df)
    
    # Standardize player coordinates
    chunk_df = standardize_play_direction(chunk_df)
    
    # Standardize ball landing coordinates
    going_left_mask = chunk_df['direction'] == 'GOING LEFT'
    if going_left_mask.any():
        chunk_df.loc[going_left_mask, 'ball_land_x'] = 120 - chunk_df.loc[going_left_mask, 'ball_land_x']
        chunk_df.loc[going_left_mask, 'ball_land_y'] = 53.3 - chunk_df.loc[going_left_mask, 'ball_land_y']
    
    chunk_results.append(chunk_df)
    
    # Free memory
    del chunk_df
    gc.collect()

print(f"\n  ✓ Processed {chunk_num} chunks")
print("  Combining chunks...")
input_df = pd.concat(chunk_results, ignore_index=True)
del chunk_results
gc.collect()

print("  ✓ Coordinates standardized (all plays now going RIGHT)")
print(f"  Memory usage: {input_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

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
input_df['isTargeted'] = (input_df['player_role'] == 'Targeted Receiver').astype('int8')
input_df['isPasser'] = (input_df['player_role'] == 'Passer').astype('int8')
input_df['isRouteRunner'] = (
    (input_df['player_role'] == 'Targeted Receiver') | 
    (input_df['player_role'] == 'Other Route Runner')
).astype('int8')

print(f"  Targeted receivers: {input_df['isTargeted'].sum():,} player-frames")
print(f"  Passers: {input_df['isPasser'].sum():,} player-frames")
print(f"  Route runners: {input_df['isRouteRunner'].sum():,} player-frames")

# Velocity and acceleration components
print("\nCalculating velocity and acceleration vectors...")
# Convert dir and o from degrees to radians
dir_rad = np.deg2rad(input_df['dir'])
o_rad = np.deg2rad(input_df['o'])

# Velocity components
input_df['v_x'] = input_df['s'] * np.cos(dir_rad)
input_df['v_y'] = input_df['s'] * np.sin(dir_rad)

# Acceleration components
input_df['a_x'] = input_df['a'] * np.cos(dir_rad)
input_df['a_y'] = input_df['a'] * np.sin(dir_rad)

# Orientation components
input_df['o_x'] = np.cos(o_rad)
input_df['o_y'] = np.sin(o_rad)

print(f"  ✓ Velocity components: v_x, v_y")
print(f"  ✓ Acceleration components: a_x, a_y")
print(f"  ✓ Orientation components: o_x, o_y")

# Clean up
del dir_rad, o_rad
gc.collect()

# Player age at game time
print("\nCalculating player age...")
if 'player_birth_date' in input_df.columns and 'game_date' in input_df.columns:
    input_df['player_birth_date'] = pd.to_datetime(input_df['player_birth_date'], errors='coerce')
    input_df['game_date'] = pd.to_datetime(input_df['game_date'], errors='coerce')
    
    input_df['player_age'] = (
        (input_df['game_date'] - input_df['player_birth_date']).dt.days / 365.25
    )
    
    age_stats = input_df.groupby('nfl_id')['player_age'].first()
    print(f"  ✓ Player ages: mean={age_stats.mean():.1f}, min={age_stats.min():.1f}, max={age_stats.max():.1f}")
else:
    print("  ⚠ Warning: Missing birth_date or game_date, skipping age calculation")
    input_df['player_age'] = np.nan

# ============================================================================
# 5. Merge SumerSports Coverage Data
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: MERGING SUMERSPORTS COVERAGE DATA")
print("-" * 80)

# -----------------------------------------------------------------------
# 5a. Merge Player-Level Coverage (per play)
# -----------------------------------------------------------------------

if has_sumer_player:
    print("Merging player-level coverage data (per play)...")
    
    # Ensure consistent types
    input_df['game_id'] = input_df['game_id'].astype(str)
    input_df['play_id'] = input_df['play_id'].astype(float)
    input_df['nfl_id'] = input_df['nfl_id'].astype(float)
    
    sumer_coverage_player['game_id'] = sumer_coverage_player['game_id'].astype(str)
    sumer_coverage_player['play_id'] = sumer_coverage_player['play_id'].astype(float)
    sumer_coverage_player['nfl_id'] = sumer_coverage_player['nfl_id'].astype(float)
    
    # Select only needed columns
    player_cov_cols = ['game_id', 'play_id', 'nfl_id', 'coverage_responsibility', 
                       'targeted_defender', 'coverage_responsibility_side', 'alignment']
    player_cov_cols = [c for c in player_cov_cols if c in sumer_coverage_player.columns]
    sumer_player_subset = sumer_coverage_player[player_cov_cols]
    
    initial_len = len(input_df)
    input_df = input_df.merge(
        sumer_player_subset,
        on=['game_id', 'play_id', 'nfl_id'],
        how='left'
    )
    
    print(f"  ✓ Merged player coverage: {len(input_df):,} rows (same: {len(input_df) == initial_len})")
    
    # Clean up
    del sumer_coverage_player, sumer_player_subset
    gc.collect()
    
    # Check coverage stats
    if 'coverage_responsibility' in input_df.columns:
        cov_counts = input_df.groupby('nfl_id')['coverage_responsibility'].first().value_counts()
        print(f"\n  Player coverage responsibility (top 5):")
        for cov_type, count in cov_counts.head(5).items():
            print(f"    {cov_type}: {count:,} players")
    
    if 'targeted_defender' in input_df.columns:
        targeted = input_df.groupby('nfl_id')['targeted_defender'].first().sum()
        total = input_df.groupby('nfl_id')['targeted_defender'].first().count()
        print(f"  Targeted defenders: {targeted:,} / {total:,} ({100*targeted/total:.1f}%)")
else:
    print("Skipping player-level coverage data (not available)")
    input_df['coverage_responsibility'] = None
    input_df['targeted_defender'] = None
    input_df['coverage_responsibility_side'] = None
    input_df['alignment'] = None

# -----------------------------------------------------------------------
# 5b. Merge Frame-Level Coverage (per frame)
# -----------------------------------------------------------------------

if has_sumer_frame:
    print("\nMerging frame-level coverage data (per frame)...")
    
    # Ensure consistent types
    sumer_coverage_frame['game_id'] = sumer_coverage_frame['game_id'].astype(str)
    sumer_coverage_frame['play_id'] = sumer_coverage_frame['play_id'].astype(float)
    sumer_coverage_frame['frame_id'] = sumer_coverage_frame['frame_id'].astype('int16')
    
    input_df['frame_id'] = input_df['frame_id'].astype('int16')
    
    # Select coverage columns
    frame_cov_cols = ['game_id', 'play_id', 'frame_id', 'coverage_scheme',
                      'coverage_scheme__COVER_0', 'coverage_scheme__COVER_1',
                      'coverage_scheme__COVER_2', 'coverage_scheme__COVER_2_MAN',
                      'coverage_scheme__COVER_3', 'coverage_scheme__COVER_4',
                      'coverage_scheme__COVER_6', 'coverage_scheme__MISC',
                      'coverage_scheme__PREVENT', 'coverage_scheme__REDZONE',
                      'coverage_scheme__SHORT']
    frame_cov_cols = [c for c in frame_cov_cols if c in sumer_coverage_frame.columns]
    sumer_frame_subset = sumer_coverage_frame[frame_cov_cols]
    
    initial_len = len(input_df)
    input_df = input_df.merge(
        sumer_frame_subset,
        on=['game_id', 'play_id', 'frame_id'],
        how='left'
    )
    
    print(f"  ✓ Merged frame coverage: {len(input_df):,} rows (same: {len(input_df) == initial_len})")
    
    # Clean up
    del sumer_coverage_frame, sumer_frame_subset
    gc.collect()
    
    # Check frame coverage stats
    if 'coverage_scheme' in input_df.columns:
        scheme_counts = input_df['coverage_scheme'].value_counts()
        print(f"\n  Frame-level coverage schemes (top 5):")
        for scheme, count in scheme_counts.head(5).items():
            print(f"    {scheme}: {count:,} frames")
        
        coverage_pct = input_df['coverage_scheme'].notna().sum() / len(input_df) * 100
        print(f"  Coverage data available: {coverage_pct:.1f}% of frames")
else:
    print("\nSkipping frame-level coverage data (not available)")
    input_df['coverage_scheme'] = None
    for cover_type in ['COVER_0', 'COVER_1', 'COVER_2', 'COVER_2_MAN', 
                       'COVER_3', 'COVER_4', 'COVER_6', 'MISC', 
                       'PREVENT', 'REDZONE', 'SHORT']:
        input_df[f'coverage_scheme__{cover_type}'] = None

# ============================================================================
# 6. Select Final Columns
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: SELECTING FINAL COLUMNS")
print("-" * 80)

df_a_cols = [
    # Identifiers
    'game_id', 'play_id', 'frame_id', 'nfl_id',
    
    # Player attributes
    'player_name', 'player_height', 'player_weight', 'player_birth_date', 'player_age',
    'player_position', 'player_side', 'player_role',
    
    # Standardized tracking data
    'x', 'y', 's', 'a', 'dir', 'o',
    
    # Standardized ball landing coordinates
    'ball_land_x', 'ball_land_y',
    
    # Velocity and acceleration vectors
    'v_x', 'v_y', 'a_x', 'a_y',
    
    # Orientation vectors
    'o_x', 'o_y',
    
    # Distance metrics
    'e_dist_ball_land', 'los_dist',
    
    # Role indicators
    'isTargeted', 'isPasser', 'isRouteRunner',
    
    # SumerSports player-level coverage data
    'coverage_responsibility', 'targeted_defender', 
    'coverage_responsibility_side', 'alignment',
    
    # SumerSports frame-level coverage data
    'coverage_scheme',
    'coverage_scheme__COVER_0', 'coverage_scheme__COVER_1',
    'coverage_scheme__COVER_2', 'coverage_scheme__COVER_2_MAN',
    'coverage_scheme__COVER_3', 'coverage_scheme__COVER_4',
    'coverage_scheme__COVER_6', 'coverage_scheme__MISC',
    'coverage_scheme__PREVENT', 'coverage_scheme__REDZONE',
    'coverage_scheme__SHORT'
]

df_a_cols = [col for col in df_a_cols if col in input_df.columns]
output_df = input_df[df_a_cols].copy()

# Clean up
del input_df
gc.collect()

print(f"Final columns ({len(df_a_cols)})")
print(f"Memory usage: {output_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# ============================================================================
# 7. Data Quality Checks
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: DATA QUALITY CHECKS")
print("-" * 80)

print("Summary statistics:")
print(f"  Total rows: {len(output_df):,}")
print(f"  Unique games: {output_df['game_id'].nunique():,}")
print(f"  Unique plays: {output_df['play_id'].nunique():,}")
print(f"  Unique players: {output_df['nfl_id'].nunique():,}")

# ============================================================================
# 8. Save Output
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: SAVING OUTPUT")
print("-" * 80)

print("Writing to parquet...")
output_df.to_parquet(OUTPUT_FILE, engine='pyarrow', index=False, compression='snappy')
print(f"✓ Saved to: {OUTPUT_FILE}")
print(f"  File size: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.1f} MB")

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