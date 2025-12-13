"""
Dataframe B: Play-Level Features (v3.2 - JOIN INDEX & SELECT METRICS)
=====================================================================
Creates play-level contextual features for each play using a pre-calculated join index.

INPUTS:
  - data/supplementary_data.csv (BDB supplementary data)
  - data/sdv_raw_pbp.parquet (nflfastR play-by-play data)
  - outputs/dataframe_a/v2.parquet (to extract passer position at release)

OUTPUTS:
  - outputs/dataframe_b/v4.parquet

CHANGELOG:
  - Merging logic simplified: Uses 'join_index' present in both datasets
  - nflfastR columns strictly limited to: xpass, cp, comp_yac_epa, shotgun, comp_air_epa
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

print("=" * 80)
print("DATAFRAME B (v3.2): JOIN INDEX + SELECT METRICS")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# File paths
INPUT_BDB_SUPP = 'data/supplementary_data.csv'
INPUT_NFLFASTR = 'data/sdv_raw_pbp.parquet'
INPUT_DF_A = 'outputs/dataframe_a/v2.parquet'
OUTPUT_DIR = 'outputs/dataframe_b'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'v1.parquet')

# Toggle for nflfastR merge
SDV_MERGE = True 

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 1. Load Data
# ============================================================================

print("STEP 1: LOADING DATA")
print("-" * 80)

print("Loading BDB supplementary data...")
sup_raw = pd.read_csv(INPUT_BDB_SUPP, low_memory=False)
print(f"  ✓ Loaded {len(sup_raw):,} rows")

print("\nLoading dataframe_a (for passer position at release)...")
df_a = pd.read_parquet(INPUT_DF_A)
print(f"  ✓ Loaded {len(df_a):,} rows")

if SDV_MERGE:
    print(f"\n✓ SDV_MERGE = True: Will merge nflfastR data")
    print("\nLoading nflfastR play-by-play data...")
    sdv_pbp = pd.read_parquet(INPUT_NFLFASTR)
    print(f"  ✓ Loaded {len(sdv_pbp):,} rows")
    
    # Check for join_index
    if 'join_index' not in sdv_pbp.columns:
        raise ValueError("CRITICAL ERROR: 'join_index' column missing from nflfastR data")
        
    # Check for join_index in BDB data
    if 'join_index' not in sup_raw.columns:
         # Fallback if the user cleaned it externally but didn't save it to the CSV path
         # Depending on your workflow, you might need to generate it or error out.
         # For now, I'll assume it is there as you stated.
         print("  ⚠ WARNING: 'join_index' missing from BDB CSV. Please ensure data is cleaned.")
    else:
         print("  ✓ 'join_index' found in both datasets")

    # Strict Column Selection for SDV
    req_cols = ["join_index", "xpass", "cp", "comp_yac_epa", "shotgun", "comp_air_epa"]
    
    # Filter to only existing columns (just in case)
    final_sdv_cols = [c for c in req_cols if c in sdv_pbp.columns]
    
    missing = set(req_cols) - set(final_sdv_cols)
    if missing:
        print(f"  ⚠ WARNING: Missing requested columns: {missing}")
    
    sdv_subset = sdv_pbp[final_sdv_cols].copy()
    print(f"  ✓ Selected {len(sdv_subset.columns)} columns from nflfastR")

else:
    print(f"\n⊗ SDV_MERGE = False: Skipping nflfastR merge")
    sdv_pbp = None

# ============================================================================
# 2. Extract Ball Trajectory Features from Dataframe A
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: EXTRACTING BALL TRAJECTORY FEATURES")
print("-" * 80)

print("Identifying passers at release frame (frame_id = 1)...")

passers_release = df_a[
    (df_a['frame_id'] == 1) & 
    (df_a['isPasser'] == 1)
].copy()

print(f"  Found {len(passers_release):,} passer positions at release")

ball_start = passers_release[[
    'game_id', 'play_id', 'x', 'y', 'o', 'ball_land_x', 'ball_land_y'
]].copy()

ball_start = ball_start.rename(columns={
    'x': 'start_ball_x',
    'y': 'start_ball_y',
    'o': 'start_ball_o'
})

# Get number of frames per play (flight duration)
frames_per_play = df_a.groupby(['game_id', 'play_id'])['frame_id'].max().reset_index()
frames_per_play = frames_per_play.rename(columns={'frame_id': 'ball_flight_frames'})

# Merge ball start position with frame counts
ball_features = ball_start.merge(frames_per_play, on=['game_id', 'play_id'], how='left')

# Calculate derived features
ball_features['ball_flight_distance'] = np.sqrt(
    (ball_features['ball_land_x'] - ball_features['start_ball_x'])**2 +
    (ball_features['ball_land_y'] - ball_features['start_ball_y'])**2
)

ball_features['throw_direction'] = np.arctan2(
    ball_features['ball_land_y'] - ball_features['start_ball_y'],
    ball_features['ball_land_x'] - ball_features['start_ball_x']
) * 180 / np.pi
ball_features['throw_direction'] = (ball_features['throw_direction'] + 360) % 360

def categorize_throw(distance):
    if pd.isna(distance): return 'unknown'
    elif distance < 10: return 'short'
    elif distance < 20: return 'medium'
    elif distance < 35: return 'deep'
    else: return 'bomb'

ball_features['throw_type'] = ball_features['ball_flight_distance'].apply(categorize_throw)
ball_features = ball_features.drop(columns=['ball_land_x', 'ball_land_y'])

print(f"  ✓ Calculated trajectory features for {len(ball_features):,} plays")

# ============================================================================
# 3. Merge Features using join_index
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: MERGING FEATURES")
print("-" * 80)

if SDV_MERGE:
    print(f"Merging BDB ({len(sup_raw)}) and nflfastR ({len(sdv_subset)}) on 'join_index'...")
    
    # Verify unique keys
    bdb_unique = sup_raw['join_index'].nunique()
    sdv_unique = sdv_subset['join_index'].nunique()
    print(f"  Unique Join Indices - BDB: {bdb_unique:,}, nflfastR: {sdv_unique:,}")
    
    # Merge
    master = sup_raw.merge(sdv_subset, on='join_index', how='left')
    print(f"  ✓ Merged: {len(master):,} plays")
    
    # Check match rate
    matched = master['cp'].notna().sum()
    print(f"  Match Rate: {matched:,} plays found in both ({100*matched/len(master):.1f}%)")
else:
    master = sup_raw.copy()
    print(f"  Using BDB data only: {len(master):,} plays")

# ============================================================================
# 4. Merge Ball Trajectory Features
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: MERGING BALL TRAJECTORY FEATURES")
print("-" * 80)

master['game_id'] = master['game_id'].astype(str)
ball_features['game_id'] = ball_features['game_id'].astype(str)

master = master.merge(ball_features, on=['game_id', 'play_id'], how='left')
print(f"  Merged ball trajectory features")

# ============================================================================
# 5. Select Final Columns
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: SELECTING FINAL COLUMNS")
print("-" * 80)

# Base columns from BDB
df_b_cols = [
    'season', 'week', 'play_id', 'game_id',
    'quarter', 'down', 'yards_to_go',
    'possession_team', 'defensive_team', 'yardline_side', 'yardline_number',
    'pass_result', 'offense_formation', 'receiver_alignment',
    'defenders_in_the_box', 'yards_gained',
    'expected_points', 'expected_points_added',
    # Trajectory
    'start_ball_x', 'start_ball_y', 'start_ball_o',
    'ball_flight_distance', 'ball_flight_frames',
    'throw_direction', 'throw_type',
]

# Add requested SDV columns if available
if SDV_MERGE:
    # Explicitly the exact columns requested
    requested_additions = ["xpass", "cp", "comp_yac_epa", "shotgun", "comp_air_epa"]
    
    # Check existence in master
    final_additions = [col for col in requested_additions if col in master.columns]
    df_b_cols.extend(final_additions)

# Filter to only columns that exist
df_b_cols = [col for col in df_b_cols if col in master.columns]
df_b = master[df_b_cols].copy()

print(f"Final columns: {len(df_b_cols)}")
for col in ["xpass", "cp", "comp_yac_epa", "shotgun", "comp_air_epa"]:
    if col in df_b.columns:
        print(f"  ✓ Retained: {col}")
    else:
        print(f"  X Missing: {col}")

# ============================================================================
# 6. Save Output
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: SAVING OUTPUT")
print("-" * 80)

df_b.to_parquet(OUTPUT_FILE, engine='pyarrow', index=False)
print(f"✓ Saved to: {OUTPUT_FILE}")

print("\n" + "=" * 80)
print("COMPLETE")
print("=" * 80)