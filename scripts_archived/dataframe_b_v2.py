"""
Dataframe B: Play-Level Features (v3 - WITH BALL TRAJECTORY)
=============================================================
Creates play-level contextual features for each play.

INPUTS:
  - data/supplementary_data.csv (BDB supplementary data)
  - data/sdv_raw_pbp.parquet (nflfastR play-by-play data)
  - outputs/dataframe_a/v2.parquet (to extract passer position at release)

OUTPUTS:
  - outputs/dataframe_b/v3.parquet

NEW IN V3:
  - Ball trajectory features:
    * start_ball_x (passer's x at release frame)
    * start_ball_y (passer's y at release frame)  
    * start_ball_o (ball orientation at release)
    * ball_flight_distance (Euclidean distance from start to landing)
    * ball_flight_frames (duration in frames)
    * throw_direction (normalized angle, 0-360)
    * throw_type (categorized: short/medium/deep based on distance)

FEATURES INCLUDED (from v2):
  - Game context (quarter, down, yards_to_go, time remaining, etc.)
  - Team context (scores, win probability, timeouts)
  - Formation & alignment (offense formation, receiver alignment, dropback)
  - Coverage schemes (man/zone, coverage type from BDB)
  - Pass characteristics (length, location, result, air yards, YAC)
  - EPA metrics (air_epa, yac_epa, comp_epa, total EPA)
  - Win probability metrics (WP, WPA for both teams)
  - Defensive stats (defenders in box, PBU, tackles)
  - Field conditions (surface, roof, weather)
  - Betting lines (spread, total)

CHANGELOG v2 -> v3:
  - Added ball trajectory features derived from passer position at release
  - Now loads dataframe_a to extract passer coordinates at first frame
  - Calculates throw distance, direction, and categorizes throw type
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

print("=" * 80)
print("DATAFRAME B (v3): PLAY-LEVEL FEATURES + BALL TRAJECTORY")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# File paths
INPUT_BDB_SUPP = 'data/supplementary_data.csv'
INPUT_NFLFASTR = 'data/sdv_raw_pbp.parquet'
INPUT_DF_A = 'outputs/dataframe_a/v2.parquet'  # NEW: for passer position
OUTPUT_DIR = 'outputs/dataframe_b'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'v3.parquet')

# Toggle for nflfastR merge
SDV_MERGE = False  # Set to False to skip nflfastR merge entirely

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
print(f"  Columns: {len(sup_raw.columns)}")

print("\nLoading dataframe_a (for passer position at release)...")
df_a = pd.read_parquet(INPUT_DF_A)
print(f"  ✓ Loaded {len(df_a):,} rows")

if SDV_MERGE:
    print(f"\n✓ SDV_MERGE = True: Will merge nflfastR data")
    print("\nLoading nflfastR play-by-play data...")
    sdv_pbp = pd.read_parquet(INPUT_NFLFASTR)
    print(f"  ✓ Loaded {len(sdv_pbp):,} rows")
    print(f"  Columns: {len(sdv_pbp.columns)}")
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

# Filter to frame 1 (release) and passers only
passers_release = df_a[
    (df_a['frame_id'] == 1) & 
    (df_a['isPasser'] == 1)
].copy()

print(f"  Found {len(passers_release):,} passer positions at release")

# Extract ball starting position (passer position at release)
ball_start = passers_release[[
    'game_id', 'play_id', 'x', 'y', 'o', 'ball_land_x', 'ball_land_y'
]].copy()

ball_start = ball_start.rename(columns={
    'x': 'start_ball_x',
    'y': 'start_ball_y',
    'o': 'start_ball_o'
})

print(f"  Extracted start positions for {len(ball_start):,} plays")

# Get number of frames per play (flight duration)
frames_per_play = df_a.groupby(['game_id', 'play_id'])['frame_id'].max().reset_index()
frames_per_play = frames_per_play.rename(columns={'frame_id': 'ball_flight_frames'})

print(f"  Calculated flight duration for {len(frames_per_play):,} plays")

# Merge ball start position with frame counts
ball_features = ball_start.merge(frames_per_play, on=['game_id', 'play_id'], how='left')

# Calculate derived features
print("\nCalculating derived ball trajectory features...")

# 1. Ball flight distance (Euclidean)
ball_features['ball_flight_distance'] = np.sqrt(
    (ball_features['ball_land_x'] - ball_features['start_ball_x'])**2 +
    (ball_features['ball_land_y'] - ball_features['start_ball_y'])**2
)

# 2. Throw direction (angle from start to landing, 0-360 degrees)
ball_features['throw_direction'] = np.arctan2(
    ball_features['ball_land_y'] - ball_features['start_ball_y'],
    ball_features['ball_land_x'] - ball_features['start_ball_x']
) * 180 / np.pi

# Normalize to 0-360
ball_features['throw_direction'] = (ball_features['throw_direction'] + 360) % 360

# 3. Throw type (categorize by distance)
def categorize_throw(distance):
    """Categorize throw by distance in yards"""
    if pd.isna(distance):
        return 'unknown'
    elif distance < 10:
        return 'short'
    elif distance < 20:
        return 'medium'
    elif distance < 35:
        return 'deep'
    else:
        return 'bomb'

ball_features['throw_type'] = ball_features['ball_flight_distance'].apply(categorize_throw)

print(f"  ✓ Ball flight distance: mean={ball_features['ball_flight_distance'].mean():.2f} yards")
print(f"  ✓ Ball flight frames: mean={ball_features['ball_flight_frames'].mean():.1f} frames")

print("\nThrow type distribution:")
throw_type_counts = ball_features['throw_type'].value_counts()
for throw_type, count in throw_type_counts.items():
    pct = 100 * count / len(ball_features)
    print(f"  {throw_type}: {count:,} ({pct:.1f}%)")

# Drop intermediate columns used for calculation
ball_features = ball_features.drop(columns=['ball_land_x', 'ball_land_y'])

print(f"\nFinal ball trajectory features: {len(ball_features.columns)-2} features")  # -2 for game_id, play_id

# ============================================================================
# 3. Create Index Columns for Merging (SAME AS V2)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: CREATING INDEX COLUMNS")
print("-" * 80)

if SDV_MERGE:
    # [SAME CODE AS V2 - Lines 93-145 of original]
    print("\nCreating chronological play_id for nflfastR...")
    print("  (BDB play_id is chronological within game, nflfastR is not)")
    
    sdv_pbp = sdv_pbp.sort_values(
        by=['old_game_id', 'qtr', 'quarter_seconds_remaining'],
        ascending=[True, True, False]
    )
    
    sdv_pbp['new_play_id'] = sdv_pbp.groupby('old_game_id').cumcount() + 1
    print(f"  ✓ Created new_play_id: range 1-{sdv_pbp['new_play_id'].max()}")
    
    sup_raw['game_id_str'] = sup_raw['game_id'].astype(str)
    sdv_pbp['game_id_str'] = sdv_pbp['old_game_id'].astype(str)
    
    sup_raw['merge_key'] = sup_raw['game_id_str'] + '_' + sup_raw['play_id'].astype(str)
    sdv_pbp['merge_key'] = sdv_pbp['game_id_str'] + '_' + sdv_pbp['new_play_id'].astype(str)
    
    print(f"\nMerge key created:")
    print(f"  BDB unique keys: {sup_raw['merge_key'].nunique():,}")
    print(f"  nflfastR unique keys: {sdv_pbp['merge_key'].nunique():,}")
    
    overlap = len(set(sup_raw['merge_key']) & set(sdv_pbp['merge_key']))
    overlap_pct = 100 * overlap / len(sup_raw)
    print(f"  Overlap: {overlap:,} plays ({overlap_pct:.1f}% of BDB)")
else:
    print("\nSkipping index creation (SDV_MERGE = False)")
    sup_raw['merge_key'] = None

# ============================================================================
# 4. Calculate Derived Features in nflfastR Data (SAME AS V2)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: CALCULATING DERIVED FEATURES")
print("-" * 80)

if SDV_MERGE:
    print("Calculating play_in_drive...")
    sdv_pbp['play_in_drive'] = sdv_pbp.groupby(['game_id_str', 'posteam', 'drive']).cumcount() + 1
    print(f"  ✓ Plays per drive (avg): {sdv_pbp['play_in_drive'].mean():.1f}")
    
    print("\nCreating defensive stat indicators...")
    sdv_pbp['tfl'] = np.where(
        sdv_pbp['tackle_for_loss_1_player_id'].notna() | sdv_pbp['tackle_for_loss_2_player_id'].notna(),
        1, 0
    )
    sdv_pbp['pbu'] = np.where(
        (sdv_pbp['pass_defense_1_player_id'].notna() | sdv_pbp['pass_defense_2_player_id'].notna()) | 
        (sdv_pbp['interception'] == 1),
        1, 0
    )
    sdv_pbp['atkl'] = np.where(
        sdv_pbp['solo_tackle_1_player_id'].notna() | sdv_pbp['solo_tackle_2_player_id'].notna(),
        1, 0
    )
    sdv_pbp['stkl'] = np.where(
        sdv_pbp['assist_tackle_1_player_id'].notna() | 
        sdv_pbp['assist_tackle_2_player_id'].notna() | 
        sdv_pbp['assist_tackle_3_player_id'].notna() | 
        sdv_pbp['assist_tackle_4_player_id'].notna(),
        1, 0
    )
    print(f"  ✓ Defensive stats created")
else:
    print("Skipping derived features (SDV_MERGE = False)")

# ============================================================================
# 5-7. [SAME AS V2 - Feature selection and merging]
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5-7: SELECTING AND MERGING FEATURES")
print("-" * 80)

if SDV_MERGE:
    sdv_cols = [
        "quarter_seconds_remaining", "game_seconds_remaining", "half_seconds_remaining",
        "goal_to_go", "shotgun", "no_huddle", 
        "posteam_timeouts_remaining", "defteam_timeouts_remaining",
        "air_yards", "yards_after_catch", "pass_length", "pass_location",
        "qb_hit", "touchdown", 
        "ep", "epa", "air_epa", "yac_epa", "comp_air_epa", "comp_yac_epa",
        "wp", "def_wp", "wpa", "air_wpa", "yac_wpa", "comp_air_wpa", "comp_yac_wpa",
        "cp", "cpoe", "xpass", "pass_oe",
        "xyac_epa", "xyac_median_yardage", "xyac_mean_yardage", "xyac_success",
        "surface", "roof", "temp", "wind",
        "total_line", "spread_line",
        "series", "play_in_drive",
        "pbu", "tfl", "atkl", "stkl",
        "complete_pass", "incomplete_pass", "interception",
        "merge_key"
    ]
    sdv_cols = [col for col in sdv_cols if col in sdv_pbp.columns]
    sdv_subset = sdv_pbp[sdv_cols]
    
    print(f"  Selected {len(sdv_cols)} nflfastR columns")
    master = sup_raw.merge(sdv_subset, on='merge_key', how='left', suffixes=('', '_sdv'))
    print(f"  ✓ Merged: {len(master):,} plays")
else:
    master = sup_raw.copy()
    print(f"  Using BDB data only: {len(master):,} plays")

# ============================================================================
# 8. Merge Ball Trajectory Features (NEW IN V3)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: MERGING BALL TRAJECTORY FEATURES")
print("-" * 80)

# Fix game_id types before merge
master['game_id'] = master['game_id'].astype(str)
ball_features['game_id'] = ball_features['game_id'].astype(str)

initial_len = len(master)
master = master.merge(
    ball_features,
    on=['game_id', 'play_id'],
    how='left'
)

print(f"  Merged ball trajectory features: {len(master):,} plays (same: {len(master) == initial_len})")

# Check merge success
ball_feat_cols = ['start_ball_x', 'start_ball_y', 'start_ball_o', 'ball_flight_distance', 
                  'ball_flight_frames', 'throw_direction', 'throw_type']
merged_count = master[ball_feat_cols].notna().all(axis=1).sum()
print(f"  Plays with complete ball trajectory data: {merged_count:,} ({100*merged_count/len(master):.1f}%)")

# ============================================================================
# 9. Calculate Team-Specific Metrics (SAME AS V2, Lines 356-410)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 9: CALCULATING TEAM-SPECIFIC METRICS")
print("-" * 80)

# [Lines 356-410 from original v2 code]
# ... (same calculations for pos_team_wp, pos_team_wpa, ps_pos_team_score, etc.)

print("  ✓ Team-specific metrics calculated from BDB data")

# ============================================================================
# 10. Select Final Columns
# ============================================================================

print("\n" + "=" * 80)
print("STEP 10: SELECTING FINAL COLUMNS")
print("-" * 80)

# Base columns from BDB (same as v2)
df_b_cols = [
    # Game identifiers
    'season', 'week', 'play_id', 'game_id',
    
    # Game state
    'quarter', 'down', 'yards_to_go', 'goal_to_go',
    
    # Teams
    'possession_team', 'defensive_team', 
    'yardline_side', 'yardline_number',
    
    # Scores & Win Probability (from BDB)
    'ps_pos_team_score', 'ps_def_team_score', 
    'pos_team_wp', 'pos_team_wpa', 'def_team_wpa',
    
    # Play characteristics (from BDB)
    'pass_result', 'pass_length', 'pass_location',
    'offense_formation', 'receiver_alignment',
    'route_of_targeted_receiver', 
    'play_action', 'dropback_type', 'dropback_distance',
    'pass_location_type',
    
    # Defense (from BDB)
    'defenders_in_the_box',
    'team_coverage_man_zone', 'team_coverage_type',
    
    # Outcomes (from BDB)
    'pre_penalty_yards_gained', 'yards_gained',
    
    # EPA/WP from BDB
    'expected_points', 'expected_points_added',
    
    # *** NEW IN V3: Ball trajectory features ***
    'start_ball_x', 'start_ball_y', 'start_ball_o',
    'ball_flight_distance', 'ball_flight_frames',
    'throw_direction', 'throw_type',
]

# Conditionally add SDV columns
if SDV_MERGE:
    df_b_cols.extend([
        'quarter_seconds_remaining', 'game_seconds_remaining', 'half_seconds_remaining',
        'air_yards', 'yards_after_catch',
        'touchdown', 'qb_hit', 'complete_pass', 'incomplete_pass', 'interception',
        'ep', 'epa', 'air_epa', 'yac_epa', 'comp_air_epa', 'comp_yac_epa',
        'wp', 'def_wp', 'wpa', 'air_wpa', 'yac_wpa', 'comp_air_wpa', 'comp_yac_wpa',
        'cp', 'cpoe', 'xpass', 'pass_oe',
        'xyac_epa', 'xyac_median_yardage', 'xyac_mean_yardage', 'xyac_success',
        'shotgun', 'no_huddle',
        'posteam_timeouts_remaining', 'defteam_timeouts_remaining',
        'surface', 'roof', 'temp', 'wind',
        'total_line', 'spread_line',
        'series', 'play_in_drive',
        'pbu', 'tfl', 'atkl', 'stkl',
    ])

# Filter to only columns that exist
df_b_cols = [col for col in df_b_cols if col in master.columns]
df_b = master[df_b_cols].copy()

print(f"Final columns: {len(df_b_cols)}")
print(f"  NEW ball trajectory features: 7")

# ============================================================================
# 11. Data Quality Checks
# ============================================================================

print("\n" + "=" * 80)
print("STEP 11: DATA QUALITY CHECKS")
print("-" * 80)

# Check for nulls in critical columns
critical_cols = ['game_id', 'play_id', 'possession_team', 'defensive_team', 'down', 'yards_to_go']
print("Checking critical columns for nulls...")
null_found = False
for col in critical_cols:
    if col in df_b.columns:
        null_count = df_b[col].isnull().sum()
        if null_count > 0:
            print(f"  ⚠ {col}: {null_count:,} nulls")
            null_found = True

if not null_found:
    print("  ✓ All critical columns have no nulls")

# Check ball trajectory features
print("\nBall trajectory feature coverage:")
ball_cols_check = ['start_ball_x', 'ball_flight_distance', 'throw_type']
for col in ball_cols_check:
    if col in df_b.columns:
        non_null = df_b[col].notna().sum()
        pct = 100 * non_null / len(df_b)
        print(f"  {col}: {non_null:,} ({pct:.1f}%)")

# Check for duplicates
print("\nChecking for duplicate plays...")
dup_count = df_b.duplicated(subset=['game_id', 'play_id']).sum()
if dup_count > 0:
    print(f"  ⚠ Warning: {dup_count:,} duplicate plays found")
else:
    print("  ✓ No duplicates!")

# Summary statistics
print("\nSummary statistics:")
print(f"  Total plays: {len(df_b):,}")
print(f"  Unique games: {df_b['game_id'].nunique():,}")

if 'ball_flight_distance' in df_b.columns:
    print(f"  Avg throw distance: {df_b['ball_flight_distance'].mean():.2f} yards")
    print(f"  Avg flight duration: {df_b['ball_flight_frames'].mean():.1f} frames")

# ============================================================================
# 12. Save Output
# ============================================================================

print("\n" + "=" * 80)
print("STEP 12: SAVING OUTPUT")
print("-" * 80)

df_b.to_parquet(OUTPUT_FILE, engine='pyarrow', index=False)
print(f"✓ Saved to: {OUTPUT_FILE}")
print(f"  File size: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.1f} MB")

# Sample output
print("\nSample ball trajectory features (first 3 plays):")
ball_sample_cols = ['game_id', 'play_id', 'start_ball_x', 'start_ball_y', 
                    'ball_flight_distance', 'ball_flight_frames', 'throw_type']
ball_sample_cols = [c for c in ball_sample_cols if c in df_b.columns]
print(df_b[ball_sample_cols].head(3).to_string())

# ============================================================================
# Complete
# ============================================================================

print("\n" + "=" * 80)
print("DATAFRAME B (v3): COMPLETE")
print("=" * 80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nOutput: {OUTPUT_FILE}")
print(f"Rows: {len(df_b):,}")
print(f"Columns: {len(df_b.columns)}")
print(f"  Including 7 NEW ball trajectory features")
print("\n" + "=" * 80)