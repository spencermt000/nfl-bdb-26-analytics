"""
Dataframe B: Play-Level Features (v2 - Simplified)
===================================================
Creates play-level contextual features for each play.

INPUTS:
  - data/supplementary_data.csv (BDB supplementary data)
  - data/sdv_raw_pbp.parquet (nflfastR play-by-play data)

OUTPUTS:
  - outputs/dataframe_b/v2.parquet

FEATURES INCLUDED:
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

NOTE: SumerSports coverage data removed - frame-level data doesn't fit play-level table.
      Coverage info from BDB supplementary data is included instead.

CHANGELOG v1 -> v2:
  - Retained MORE nflfastR features (EPA breakdowns, xYAC metrics)
  - Better handling of missing values
  - Fixed game_id merging issues
  - Removed SumerSports frame data (not applicable at play level)
  - Improved documentation and progress tracking
  - Added data quality checks
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

print("=" * 80)
print("DATAFRAME B (v2): PLAY-LEVEL FEATURES")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# File paths
INPUT_BDB_SUPP = 'data/supplementary_data.csv'
INPUT_NFLFASTR = 'data/sdv_raw_pbp.parquet'
OUTPUT_DIR = 'outputs/dataframe_b'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'v2.parquet')

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

print("\nLoading nflfastR play-by-play data...")
sdv_pbp = pd.read_parquet(INPUT_NFLFASTR)
print(f"  ✓ Loaded {len(sdv_pbp):,} rows")
print(f"  Columns: {len(sdv_pbp.columns)}")

# ============================================================================
# 2. Create Index Columns for Merging
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: CREATING INDEX COLUMNS")
print("-" * 80)

# Convert BDB game_id to string for merging
sup_raw['game_id_str'] = sup_raw['game_id'].astype(str)
sup_raw['index'] = sup_raw['game_id_str'] + '_' + sup_raw['play_id'].astype(str)
print(f"Created BDB index: {sup_raw['index'].nunique():,} unique plays")

# nflfastR uses old_game_id (already string) + play_id
sdv_pbp['index'] = sdv_pbp['old_game_id'].astype(str) + '_' + sdv_pbp['play_id'].astype(str)
print(f"Created nflfastR index: {sdv_pbp['index'].nunique():,} unique plays")

# Check overlap
overlap = len(set(sup_raw['index']) & set(sdv_pbp['index']))
print(f"\nOverlap check: {overlap:,} plays match")

if overlap == 0:
    print("  ⚠ WARNING: No plays match - check game_id formats!")
else:
    print(f"  ✓ Good merge expected: {100*overlap/len(sup_raw):.1f}% of BDB plays should match")

# ============================================================================
# 3. Calculate Derived Features in nflfastR Data
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: CALCULATING DERIVED FEATURES")
print("-" * 80)

# Play number in drive
print("Calculating play_in_drive...")
sdv_pbp['play_in_drive'] = sdv_pbp.groupby(['game_id', 'posteam', 'drive']).cumcount() + 1
print(f"  ✓ Plays per drive (avg): {sdv_pbp['play_in_drive'].mean():.1f}")

# Binary defensive stats
print("\nCreating defensive stat indicators...")
sdv_pbp['tfl'] = np.where(
    sdv_pbp['tackle_for_loss_1_player_id'].notna() | sdv_pbp['tackle_for_loss_2_player_id'].notna(),
    1, 0
)
print(f"  Tackles for loss: {sdv_pbp['tfl'].sum():,} plays")

sdv_pbp['pbu'] = np.where(
    (sdv_pbp['pass_defense_1_player_id'].notna() | sdv_pbp['pass_defense_2_player_id'].notna()) | 
    (sdv_pbp['interception'] == 1),
    1, 0
)
print(f"  Pass breakups: {sdv_pbp['pbu'].sum():,} plays")

sdv_pbp['atkl'] = np.where(
    sdv_pbp['solo_tackle_1_player_id'].notna() | sdv_pbp['solo_tackle_2_player_id'].notna(),
    1, 0
)
print(f"  Solo tackles: {sdv_pbp['atkl'].sum():,} plays")

sdv_pbp['stkl'] = np.where(
    sdv_pbp['assist_tackle_1_player_id'].notna() | 
    sdv_pbp['assist_tackle_2_player_id'].notna() | 
    sdv_pbp['assist_tackle_3_player_id'].notna() | 
    sdv_pbp['assist_tackle_4_player_id'].notna(),
    1, 0
)
print(f"  Assist tackles: {sdv_pbp['stkl'].sum():,} plays")

# ============================================================================
# 4. Select Features from nflfastR
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: SELECTING NFLFASTR FEATURES")
print("-" * 80)

# Expanded feature set - keeping MORE than v1
sdv_cols = [
    # Time & game state
    "quarter_seconds_remaining", "game_seconds_remaining", "half_seconds_remaining",
    "goal_to_go", "shotgun", "no_huddle", 
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    
    # Pass characteristics
    "air_yards", "yards_after_catch", "pass_length", "pass_location",
    "qb_hit", "touchdown", 
    
    # EPA metrics (EXPANDED from v1)
    "ep", "epa",
    "air_epa", "yac_epa", 
    "comp_air_epa", "comp_yac_epa",
    
    # Win probability metrics (EXPANDED from v1)
    "wp", "def_wp", "wpa",
    "air_wpa", "yac_wpa",
    "comp_air_wpa", "comp_yac_wpa",
    
    # Expected metrics
    "cp", "cpoe", "xpass", "pass_oe",
    "xyac_epa", "xyac_median_yardage", "xyac_mean_yardage", "xyac_success",
    
    # Field conditions
    "surface", "roof", "temp", "wind",
    
    # Betting lines
    "total_line", "spread_line",
    
    # Drive context
    "series", "play_in_drive",
    
    # Defensive stats
    "tfl", "pbu", "atkl", "stkl",
    
    # Completion/result
    "complete_pass", "incomplete_pass", "interception",
    
    # Index
    "index"
]

# Filter to only columns that exist
sdv_cols = [col for col in sdv_cols if col in sdv_pbp.columns]
sdv_subset = sdv_pbp[sdv_cols].copy()

print(f"Selected {len(sdv_cols)} columns from nflfastR data")

# ============================================================================
# 5. Merge BDB and nflfastR Data
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: MERGING BDB AND NFLFASTR DATA")
print("-" * 80)

initial_len = len(sup_raw)
master = sup_raw.merge(sdv_subset, on='index', how='left')
print(f"Merged: {len(master):,} rows (same as BDB: {len(master) == initial_len})")

# Check merge quality
merge_quality = master['epa'].notna().sum() / len(master) * 100
print(f"  Merge quality: {merge_quality:.1f}% of plays have nflfastR data")

# ============================================================================
# 6. Handle Missing Values
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: HANDLING MISSING VALUES")
print("-" * 80)

print("Filling missing values with appropriate defaults...")

# YAC-related: 0 if incomplete
master['yards_after_catch'] = master['yards_after_catch'].fillna(0)
master['xyac_epa'] = master['xyac_epa'].fillna(0)
master['xyac_median_yardage'] = master['xyac_median_yardage'].fillna(0)
master['xyac_mean_yardage'] = master['xyac_mean_yardage'].fillna(0)
master['xyac_success'] = master['xyac_success'].fillna(0)
master['yac_epa'] = master['yac_epa'].fillna(0)
master['comp_yac_epa'] = master['comp_yac_epa'].fillna(0)
master['yac_wpa'] = master['yac_wpa'].fillna(0)
master['comp_yac_wpa'] = master['comp_yac_wpa'].fillna(0)

# Air yards: use pass_length as fallback
if 'air_yards' in master.columns and 'pass_length' in master.columns:
    master['air_yards'] = master['air_yards'].fillna(master['pass_length'])

# EPA: 0 for missing
master['air_epa'] = master['air_epa'].fillna(0)
master['comp_air_epa'] = master['comp_air_epa'].fillna(0)
master['air_wpa'] = master['air_wpa'].fillna(0)
master['comp_air_wpa'] = master['comp_air_wpa'].fillna(0)

# Binary flags: 0 for missing
for col in ['tfl', 'pbu', 'atkl', 'stkl', 'complete_pass', 'incomplete_pass', 'interception']:
    if col in master.columns:
        master[col] = master[col].fillna(0)

print("  ✓ Missing value handling complete")

# ============================================================================
# 7. Create Team-Specific Metrics
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: CREATING TEAM-SPECIFIC METRICS")
print("-" * 80)

print("Calculating possession team win probability...")
master['pos_team_wp'] = master.apply(
    lambda row: row['pre_snap_home_team_win_probability'] 
    if row['possession_team'] == row['home_team_abbr'] 
    else row['pre_snap_visitor_team_win_probability'], 
    axis=1
)

print("Calculating possession team WPA...")
master['pos_team_wpa'] = master.apply(
    lambda row: row['home_team_win_probability_added'] 
    if row['possession_team'] == row['home_team_abbr'] 
    else row['visitor_team_win_probility_added'], 
    axis=1
)

print("Calculating defensive team WPA...")
master['def_team_wpa'] = master.apply(
    lambda row: row['visitor_team_win_probility_added'] 
    if row['possession_team'] == row['home_team_abbr'] 
    else row['home_team_win_probability_added'], 
    axis=1
)

print("Calculating team scores...")
master['ps_pos_team_score'] = master.apply(
    lambda row: row['pre_snap_home_score'] 
    if row['possession_team'] == row['home_team_abbr'] 
    else row['pre_snap_visitor_score'], 
    axis=1
)

master['ps_def_team_score'] = master.apply(
    lambda row: row['pre_snap_visitor_score'] 
    if row['possession_team'] == row['home_team_abbr'] 
    else row['pre_snap_home_score'], 
    axis=1
)

print("  ✓ Team-specific metrics calculated")

# ============================================================================
# 8. Select Final Columns
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: SELECTING FINAL COLUMNS")
print("-" * 80)

df_b_cols = [
    # Game identifiers
    'season', 'week', 'play_id', 'game_id',
    
    # Game state
    'quarter', 'down', 'yards_to_go',
    'quarter_seconds_remaining', 'game_seconds_remaining', 'half_seconds_remaining',
    'goal_to_go',
    
    # Teams
    'possession_team', 'defensive_team', 
    'yardline_side', 'yardline_number',
    
    # Scores & Win Probability
    'ps_pos_team_score', 'ps_def_team_score', 
    'pos_team_wp', 'pos_team_wpa', 'def_team_wpa',
    
    # Play characteristics
    'pass_result', 'pass_length', 'pass_location',
    'offense_formation', 'receiver_alignment',
    'route_of_targeted_receiver', 
    'play_action', 'dropback_type', 'dropback_distance',
    'pass_location_type',
    
    # Defense (from BDB)
    'defenders_in_the_box',
    'team_coverage_man_zone', 'team_coverage_type',
    
    # Outcomes
    'pre_penalty_yards_gained', 'yards_gained',
    'air_yards', 'yards_after_catch',
    'touchdown', 'qb_hit', 'complete_pass', 'incomplete_pass', 'interception',
    
    # EPA metrics
    'expected_points', 'expected_points_added',
    'ep', 'epa',
    'air_epa', 'yac_epa',
    'comp_air_epa', 'comp_yac_epa',
    
    # Win probability metrics
    'wp', 'def_wp', 'wpa',
    'air_wpa', 'yac_wpa',
    'comp_air_wpa', 'comp_yac_wpa',
    
    # Expected metrics
    'cp', 'cpoe', 'xpass', 'pass_oe',
    'xyac_epa', 'xyac_median_yardage', 'xyac_mean_yardage', 'xyac_success',
    
    # Situational
    'shotgun', 'no_huddle',
    'posteam_timeouts_remaining', 'defteam_timeouts_remaining',
    
    # Field conditions
    'surface', 'roof', 'temp', 'wind',
    
    # Betting
    'total_line', 'spread_line',
    
    # Drive context
    'series', 'play_in_drive',
    
    # Defensive stats
    'pbu', 'tfl', 'atkl', 'stkl',
]

# Filter to only columns that exist
df_b_cols = [col for col in df_b_cols if col in master.columns]

df_b = master[df_b_cols].copy()

print(f"Final columns: {len(df_b_cols)}")

# ============================================================================
# 9. Data Quality Checks
# ============================================================================

print("\n" + "=" * 80)
print("STEP 9: DATA QUALITY CHECKS")
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
print(f"  Weeks: {df_b['week'].nunique() if 'week' in df_b.columns else 'N/A'}")

if 'epa' in df_b.columns and df_b['epa'].notna().sum() > 0:
    print(f"  Average EPA: {df_b['epa'].mean():.3f}")
    print(f"  Plays with EPA data: {df_b['epa'].notna().sum():,} ({100*df_b['epa'].notna().sum()/len(df_b):.1f}%)")
else:
    print(f"  ⚠ No EPA data available")

if 'complete_pass' in df_b.columns and df_b['complete_pass'].notna().sum() > 0:
    completion_rate = df_b['complete_pass'].sum() / (df_b['complete_pass'].sum() + df_b['incomplete_pass'].sum())
    print(f"  Completion rate: {completion_rate:.1%}")
else:
    print(f"  ⚠ No completion data available")

# ============================================================================
# 10. Save Output
# ============================================================================

print("\n" + "=" * 80)
print("STEP 10: SAVING OUTPUT")
print("-" * 80)

df_b.to_parquet(OUTPUT_FILE, engine='pyarrow', index=False)
print(f"✓ Saved to: {OUTPUT_FILE}")
print(f"  File size: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.1f} MB")

# Sample output
print("\nSample output (first 3 rows):")
print(df_b.head(3).to_string())

# ============================================================================
# Complete
# ============================================================================

print("\n" + "=" * 80)
print("DATAFRAME B (v2): COMPLETE")
print("=" * 80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nOutput: {OUTPUT_FILE}")
print(f"Rows: {len(df_b):,}")
print(f"Columns: {len(df_b.columns)}")
print("\n" + "=" * 80)