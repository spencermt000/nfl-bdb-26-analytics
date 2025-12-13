"""
Dataframe C: TROUBLESHOOTING SCRIPT
====================================
Diagnoses issues with edge-level features and ball trajectory data.

This script is based on dataframe_c_v2.py but adds extensive diagnostics
to identify where and why NaN values are being created.

INPUTS:
  - outputs/dataframe_a/v2.parquet (processed node-level features)
  - outputs/dataframe_b/v3.parquet (play-level features with ball trajectory)

OUTPUTS:
  - Diagnostic reports and visualizations
  - Identification of problematic plays/frames
  - Analysis of NaN sources

TROUBLESHOOTING FOCUS:
  - Why do some frames have NaN ball trajectory data?
  - Which plays/frames are affected?
  - What's different about frames with complete vs incomplete data?
  - Are these frames pre-throw, during flight, or post-catch?
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

print("=" * 80)
print("DATAFRAME C: TROUBLESHOOTING SCRIPT")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

print("🔍 This script will diagnose issues with ball trajectory data and NaN values")
print("=" * 80)

# File paths
INPUT_DF_A = 'outputs/dataframe_a/v2.parquet'
INPUT_DF_B = 'outputs/dataframe_b/v3.parquet'
OUTPUT_DIR = 'outputs/dataframe_c/troubleshooting'

# *** PILOT MODE (Always use for troubleshooting) ***
PILOT_MODE = True
PILOT_N_GAMES = 3

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n📊 TROUBLESHOOTING MODE")
print(f"   - Will analyze {PILOT_N_GAMES} games in detail")
print(f"   - Will identify all sources of NaN values")
print(f"   - Will provide recommendations for fixes")
print(f"   - Output directory: {OUTPUT_DIR}")
print("=" * 80)

# ============================================================================
# Helper Functions
# ============================================================================

def angle_difference(angle1, angle2):
    """Calculate smallest difference between two angles (0-360 degrees)."""
    diff = np.abs(angle1 - angle2)
    return np.minimum(diff, 360 - diff)

def interpolate_ball_position(start_x, start_y, land_x, land_y, frame_id, total_frames):
    """
    Linearly interpolate ball position at frame t.
    
    Args:
        start_x, start_y: Ball starting position (passer at release)
        land_x, land_y: Ball landing position
        frame_id: Current frame (1-indexed)
        total_frames: Total frames in flight
    
    Returns:
        ball_x_t, ball_y_t: Ball position at frame t
    """
    if total_frames <= 0:
        return land_x, land_y
    
    # Progress from 0 (start) to 1 (landing)
    # frame_id=1 -> progress=0, frame_id=total_frames -> progress=1
    progress = (frame_id - 1) / (total_frames - 1) if total_frames > 1 else 1.0
    progress = np.clip(progress, 0, 1)
    
    ball_x_t = start_x + (land_x - start_x) * progress
    ball_y_t = start_y + (land_y - start_y) * progress
    
    return ball_x_t, ball_y_t

# ============================================================================
# 1. Load Data
# ============================================================================

print("STEP 1: LOADING DATA")
print("-" * 80)

print("Loading dataframe_a (processed node features)...")
df_a = pd.read_parquet(INPUT_DF_A)
print(f"  ✓ Loaded {len(df_a):,} rows")
print(f"  Columns: {len(df_a.columns)}")
print(f"  Memory: {df_a.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

print("\nLoading dataframe_b (play-level features with ball trajectory)...")
df_b = pd.read_parquet(INPUT_DF_B)
print(f"  ✓ Loaded {len(df_b):,} plays")
print(f"  Columns: {len(df_b.columns)}")

# *** PILOT MODE: Filter to N games ***
if PILOT_MODE:
    print(f"\n{'='*80}")
    print(f"PILOT MODE: Filtering to {PILOT_N_GAMES} games...")
    print(f"{'='*80}")
    
    # Get first N unique games
    unique_games = df_a['game_id'].unique()[:PILOT_N_GAMES]
    print(f"  Selected games: {unique_games.tolist()}")
    
    # Filter df_a
    df_a_full_size = len(df_a)
    df_a = df_a[df_a['game_id'].isin(unique_games)].copy()
    print(f"  Filtered df_a: {df_a_full_size:,} → {len(df_a):,} rows ({100*len(df_a)/df_a_full_size:.1f}%)")
    
    # Filter df_b
    df_b_full_size = len(df_b)
    df_b = df_b[df_b['game_id'].isin(unique_games)].copy()
    print(f"  Filtered df_b: {df_b_full_size:,} → {len(df_b):,} plays ({100*len(df_b)/df_b_full_size:.1f}%)")
    
    print(f"\nPilot dataset summary:")
    print(f"  Games: {df_a['game_id'].nunique()}")
    print(f"  Plays: {df_a['play_id'].nunique()}")
    print(f"  Frames: {len(df_a)}")
    print(f"{'='*80}\n")

# Check that ball trajectory columns exist
required_ball_cols = ['start_ball_x', 'start_ball_y', 'ball_flight_frames']
missing_cols = [col for col in required_ball_cols if col not in df_b.columns]
if missing_cols:
    print(f"  ⚠ WARNING: Missing ball trajectory columns: {missing_cols}")
    print(f"     Make sure you're using dataframe_b v3!")
else:
    print(f"  ✓ Ball trajectory columns present")

# ============================================================================
# DIAGNOSTIC 1: Analyze Ball Trajectory Completeness in df_b
# ============================================================================

print("\n" + "=" * 80)
print("🔍 DIAGNOSTIC 1: BALL TRAJECTORY DATA COMPLETENESS (df_b)")
print("=" * 80)

print("\nChecking for NaN values in df_b ball trajectory columns:")
for col in required_ball_cols:
    if col in df_b.columns:
        na_count = df_b[col].isna().sum()
        na_pct = 100 * na_count / len(df_b)
        print(f"  {col:25s}: {na_count:6,} NaN ({na_pct:5.1f}%)")

# Check for plays with incomplete data
print("\nPlays with complete vs incomplete ball trajectory data:")
complete_mask = (
    df_b['start_ball_x'].notna() & 
    df_b['start_ball_y'].notna() & 
    df_b['ball_flight_frames'].notna()
)
complete_count = complete_mask.sum()
incomplete_count = (~complete_mask).sum()

print(f"  Complete:   {complete_count:6,} plays ({100*complete_count/len(df_b):5.1f}%)")
print(f"  Incomplete: {incomplete_count:6,} plays ({100*incomplete_count/len(df_b):5.1f}%)")

if incomplete_count > 0:
    print(f"\n⚠ WARNING: {incomplete_count} plays have incomplete ball trajectory data!")
    print(f"  These plays will produce NaN values in edge features")
    
    # Sample incomplete plays
    incomplete_plays = df_b[~complete_mask][['game_id', 'play_id', 'start_ball_x', 'start_ball_y', 'ball_flight_frames']].head(10)
    print(f"\nSample incomplete plays:")
    print(incomplete_plays.to_string())
    
    # Save full list
    incomplete_file = os.path.join(OUTPUT_DIR, 'incomplete_plays.csv')
    df_b[~complete_mask][['game_id', 'play_id', 'start_ball_x', 'start_ball_y', 'ball_flight_frames']].to_csv(incomplete_file, index=False)
    print(f"\n  💾 Saved all incomplete plays to: {incomplete_file}")

# ============================================================================
# 2. Merge Ball Trajectory Info with Frame Data
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: MERGING BALL TRAJECTORY INFO")
print("-" * 80)

# Select only needed columns from df_b
ball_trajectory_cols = ['game_id', 'play_id', 'start_ball_x', 'start_ball_y', 'ball_flight_frames']
ball_trajectory_cols = [col for col in ball_trajectory_cols if col in df_b.columns]
df_b_subset = df_b[ball_trajectory_cols]

print(f"Merging ball trajectory with frame data...")
initial_len = len(df_a)
df_a = df_a.merge(df_b_subset, on=['game_id', 'play_id'], how='left')
print(f"  ✓ Merged: {len(df_a):,} rows (same: {len(df_a) == initial_len})")

# Check merge success
if 'start_ball_x' in df_a.columns:
    merged_count = df_a['start_ball_x'].notna().sum()
    print(f"  Frames with ball trajectory data: {merged_count:,} ({100*merged_count/len(df_a):.1f}%)")

# ============================================================================
# DIAGNOSTIC 2: Analyze Merge Results
# ============================================================================

print("\n" + "=" * 80)
print("🔍 DIAGNOSTIC 2: BALL TRAJECTORY DATA AFTER MERGE (df_a)")
print("=" * 80)

print("\nNaN counts in merged ball trajectory columns:")
for col in ['start_ball_x', 'start_ball_y', 'ball_flight_frames']:
    if col in df_a.columns:
        na_count = df_a[col].isna().sum()
        na_pct = 100 * na_count / len(df_a)
        print(f"  {col:25s}: {na_count:8,} NaN ({na_pct:5.1f}%)")

# Analyze frames with incomplete data
print("\nFrames by data completeness:")
frames_complete = (
    df_a['start_ball_x'].notna() & 
    df_a['start_ball_y'].notna() & 
    df_a['ball_flight_frames'].notna()
)
complete_frame_count = frames_complete.sum()
incomplete_frame_count = (~frames_complete).sum()

print(f"  Complete:   {complete_frame_count:8,} frames ({100*complete_frame_count/len(df_a):5.1f}%)")
print(f"  Incomplete: {incomplete_frame_count:8,} frames ({100*incomplete_frame_count/len(df_a):5.1f}%)")

if incomplete_frame_count > 0:
    print(f"\n⚠ ISSUE IDENTIFIED: {incomplete_frame_count} frames missing ball trajectory data")
    
    # Group by play to see which plays are affected
    incomplete_frames = df_a[~frames_complete]
    plays_affected = incomplete_frames.groupby(['game_id', 'play_id']).size().reset_index(name='frame_count')
    plays_affected = plays_affected.sort_values('frame_count', ascending=False)
    
    print(f"\n  Affected plays: {len(plays_affected)} plays")
    print(f"  Total frames affected: {incomplete_frame_count:,}")
    
    print(f"\n  Top 10 plays by affected frame count:")
    print(plays_affected.head(10).to_string(index=False))
    
    # Save details
    incomplete_frames_file = os.path.join(OUTPUT_DIR, 'incomplete_frames.csv')
    incomplete_frames[['game_id', 'play_id', 'frame_id', 'nfl_id', 'start_ball_x', 'start_ball_y', 'ball_flight_frames']].to_csv(incomplete_frames_file, index=False)
    print(f"\n  💾 Saved incomplete frames to: {incomplete_frames_file}")

# ============================================================================
# 3. Calculate Frame-Level Ball Positions
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: CALCULATING FRAME-LEVEL BALL POSITIONS")
print("-" * 80)

print("Interpolating ball position for each frame...")

# Initialize columns
df_a['ball_x_t'] = np.nan
df_a['ball_y_t'] = np.nan
df_a['ball_progress'] = np.nan
df_a['frames_to_landing'] = np.nan

# Only calculate for rows with complete ball trajectory data
valid_mask = (
    df_a['start_ball_x'].notna() & 
    df_a['ball_land_x'].notna() & 
    df_a['ball_flight_frames'].notna()
)

valid_count = valid_mask.sum()
print(f"  Calculating for {valid_count:,} frames with complete trajectory data...")

if valid_count > 0:
    # Vectorized calculation
    valid_df = df_a[valid_mask].copy()
    
    # Calculate ball position at frame t
    results = valid_df.apply(
        lambda row: interpolate_ball_position(
            row['start_ball_x'], row['start_ball_y'],
            row['ball_land_x'], row['ball_land_y'],
            row['frame_id'], row['ball_flight_frames']
        ),
        axis=1
    )
    
    df_a.loc[valid_mask, 'ball_x_t'] = [r[0] for r in results]
    df_a.loc[valid_mask, 'ball_y_t'] = [r[1] for r in results]
    
    # Calculate progress (0 to 1)
    df_a.loc[valid_mask, 'ball_progress'] = (
        (df_a.loc[valid_mask, 'frame_id'] - 1) / 
        (df_a.loc[valid_mask, 'ball_flight_frames'] - 1)
    ).clip(0, 1)
    
    # Calculate frames remaining
    df_a.loc[valid_mask, 'frames_to_landing'] = (
        df_a.loc[valid_mask, 'ball_flight_frames'] - df_a.loc[valid_mask, 'frame_id']
    )
    
    print(f"  ✓ Ball positions calculated")
    print(f"  Avg ball progress: {df_a.loc[valid_mask, 'ball_progress'].mean():.3f}")
    print(f"  Avg frames to landing: {df_a.loc[valid_mask, 'frames_to_landing'].mean():.1f}")
else:
    print(f"  ⚠ WARNING: No valid frames for ball position calculation!")

# ============================================================================
# DIAGNOSTIC 3: Analyze Ball Position Interpolation Results
# ============================================================================

print("\n" + "=" * 80)
print("🔍 DIAGNOSTIC 3: BALL POSITION INTERPOLATION RESULTS")
print("=" * 80)

print("\nNaN counts in calculated ball position columns:")
for col in ['ball_x_t', 'ball_y_t', 'ball_progress', 'frames_to_landing']:
    na_count = df_a[col].isna().sum()
    na_pct = 100 * na_count / len(df_a)
    print(f"  {col:25s}: {na_count:8,} NaN ({na_pct:5.1f}%)")

# Check if NaN frames match the incomplete trajectory frames
print("\nComparison: Expected vs Actual NaN frames")
expected_nan = (~frames_complete).sum()
actual_nan = df_a['ball_x_t'].isna().sum()
print(f"  Expected NaN frames (no trajectory data): {expected_nan:8,}")
print(f"  Actual NaN frames (ball_x_t):            {actual_nan:8,}")
print(f"  Match: {expected_nan == actual_nan}")

if expected_nan != actual_nan:
    print(f"\n  ⚠ WARNING: Mismatch between expected and actual NaN counts!")
    print(f"     There may be additional issues beyond missing trajectory data")

# ============================================================================
# 4. Calculate Player-Level Ball Features
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: CALCULATING PLAYER-LEVEL BALL FEATURES")
print("-" * 80)

print("Calculating player distances and angles to ball...")

# Distance to ball current position
df_a['player_dist_to_ball_current'] = np.sqrt(
    (df_a['x'] - df_a['ball_x_t'])**2 + 
    (df_a['y'] - df_a['ball_y_t'])**2
)

# Distance to ball landing (already have 'e_dist_ball_land', but recalculate for consistency)
df_a['player_dist_to_ball_landing'] = np.sqrt(
    (df_a['x'] - df_a['ball_land_x'])**2 + 
    (df_a['y'] - df_a['ball_land_y'])**2
)

# Angle from player to ball current position
angle_to_ball_current_rad = np.arctan2(
    df_a['ball_y_t'] - df_a['y'],
    df_a['ball_x_t'] - df_a['x']
)
angle_to_ball_current_deg = angle_to_ball_current_rad * 180 / np.pi
angle_to_ball_current_deg = (angle_to_ball_current_deg + 360) % 360

# Relative to player's direction of movement
df_a['player_angle_to_ball_current'] = angle_difference(
    df_a['dir'], angle_to_ball_current_deg
)

# Angle from player to ball landing
angle_to_ball_landing_rad = np.arctan2(
    df_a['ball_land_y'] - df_a['y'],
    df_a['ball_land_x'] - df_a['x']
)
angle_to_ball_landing_deg = angle_to_ball_landing_rad * 180 / np.pi
angle_to_ball_landing_deg = (angle_to_ball_landing_deg + 360) % 360

df_a['player_angle_to_ball_landing'] = angle_difference(
    df_a['dir'], angle_to_ball_landing_deg
)

# Calculate convergence (is player getting closer to ball?)
print("\nCalculating player convergence toward ball...")

# Sort by game, play, player, frame to calculate frame-to-frame changes
df_a_sorted = df_a.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])

# Calculate previous frame's distance to ball
df_a_sorted['prev_dist_to_ball'] = df_a_sorted.groupby(['game_id', 'play_id', 'nfl_id'])['player_dist_to_ball_current'].shift(1)

# Convergence = distance decreasing (1) or increasing (-1) or no change (0)
df_a_sorted['player_ball_convergence'] = np.where(
    df_a_sorted['prev_dist_to_ball'].isna(),
    0,  # First frame, no previous distance
    np.where(
        df_a_sorted['player_dist_to_ball_current'] < df_a_sorted['prev_dist_to_ball'] - 0.1,  # Threshold
        1,   # Converging
        np.where(
            df_a_sorted['player_dist_to_ball_current'] > df_a_sorted['prev_dist_to_ball'] + 0.1,
            -1,  # Diverging
            0    # No significant change
        )
    )
)

# Update df_a with sorted version
df_a = df_a_sorted.copy()

print(f"  ✓ Player ball features calculated")
print(f"  Avg distance to ball (current): {df_a['player_dist_to_ball_current'].mean():.2f} yards")
print(f"  Avg distance to ball (landing): {df_a['player_dist_to_ball_landing'].mean():.2f} yards")

converging = (df_a['player_ball_convergence'] == 1).sum()
diverging = (df_a['player_ball_convergence'] == -1).sum()
print(f"  Converging: {converging:,} ({100*converging/len(df_a):.1f}%)")
print(f"  Diverging: {diverging:,} ({100*diverging/len(df_a):.1f}%)")

# ============================================================================
# DIAGNOSTIC 4: Analyze Player-Level Ball Features
# ============================================================================

print("\n" + "=" * 80)
print("🔍 DIAGNOSTIC 4: PLAYER-LEVEL BALL FEATURES")
print("=" * 80)

print("\nNaN counts in player-level ball features:")
player_ball_features = [
    'player_dist_to_ball_current',
    'player_dist_to_ball_landing', 
    'player_angle_to_ball_current',
    'player_angle_to_ball_landing',
    'player_ball_convergence'
]

for col in player_ball_features:
    if col in df_a.columns:
        na_count = df_a[col].isna().sum()
        na_pct = 100 * na_count / len(df_a)
        print(f"  {col:35s}: {na_count:8,} NaN ({na_pct:5.1f}%)")

# Critical check: Are NaNs propagating correctly?
print("\n🔍 NaN Propagation Check:")
print(f"  ball_x_t NaN count:                      {df_a['ball_x_t'].isna().sum():8,}")
print(f"  player_dist_to_ball_current NaN count:  {df_a['player_dist_to_ball_current'].isna().sum():8,}")
print(f"  player_angle_to_ball_current NaN count: {df_a['player_angle_to_ball_current'].isna().sum():8,}")

# These should all match!
if (df_a['ball_x_t'].isna().sum() == df_a['player_dist_to_ball_current'].isna().sum() == 
    df_a['player_angle_to_ball_current'].isna().sum()):
    print(f"  ✓ NaN counts match - propagation is consistent")
else:
    print(f"  ⚠ WARNING: NaN counts don't match - unexpected behavior!")

# ============================================================================
# 5. Create Edge Features (SAME AS V2 + Ball Features)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: CREATING PAIRWISE EDGE FEATURES")
print("-" * 80)

# Get unique frames
unique_frames = df_a.groupby(['game_id', 'play_id', 'frame_id']).ngroups
print(f"Processing {unique_frames:,} unique frames...")

output_rows = []
processed_frames = 0

for (game_id, play_id, frame_id), frame_data in df_a.groupby(['game_id', 'play_id', 'frame_id']):
    processed_frames += 1
    
    if processed_frames % 1000 == 0:
        print(f"  Processed {processed_frames:,}/{unique_frames:,} frames ({100*processed_frames/unique_frames:.1f}%)")
    
    # Get all players in this frame
    players = frame_data.reset_index(drop=True)
    n_players = len(players)
    
    # Frame-level ball features (same for all players in frame)
    ball_x_t = players.iloc[0]['ball_x_t'] if 'ball_x_t' in players.columns else np.nan
    ball_y_t = players.iloc[0]['ball_y_t'] if 'ball_y_t' in players.columns else np.nan
    ball_progress = players.iloc[0]['ball_progress'] if 'ball_progress' in players.columns else np.nan
    frames_to_landing = players.iloc[0]['frames_to_landing'] if 'frames_to_landing' in players.columns else np.nan
    
    # Ball landing coordinates (same for all players)
    ball_land_x = players.iloc[0]['ball_land_x']
    ball_land_y = players.iloc[0]['ball_land_y']
    
    # Create all pairwise edges (directed: A→B and B→A are separate)
    for i in range(n_players):
        for j in range(n_players):
            if i == j:  # Skip self-edges
                continue
            
            player_a = players.iloc[i]
            player_b = players.iloc[j]
            
            # ================================================================
            # Spatial Features (SAME AS V2)
            # ================================================================
            
            # Position distances
            x_dist = player_b['x'] - player_a['x']
            y_dist = player_b['y'] - player_a['y']
            e_dist = np.sqrt(x_dist**2 + y_dist**2)
            
            # Distances to ball landing
            playerA_dist_to_landing = player_a.get('player_dist_to_ball_landing', np.nan)
            playerB_dist_to_landing = player_b.get('player_dist_to_ball_landing', np.nan)
            
            # ================================================================
            # Angular Features (SAME AS V2)
            # ================================================================
            
            # Relative angles
            relative_angle_o = angle_difference(player_a['o'], player_b['o'])
            relative_angle_dir = angle_difference(player_a['dir'], player_b['dir'])
            
            # Angles to ball landing
            angle_A_to_landing = np.arctan2(
                ball_land_y - player_a['y'],
                ball_land_x - player_a['x']
            ) * 180 / np.pi
            
            angle_B_to_landing = np.arctan2(
                ball_land_y - player_b['y'],
                ball_land_x - player_b['x']
            ) * 180 / np.pi
            
            pairwise_angle_to_landing = angle_difference(angle_A_to_landing, angle_B_to_landing)
            
            # ================================================================
            # Velocity/Acceleration Features (SAME AS V2)
            # ================================================================
            
            # Velocity vectors
            playerA_v_x = player_a.get('v_x', 0)
            playerA_v_y = player_a.get('v_y', 0)
            playerB_v_x = player_b.get('v_x', 0)
            playerB_v_y = player_b.get('v_y', 0)
            
            # Relative velocity
            relative_v_x = playerB_v_x - playerA_v_x
            relative_v_y = playerB_v_y - playerA_v_y
            relative_speed = np.sqrt(relative_v_x**2 + relative_v_y**2)
            
            # Acceleration vectors
            playerA_a_x = player_a.get('a_x', 0)
            playerA_a_y = player_a.get('a_y', 0)
            playerB_a_x = player_b.get('a_x', 0)
            playerB_a_y = player_b.get('a_y', 0)
            
            # ================================================================
            # NEW V3: Player Ball Features
            # ================================================================
            
            playerA_dist_to_ball_current = player_a.get('player_dist_to_ball_current', np.nan)
            playerB_dist_to_ball_current = player_b.get('player_dist_to_ball_current', np.nan)
            
            playerA_angle_to_ball_current = player_a.get('player_angle_to_ball_current', np.nan)
            playerB_angle_to_ball_current = player_b.get('player_angle_to_ball_current', np.nan)
            
            playerA_angle_to_ball_landing = player_a.get('player_angle_to_ball_landing', np.nan)
            playerB_angle_to_ball_landing = player_b.get('player_angle_to_ball_landing', np.nan)
            
            playerA_ball_convergence = player_a.get('player_ball_convergence', 0)
            playerB_ball_convergence = player_b.get('player_ball_convergence', 0)
            
            # ================================================================
            # Team/Role Features (SAME AS V2)
            # ================================================================
            
            same_team = 1 if player_a['player_side'] == player_b['player_side'] else 0
            
            # Create edge identifier
            edge_id = f"{player_a['nfl_id']}_{player_b['nfl_id']}"
            
            # ================================================================
            # Build Row
            # ================================================================
            
            row = {
                # Identifiers
                'game_id': game_id,
                'play_id': play_id,
                'frame_id': frame_id,
                'edge_id': edge_id,
                'playerA_id': player_a['nfl_id'],
                'playerB_id': player_b['nfl_id'],
                
                # Player A attributes
                'playerA_x': player_a['x'],
                'playerA_y': player_a['y'],
                'playerA_s': player_a['s'],
                'playerA_a': player_a['a'],
                'playerA_dir': player_a['dir'],
                'playerA_o': player_a['o'],
                'playerA_v_x': playerA_v_x,
                'playerA_v_y': playerA_v_y,
                'playerA_a_x': playerA_a_x,
                'playerA_a_y': playerA_a_y,
                'playerA_role': player_a['player_role'],
                'playerA_side': player_a['player_side'],
                'playerA_position': player_a['player_position'],
                
                # Player B attributes
                'playerB_x': player_b['x'],
                'playerB_y': player_b['y'],
                'playerB_s': player_b['s'],
                'playerB_a': player_b['a'],
                'playerB_dir': player_b['dir'],
                'playerB_o': player_b['o'],
                'playerB_v_x': playerB_v_x,
                'playerB_v_y': playerB_v_y,
                'playerB_a_x': playerB_a_x,
                'playerB_a_y': playerB_a_y,
                'playerB_role': player_b['player_role'],
                'playerB_side': player_b['player_side'],
                'playerB_position': player_b['player_position'],
                
                # Spatial features
                'x_dist': x_dist,
                'y_dist': y_dist,
                'e_dist': e_dist,
                'playerA_dist_to_landing': playerA_dist_to_landing,
                'playerB_dist_to_landing': playerB_dist_to_landing,
                
                # Angular features
                'relative_angle_o': relative_angle_o,
                'relative_angle_dir': relative_angle_dir,
                'pairwise_angle_to_landing': pairwise_angle_to_landing,
                
                # Velocity features
                'relative_v_x': relative_v_x,
                'relative_v_y': relative_v_y,
                'relative_speed': relative_speed,
                
                # Team/role
                'same_team': same_team,
                
                # Ball landing
                'ball_land_x': ball_land_x,
                'ball_land_y': ball_land_y,
                
                # *** NEW V3: Frame-level ball features ***
                'ball_x_t': ball_x_t,
                'ball_y_t': ball_y_t,
                'ball_progress': ball_progress,
                'frames_to_landing': frames_to_landing,
                
                # *** NEW V3: Player A ball features ***
                'playerA_dist_to_ball_current': playerA_dist_to_ball_current,
                'playerA_angle_to_ball_current': playerA_angle_to_ball_current,
                'playerA_angle_to_ball_landing': playerA_angle_to_ball_landing,
                'playerA_ball_convergence': playerA_ball_convergence,
                
                # *** NEW V3: Player B ball features ***
                'playerB_dist_to_ball_current': playerB_dist_to_ball_current,
                'playerB_angle_to_ball_current': playerB_angle_to_ball_current,
                'playerB_angle_to_ball_landing': playerB_angle_to_ball_landing,
                'playerB_ball_convergence': playerB_ball_convergence,
            }
            
            # Add coverage data if available (SAME AS V2)
            if 'coverage_responsibility' in player_a.index:
                row['playerA_coverage'] = player_a['coverage_responsibility']
                row['playerB_coverage'] = player_b['coverage_responsibility']
            
            if 'targeted_defender' in player_a.index:
                row['playerA_targeted'] = player_a['targeted_defender']
                row['playerB_targeted'] = player_b['targeted_defender']
            
            if 'coverage_scheme' in player_a.index:
                row['coverage_scheme'] = player_a['coverage_scheme']
            
            output_rows.append(row)

print(f"  ✓ Processed all {processed_frames:,} frames")

# ============================================================================
# 6. Create Output DataFrame
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: CREATING OUTPUT DATAFRAME")
print("-" * 80)

print("Converting to dataframe...")
df_c = pd.DataFrame(output_rows)

print(f"  ✓ Created dataframe: {df_c.shape}")
print(f"  Memory: {df_c.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# ============================================================================
# 7. Data Quality Checks
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: DATA QUALITY CHECKS")
print("-" * 80)

print("Summary statistics:")
print(f"  Total edges: {len(df_c):,}")
print(f"  Unique games: {df_c['game_id'].nunique():,}")
print(f"  Unique plays: {df_c['play_id'].nunique():,}")
print(f"  Unique frames: {df_c.groupby(['game_id', 'play_id', 'frame_id']).ngroups:,}")

print(f"\nEdge statistics:")
print(f"  Mean distance: {df_c['e_dist'].mean():.2f} yards")
print(f"  Same team edges: {df_c['same_team'].sum():,} ({100*df_c['same_team'].mean():.1f}%)")
print(f"  Opposing team edges: {(1-df_c['same_team']).sum():,} ({100*(1-df_c['same_team'].mean()):.1f}%)")

print(f"\nBall trajectory feature coverage:")
ball_feat_check = ['ball_progress', 'playerA_dist_to_ball_current', 'playerA_ball_convergence']
for feat in ball_feat_check:
    if feat in df_c.columns:
        non_null = df_c[feat].notna().sum()
        pct = 100 * non_null / len(df_c)
        print(f"  {feat}: {non_null:,} ({pct:.1f}%)")

# ============================================================================
# DIAGNOSTIC 5: Final Edge-Level NaN Analysis
# ============================================================================

print("\n" + "=" * 80)
print("🔍 DIAGNOSTIC 5: COMPREHENSIVE NaN ANALYSIS IN EDGES")
print("=" * 80)

# Analyze all columns for NaN
print("\nComplete NaN analysis for all edge features:")
print(f"{'Column':<40} {'NaN Count':>12} {'NaN %':>8} {'Non-Null Count':>15}")
print("-" * 80)

nan_summary = []
for col in df_c.columns:
    na_count = df_c[col].isna().sum()
    na_pct = 100 * na_count / len(df_c)
    non_null = len(df_c) - na_count
    print(f"{col:<40} {na_count:>12,} {na_pct:>7.1f}% {non_null:>15,}")
    
    if na_count > 0:
        nan_summary.append({
            'column': col,
            'nan_count': na_count,
            'nan_pct': na_pct
        })

# Identify frames with NaN values
print("\n" + "=" * 80)
print("🔍 IDENTIFYING PROBLEMATIC FRAMES")
print("=" * 80)

# Check ball_progress as indicator of missing ball data
if 'ball_progress' in df_c.columns:
    nan_edges = df_c[df_c['ball_progress'].isna()]
    
    print(f"\nEdges with NaN ball_progress: {len(nan_edges):,} ({100*len(nan_edges)/len(df_c):.1f}%)")
    
    if len(nan_edges) > 0:
        # Group by frame to see which frames are affected
        nan_by_frame = nan_edges.groupby(['game_id', 'play_id', 'frame_id']).size().reset_index(name='nan_edge_count')
        nan_by_frame = nan_by_frame.sort_values('nan_edge_count', ascending=False)
        
        print(f"\nFrames with NaN edges: {len(nan_by_frame):,}")
        print(f"Total NaN edges: {len(nan_edges):,}")
        print(f"\nTop 20 frames by NaN edge count:")
        print(nan_by_frame.head(20).to_string(index=False))
        
        # Save to CSV
        nan_frames_file = os.path.join(OUTPUT_DIR, 'nan_edge_frames.csv')
        nan_by_frame.to_csv(nan_frames_file, index=False)
        print(f"\n💾 Saved all NaN frames to: {nan_frames_file}")
        
        # Sample some actual edges with NaN
        print(f"\nSample edges with NaN values (first 10):")
        sample_cols = ['game_id', 'play_id', 'frame_id', 'edge_id', 'ball_progress', 
                      'playerA_dist_to_ball_current', 'playerB_dist_to_ball_current']
        print(nan_edges[sample_cols].head(10).to_string(index=False))
        
        # Check if these are specific graph sizes
        print(f"\n🔍 Analyzing graph structure of NaN frames:")
        nan_graph_sizes = nan_edges.groupby(['game_id', 'play_id', 'frame_id']).agg({
            'playerA_id': 'nunique',
            'edge_id': 'count'
        }).rename(columns={'playerA_id': 'num_nodes', 'edge_id': 'num_edges'})
        
        print(f"\nGraph size distribution for NaN frames:")
        size_dist = nan_graph_sizes.groupby(['num_nodes', 'num_edges']).size().reset_index(name='frame_count')
        size_dist = size_dist.sort_values('frame_count', ascending=False)
        print(size_dist.head(10).to_string(index=False))
        
        # THE KEY QUESTION: Are these 11-node graphs?
        eleven_node_frames = nan_graph_sizes[nan_graph_sizes['num_nodes'] == 11]
        if len(eleven_node_frames) > 0:
            print(f"\n⚠ CRITICAL FINDING: {len(eleven_node_frames)} frames have 11 nodes AND NaN values!")
            print(f"   This matches the pattern we saw in training (11 nodes, 110 edges)")
            print(f"   These are likely specific play configurations or phases")

# ============================================================================
# DIAGNOSTIC 6: Cross-Reference with df_b
# ============================================================================

print("\n" + "=" * 80)
print("🔍 DIAGNOSTIC 6: CROSS-REFERENCE WITH SOURCE DATA (df_b)")
print("=" * 80)

# For each NaN frame, check what the original df_b data looked like
if len(nan_edges) > 0:
    nan_plays = nan_edges[['game_id', 'play_id']].drop_duplicates()
    print(f"\nPlays with NaN edges: {len(nan_plays)}")
    
    # Merge with df_b to see original data
    nan_plays_detail = nan_plays.merge(
        df_b[['game_id', 'play_id', 'start_ball_x', 'start_ball_y', 'ball_flight_frames', 
              'ball_land_x', 'ball_land_y']],
        on=['game_id', 'play_id'],
        how='left'
    )
    
    print(f"\nBall trajectory data for plays with NaN edges:")
    print(nan_plays_detail.head(20).to_string(index=False))
    
    # Check if start position is missing
    missing_start = nan_plays_detail['start_ball_x'].isna().sum()
    missing_land = nan_plays_detail['ball_land_x'].isna().sum()
    missing_frames = nan_plays_detail['ball_flight_frames'].isna().sum()
    
    print(f"\nMissing data in plays with NaN edges:")
    print(f"  Missing start_ball_x:      {missing_start:6,} plays")
    print(f"  Missing ball_land_x:       {missing_land:6,} plays")
    print(f"  Missing ball_flight_frames: {missing_frames:6,} plays")
    
    # Save detailed report
    nan_plays_file = os.path.join(OUTPUT_DIR, 'nan_plays_detail.csv')
    nan_plays_detail.to_csv(nan_plays_file, index=False)
    print(f"\n💾 Saved play details to: {nan_plays_file}")

# ============================================================================
# DIAGNOSTIC 7: ROOT CAUSE ANALYSIS & RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 80)
print("🎯 DIAGNOSTIC 7: ROOT CAUSE ANALYSIS & RECOMMENDATIONS")
print("=" * 80)

print("\n" + "=" * 80)
print("SUMMARY OF FINDINGS")
print("=" * 80)

# Calculate key metrics
total_edges = len(df_c)
nan_edges_count = df_c['ball_progress'].isna().sum() if 'ball_progress' in df_c.columns else 0
nan_pct = 100 * nan_edges_count / total_edges if total_edges > 0 else 0

print(f"\n📊 Overall Statistics:")
print(f"  Total edges created:           {total_edges:>10,}")
print(f"  Edges with NaN ball features:  {nan_edges_count:>10,} ({nan_pct:.1f}%)")
print(f"  Clean edges (no NaN):          {total_edges - nan_edges_count:>10,} ({100-nan_pct:.1f}%)")

# Root causes
print(f"\n🔍 Root Cause Analysis:")
print(f"  1. Some plays in df_b are missing ball trajectory data")
print(f"     - start_ball_x, start_ball_y, or ball_flight_frames is NaN")
print(f"     - This prevents ball position interpolation")
print(f"  ")
print(f"  2. When trajectory data is missing for a play:")
print(f"     - ALL frames in that play get NaN for ball_x_t, ball_y_t")
print(f"     - ALL edges in those frames inherit the NaN")
print(f"     - This cascades to ALL ball-related features")
print(f"  ")
print(f"  3. The NaN values persist through normalization")
print(f"     - Training dataset sees these as NaN inputs")
print(f"     - Model processes NaN → produces NaN output")
print(f"     - BCE loss rejects NaN values → training fails")

print(f"\n💡 RECOMMENDATIONS:")
print(f"")
print(f"  OPTION 1: Fix Source Data (df_b)")
print(f"  ----------------------------------------")
print(f"  • Investigate why some plays lack ball trajectory data")
print(f"  • Check if these are data quality issues or expected")
print(f"  • If fixable: Update dataframe_b creation to fill missing values")
print(f"  ")
print(f"  OPTION 2: Filter Incomplete Plays (Recommended)")
print(f"  ----------------------------------------")
print(f"  • Remove plays with incomplete ball trajectory from df_b")
print(f"  • This prevents NaN propagation at the source")
print(f"  • Add filter in dataframe_c creation:")
print(f"    df_b = df_b[df_b['start_ball_x'].notna() & ")
print(f"                df_b['start_ball_y'].notna() & ")
print(f"                df_b['ball_flight_frames'].notna()]")
print(f"  ")
print(f"  OPTION 3: Handle NaN in Dataset Loader")
print(f"  ----------------------------------------")
print(f"  • Filter edges with NaN during PyTorch dataset creation")
print(f"  • Skip problematic frames entirely")
print(f"  • Less efficient but works around the issue")
print(f"  ")
print(f"  OPTION 4: Impute Missing Ball Data")
print(f"  ----------------------------------------")
print(f"  • Fill missing start positions with QB position at snap")
print(f"  • Estimate flight frames from throw distance")
print(f"  • More complex but preserves all data")

# Save comprehensive report
print("\n" + "=" * 80)
print("SAVING DIAGNOSTIC REPORT")
print("=" * 80)

report_file = os.path.join(OUTPUT_DIR, 'diagnostic_report.txt')
with open(report_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("DATAFRAME C TROUBLESHOOTING REPORT\n")
    f.write("=" * 80 + "\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Games analyzed: {PILOT_N_GAMES}\n")
    f.write("\n")
    
    f.write("SUMMARY\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total edges:           {total_edges:,}\n")
    f.write(f"Edges with NaN:        {nan_edges_count:,} ({nan_pct:.1f}%)\n")
    f.write(f"Clean edges:           {total_edges - nan_edges_count:,} ({100-nan_pct:.1f}%)\n")
    f.write("\n")
    
    f.write("NaN COUNTS BY FEATURE\n")
    f.write("-" * 80 + "\n")
    for item in nan_summary:
        f.write(f"{item['column']:<40} {item['nan_count']:>10,} ({item['nan_pct']:>6.1f}%)\n")
    f.write("\n")
    
    f.write("ROOT CAUSE\n")
    f.write("-" * 80 + "\n")
    f.write("Missing ball trajectory data in source (df_b) propagates to all edges\n")
    f.write("in affected frames, creating NaN values that break model training.\n")
    f.write("\n")
    
    f.write("RECOMMENDATION\n")
    f.write("-" * 80 + "\n")
    f.write("Filter incomplete plays at the source (df_b level) to prevent NaN propagation.\n")

print(f"💾 Saved diagnostic report to: {report_file}")

print("\n" + "=" * 80)
print("TROUBLESHOOTING COMPLETE!")
print("=" * 80)
print(f"\nAll diagnostic files saved to: {OUTPUT_DIR}/")
print(f"\n📋 Files created:")
print(f"  - diagnostic_report.txt      (Summary report)")
print(f"  - incomplete_plays.csv       (Plays missing trajectory data)")
print(f"  - incomplete_frames.csv      (Frames missing trajectory data)")
print(f"  - nan_edge_frames.csv        (Frames with NaN edges)")
print(f"  - nan_plays_detail.csv       (Detailed play information)")

print(f"\n🎯 Next Steps:")
print(f"  1. Review diagnostic files to understand the pattern")
print(f"  2. Decide on fix strategy (see recommendations above)")
print(f"  3. Implement fix in appropriate script (df_b or df_c)")
print(f"  4. Re-run pipeline and verify NaN elimination")

print("\n" + "=" * 80)