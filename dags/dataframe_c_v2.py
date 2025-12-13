"""
Dataframe C: Edge-Level Features (v3 - WITH BALL TRAJECTORY)
=============================================================
Creates pairwise edge features between all players in each frame.

INPUTS:
  - outputs/dataframe_a/v2.parquet (processed node-level features)
  - outputs/dataframe_b/v3.parquet (play-level features with ball trajectory)

OUTPUTS:
  - outputs/dataframe_c/v3.parquet

NEW IN V3:
  - Frame-level ball trajectory features:
    * ball_x_t (interpolated x position at frame t)
    * ball_y_t (interpolated y position at frame t)
    * ball_progress (0 to 1, how far along trajectory)
    * frames_to_landing (remaining frames)
  
  - Player-level ball features (for each player in each frame):
    * player_dist_to_ball_current (distance to ball at frame t)
    * player_dist_to_ball_landing (distance to landing spot)
    * player_angle_to_ball_current (relative to player's dir)
    * player_angle_to_ball_landing (relative to player's dir)
    * player_ball_convergence (is distance to ball decreasing?)

FEATURES FROM V2:
  - Pairwise distances (x, y, euclidean)
  - Relative angles (orientation, direction)
  - Same team indicator
  - Player attributes for both nodes
  - Coverage data (if available)
  - Velocity/acceleration vectors

CHANGELOG v2 -> v3:
  - Added 4 frame-level ball trajectory features
  - Added 5 player-level ball interaction features per player
  - Now loads dataframe_b to get ball start position and flight frames
  - Calculates ball position interpolation for each frame
  - Tracks player convergence toward ball
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

print("=" * 80)
print("DATAFRAME C (v3): EDGE-LEVEL FEATURES + BALL TRAJECTORY")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# File paths
INPUT_DF_A = 'outputs/dataframe_a/v2.parquet'
INPUT_DF_B = 'outputs/dataframe_b/v3.parquet'  # NEW: for ball trajectory
OUTPUT_DIR = 'outputs/dataframe_c'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'v3.parquet')

# *** PILOT MODE ***
PILOT_MODE = True  # Set to True to process only 3 games for testing
PILOT_N_GAMES = 3  # Number of games to process in pilot mode

# Processing settings
CHUNK_SIZE = 1000  # Process 1000 frames at a time

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

if PILOT_MODE:
    print("\n" + "⚠" * 40)
    print("PILOT MODE ENABLED".center(80))
    print(f"Will process only {PILOT_N_GAMES} games for testing".center(80))
    print("Set PILOT_MODE = False for full processing".center(80))
    print("⚠" * 40 + "\n")

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
# 8. Save Output
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: SAVING OUTPUT")
print("-" * 80)

# Adjust output filename for pilot mode
if PILOT_MODE:
    output_file_final = OUTPUT_FILE.replace('v3.parquet', f'v3_pilot_{PILOT_N_GAMES}games.parquet')
    print(f"PILOT MODE: Saving to pilot filename...")
else:
    output_file_final = OUTPUT_FILE

print("Writing to parquet...")
df_c.to_parquet(output_file_final, engine='pyarrow', index=False, compression='snappy')
print(f"✓ Saved to: {output_file_final}")
print(f"  File size: {os.path.getsize(output_file_final) / 1024 / 1024:.1f} MB")

print("\nSample output (first 2 rows):")
print(df_c.head(2).to_string())

# ============================================================================
# Complete
# ============================================================================

print("\n" + "=" * 80)
if PILOT_MODE:
    print("DATAFRAME C (v3): COMPLETE [PILOT MODE]")
    print("=" * 80)
    print(f"⚠ Processed {PILOT_N_GAMES} games only")
    print(f"⚠ Set PILOT_MODE = False for full processing")
else:
    print("DATAFRAME C (v3): COMPLETE")
    print("=" * 80)

print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nOutput: {output_file_final}")
print(f"Rows: {len(df_c):,}")
print(f"Columns: {len(df_c.columns)}")
print(f"  NEW ball trajectory features:")
print(f"    - 4 frame-level: ball_x_t, ball_y_t, ball_progress, frames_to_landing")
print(f"    - 8 player-level (×2 players): dist/angle to ball current/landing, convergence")
print("\n" + "=" * 80)