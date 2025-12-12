"""
Dataframe C: Edge-Level Features (v2)
======================================
Creates pairwise edge features between all players in each frame.

INPUTS:
  - outputs/dataframe_a/v2.parquet (processed node-level features)

OUTPUTS:
  - outputs/dataframe_c/v2.parquet

FEATURES:
  - Pairwise distances (x, y, euclidean)
  - Relative angles (orientation, direction)
  - Same team indicator
  - Player attributes for both nodes
  - Ball landing distances for both players
  - Angle to ball landing spot
  - Coverage data (if available)
  - Velocity/acceleration vectors

CHANGELOG v1 -> v2:
  - Now uses dataframe_a output (standardized coordinates)
  - Includes velocity and acceleration vectors
  - Includes coverage data from both SumerSports datasets
  - More efficient (single parquet instead of 18 CSVs)
  - Better documentation and progress tracking
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

print("=" * 80)
print("DATAFRAME C (v2): EDGE-LEVEL FEATURES")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# File paths
INPUT_DF_A = 'outputs/dataframe_a/v2.parquet'
OUTPUT_DIR = 'outputs/dataframe_c'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'v2.parquet')

# Processing settings
CHUNK_SIZE = 1000  # Process 1000 frames at a time

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Helper Functions
# ============================================================================

def angle_difference(angle1, angle2):
    """Calculate smallest difference between two angles (0-360 degrees)."""
    diff = np.abs(angle1 - angle2)
    return np.minimum(diff, 360 - diff)

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

# ============================================================================
# 2. Create Edge Features
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: CREATING PAIRWISE EDGE FEATURES")
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
            # Spatial Features
            # ================================================================
            
            # Position distances
            x_dist = player_b['x'] - player_a['x']
            y_dist = player_b['y'] - player_a['y']
            e_dist = np.sqrt(x_dist**2 + y_dist**2)
            
            # Distances to ball landing
            playerA_dist_to_landing = np.sqrt(
                (player_a['x'] - ball_land_x)**2 + 
                (player_a['y'] - ball_land_y)**2
            )
            playerB_dist_to_landing = np.sqrt(
                (player_b['x'] - ball_land_x)**2 + 
                (player_b['y'] - ball_land_y)**2
            )
            
            # ================================================================
            # Angular Features
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
            # Velocity/Acceleration Features
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
            # Team/Role Features
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
            }
            
            # Add coverage data if available
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
# 3. Create Output DataFrame
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: CREATING OUTPUT DATAFRAME")
print("-" * 80)

print("Converting to dataframe...")
df_c = pd.DataFrame(output_rows)

print(f"  ✓ Created dataframe: {df_c.shape}")
print(f"  Memory: {df_c.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# ============================================================================
# 4. Data Quality Checks
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: DATA QUALITY CHECKS")
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

# ============================================================================
# 5. Save Output
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: SAVING OUTPUT")
print("-" * 80)

print("Writing to parquet...")
df_c.to_parquet(OUTPUT_FILE, engine='pyarrow', index=False, compression='snappy')
print(f"✓ Saved to: {OUTPUT_FILE}")
print(f"  File size: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.1f} MB")

print("\nSample output (first 2 rows):")
print(df_c.head(2).to_string())

# ============================================================================
# Complete
# ============================================================================

print("\n" + "=" * 80)
print("DATAFRAME C (v2): COMPLETE")
print("=" * 80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nOutput: {OUTPUT_FILE}")
print(f"Rows: {len(df_c):,}")
print(f"Columns: {len(df_c.columns)}")
print("\n" + "=" * 80)