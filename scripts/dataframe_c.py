import pandas as pd
import numpy as np
from pathlib import Path
from utils import angle_difference
import os

# Create output directory if it doesn't exist
os.makedirs('outputs/dataframe_c', exist_ok=True)

# Process each week
for week in range(1, 19):  # weeks 1-18
    week_str = f"w{week:02d}"  # Format as w01, w02, etc.
    input_file = f'data/train/input_2023_{week_str}.csv'
    output_file = f'outputs/dataframe_c/{week_str}_v1.parquet'
    
    print(f"\n{'='*60}")
    print(f"Processing {week_str}...")
    print(f"{'='*60}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"WARNING: {input_file} not found, skipping...")
        continue
    
    # Load input data
    input_df = pd.read_csv(input_file)
    print(f"Loaded {input_file}: {input_df.shape}")
    
    # Get unique combinations of game, play, frame
    groups = input_df.groupby(['game_id', 'play_id', 'frame_id'])
    
    output_rows = []
    
    for (game_id, play_id, frame_id), frame_data in groups:
        # Get all players in this frame
        players = frame_data.reset_index(drop=True)
        n_players = len(players)
        
        # Create all pairwise combinations (directed: A->B and B->A are separate)
        for i in range(n_players):
            for j in range(n_players):
                if i == j:  # Skip self-edges
                    continue
                
                player_a = players.iloc[i]
                player_b = players.iloc[j]
                
                # Calculate distances
                x_dist = player_b['x'] - player_a['x']
                y_dist = player_b['y'] - player_a['y']
                e_dist = np.sqrt(x_dist**2 + y_dist**2)
                
                # Ball landing coordinates (same for all players in the frame)
                ball_land_x = player_a['ball_land_x']  # These are the same for all players
                ball_land_y = player_a['ball_land_y']
                
                # Calculate distances from each player to ball landing spot
                playerA_dist_to_landing = np.sqrt(
                    (player_a['x'] - ball_land_x)**2 + 
                    (player_a['y'] - ball_land_y)**2
                )
                playerB_dist_to_landing = np.sqrt(
                    (player_b['x'] - ball_land_x)**2 + 
                    (player_b['y'] - ball_land_y)**2
                )
                
                # Calculate angle from each player to landing spot
                # Angle from player A to landing
                angle_A_to_landing = np.arctan2(
                    ball_land_y - player_a['y'],
                    ball_land_x - player_a['x']
                ) * 180 / np.pi  # Convert to degrees
                
                # Angle from player B to landing
                angle_B_to_landing = np.arctan2(
                    ball_land_y - player_b['y'],
                    ball_land_x - player_b['x']
                ) * 180 / np.pi  # Convert to degrees
                
                # Relative angle between their angles to landing spot
                pairwise_angle_to_landing = angle_difference(angle_A_to_landing, angle_B_to_landing)
                
                # Calculate relative angles
                relative_angle_o = angle_difference(player_a['o'], player_b['o'])
                relative_angle_dir = angle_difference(player_a['dir'], player_b['dir'])
                
                # Same team check
                same_team = 1 if player_a['player_side'] == player_b['player_side'] else 0
                
                # Create player relationship index
                player_rel_index = f"{player_a['nfl_id']}_{player_b['nfl_id']}"
                
                # Build row
                row = {
                    'game_id': game_id,
                    'play_id': play_id,
                    'frame_id': frame_id,
                    'playerA_id': player_a['nfl_id'],
                    'playerA_x': player_a['x'],
                    'playerA_y': player_a['y'],
                    'playerA_s': player_a['s'],
                    'playerA_a': player_a['a'],
                    'playerA_dir': player_a['dir'],
                    'playerA_o': player_a['o'],
                    'playerA_role': player_a['player_role'],
                    'playerA_side': player_a['player_side'],
                    'playerA_position': player_a['player_position'],
                    'playerA_height': player_a['player_height'],
                    'playerA_weight': player_a['player_weight'],
                    'playerB_id': player_b['nfl_id'],
                    'playerB_x': player_b['x'],
                    'playerB_y': player_b['y'],
                    'playerB_s': player_b['s'],
                    'playerB_a': player_b['a'],
                    'playerB_dir': player_b['dir'],
                    'playerB_o': player_b['o'],
                    'playerB_role': player_b['player_role'],
                    'playerB_side': player_b['player_side'],
                    'playerB_position': player_b['player_position'],
                    'playerB_height': player_b['player_height'],
                    'playerB_weight': player_b['player_weight'],
                    'x_dist': x_dist,
                    'y_dist': y_dist,
                    'e_dist': e_dist,
                    'relative_angle_o': relative_angle_o,
                    'relative_angle_dir': relative_angle_dir,
                    'same_team': same_team,
                    'player_rel_index': player_rel_index,
                    # NEW: Ball landing coordinates
                    'ball_land_x': ball_land_x,
                    'ball_land_y': ball_land_y,
                    # NEW: Derived features - distances to landing spot
                    'playerA_dist_to_landing': playerA_dist_to_landing,
                    'playerB_dist_to_landing': playerB_dist_to_landing,
                    'pairwise_angle_to_landing': pairwise_angle_to_landing,
                }
                
                output_rows.append(row)
    
    # Create output dataframe
    output_df = pd.DataFrame(output_rows)
    
    print(f"Output shape: {output_df.shape}")
    print(f"\nSample output:")
    print(output_df.head(2))
    
    # Save to parquet
    output_df.to_parquet(output_file, engine='pyarrow', index=False)
    print(f"✓ Saved to {output_file}")

print(f"\n{'='*60}")
print("All weeks processed!")
print(f"{'='*60}")



parquet_files = sorted(Path('outputs/dataframe_c').glob('*.parquet'))

print(f"Found {len(parquet_files)} parquet files")

# Read and combine all parquet files
dfs = []
for file in parquet_files:
    print(f"Reading {file.name}...")
    df = pd.read_parquet(file)
    dfs.append(df)
    print(f"  Shape: {df.shape}")

# Concatenate all dataframes
combined_df = pd.concat(dfs, ignore_index=True)
print(f"\nCombined dataframe shape: {combined_df.shape}")

# Save combined dataframe
output_file = 'outputs/dataframe_c/v1.parquet'
combined_df.to_parquet(output_file, engine='pyarrow', index=False)
print(f"Saved to {output_file}")