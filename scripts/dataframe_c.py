import pandas as pd
import numpy as np
from utils import angle_difference

input_df = pd.read_parquet('data/train/2023_input_all.parquet')

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
                'player_rel_index': player_rel_index
            }
            
            output_rows.append(row)

# Create output dataframe
output_df = pd.DataFrame(output_rows)

print(f"Input dataframe shape: {input_df.shape}")
print(f"Output dataframe shape: {output_df.shape}")
print(f"\nSample output:")
print(output_df.head(2))

output_df.to_parquet('outputs/dataframe_c_v1.parquet', engine='pyarrow', index=False)
print("Saved to dataframe_c_v1.parquet")
