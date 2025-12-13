# node-level features
import pandas as pd
import numpy as np
import os
from utils import add_play_direction, standardize_play_direction

# read in the player tracking data
input_df = pd.read_parquet('data/train/2023_input_all.parquet')

# read in the supplementary data
supplementary_data = pd.read_csv('data/supplementary_data.csv', low_memory=False)

# merge with supplementary data
input_df = input_df.merge(
    supplementary_data,
    on=['game_id', 'play_id'],
    how='left'
)

# add play direction
input_df = add_play_direction(input_df)

# standardize x, y, dir, o columns
input_df = standardize_play_direction(input_df)
output_df = input_df

cols = ['game_id', 'play_id', 'player_to_predict', 'nfl_id', 'frame_id', 'play_direction', 'absolute_yardline_number', 'player_name', 'player_height', 'player_weight', 'player_birth_date', 'player_position', 'player_side', 'player_role', 'x', 'y', 's', 'a', 'dir', 'o',
              'num_frames_output', 'ball_land_x', 'ball_land_y', 'season', 'week']
input_df = input_df[cols]

#print(f"Loaded and standardized {len(input_df)} rows")
#print(f"Columns: {input_df.columns.tolist()}")

# save the standardized dataframe
#input_df.to_parquet('data/train/2023_input_standardized.parquet', engine='pyarrow', index=False)
#print("Saved to 2023_input_standardized.parquet")

output_df['e_dist_ball_land'] = np.sqrt(
    (output_df['x'] - output_df['ball_land_x'])**2 + 
    (output_df['y'] - output_df['ball_land_y'])**2)

output_df['isTargeted'] = (output_df['player_role'] == 'Targeted Receiver').astype(int)
output_df['isPasser'] = (output_df['player_role'] == 'Passer').astype(int)
output_df['isRouteRunner'] = (
    (output_df['player_role'] == 'Targeted Receiver') | 
    (output_df['player_role'] == 'Other Route Runner')
).astype(int)
output_df['los_dist'] = np.abs(output_df['x'] - output_df['absolute_yardline_number'])

# List of columns to keep
df_a_cols = [
    'game_id', 'play_id', 'frame_id', 'nfl_id',
    'player_height', 'player_weight', 'player_birth_date',
    'player_position', 'player_side', 'player_role',
    'x', 'y', 's', 'a', 'dir', 'o',
    'e_dist_ball_land', 'los_dist', 'isTargeted', 'isPasser', 'isRouteRunner'
]

output_df = output_df[df_a_cols]

os.makedirs('outputs/dataframe_a', exist_ok=True)
output_df.to_parquet('outputs/dataframe_a/v1.parquet', engine='pyarrow', index=False)