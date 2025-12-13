import pandas as pd

# Read the parquet file
input_df = pd.read_parquet('data/train/2023_input_all.parquet')

# Group by game_id, play_id, and frame_id to count players
player_counts = input_df.groupby(['game_id', 'play_id', 'frame_id']).agg(
    n_players_tot=('nfl_id', 'count'),
    n_players_off=('player_side', lambda x: (x == 'Offense').sum()),
    n_players_def=('player_side', lambda x: (x == 'Defense').sum())
).reset_index()

# Get num_frames_output (assuming it's already in the original dataframe)
# Take the first occurrence per game_id, play_id, frame_id
num_frames = input_df.groupby(['game_id', 'play_id', 'frame_id'])['num_frames_output'].first().reset_index()

# Merge the counts with num_frames_output
output_df = player_counts.merge(num_frames, on=['game_id', 'play_id', 'frame_id'])

# Reorder columns to match ideal output
output_df = output_df[['game_id', 'play_id', 'num_frames_output', 'frame_id', 
                        'n_players_tot', 'n_players_off', 'n_players_def']]

print(f"Output shape: {output_df.shape}")
print(f"\nSample output:")
print(output_df.head(10))
print(f"\nSummary statistics:")
print(output_df[['n_players_tot', 'n_players_off', 'n_players_def']].describe())

output_df.to_parquet('outputs/dataframe_d/v1.parquet', engine='pyarrow', index=False)
print("Saved to dataframe_d/v1.parquet")