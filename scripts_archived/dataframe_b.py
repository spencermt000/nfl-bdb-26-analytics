import pandas as pd
import os
import numpy as np

# Set display options
pd.set_option('display.float_format', lambda x: '%.10f' % x)

# Read supplementary data
sup_file = 'data/supplementary_data.csv'
sup_raw = pd.read_csv(sup_file)

# Read sdv_pbp from parquet file
sdv_pbp = pd.read_parquet('data/sdv_raw_pbp.parquet')

# Create index columns
sup_raw['index'] = sup_raw['game_id'].astype(str) + '_' + sup_raw['play_id'].astype(str)
sdv_pbp['index'] = sdv_pbp['old_game_id'].astype(str) + '_' + sdv_pbp['play_id'].astype(str)

# Create play_in_drive column
sdv_pbp['play_in_drive'] = sdv_pbp.groupby(['game_id', 'posteam', 'drive']).cumcount() + 1

# Create binary indicator columns
sdv_pbp['tfl'] = np.where(
    sdv_pbp['tackle_for_loss_1_player_id'].notna() & sdv_pbp['tackle_for_loss_2_player_id'].notna(),
    1, 0
)

sdv_pbp['pbu'] = np.where(
    (sdv_pbp['pass_defense_1_player_id'].notna() & sdv_pbp['pass_defense_2_player_id'].notna()) | 
    (sdv_pbp['interception'] == 1),
    1, 0
)

sdv_pbp['atkl'] = np.where(
    sdv_pbp['solo_tackle_1_player_id'].notna() & sdv_pbp['solo_tackle_2_player_id'].notna(),
    1, 0
)

sdv_pbp['stkl'] = np.where(
    sdv_pbp['assist_tackle_1_player_id'].notna() | 
    sdv_pbp['assist_tackle_2_player_id'].notna() | 
    sdv_pbp['assist_tackle_3_player_id'].notna() | 
    sdv_pbp['assist_tackle_4_player_id'].notna(),
    1, 0
)

# Select specific columns
sdv_cols = [
    "quarter_seconds_remaining", "game_seconds_remaining", "half_seconds_remaining",
    "goal_to_go", "shotgun", "no_huddle", "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "air_yards", "yards_after_catch", "air_epa", "comp_air_epa", "comp_yac_epa", "qb_hit", "touchdown", 
    "surface", "roof", "total_line", "spread_line", "series", "play_in_drive",
    "tfl", "pbu", "atkl", "stkl", "cp", "cpoe", "xpass", "pass_oe", "old_game_id",
    "xyac_epa", "xyac_median_yardage", "xyac_mean_yardage", "xyac_success", "index"
]

test = sdv_pbp[sdv_cols]

# Left join and clean up
master = sup_raw.merge(test, on='index', how='left')
master = master.drop(columns=['penalty_yards'])

# Replace NA values
master['yards_after_catch'] = master['yards_after_catch'].fillna(0)
master['xyac_epa'] = master['xyac_epa'].fillna(0)
master['xyac_median_yardage'] = master['xyac_median_yardage'].fillna(0)
master['xyac_mean_yardage'] = master['xyac_mean_yardage'].fillna(0)
master['xyac_success'] = master['xyac_success'].fillna(0)
master['air_yards'] = master['air_yards'].fillna(master['pass_length'])
master['air_epa'] = master['air_epa'].fillna(0)

df_b = master

# Create pos_team_wp
df_b['pos_team_wp'] = df_b.apply(
    lambda row: row['pre_snap_home_team_win_probability'] 
    if row['possession_team'] == row['home_team_abbr'] 
    else row['pre_snap_visitor_team_win_probability'], 
    axis=1
)

# Create pos_team_wpa
df_b['pos_team_wpa'] = df_b.apply(
    lambda row: row['home_team_win_probability_added'] 
    if row['possession_team'] == row['home_team_abbr'] 
    else row['visitor_team_win_probility_added'], 
    axis=1
)

# Create def_team_wpa
df_b['def_team_wpa'] = df_b.apply(
    lambda row: row['visitor_team_win_probility_added'] 
    if row['possession_team'] == row['home_team_abbr'] 
    else row['home_team_win_probability_added'], 
    axis=1
)

# Create ps_pos_team_score
df_b['ps_pos_team_score'] = df_b.apply(
    lambda row: row['pre_snap_home_score'] 
    if row['possession_team'] == row['home_team_abbr'] 
    else row['pre_snap_visitor_score'], 
    axis=1
)

# Create ps_def_team_score
df_b['ps_def_team_score'] = df_b.apply(
    lambda row: row['pre_snap_visitor_score'] 
    if row['possession_team'] == row['home_team_abbr'] 
    else row['pre_snap_home_score'], 
    axis=1
)

# List of columns to keep
df_b_cols = [
    'season', 'week', 'play_id', 'quarter', 'down', 'yards_to_go',
    'possession_team', 'defensive_team', 'yardline_side', 'yardline_number',
    'ps_pos_team_score', 'ps_def_team_score', 'pos_team_wp',
    'pass_result', 'pass_length', 'offense_formation', 'receiver_alignment',
    'route_of_targeted_receiver', 'play_action', 'dropback_type', 'dropback_distance',
    'pass_location_type', 'defenders_in_the_box', 'team_coverage_man_zone',
    'team_coverage_type', 'pre_penalty_yards_gained', 'yards_gained',
    'expected_points', 'expected_points_added', 'pos_team_wpa', 'def_team_wpa',
    'quarter_seconds_remaining', 'game_seconds_remaining', 'half_seconds_remaining',
    'goal_to_go', 'shotgun', 'no_huddle', 'posteam_timeouts_remaining',
    'defteam_timeouts_remaining', 'air_yards', 'yards_after_catch', 'air_epa',
    'comp_air_epa', 'comp_yac_epa', 'qb_hit', 'touchdown', 'surface', 'roof',
    'total_line', 'spread_line', 'series', 'play_in_drive', 'pbu', 'cp', 'cpoe',
    'xpass', 'old_game_id', 'xyac_epa', 'xyac_median_yardage', 'xyac_mean_yardage',
    'xyac_success'
]

# Keep only specified columns
df_b = df_b[df_b_cols]

# Save to parquet
os.makedirs('outputs/dataframe_b', exist_ok=True)
df_b.to_parquet('outputs/dataframe_b/v1.parquet', engine='pyarrow', index=False)