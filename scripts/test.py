import pandas as pd
import numpy as np
import pyarrow as pa
from utils import supplement_data, add_play_direction, standardize_play_direction, find_nearest_v2, angle_difference

supplementary_data = pd.read_csv('data/supplementary_data.csv')

master = pd.DataFrame()
all_input = pd.DataFrame()

for week in range(1, 3): #19):
    test_file_path = f'data/train/output_2023_w{week:02d}.csv'
    train_file_path = f'data/train/input_2023_w{week:02d}.csv'

    temp_test = pd.read_csv(test_file_path)
    temp_train = pd.read_csv(train_file_path)

    all_input = pd.concat([all_input, temp_train], ignore_index=True)

    better_train, supplemented = supplement_data(supplementary_data, temp_train, temp_test)
    complete = add_play_direction(supplemented)
    final = standardize_play_direction(complete)

    master = pd.concat([master, final], ignore_index=True)

    master['nfl_id'] = master['nfl_id'].fillna('BALL')
    print(f"done with week {week}")

test_df = find_nearest_v2(all_input, n_same_team=5, n_opp_team=5)

test_df.to_parquet('test.parquet', engine='pyarrow', index=False)
