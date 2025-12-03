import pandas as pd
import numpy as np

def supplement_data(supplemental_df, train_df, test_df):
    test_df = test_df.rename(columns={'x': 'x_output', 'y': 'y_output'})

    better_train_df = train_df.merge(
        supplemental_df,
        on=['game_id', 'play_id'],
        how='left')
    
    complete_df = better_train_df.merge(
        test_df,
        on=['game_id', 'play_id', 'nfl_id', 'frame_id'])
    
    return better_train_df, complete_df


def add_play_direction(df):
    """
    Add a 'direction' column to the dataframe indicating play direction.

    For plays at midfield (absolute_yardline_number == 60), uses mean and median
    of orientation values to determine direction.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: game_id, play_id, absolute_yardline_number,
        yardline_side, defensive_team, possession_team, o (orientation)

    Returns:
    --------
    pd.DataFrame
        DataFrame with added 'direction' column
    """
    df = df.copy()

    def simple_dir(row):
        """Helper function to determine direction for a single row."""
        if row['absolute_yardline_number'] > 60:
            if row['yardline_side'] == row['defensive_team']:
                return 'GOING RIGHT'
            if row['yardline_side'] == row['possession_team']:
                return 'GOING LEFT'
        elif row['absolute_yardline_number'] < 60:
            if row['yardline_side'] == row['defensive_team']:
                return 'GOING LEFT'
            if row['yardline_side'] == row['possession_team']:
                return 'GOING RIGHT'
        else:
            return 'ERROR'

        return 'BLANK'

    df['direction'] = df.apply(simple_dir, axis=1)

    error_mask = df['direction'] == 'ERROR'

    if error_mask.any():
        error_plays = df[error_mask].copy()

        orientation_stats = error_plays.groupby(['game_id', 'play_id'])['o'].agg(['mean', 'median']).reset_index()

        orientation_stats['error_direction'] = orientation_stats.apply(
            lambda x: 'GOING RIGHT' if (x['mean'] > 180 and x['median'] > 180) else 'GOING LEFT',
            axis=1
        )

        df = df.merge(
            orientation_stats[['game_id', 'play_id', 'error_direction']],
            on=['game_id', 'play_id'],
            how='left'
        )

        df.loc[error_mask, 'direction'] = df.loc[error_mask, 'error_direction']

        df = df.drop(columns=['error_direction'])

    return df

import pandas as pd
import numpy as np

def standardize_play_direction(df):
    """
    Standardizes X, Y, direction (dir), and orientation (o) so all plays 
    appear as if the offense is moving right (increasing X).
    
    Parameters:
    -----------
    df : pandas DataFrame
        Must contain columns: 'x', 'y', 'dir', 'o', 'direction'
        where 'direction' is either 'GOING LEFT' or 'GOING RIGHT'
    
    Returns:
    --------
    pandas DataFrame with standardized coordinates and angles
    """
    df_standardized = df.copy()
    
    going_left_mask = df_standardized['direction'] == 'GOING LEFT'
    
    df_standardized.loc[going_left_mask, 'x'] = 120 - df_standardized.loc[going_left_mask, 'x']
    df_standardized.loc[going_left_mask, 'y'] = 53.3 - df_standardized.loc[going_left_mask, 'y']
    df_standardized.loc[going_left_mask, 'dir'] = (180 - df_standardized.loc[going_left_mask, 'dir']) % 360
    df_standardized.loc[going_left_mask, 'o'] = (180 - df_standardized.loc[going_left_mask, 'o']) % 360
    
    return df_standardized

def e_dist(x1, x2, y1, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def angle_difference(angle1, angle2):
    diff = np.abs(angle1 - angle2)
    return np.minimum(diff, 360 - diff)

def find_nearest_v2(df, n_same_team, n_opp_team):
    df = df.copy()
    output_rows = []

    for (game_id, play_id, frame_id), working_df in df.groupby(['game_id', 'play_id', 'frame_id']):
        
        for root_idx in working_df.index:
            root_row = working_df.loc[root_idx]
            
            same_team_distances = []
            opp_team_distances = []
            
            for test_idx in working_df.index:
                if root_idx == test_idx:
                    continue
                
                test_row = working_df.loc[test_idx]
                
                distance = ((root_row['x'] - test_row['x'])**2 + 
                           (root_row['y'] - test_row['y'])**2)**0.5
                
                angle_dif = angle_difference(root_row['dir'], test_row['dir'])
                
                if root_row['player_side'] == test_row['player_side']:
                    same_team_distances.append((test_row['nfl_id'], distance))
                else:
                    opp_team_distances.append((test_row['nfl_id'], distance))
            
            same_team_distances.sort(key=lambda x: x[1])
            opp_team_distances.sort(key=lambda x: x[1])
            
            result = {
                'game_id': game_id,
                'play_id': play_id,
                'frame_id': frame_id,
                'root_player_id': root_row['nfl_id'],
                'total_players': len(working_df),
                'total_same_team': len(same_team_distances),
                'total_opp_team': len(opp_team_distances)
            }
            
            # Create separate columns for each neighbor
            for i in range(n_same_team):
                result[f'nearest_same_{i+1}'] = same_team_distances[i][0] if i < len(same_team_distances) else None
            
            for i in range(n_opp_team):
                result[f'nearest_opp_{i+1}'] = opp_team_distances[i][0] if i < len(opp_team_distances) else None
            
            output_rows.append(result)

    output_df = pd.DataFrame(output_rows)
    return output_df

def angle_difference(angle1, angle2):
    """
    Calculate the smallest difference between two angles (0-360 degrees).
    Returns value in range [0, 180].
    """
    diff = np.abs(angle1 - angle2)
    return np.minimum(diff, 360 - diff)