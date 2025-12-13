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
    """
    Calculate the smallest difference between two angles (0-360 degrees).
    Returns value in range [0, 180].
    """
    diff = np.abs(angle1 - angle2)
    return np.minimum(diff, 360 - diff)


def create_play_level_coverage(input_tracking_df, output_tracking_df, sumer_coverage_frame_df):
    """
    Create play-level coverage data by extracting coverage at ball release (start) 
    and ball landing (end) from frame-level SumerSports coverage data.
    
    Parameters:
    -----------
    input_tracking_df : pd.DataFrame
        Player tracking data BEFORE ball lands (from input files)
        Must contain: game_id, play_id, frame_id
        
    output_tracking_df : pd.DataFrame
        Player tracking data AFTER ball lands (from output files)
        Must contain: game_id, play_id, frame_id
        
    sumer_coverage_frame_df : pd.DataFrame
        SumerSports frame-level coverage data
        Must contain: game_id, play_id, frame_id, coverage_scheme, coverage_scheme__*
    
    Returns:
    --------
    pd.DataFrame with columns:
        - game_id, play_id
        - start_scheme, start_cover_0, start_cover_1, ..., start_cover_SHORT
        - land_scheme, land_cover_0, land_cover_1, ..., land_cover_SHORT
    """
    print("Creating play-level coverage data from frame-level SumerSports data...")
    
    # ========================================================================
    # 1. Get start frame (ball release) - first frame in input data
    # ========================================================================
    print("  Finding ball release frames (first frame per play)...")
    start_frames = input_tracking_df.groupby(['game_id', 'play_id']).agg(
        start_frame_id=('frame_id', 'min')
    ).reset_index()
    
    print(f"    ✓ Found {len(start_frames):,} plays with start frames")
    
    # ========================================================================
    # 2. Get landing frame (ball catch/land) - first frame in output data
    # ========================================================================
    print("  Finding ball landing frames (first frame in output data)...")
    land_frames = output_tracking_df.groupby(['game_id', 'play_id']).agg(
        land_frame_id=('frame_id', 'min')
    ).reset_index()
    
    print(f"    ✓ Found {len(land_frames):,} plays with landing frames")
    
    # ========================================================================
    # 3. Merge to get both start and land frames per play
    # ========================================================================
    print("  Combining start and land frame info...")
    play_frames = start_frames.merge(
        land_frames,
        on=['game_id', 'play_id'],
        how='inner'
    )
    
    print(f"    ✓ {len(play_frames):,} plays have both start and land frames")
    
    # ========================================================================
    # 4. Extract coverage at start frame
    # ========================================================================
    print("  Extracting coverage at ball release (start)...")
    
    # Merge play_frames with coverage data to get start frame coverage
    start_coverage = play_frames.merge(
        sumer_coverage_frame_df,
        left_on=['game_id', 'play_id', 'start_frame_id'],
        right_on=['game_id', 'play_id', 'frame_id'],
        how='left'
    )
    
    # Select and rename coverage columns
    start_cols = {
        'coverage_scheme': 'start_scheme',
        'coverage_scheme__COVER_0': 'start_cover_0',
        'coverage_scheme__COVER_1': 'start_cover_1',
        'coverage_scheme__COVER_2': 'start_cover_2',
        'coverage_scheme__COVER_2_MAN': 'start_cover_2_MAN',
        'coverage_scheme__COVER_3': 'start_cover_3',
        'coverage_scheme__COVER_4': 'start_cover_4',
        'coverage_scheme__COVER_6': 'start_cover_6',
        'coverage_scheme__MISC': 'start_cover_MISC',
        'coverage_scheme__PREVENT': 'start_cover_PREVENT',
        'coverage_scheme__REDZONE': 'start_cover_REDZONE',
        'coverage_scheme__SHORT': 'start_cover_SHORT'
    }
    
    start_coverage = start_coverage.rename(columns=start_cols)
    start_coverage = start_coverage[['game_id', 'play_id'] + list(start_cols.values())]
    
    print(f"    ✓ Coverage at start: {start_coverage['start_scheme'].notna().sum():,} plays")
    
    # ========================================================================
    # 5. Extract coverage at landing frame
    # ========================================================================
    print("  Extracting coverage at ball landing (end)...")
    
    # Merge play_frames with coverage data to get landing frame coverage
    land_coverage = play_frames.merge(
        sumer_coverage_frame_df,
        left_on=['game_id', 'play_id', 'land_frame_id'],
        right_on=['game_id', 'play_id', 'frame_id'],
        how='left'
    )
    
    # Select and rename coverage columns
    land_cols = {
        'coverage_scheme': 'land_scheme',
        'coverage_scheme__COVER_0': 'land_cover_0',
        'coverage_scheme__COVER_1': 'land_cover_1',
        'coverage_scheme__COVER_2': 'land_cover_2',
        'coverage_scheme__COVER_2_MAN': 'land_cover_2_MAN',
        'coverage_scheme__COVER_3': 'land_cover_3',
        'coverage_scheme__COVER_4': 'land_cover_4',
        'coverage_scheme__COVER_6': 'land_cover_6',
        'coverage_scheme__MISC': 'land_cover_MISC',
        'coverage_scheme__PREVENT': 'land_cover_PREVENT',
        'coverage_scheme__REDZONE': 'land_cover_REDZONE',
        'coverage_scheme__SHORT': 'land_cover_SHORT'
    }
    
    land_coverage = land_coverage.rename(columns=land_cols)
    land_coverage = land_coverage[['game_id', 'play_id'] + list(land_cols.values())]
    
    print(f"    ✓ Coverage at landing: {land_coverage['land_scheme'].notna().sum():,} plays")
    
    # ========================================================================
    # 6. Merge start and land coverage into single play-level dataframe
    # ========================================================================
    print("  Combining start and land coverage...")
    
    play_coverage = start_coverage.merge(
        land_coverage,
        on=['game_id', 'play_id'],
        how='outer'
    )
    
    # Add coverage_changed flag
    play_coverage['coverage_changed'] = (
        play_coverage['start_scheme'] != play_coverage['land_scheme']
    ).astype(int)
    
    print(f"    ✓ Final play-level coverage: {len(play_coverage):,} plays")
    print(f"    ✓ Coverage changed during play: {play_coverage['coverage_changed'].sum():,} plays")
    
    # Show coverage distribution
    if play_coverage['start_scheme'].notna().sum() > 0:
        print(f"\n  Coverage at ball release (top 5):")
        for scheme, count in play_coverage['start_scheme'].value_counts().head(5).items():
            print(f"    {scheme}: {count:,} plays")
    
    if play_coverage['land_scheme'].notna().sum() > 0:
        print(f"\n  Coverage at ball landing (top 5):")
        for scheme, count in play_coverage['land_scheme'].value_counts().head(5).items():
            print(f"    {scheme}: {count:,} plays")
    
    return play_coverage


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


def aggregate_coverage_to_play_level(coverage_frame_df, tracking_df):
    """
    Aggregate frame-level coverage data to play-level by extracting coverage
    at ball release (first frame) and ball arrival (last frame before catch).
    
    Parameters:
    -----------
    coverage_frame_df : pd.DataFrame
        SumerSports frame-level coverage data with columns:
        - game_id, play_id, frame_id
        - coverage_scheme
        - coverage_scheme__COVER_0, COVER_1, COVER_2, etc. (probabilities)
    
    tracking_df : pd.DataFrame
        Tracking data with columns:
        - game_id, play_id, frame_id, num_frames_output
        Used to determine first and last frames
    
    Returns:
    --------
    pd.DataFrame with play-level coverage data:
        - game_id
        - play_id
        - coverage_at_release (coverage scheme at ball release)
        - coverage_at_arrival (coverage scheme at ball arrival/catch)
        - coverage_changed (1 if different, 0 if same)
        - prob_cover_0_release, prob_cover_1_release, etc. (probabilities at release)
        - prob_cover_0_arrival, prob_cover_1_arrival, etc. (probabilities at arrival)
    """
    
    print("Aggregating coverage frame data to play level...")
    
    # Get first and last frame for each play from tracking data
    frame_info = tracking_df.groupby(['game_id', 'play_id']).agg({
        'frame_id': ['min', 'max']
    }).reset_index()
    
    # Flatten column names
    frame_info.columns = ['game_id', 'play_id', 'first_frame', 'last_frame']
    
    print(f"  Found {len(frame_info):,} unique plays")
    print(f"  First frame range: {frame_info['first_frame'].min()}-{frame_info['first_frame'].max()}")
    print(f"  Last frame range: {frame_info['last_frame'].min()}-{frame_info['last_frame'].max()}")
    
    # Ensure consistent types for merging
    coverage_frame_df['game_id'] = coverage_frame_df['game_id'].astype(str)
    coverage_frame_df['play_id'] = coverage_frame_df['play_id'].astype(float)
    frame_info['game_id'] = frame_info['game_id'].astype(str)
    frame_info['play_id'] = frame_info['play_id'].astype(float)
    
    # Merge coverage data with frame info
    coverage_with_frames = coverage_frame_df.merge(
        frame_info,
        on=['game_id', 'play_id'],
        how='inner'
    )
    
    print(f"  Matched {len(coverage_with_frames):,} coverage frames to plays")
    
    # Extract coverage at release (first frame)
    coverage_at_release = coverage_with_frames[
        coverage_with_frames['frame_id'] == coverage_with_frames['first_frame']
    ].copy()
    
    # Rename columns for release
    release_cols = {
        'coverage_scheme': 'coverage_at_release'
    }
    
    # Add probability columns for release
    prob_cols = [col for col in coverage_at_release.columns if col.startswith('coverage_scheme__')]
    for col in prob_cols:
        coverage_name = col.replace('coverage_scheme__', '').lower()
        release_cols[col] = f'prob_{coverage_name}_release'
    
    coverage_at_release = coverage_at_release.rename(columns=release_cols)
    release_keep_cols = ['game_id', 'play_id', 'coverage_at_release'] + [release_cols[col] for col in prob_cols]
    coverage_at_release = coverage_at_release[release_keep_cols]
    
    print(f"  Coverage at release: {len(coverage_at_release):,} plays")
    
    # Extract coverage at arrival (last frame)
    coverage_at_arrival = coverage_with_frames[
        coverage_with_frames['frame_id'] == coverage_with_frames['last_frame']
    ].copy()
    
    # Rename columns for arrival
    arrival_cols = {
        'coverage_scheme': 'coverage_at_arrival'
    }
    
    for col in prob_cols:
        coverage_name = col.replace('coverage_scheme__', '').lower()
        arrival_cols[col] = f'prob_{coverage_name}_arrival'
    
    coverage_at_arrival = coverage_at_arrival.rename(columns=arrival_cols)
    arrival_keep_cols = ['game_id', 'play_id', 'coverage_at_arrival'] + [arrival_cols[col] for col in prob_cols]
    coverage_at_arrival = coverage_at_arrival[arrival_keep_cols]
    
    print(f"  Coverage at arrival: {len(coverage_at_arrival):,} plays")
    
    # Merge release and arrival coverage
    play_coverage = coverage_at_release.merge(
        coverage_at_arrival,
        on=['game_id', 'play_id'],
        how='outer'
    )
    
    # Calculate if coverage changed
    play_coverage['coverage_changed'] = (
        play_coverage['coverage_at_release'] != play_coverage['coverage_at_arrival']
    ).astype(int)
    
    # Handle cases where one is missing
    play_coverage['coverage_changed'] = play_coverage['coverage_changed'].fillna(0).astype(int)
    
    print(f"\nPlay-level coverage summary:")
    print(f"  Total plays: {len(play_coverage):,}")
    print(f"  Plays with coverage change: {play_coverage['coverage_changed'].sum():,} ({100*play_coverage['coverage_changed'].mean():.1f}%)")
    
    if 'coverage_at_release' in play_coverage.columns:
        print(f"\n  Coverage at release distribution:")
        for cov, count in play_coverage['coverage_at_release'].value_counts().head(5).items():
            print(f"    {cov}: {count:,}")
    
    if 'coverage_at_arrival' in play_coverage.columns:
        print(f"\n  Coverage at arrival distribution:")
        for cov, count in play_coverage['coverage_at_arrival'].value_counts().head(5).items():
            print(f"    {cov}: {count:,}")
    
    return play_coverage

def load_parquet_to_df(file_path, df_name=None):
    """
    Load a parquet file into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the parquet file
        df_name (str, optional): Name for the resulting DataFrame (for logging)
    
    Returns:
        pd.DataFrame: Loaded dataframe
    
    Example:
        df_a = load_parquet_to_df('outputs/dataframe_a/v2.parquet', 'df_a')
    """
    import pandas as pd
    import os
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Parquet file not found: {file_path}")
    
    # Load parquet
    print(f"Loading {df_name or 'dataframe'} from {file_path}...")
    df = pd.read_parquet(file_path)
    
    # Report stats
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"  ✓ Loaded {len(df):,} rows, {len(df.columns)} columns ({size_mb:.1f} MB)")
    
    return df