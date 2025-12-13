"""
Dataframe C v2: Structured Edge-Level Features with Domain-Informed Priors
===========================================================================
Creates STRUCTURED pairwise edges between players with:
  - Edge type taxonomy (8 types: qb_trr, trr_def, etc.)
  - Time-varying distance thresholds (decay as ball approaches landing)
  - Attention priors (P_ij) based on football domain knowledge
  - Temporal edge continuity tracking (edge_id)
  - Coverage scheme probability distributions (11 columns)

INPUTS:
  - outputs/dataframe_a/v1.parquet (processed node features)
    * Provides: ball_land_x, ball_land_y, coverage_scheme, coverage probabilities
  - outputs/dataframe_b/v1.parquet (play-level features with ball trajectory)
    * Provides: start_ball_x, start_ball_y, ball_flight_frames

OUTPUTS:
  - outputs/dataframe_c/v2.parquet (structured edges)
  - outputs/dataframe_c/v2_diagnostics/ (optional diagnostic outputs)

KEY DIFFERENCES FROM V1:
  ✨ STRUCTURED EDGES: Not fully connected - filtered by football relevance
  ✨ EDGE TYPES: 8 semantic types (qb_trr, trr_def, def_def, etc.)
  ✨ TIME-VARYING THRESHOLDS: Distance filtering adapts based on ball progress
  ✨ ATTENTION PRIORS: P_ij values (0-1) inform model which edges are critical
  ✨ TEMPORAL CONTINUITY: edge_id tracks same relationship across frames
  ✨ COVERAGE CONTEXT: 11 coverage columns (scheme + 10 probability distributions)

EDGE TYPE TAXONOMY:
  - qb_rr: QB → non-targeted route runner (IGNORED)
  - qb_trr: QB → targeted route runner (CRITICAL)
  - qb_def: QB → coverage defender
  - rr_rr: Route runner → route runner (IGNORED)
  - rr_trr: Route runner → targeted route runner (IGNORED)
  - rr_def: Route runner → coverage defender
  - trr_def: Targeted route runner → coverage defender (CRITICAL)
  - def_def: Coverage defender → coverage defender

EDGE CREATION LOGIC (TIERED):
  Tier 1 (ALWAYS include):
    - qb_trr: QB to targeted receiver
    - trr_def: Targeted receiver to covering defender
  
  Tier 2 (Distance-filtered):
    - qb_def, rr_def, def_def: If distance < threshold_at_t
  
  Tier 3 (IGNORED for computational efficiency):
    - rr_rr, rr_trr, qb_rr: Not created

COVERAGE SCHEME COLUMNS (11 total):
  - coverage_scheme: Primary coverage type (categorical)
  - coverage_scheme__COVER_0: Probability of Cover 0
  - coverage_scheme__COVER_1: Probability of Cover 1
  - coverage_scheme__COVER_2: Probability of Cover 2
  - coverage_scheme__COVER_3: Probability of Cover 3
  - coverage_scheme__COVER_4: Probability of Cover 4
  - coverage_scheme__COVER_6: Probability of Cover 6
  - coverage_scheme__PREVENT: Probability of Prevent
  - coverage_scheme__GOAL_LINE: Probability of Goal Line
  - coverage_scheme__MISC: Probability of Misc coverage
  - coverage_scheme__SHORT: Probability of Short coverage
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# Configuration
# ============================================================================

print("=" * 80)
print("DATAFRAME C v2: STRUCTURED EDGES + ATTENTION PRIORS")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# File paths
INPUT_DF_A = 'outputs/dataframe_a/v1.parquet'
INPUT_DF_B = 'outputs/dataframe_b/v1.parquet'
OUTPUT_DIR = 'outputs/dataframe_c'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'v2.parquet')
DIAGNOSTICS_DIR = os.path.join(OUTPUT_DIR, 'v2_diagnostics')

# *** PILOT MODE ***
PILOT_MODE = True  # Set to True to process only N games for testing
PILOT_N_GAMES = 3

# *** DIAGNOSTICS MODE ***
GENERATE_DIAGNOSTICS = False  # Set to True to generate diagnostic outputs

# === EDGE CREATION PARAMETERS ===
MAX_DIST_EARLY = 15.0       # Max distance (yards) early in ball flight
MIN_DIST_LATE = 5.0         # Min distance (yards) near landing
DECAY_RATE = 2.0            # How quickly distance threshold decays (higher = steeper)

BALL_PROXIMITY_THRESHOLD = 10.0  # Consider "near ball landing" if within this distance

# === ATTENTION PRIOR PARAMETERS ===
DISTANCE_CHARACTERISTIC_SCALE = 10.0  # For distance decay in priors (yards)
BALL_PROXIMITY_BONUS_START = 0.5      # Apply bonus when ball_progress > this

# === BASE PRIOR WEIGHTS (by edge type) ===
BASE_PRIORS = {
    'qb_trr': 0.95,      # QB to target = critical
    'trr_def': 1.00,     # Target vs defender = THE key matchup
    'qb_def': 0.85,      # QB reading defense
    'rr_def': 0.85,      # Other WR matchups matter
    'def_def': 0.85,     # Defender coordination
    # Note: qb_rr, rr_trr, rr_rr are not created (low priority)
}

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
if GENERATE_DIAGNOSTICS:
    os.makedirs(DIAGNOSTICS_DIR, exist_ok=True)

if PILOT_MODE:
    print("\n" + "⚠" * 40)
    print("PILOT MODE ENABLED".center(80))
    print(f"Will process only {PILOT_N_GAMES} games for testing".center(80))
    print("Set PILOT_MODE = False for full processing".center(80))
    print("⚠" * 40 + "\n")

if GENERATE_DIAGNOSTICS:
    print("📊 DIAGNOSTICS MODE ENABLED")
    print(f"   Will generate diagnostic outputs to: {DIAGNOSTICS_DIR}\n")

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
    """
    if total_frames <= 0:
        return land_x, land_y
    
    # Progress from 0 (start) to 1 (landing)
    progress = (frame_id - 1) / (total_frames - 1) if total_frames > 1 else 1.0
    progress = np.clip(progress, 0, 1)
    
    ball_x_t = start_x + (land_x - start_x) * progress
    ball_y_t = start_y + (land_y - start_y) * progress
    
    return ball_x_t, ball_y_t

def get_distance_threshold(frames_to_landing, total_frames):
    """
    Time-varying distance threshold: stricter as ball approaches landing.
    
    Early in flight (ball just released): MAX_DIST_EARLY (15 yards)
    Late in flight (ball about to land): MIN_DIST_LATE (5 yards)
    """
    if total_frames <= 0:
        return MIN_DIST_LATE
    
    ball_progress = 1 - (frames_to_landing / total_frames)  # 0 at release, 1 at landing
    ball_progress = np.clip(ball_progress, 0, 1)
    
    # Exponential decay
    threshold = MIN_DIST_LATE + (MAX_DIST_EARLY - MIN_DIST_LATE) * np.exp(-DECAY_RATE * ball_progress)
    return threshold

def create_edge_id(player_a_id, player_b_id, game_id, play_id):
    """
    Create undirected edge ID (A↔B same as B↔A) for temporal continuity tracking.
    """
    id_pair = tuple(sorted([player_a_id, player_b_id]))
    return f"{game_id}_{play_id}_{id_pair[0]}_{id_pair[1]}"

def determine_edge_type(player_a, player_b):
    """
    Determine edge type based on player roles and targeting.
    
    Returns:
        edge_type (str): One of 8 edge types, or None if edge should not be created
    """
    # Extract roles
    a_is_qb = (player_a.get('isPasser', 0) == 1)
    a_is_rr = (player_a.get('isRouteRunner', 0) == 1)
    a_is_trr = (player_a.get('isTargeted', 0) == 1)
    a_is_def = pd.notna(player_a.get('coverage_responsibility'))
    
    b_is_qb = (player_b.get('isPasser', 0) == 1)
    b_is_rr = (player_b.get('isRouteRunner', 0) == 1)
    b_is_trr = (player_b.get('isTargeted', 0) == 1)
    b_is_def = pd.notna(player_b.get('coverage_responsibility'))
    
    # QB edges
    if a_is_qb and b_is_trr:
        return 'qb_trr'
    if b_is_qb and a_is_trr:
        return 'qb_trr'
    
    if a_is_qb and b_is_def:
        return 'qb_def'
    if b_is_qb and a_is_def:
        return 'qb_def'
    
    # IGNORED: qb_rr (QB to non-targeted route runner)
    # if a_is_qb and b_is_rr and not b_is_trr:
    #     return 'qb_rr'
    # if b_is_qb and a_is_rr and not a_is_trr:
    #     return 'qb_rr'
    
    # Targeted receiver to defender (CRITICAL)
    if a_is_trr and b_is_def:
        return 'trr_def'
    if b_is_trr and a_is_def:
        return 'trr_def'
    
    # Route runner to defender
    if a_is_rr and b_is_def and not a_is_trr:
        return 'rr_def'
    if b_is_rr and a_is_def and not b_is_trr:
        return 'rr_def'
    
    # Defender to defender
    if a_is_def and b_is_def:
        return 'def_def'
    
    # IGNORED: rr_rr, rr_trr (low priority)
    # if a_is_rr and b_is_rr:
    #     return 'rr_rr' or 'rr_trr' depending on targeting
    
    return None  # Edge not relevant

def compute_attention_prior(edge_type, distance, ball_progress, 
                            player_a_ball_dist, player_b_ball_dist):
    """
    Compute domain-informed attention prior P_ij ∈ [0, 1].
    
    Args:
        edge_type: One of the edge types
        distance: Euclidean distance between players (yards)
        ball_progress: 0 (release) to 1 (landing)
        player_a_ball_dist: Player A distance to ball landing spot
        player_b_ball_dist: Player B distance to ball landing spot
    
    Returns:
        prior (float): Attention prior in [0.05, 1.0]
    """
    # Start with base prior
    if edge_type not in BASE_PRIORS:
        return 0.5  # Default for unknown types
    
    prior = BASE_PRIORS[edge_type]
    
    # Distance penalty (exponential decay)
    distance_weight = np.exp(-distance / DISTANCE_CHARACTERISTIC_SCALE)
    
    # Ball proximity bonus (late in flight, being near landing matters more)
    ball_proximity_weight = 1.0
    if ball_progress > BALL_PROXIMITY_BONUS_START:
        avg_ball_dist = (player_a_ball_dist + player_b_ball_dist) / 2
        # Bonus grows as ball approaches AND players are near landing
        ball_proximity_weight = 1.0 + (ball_progress - BALL_PROXIMITY_BONUS_START) * np.exp(-avg_ball_dist / 5.0)
    
    # Combine multiplicatively
    final_prior = prior * distance_weight * ball_proximity_weight
    
    # Clip to reasonable range
    return np.clip(final_prior, 0.05, 1.0)

# ============================================================================
# 1. Load Data
# ============================================================================

print("STEP 1: LOADING DATA")
print("-" * 80)

print("Loading dataframe_a (processed node features)...")
df_a = pd.read_parquet(INPUT_DF_A)
print(f"  ✓ Loaded {len(df_a):,} rows")

print("\nLoading dataframe_b (play-level features with ball trajectory)...")
df_b = pd.read_parquet(INPUT_DF_B)
print(f"  ✓ Loaded {len(df_b):,} plays")

# ============================================================================
# 2. Filter Plays with Incomplete Ball Trajectory
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: FILTERING INCOMPLETE BALL TRAJECTORY DATA")
print("-" * 80)

required_ball_cols = ['start_ball_x', 'start_ball_y', 'ball_flight_frames', 
                      'ball_land_x', 'ball_land_y']

complete_trajectory_mask = True
for col in required_ball_cols:
    if col in df_b.columns:
        complete_trajectory_mask &= df_b[col].notna()

incomplete_count = (~complete_trajectory_mask).sum()

if incomplete_count > 0:
    print(f"⚠ Found {incomplete_count} play(s) with incomplete ball trajectory data")
    df_b = df_b[complete_trajectory_mask].copy()
    print(f"✓ Filtered to {len(df_b):,} plays with complete trajectory")
else:
    print(f"✓ All plays have complete ball trajectory data")

# Filter df_a to match
valid_plays = df_b[['game_id', 'play_id']].drop_duplicates()
df_a = df_a.merge(valid_plays, on=['game_id', 'play_id'], how='inner')
print(f"✓ Filtered df_a to {len(df_a):,} rows matching valid plays")

# ============================================================================
# 3. Pilot Mode Filtering
# ============================================================================

if PILOT_MODE:
    print("\n" + "=" * 80)
    print("STEP 3: PILOT MODE FILTERING")
    print("-" * 80)
    
    unique_games = df_a['game_id'].unique()
    pilot_games = unique_games[:PILOT_N_GAMES]
    
    df_a = df_a[df_a['game_id'].isin(pilot_games)].copy()
    df_b = df_b[df_b['game_id'].isin(pilot_games)].copy()
    
    print(f"✓ Filtered to {PILOT_N_GAMES} games: {pilot_games}")
    print(f"  df_a: {len(df_a):,} rows")
    print(f"  df_b: {len(df_b):,} plays")

# ============================================================================
# 4. Create Target Relationship Indicator
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: CREATING TARGET RELATIONSHIP INDICATORS")
print("-" * 80)

# Unified indicator: player is in THE critical relationship (targeted or covering)
df_a['in_tgt_relationship'] = (
    (df_a['isTargeted'] == 1) | 
    (df_a['targeted_defender'] == 1)
)

print(f"Players in target relationship: {df_a['in_tgt_relationship'].sum():,} / {len(df_a):,}")

# ============================================================================
# 5. Process Frames and Create Structured Edges
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: CREATING STRUCTURED EDGES")
print("-" * 80)

# Get unique frames
unique_frames = df_a.groupby(['game_id', 'play_id', 'frame_id']).size().reset_index()[
    ['game_id', 'play_id', 'frame_id']
]

print(f"Total frames to process: {len(unique_frames):,}")

output_rows = []
edge_type_counts = {et: 0 for et in BASE_PRIORS.keys()}
edge_type_counts['total'] = 0
frames_with_no_critical_edges = []

for idx, frame_row in unique_frames.iterrows():
    if idx % 1000 == 0 and idx > 0:
        print(f"  Processed {idx:,} / {len(unique_frames):,} frames ({100*idx/len(unique_frames):.1f}%)")
    
    game_id = frame_row['game_id']
    play_id = frame_row['play_id']
    frame_id = frame_row['frame_id']
    
    # Get players in this frame
    players = df_a[
        (df_a['game_id'] == game_id) &
        (df_a['play_id'] == play_id) &
        (df_a['frame_id'] == frame_id)
    ].copy()
    
    # Get play-level data
    play_data = df_b[
        (df_b['game_id'] == game_id) &
        (df_b['play_id'] == play_id)
    ]
    
    if play_data.empty:
        continue
    
    play_info = play_data.iloc[0]
    
    # Ball trajectory info from df_b
    start_ball_x = play_info['start_ball_x']
    start_ball_y = play_info['start_ball_y']
    total_frames = play_info['ball_flight_frames']
    
    # Ball landing coordinates from df_a (should be same for all players in frame)
    ball_land_x = players.iloc[0]['ball_land_x']
    ball_land_y = players.iloc[0]['ball_land_y']
    
    # Interpolate ball position at this frame
    ball_x_t, ball_y_t = interpolate_ball_position(
        start_ball_x, start_ball_y, ball_land_x, ball_land_y, frame_id, total_frames
    )
    
    # Calculate ball progress and frames to landing
    ball_progress = (frame_id - 1) / (total_frames - 1) if total_frames > 1 else 1.0
    ball_progress = np.clip(ball_progress, 0, 1)
    frames_to_landing = max(0, total_frames - frame_id)
    
    # Get time-varying distance threshold
    distance_threshold = get_distance_threshold(frames_to_landing, total_frames)
    
    # Create edges between all player pairs
    frame_has_critical_edge = False
    
    for i, player_a in players.iterrows():
        for j, player_b in players.iterrows():
            if i >= j:  # Skip self-loops and duplicate pairs (undirected)
                continue
            
            # Determine edge type
            edge_type = determine_edge_type(player_a, player_b)
            
            if edge_type is None:
                continue  # Skip irrelevant edges
            
            # Extract positions
            x_a, y_a = player_a['x'], player_a['y']
            x_b, y_b = player_b['x'], player_b['y']
            
            # Calculate distance
            x_dist = x_b - x_a
            y_dist = y_b - y_a
            e_dist = np.sqrt(x_dist**2 + y_dist**2)
            
            # TIERED FILTERING LOGIC
            is_critical_edge = edge_type in ['qb_trr', 'trr_def']
            
            if is_critical_edge:
                # ALWAYS include critical edges
                include_edge = True
                frame_has_critical_edge = True
            else:
                # For other edges, apply distance threshold
                include_edge = e_dist < distance_threshold
            
            if not include_edge:
                continue
            
            # Calculate ball proximity for both players
            playerA_dist_to_landing = np.sqrt(
                (x_a - ball_land_x)**2 + (y_a - ball_land_y)**2
            )
            playerB_dist_to_landing = np.sqrt(
                (x_b - ball_land_x)**2 + (y_b - ball_land_y)**2
            )
            
            playerA_dist_to_ball_current = np.sqrt(
                (x_a - ball_x_t)**2 + (y_a - ball_y_t)**2
            )
            playerB_dist_to_ball_current = np.sqrt(
                (x_b - ball_x_t)**2 + (y_b - ball_y_t)**2
            )
            
            # Compute attention prior
            attention_prior = compute_attention_prior(
                edge_type, e_dist, ball_progress,
                playerA_dist_to_landing, playerB_dist_to_landing
            )
            
            # Create edge ID for temporal continuity
            edge_id = create_edge_id(
                player_a['nfl_id'], player_b['nfl_id'], game_id, play_id
            )
            
            # Angular features
            relative_angle_o = angle_difference(player_a['o'], player_b['o'])
            relative_angle_dir = angle_difference(player_a['dir'], player_b['dir'])
            
            # Angle to ball landing (for both players)
            playerA_angle_to_ball_landing = np.degrees(np.arctan2(
                ball_land_y - y_a, ball_land_x - x_a
            )) % 360
            playerB_angle_to_ball_landing = np.degrees(np.arctan2(
                ball_land_y - y_b, ball_land_x - x_b
            )) % 360
            
            playerA_angle_to_ball_current = np.degrees(np.arctan2(
                ball_y_t - y_a, ball_x_t - x_a
            )) % 360
            playerB_angle_to_ball_current = np.degrees(np.arctan2(
                ball_y_t - y_b, ball_x_t - x_b
            )) % 360
            
            pairwise_angle_to_landing = angle_difference(
                playerA_angle_to_ball_landing, playerB_angle_to_ball_landing
            )
            
            # Velocity vectors
            playerA_v_x = player_a.get('v_x', 0)
            playerA_v_y = player_a.get('v_y', 0)
            playerB_v_x = player_b.get('v_x', 0)
            playerB_v_y = player_b.get('v_y', 0)
            
            relative_v_x = playerB_v_x - playerA_v_x
            relative_v_y = playerB_v_y - playerA_v_y
            relative_speed = np.sqrt(relative_v_x**2 + relative_v_y**2)
            
            # Acceleration vectors
            playerA_a_x = player_a.get('a_x', 0)
            playerA_a_y = player_a.get('a_y', 0)
            playerB_a_x = player_b.get('a_x', 0)
            playerB_a_y = player_b.get('a_y', 0)
            
            # Ball convergence (is distance to ball decreasing?)
            # Simplified: check if player is moving toward ball
            playerA_ball_convergence = 0
            if playerA_dist_to_ball_current > 0:
                ball_vec_x = ball_x_t - x_a
                ball_vec_y = ball_y_t - y_a
                dot_product = playerA_v_x * ball_vec_x + playerA_v_y * ball_vec_y
                playerA_ball_convergence = 1 if dot_product > 0 else 0
            
            playerB_ball_convergence = 0
            if playerB_dist_to_ball_current > 0:
                ball_vec_x = ball_x_t - x_b
                ball_vec_y = ball_y_t - y_b
                dot_product = playerB_v_x * ball_vec_x + playerB_v_y * ball_vec_y
                playerB_ball_convergence = 1 if dot_product > 0 else 0
            
            # Same team check
            same_team = (player_a.get('player_side') == player_b.get('player_side'))
            
            # Build edge row
            row = {
                # Identifiers
                'game_id': game_id,
                'play_id': play_id,
                'frame_id': frame_id,
                'edge_id': edge_id,
                'playerA_id': player_a['nfl_id'],
                'playerB_id': player_b['nfl_id'],
                
                # NEW v2: Edge type and priority
                'edge_type': edge_type,
                'attention_prior': attention_prior,
                'distance_threshold_at_t': distance_threshold,
                
                # NEW v2: Target relationship indicators
                'playerA_in_tgt_relationship': player_a['in_tgt_relationship'],
                'playerB_in_tgt_relationship': player_b['in_tgt_relationship'],
                'edge_involves_target': (player_a['in_tgt_relationship'] or player_b['in_tgt_relationship']),
                
                # Player A attributes
                'playerA_x': x_a,
                'playerA_y': y_a,
                'playerA_s': player_a['s'],
                'playerA_a': player_a['a'],
                'playerA_dir': player_a['dir'],
                'playerA_o': player_a['o'],
                'playerA_v_x': playerA_v_x,
                'playerA_v_y': playerA_v_y,
                'playerA_a_x': playerA_a_x,
                'playerA_a_y': playerA_a_y,
                'playerA_role': player_a.get('player_role'),
                'playerA_side': player_a.get('player_side'),
                'playerA_position': player_a.get('player_position'),
                
                # Player B attributes
                'playerB_x': x_b,
                'playerB_y': y_b,
                'playerB_s': player_b['s'],
                'playerB_a': player_b['a'],
                'playerB_dir': player_b['dir'],
                'playerB_o': player_b['o'],
                'playerB_v_x': playerB_v_x,
                'playerB_v_y': playerB_v_y,
                'playerB_a_x': playerB_a_x,
                'playerB_a_y': playerB_a_y,
                'playerB_role': player_b.get('player_role'),
                'playerB_side': player_b.get('player_side'),
                'playerB_position': player_b.get('player_position'),
                
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
                
                # Ball trajectory
                'ball_land_x': ball_land_x,
                'ball_land_y': ball_land_y,
                'ball_x_t': ball_x_t,
                'ball_y_t': ball_y_t,
                'ball_progress': ball_progress,
                'frames_to_landing': frames_to_landing,
                
                # Player-ball features
                'playerA_dist_to_ball_current': playerA_dist_to_ball_current,
                'playerA_angle_to_ball_current': playerA_angle_to_ball_current,
                'playerA_angle_to_ball_landing': playerA_angle_to_ball_landing,
                'playerA_ball_convergence': playerA_ball_convergence,
                
                'playerB_dist_to_ball_current': playerB_dist_to_ball_current,
                'playerB_angle_to_ball_current': playerB_angle_to_ball_current,
                'playerB_angle_to_ball_landing': playerB_angle_to_ball_landing,
                'playerB_ball_convergence': playerB_ball_convergence,
            }
            
            # Add coverage data if available
            if 'coverage_responsibility' in player_a.index:
                row['playerA_coverage'] = player_a['coverage_responsibility']
                row['playerB_coverage'] = player_b['coverage_responsibility']
            
            if 'targeted_defender' in player_a.index:
                row['playerA_targeted'] = player_a.get('targeted_defender', 0)
                row['playerB_targeted'] = player_b.get('targeted_defender', 0)
            
            # Add coverage scheme and probability distributions
            if 'coverage_scheme' in player_a.index:
                row['coverage_scheme'] = player_a['coverage_scheme']
            
            # Add coverage scheme probability columns (10 columns)
            coverage_prob_cols = [
                'COVER_0', 'COVER_1', 'COVER_2', 'COVER_3', 'COVER_4', 
                'COVER_6', 'PREVENT', 'GOAL_LINE', 'MISC', 'SHORT'
            ]
            for prob_col in coverage_prob_cols:
                full_col_name = f'coverage_scheme__{prob_col}'
                if full_col_name in player_a.index:
                    row[full_col_name] = player_a[full_col_name]
            
            output_rows.append(row)
            
            # Track edge type counts
            edge_type_counts[edge_type] += 1
            edge_type_counts['total'] += 1
    
    # Track frames with no critical edges (for diagnostics)
    if not frame_has_critical_edge:
        frames_with_no_critical_edges.append((game_id, play_id, frame_id))

print(f"  ✓ Processed all {len(unique_frames):,} frames")
print(f"  ✓ Created {len(output_rows):,} edges")

# ============================================================================
# 6. Create Output DataFrame
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: CREATING OUTPUT DATAFRAME")
print("-" * 80)

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
print(f"  Unique edge IDs: {df_c['edge_id'].nunique():,}")

print(f"\nEdge type distribution:")
for edge_type, count in edge_type_counts.items():
    if edge_type == 'total':
        continue
    pct = 100 * count / edge_type_counts['total'] if edge_type_counts['total'] > 0 else 0
    print(f"  {edge_type:12s}: {count:8,} ({pct:5.1f}%)")

print(f"\nEdge statistics:")
print(f"  Mean distance: {df_c['e_dist'].mean():.2f} yards")
print(f"  Mean attention prior: {df_c['attention_prior'].mean():.3f}")
print(f"  Same team edges: {df_c['same_team'].sum():,} ({100*df_c['same_team'].mean():.1f}%)")
print(f"  Edges involving target: {df_c['edge_involves_target'].sum():,} ({100*df_c['edge_involves_target'].mean():.1f}%)")

if len(frames_with_no_critical_edges) > 0:
    print(f"\n⚠ WARNING: {len(frames_with_no_critical_edges)} frames have NO critical edges (qb_trr or trr_def)")
    print(f"   First 5 frames: {frames_with_no_critical_edges[:5]}")

# ============================================================================
# 8. Generate Diagnostics (Optional)
# ============================================================================

if GENERATE_DIAGNOSTICS:
    print("\n" + "=" * 80)
    print("STEP 8: GENERATING DIAGNOSTICS")
    print("-" * 80)
    
    # 1. Edge Type Distribution CSV
    print("Creating edge type distribution CSV...")
    edge_stats = []
    for edge_type in BASE_PRIORS.keys():
        subset = df_c[df_c['edge_type'] == edge_type]
        if len(subset) > 0:
            edge_stats.append({
                'edge_type': edge_type,
                'count': len(subset),
                'pct_of_total': 100 * len(subset) / len(df_c),
                'avg_distance': subset['e_dist'].mean(),
                'avg_attention_prior': subset['attention_prior'].mean(),
                'avg_ball_progress': subset['ball_progress'].mean(),
            })
    
    edge_stats_df = pd.DataFrame(edge_stats)
    edge_stats_file = os.path.join(DIAGNOSTICS_DIR, 'edge_type_distribution.csv')
    edge_stats_df.to_csv(edge_stats_file, index=False)
    print(f"  ✓ Saved: {edge_stats_file}")
    
    # 2. Frame-Level Stats CSV
    print("Creating frame-level statistics CSV...")
    frame_stats = df_c.groupby(['game_id', 'play_id', 'frame_id']).agg({
        'edge_id': 'count',  # Total edges
        'attention_prior': 'mean',
        'e_dist': 'mean',
        'ball_progress': 'first',
    }).rename(columns={'edge_id': 'total_edges'}).reset_index()
    
    # Add counts by edge type
    for edge_type in BASE_PRIORS.keys():
        edge_type_counts_per_frame = df_c[df_c['edge_type'] == edge_type].groupby(
            ['game_id', 'play_id', 'frame_id']
        ).size().reset_index(name=f'n_{edge_type}')
        
        frame_stats = frame_stats.merge(
            edge_type_counts_per_frame,
            on=['game_id', 'play_id', 'frame_id'],
            how='left'
        )
        frame_stats[f'n_{edge_type}'] = frame_stats[f'n_{edge_type}'].fillna(0).astype(int)
    
    frame_stats_file = os.path.join(DIAGNOSTICS_DIR, 'frame_level_stats.csv')
    frame_stats.to_csv(frame_stats_file, index=False)
    print(f"  ✓ Saved: {frame_stats_file}")
    
    # 3. Sample Play Visualization (text)
    print("Creating sample play visualization...")
    sample_play = df_c.groupby(['game_id', 'play_id']).size().reset_index().iloc[0]
    sample_game_id = sample_play['game_id']
    sample_play_id = sample_play['play_id']
    
    sample_edges = df_c[
        (df_c['game_id'] == sample_game_id) &
        (df_c['play_id'] == sample_play_id)
    ]
    
    viz_file = os.path.join(DIAGNOSTICS_DIR, 'sample_play_visualization.txt')
    with open(viz_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"SAMPLE PLAY VISUALIZATION\n")
        f.write(f"Game: {sample_game_id}, Play: {sample_play_id}\n")
        f.write("=" * 80 + "\n\n")
        
        frames = sample_edges['frame_id'].unique()
        for frame in sorted(frames):
            frame_edges = sample_edges[sample_edges['frame_id'] == frame]
            f.write(f"\nFrame {frame}:\n")
            f.write(f"  Total edges: {len(frame_edges)}\n")
            
            for edge_type in BASE_PRIORS.keys():
                count = (frame_edges['edge_type'] == edge_type).sum()
                if count > 0:
                    f.write(f"    {edge_type}: {count}\n")
            
            # Check for missing critical edges
            has_qb_trr = (frame_edges['edge_type'] == 'qb_trr').any()
            has_trr_def = (frame_edges['edge_type'] == 'trr_def').any()
            
            if not has_qb_trr:
                f.write(f"    ⚠ WARNING: No qb_trr edges\n")
            if not has_trr_def:
                f.write(f"    ⚠ WARNING: No trr_def edges\n")
    
    print(f"  ✓ Saved: {viz_file}")
    
    # 4. Attention Prior Distribution Plot
    print("Creating attention prior distribution plot...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, edge_type in enumerate(BASE_PRIORS.keys()):
        if idx >= len(axes):
            break
        
        subset = df_c[df_c['edge_type'] == edge_type]['attention_prior']
        
        if len(subset) > 0:
            axes[idx].hist(subset, bins=30, alpha=0.7, edgecolor='black')
            axes[idx].set_title(f'{edge_type} (n={len(subset):,})')
            axes[idx].set_xlabel('Attention Prior')
            axes[idx].set_ylabel('Frequency')
            axes[idx].axvline(subset.mean(), color='red', linestyle='--', 
                             label=f'Mean: {subset.mean():.3f}')
            axes[idx].legend()
    
    # Remove unused subplots
    for idx in range(len(BASE_PRIORS), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plot_file = os.path.join(DIAGNOSTICS_DIR, 'attention_prior_distribution.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_file}")
    
    print("\n✅ All diagnostics generated successfully!")

# ============================================================================
# 9. Save Output
# ============================================================================

print("\n" + "=" * 80)
print("STEP 9: SAVING OUTPUT")
print("-" * 80)

# Adjust filename for pilot mode
if PILOT_MODE:
    output_file_final = OUTPUT_FILE.replace('v2.parquet', f'v2_pilot_{PILOT_N_GAMES}games.parquet')
else:
    output_file_final = OUTPUT_FILE

print("Writing to parquet...")
df_c.to_parquet(output_file_final, engine='pyarrow', index=False, compression='snappy')
print(f"✓ Saved to: {output_file_final}")
print(f"  File size: {os.path.getsize(output_file_final) / 1024 / 1024:.1f} MB")

# ============================================================================
# Complete
# ============================================================================

print("\n" + "=" * 80)
if PILOT_MODE:
    print("DATAFRAME C v2: COMPLETE [PILOT MODE]")
else:
    print("DATAFRAME C v2: COMPLETE")
print("=" * 80)

print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nOutput: {output_file_final}")
print(f"Rows: {len(df_c):,}")
print(f"Columns: {len(df_c.columns)}")

print(f"\n✨ V2 IMPROVEMENTS:")
print(f"  ✓ Structured edge creation (not fully connected)")
print(f"  ✓ 5 edge types created: {list(BASE_PRIORS.keys())}")
print(f"  ✓ Time-varying distance thresholds")
print(f"  ✓ Attention priors (P_ij) for domain-informed weighting")
print(f"  ✓ Temporal edge continuity tracking (edge_id)")
print(f"  ✓ Target relationship indicators")

if GENERATE_DIAGNOSTICS:
    print(f"\n📊 Diagnostics generated in: {DIAGNOSTICS_DIR}")

print("\n" + "=" * 80)