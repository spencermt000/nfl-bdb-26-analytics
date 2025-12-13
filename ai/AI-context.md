- [Introduction](#introduction)
	- [Project](#project)
	- [Goal](#goal)
	- [Approach](#approach)
- [STATUS](#status)
- [Decided-Upon](#decided-upon)
	- [Framework Breakdown](#framework-breakdown)
		- [Attention-Based Relationship Importance](#attention-based-relationship-importance)
		- [Pass Play Spatio-Temporal Interaction Graph Model](#pass-play-spatio-temporal-interaction-graph-model)
		- [Graph Structure](#graph-structure)
		- [Temporal Architecture](#temporal-architecture)
		- [Prediction Heads](#prediction-heads)
		- [Defender Impact Quantification](#defender-impact-quantification)
		- [Ideal Pursuit Trajectories](#ideal-pursuit-trajectories)
		- [References](#references)
	- [Final Data](#final-data)
		- [Standardization Process](#standardization-process)
		- [Dataframe A: Node-level information (Player-Frame Level)](#dataframe-a-node-level-information-player-frame-level)
		- [Dataframe B: Play-level information](#dataframe-b-play-level-information)
		- [Dataframe C: Edge-level relationships (Player-Player Interactions)](#dataframe-c-edge-level-relationships-player-player-interactions)
		- [Dataframe D: Temporal information (Frame-Level Aggregates)](#dataframe-d-temporal-information-frame-level-aggregates)
- [Scripts](#scripts)
		- [Code Guidelines](#code-guidelines)
	- [utils.py](#utilspy)
		- [function index](#function-index)
			- [supplement\_data](#supplement_data)
			- [add\_play\_direction](#add_play_direction)
			- [standardize\_play\_direction](#standardize_play_direction)
			- [e\_dist](#e_dist)
			- [angle\_difference](#angle_difference)
			- [create\_play\_level\_coverage](#create_play_level_coverage)
			- [find\_nearest\_v2](#find_nearest_v2)
			- [aggregate\_coverage\_to\_play\_level](#aggregate_coverage_to_play_level)
			- [load\_parquet\_to\_df](#load_parquet_to_df)
	- [dataframe\_a.py](#dataframe_apy)
		- [inputs](#inputs)
		- [outputs](#outputs)
	- [dataframe\_b.py](#dataframe_bpy)
		- [inputs](#inputs-1)
		- [outputs](#outputs-1)
	- [dataframe\_c.py](#dataframe_cpy)
		- [inputs](#inputs-2)
		- [outputs](#outputs-2)
	- [dataframe\_d.py](#dataframe_dpy)
		- [inputs](#inputs-3)
		- [outputs](#outputs-3)
	- [train\_completion.py](#train_completionpy)
		- [inputs](#inputs-4)
		- [outputs](#outputs-4)
		- [approach](#approach-1)
	- [train\_yac\_epa.py](#train_yac_epapy)
		- [inputs](#inputs-5)
		- [outputs](#outputs-5)
		- [approach](#approach-2)
- [Converting into a Metric](#converting-into-a-metric)
- [Conclusion \& Write Up](#conclusion--write-up)


# Introduction
## Project
**2026 NFL BigDataBowl, analytics submission**
*By Spencer Thompson*
**Due on: 12/16/2025**

## Goal
Quantify how a defender's actions while the ball is in the air affects the end result of the play.

## Approach
Looking specifically at how the defender impacts the completion of a pass, the YAC production after a catch is made, and the overall EPA of the play.  
We also are looking at different characteristics of the defender's movement as a proxy for "difficulty". These characteristics include the total distance the defender covers, the rate of his movement, the difficulty and quantity of changes of direction he makes, and the deviation from an "ideal" pursuit/action path.

# STATUS
- all the data processing scripts are complete! 
- the training DAG for training scripts train_yac_epa.py and train_completion.py is complete and working. 
- we need to adjust our framework though. 


# Decided-Upon
## Framework Breakdown
### Attention-Based Relationship Importance
**Learned Attention Adjacency**  
At each frame, the model builds a fully connected graph over players and uses a graph attention network to learn how important each edge (i,j) is, given the node and edge features. The resulting attention weights alpha_ij form a data-driven adjacency matrix that says "how much does player j matter for updating player i right now?".

**Domain-Biased Attention**  
Instead of leaving this attention purely free, you bias it with football priors via a multiplicative mask M_ij. You up-weight edges that are structurally important (e.g., targeted WR–defender pairs, defenders near the ball's landing point, key role pairings like WR–CB), and keep others at neutral weight. The final attention is alpha_ij = softmax(e_ij \* M_ij) so the model still learns how strong each relationship is, but is nudged to focus first on the interactions that make sense from a coverage and ball-tracking perspective.

### Pass Play Spatio-Temporal Interaction Graph Model
This project models each pass play as a spatio-temporal interaction graph over all players and the ball, and learns how defender behavior while the ball is in the air affects completion, YAC, and EPA outcomes.

### Graph Structure
At each frame t, players are nodes with rich features (tracking kinematics, role, route/coverage context, ball proximity), and pairwise relations (e.g., relative position, separation, matchup type) form fully connected edges.

A spatial graph-attention layer computes learnable attention weights alpha_ij(t) over player–player edges, optionally biased by domain priors on target–defender and ball-proximity interactions via a log-prior term log P_ij(t).

### Temporal Architecture
These spatial embeddings are then passed through temporal sequence modules (temporal GNNs and/or node-wise RNNs) organized as a Structural-RNN: nodeRNNs for semantic player groups (e.g., passer, targeted WR, CB/S/LB) and edgeRNNs for interaction types (e.g., target-WR–DB, WR–S), with factor sharing across similar roles.

### Prediction Heads
On top of this shared spatio-temporal representation, three heads are trained:

1.  Time-evolving completion head\*\* estimates p_t = P(completion | tracking up to t), allowing an air-time completion shift Delta_p = p_arrival - p_release.
2.  YAC-EPA head\*\* models f(state at catch) = E\[YAC-EPA\] for completed passes, using post-catch tracking to learn how coverage at the catch point translates into downstream value.
3.  EPA head\*\* estimates a value function V_t = E\[EPA | tracking up to t\], from which an air-time EPA shift Delta_V = V_arrival - V_release is derived.

### Defender Impact Quantification
Defender-level impact is quantified using a combination of interventions and attention-based credit assignment.  
For each defender d, "zeroing" or damping that node and its incident edges during the forward pass yields counterfactual curves p_t^(-d) and V_t^(-d), and marginal effects such as I_d^comp = Delta_p - Delta_p^(-d) and I_d^EPA = Delta_V - Delta_V^(-d).  
Attention weights from the spatial and temporal modules are integrated over air-time frames to form an attribution distribution over defenders, which can be used alone or combined with these intervention deltas.

### Ideal Pursuit Trajectories
Finally, a multimodal trajectory module inspired by Social-BiGAT and STGAT generates alternative air-time defender paths and, via a learned value head V(tau_d, others), defines "ideal pursuit" trajectories that minimize offensive EPA, allowing comparison between realized and ideal defender pursuit in both geometric and value terms.

### References
\[1\] AST-GNN.pdf  
\[2\] STGAT_Modeling_Spatial-Temporal_Interactions_for_Human_Trajectory_Prediction.pdf  
\[3\] Structural-RNN_Deep_Learning_on_Spatio-Temporal_Graphs.pdf  
\[4\] SocialBiGAT.pdf

&nbsp;
&nbsp;

&nbsp;
## Final Data
### Standardization Process
- uses the supplementary data to get yardline, possession team, and side of field
- standardizes gps data (x, y, direction of movement, direction of player orientation) to make each play offense on the left -> trying to score on the going right (assuming the field is horizontal with an endzone on the left and right sides)

### Dataframe A: Node-level information (Player-Frame Level)
**File Location:** `outputs/dataframe_a/v1.parquet`

**Summary:** Player-level tracking and contextual data at each frame, including kinematics, roles, coverage assignments, and ball proximity.

**Total Columns:** 51

**Column Categories:**

*Identifiers & Context (12):*
- game_id, play_id, frame_id, nfl_id
- player_name, player_height, player_weight, player_birth_date, player_age
- player_position, player_side, player_role

*Spatial Coordinates & Kinematics (6):*
- x, y - Current position
- s - Speed
- a - Acceleration
- dir - Direction of movement (degrees)
- o - Orientation/body facing direction (degrees)

*Post-Catch Position (3):*
- output_x, output_y - Final position after play ends
- dist_to_final_pos - Distance from current position to final position

*Ball Proximity (3):*
- ball_land_x, ball_land_y - Ball landing coordinates
- e_dist_ball_land - Euclidean distance to ball landing point

*Derived Kinematic Vectors (6):*
- v_x, v_y - Velocity components
- a_x, a_y - Acceleration components
- o_x, o_y - Orientation unit vectors

*Field Position (1):*
- los_dist - Distance to line of scrimmage

*Role Indicators (3):*
- isTargeted - Binary flag for targeted receiver
- isPasser - Binary flag for quarterback
- isRouteRunner - Binary flag for route-running receivers

*Coverage Assignment (4):*
- coverage_responsibility - Defender's coverage assignment
- targeted_defender - Binary flag if defender is covering targeted receiver
- coverage_responsibility_side - Side of field for coverage
- alignment - Defender's pre-snap alignment

*Coverage Scheme (11):*
- coverage_scheme - Primary coverage type
- coverage_scheme__COVER_0 through coverage_scheme__SHORT - Probability distributions for 10 coverage scheme types

### Dataframe B: Play-level information
**File Location:** `outputs/dataframe_b/v1.parquet`

**Summary:** Play-level context and outcomes, including game situation, formation, ball trajectory, and EPA metrics.

**Total Columns:** 30

**Column Categories:**

*Game Context (5):*
- season, week, play_id, game_id, quarter

*Down & Distance (3):*
- down, yards_to_go, yardline_number

*Teams & Field Position (3):*
- possession_team, defensive_team, yardline_side

*Play Outcome (3):*
- pass_result - Completion status
- yards_gained - Actual yards gained
- expected_points_added - EPA for the play

*Formation & Pre-Snap (4):*
- offense_formation - Offensive formation type
- receiver_alignment - Receiver alignment
- defenders_in_the_box - Number of defenders in box
- shotgun - Binary flag for shotgun formation

*Ball Trajectory (5):*
- start_ball_x, start_ball_y, start_ball_o - Ball release position and orientation
- ball_flight_distance - Total distance ball travels
- ball_flight_frames - Duration of ball flight in frames

*Pass Characteristics (3):*
- throw_direction - Direction of throw relative to field
- throw_type - Type of throw (e.g., short, deep)
- expected_points - EPA before play

*Advanced Metrics (7):*
- xpass - Expected pass completion probability
- cp - Actual completion probability
- comp_yac_epa - **TARGET for YAC model** - Expected points added from yards after catch on completions
- comp_air_epa - Expected points added from air yards on completions

### Dataframe C: Edge-level relationships (Player-Player Interactions)
**File Location:** `outputs/dataframe_c/v1.parquet`

**Summary:** Pairwise player-player interaction features at each frame, capturing spatial relationships, relative motion, and contextual matchup information.

**Total Columns:** 63

**Column Categories:**

*Identifiers (5):*
- game_id, play_id, frame_id, edge_id
- playerA_id, playerB_id

*Player A Features (10):*
- playerA_x, playerA_y - Position
- playerA_s, playerA_a - Speed and acceleration
- playerA_dir, playerA_o - Direction and orientation
- playerA_v_x, playerA_v_y - Velocity components
- playerA_a_x, playerA_a_y - Acceleration components

*Player A Context (3):*
- playerA_role, playerA_side, playerA_position

*Player B Features (10):*
- playerB_x, playerB_y - Position
- playerB_s, playerB_a - Speed and acceleration
- playerB_dir, playerB_o - Direction and orientation
- playerB_v_x, playerB_v_y - Velocity components
- playerB_a_x, playerB_a_y - Acceleration components

*Player B Context (3):*
- playerB_role, playerB_side, playerB_position

*Pairwise Spatial Features (5):*
- x_dist, y_dist - Component distances
- e_dist - Euclidean distance between players
- relative_angle_o - Relative body orientation angle
- relative_angle_dir - Relative movement direction angle

*Ball Proximity - Player A (4):*
- playerA_dist_to_landing - Distance to ball landing point
- playerA_dist_to_ball_current - Distance to current ball position
- playerA_angle_to_ball_current - Angle to current ball position
- playerA_angle_to_ball_landing - Angle to ball landing point

*Ball Proximity - Player B (4):*
- playerB_dist_to_landing - Distance to ball landing point
- playerB_dist_to_ball_current - Distance to current ball position
- playerB_angle_to_ball_current - Angle to current ball position
- playerB_angle_to_ball_landing - Angle to ball landing point

*Ball Convergence (3):*
- pairwise_angle_to_landing - Angle between players relative to ball landing
- playerA_ball_convergence - Rate at which player A approaches ball
- playerB_ball_convergence - Rate at which player B approaches ball

*Relative Motion (3):*
- relative_v_x, relative_v_y - Relative velocity components
- relative_speed - Relative speed magnitude

*Team Affiliation (1):*
- same_team - Binary flag indicating if players are on same team

*Ball State (5):*
- ball_land_x, ball_land_y - Ball landing coordinates
- ball_x_t, ball_y_t - Current ball position
- ball_progress - Proportion of ball flight completed (0-1)

*Temporal (1):*
- frames_to_landing - Frames remaining until ball lands

*Coverage Context (4):*
- playerA_coverage, playerB_coverage - Coverage assignments
- playerA_targeted, playerB_targeted - Binary flags for targeting
- coverage_scheme - Coverage scheme for this frame

### Dataframe D: Temporal information (Frame-Level Aggregates)
**File Location:** `outputs/dataframe_d/v1.parquet`

**Summary:** Frame-level aggregate statistics tracking the number of players at each frame, useful for temporal modeling and graph size tracking.

**Total Columns:** 7

**Columns:**
- game_id, play_id, frame_id - Identifiers
- num_frames_output - Total number of post-catch frames for this play
- n_players_tot - Total number of players tracked at this frame
- n_players_off - Number of offensive players at this frame
- n_players_def - Number of defensive players at this frame

**Purpose:** 
- Track player counts over time for graph construction
- Validate data completeness (should have 22 players typically)
- Support temporal sequence modeling by providing context on graph size changes
  
&nbsp;
&nbsp;
# Scripts
### Code Guidelines
- Modularize when possible
- Clearly defined inputs and outputs
- utilize print statements for debugging, troubleshooting, progress monitoring, and checkpoints
- Save progress as checkpoints, keep track of hyperparameters, etc. better safe than sorry
- Simplify where possible
- Maximize computational efficiency

## utils.py
General source of commonly used functions to be called across other scripts.

### function index

#### supplement_data
**What it does:** Merges supplemental play-by-play data with training data, then merges test output data (renaming x/y columns to x_output/y_output)

**Inputs:**
- supplemental_df: Supplemental play-by-play data
- train_df: Training tracking data
- test_df: Test output data with x, y coordinates

**Outputs:**
- better_train_df: Training data merged with supplemental data
- complete_df: Fully merged dataset with training, supplemental, and test data

#### add_play_direction
**What it does:** Determines whether each play is going left or right based on field position and team possession. Uses orientation stats to resolve midfield edge cases.

**Inputs:**
- df: DataFrame with columns game_id, play_id, absolute_yardline_number, yardline_side, defensive_team, possession_team, o

**Outputs:**
- DataFrame with added 'direction' column ('GOING LEFT' or 'GOING RIGHT')

#### standardize_play_direction
**What it does:** Standardizes coordinates and angles so all plays appear as if offense is moving right (increasing X)

**Inputs:**
- df: DataFrame with columns x, y, dir, o, direction

**Outputs:**
- DataFrame with standardized x, y, dir, o values (all plays oriented right)

#### e_dist
**What it does:** Calculates Euclidean distance between two points

**Inputs:**
- x1, x2, y1, y2: Coordinates of two points

**Outputs:**
- Float: Euclidean distance

#### angle_difference
**What it does:** Calculates the smallest difference between two angles (0-360 degrees)

**Inputs:**
- angle1: First angle in degrees
- angle2: Second angle in degrees

**Outputs:**
- Float: Smallest angular difference in range [0, 180]

#### create_play_level_coverage
**What it does:** Creates play-level coverage data by extracting coverage scheme at ball release (first frame of input data) and ball landing (first frame of output data) from SumerSports frame-level coverage data.

**Inputs:**
- input_tracking_df: Player tracking data before ball lands (input files) with game_id, play_id, frame_id
- output_tracking_df: Player tracking data after ball lands (output files) with game_id, play_id, frame_id
- sumer_coverage_frame_df: SumerSports frame-level coverage data with coverage schemes and probabilities

**Outputs:**
- DataFrame with play-level coverage featuring:
  - game_id, play_id
  - start_scheme, start_cover_0, start_cover_1, ..., start_cover_SHORT (coverage at ball release)
  - land_scheme, land_cover_0, land_cover_1, ..., land_cover_SHORT (coverage at ball landing)

#### find_nearest_v2
**What it does:** For each player at each frame, finds their n nearest same-team and opposing-team players by Euclidean distance

**Inputs:**
- df: DataFrame with columns game_id, play_id, frame_id, nfl_id, x, y, dir, player_side
- n_same_team: Number of nearest same-team players to find
- n_opp_team: Number of nearest opposing-team players to find

**Outputs:**
- DataFrame with columns for game_id, play_id, frame_id, root_player_id, total_players, total_same_team, total_opp_team, and nearest_same_1, nearest_same_2, ..., nearest_opp_1, nearest_opp_2, etc.

#### aggregate_coverage_to_play_level
**What it does:** Aggregates frame-level coverage data to play-level by extracting coverage at ball release (first frame) and ball arrival (last frame before catch). Also calculates if coverage changed during the play.

**Inputs:**
- coverage_frame_df: SumerSports frame-level coverage data with game_id, play_id, frame_id, coverage_scheme, and probability columns
- tracking_df: Tracking data with game_id, play_id, frame_id, num_frames_output (used to determine first/last frames)

**Outputs:**
- DataFrame with play-level coverage data:
  - game_id, play_id
  - coverage_at_release, coverage_at_arrival (categorical coverage schemes)
  - coverage_changed (1 if different, 0 if same)
  - prob_cover_0_release, prob_cover_1_release, etc. (probabilities at release)
  - prob_cover_0_arrival, prob_cover_1_arrival, etc. (probabilities at arrival)

#### load_parquet_to_df
**What it does:** Loads a parquet file into a pandas DataFrame with error checking and logging.

**Inputs:**
- file_path (str): Path to the parquet file
- df_name (str, optional): Name for the DataFrame (for logging purposes)

**Outputs:**
- pd.DataFrame: Loaded dataframe

**Example usage:**
```python
df_a = load_parquet_to_df('outputs/dataframe_a/v2.parquet', 'df_a')
```

## dataframe_a.py
used to create the node/player level attributes, involves player coordinate data, player role information, etc
### inputs

### outputs

## dataframe_b.py
### inputs
### outputs

## dataframe_c.py
### inputs
### outputs

## dataframe_d.py
### inputs
### outputs

&nbsp;
## train_completion.py
Script that trains a graph attention network to predict **pass completion probability** as a **binary classification task**. This is the first of two prediction models in the project.

### inputs
- `outputs/dataframe_a/v1.parquet` - Node features (player-level data)
- `outputs/dataframe_b/v1.parquet` - Play context and ball trajectory data
- `outputs/dataframe_c/v1.parquet` or `v1_pilot_3games_old.parquet` - Edge features (player-player interaction data)
- `outputs/dataframe_d/v1.parquet` - Additional data (loaded but not actively used in current implementation)

### outputs
- `model_outputs/attention/best_model.pth` - Best model checkpoint (lowest validation loss)
- `model_outputs/attention/final_model.pth` - Final model state after training completes
- `model_outputs/attention/nan_batch_{epoch}_{batch_idx}.pt` - Debugging data for any batches that produce NaN/Inf values

### approach

**Model Task:** Binary classification to predict pass completion (0 = incomplete, 1 = complete)

**Architecture:**
1. **Multi-Head Graph Attention Network** with 4 attention heads
   - Node encoder: Processes player features (16 features including position, velocity, acceleration, orientation, ball distance, role indicators)
   - Edge encoder: Processes interaction features (21 features including distances, angles, ball-related metrics, team affiliation, temporal progress)
   - Query/Key/Value projections for multi-head attention
   - Edge-aware attention scoring that incorporates both node and edge information
   - Softmax aggregation per destination node
   - Residual connection from input to output

2. **Completion Prediction Head**
   - Takes graph-level embedding (mean of all node embeddings)
   - 2-layer MLP: 128 → 64 (ReLU + Dropout 0.2) → 1 (Sigmoid)
   - Outputs probability of completion

**Node Features (16 total):**
- Spatial: x, y, s (speed), a (acceleration), dir (direction), o (orientation)
- Velocity vectors: v_x, v_y
- Acceleration vectors: a_x, a_y
- Orientation vectors: o_x, o_y
- Ball proximity: e_dist_ball_land (Euclidean distance to ball landing point)
- Role indicators: isPasser, isTargeted, isRouteRunner

**Edge Features (21 total):**
- Spatial distances: e_dist (Euclidean distance), x_dist, y_dist
- Relative angles: relative_angle_o, relative_angle_dir
- Ball-related distances: playerA_dist_to_landing, playerB_dist_to_landing, playerA_dist_to_ball_current, playerB_dist_to_ball_current
- Ball-related angles: playerA_angle_to_ball_current, playerB_angle_to_ball_current, playerA_angle_to_ball_landing, playerB_angle_to_ball_landing
- Ball convergence: playerA_ball_convergence, playerB_ball_convergence
- Relative velocity: relative_v_x, relative_v_y, relative_speed
- Team affiliation: same_team (binary)
- Temporal: ball_progress, frames_to_landing

**Data Processing:**
1. Filters to relevant players only: Passers, route runners, and coverage defenders
2. Filters edges to only include interactions between relevant players
3. Z-score normalizes all features using mean and std from training data
4. Creates graph structure with variable number of nodes/edges per frame
5. Uses 80/20 train/validation split

**Training Configuration:**
- Loss function: Binary Cross Entropy (BCE)
- Optimizer: Adam with learning rate 0.001
- Batch size: 1 (due to variable graph sizes)
- Training time limit: 5 minutes (configurable via MAX_TRAIN_TIME_MINUTES)
- Pilot mode available: Can train on 3-game subset for rapid testing
- Device: GPU if available, otherwise CPU

**Robustness Features:**
- Handles empty graphs by skipping batches
- Transposes edge_index to correct [2, N] shape if needed
- Detects NaN/Inf in predictions and saves problematic batches for debugging
- Clamps predictions and targets to [1e-7, 1-1e-7] range to prevent numerical instability
- Logs progress every 100 batches
- Saves best model based on validation loss

**Key Implementation Details:**
- Each graph represents one frame of one play
- Graph nodes = filtered players (QB + route runners + coverage defenders)
- Graph edges = all pairwise interactions between these players
- Target = binary completion outcome from df_b
- Model learns to predict completion probability given the spatial-temporal state at each frame

## train_yac_epa.py
Script that trains a graph attention network to predict **completed YAC EPA** as a **continuous regression task**. This is the second prediction model that focuses specifically on yards-after-catch expected points added for completed passes.

### inputs
- `outputs/dataframe_a/v1.parquet` - Node features (player-level data)
- `outputs/dataframe_b/v1.parquet` - Play context data including **comp_yac_epa** target variable
- `outputs/dataframe_c/v1.parquet` or `v1_pilot_3games.parquet` - Edge features (player-player interaction data)

### outputs
- `model_outputs/attention_yac/yac_model.pth` - Trained YAC EPA prediction model (saved at best validation loss)

### approach

**Model Task:** Regression to predict comp_yac_epa (continuous value representing expected points added from yards after catch on completed passes)

**Architecture:**
1. **Multi-Head Graph Attention Network** with 4 attention heads
   - Identical attention mechanism to completion model
   - Node encoder: Processes 16 player features
   - Edge encoder: Processes 21 interaction features (UPDATED to match completion model)
   - Query/Key/Value projections for multi-head attention
   - Edge-aware attention scoring
   - Softmax aggregation per destination node
   - Residual connection

2. **YAC EPA Prediction Head**
   - Takes graph-level embedding (mean of all node embeddings)
   - 2-layer MLP: 128 → 64 (ReLU + Dropout 0.2) → 1 (Linear output)
   - Outputs continuous YAC EPA value (no activation on final layer for regression)

**Node Features (16 total - identical to completion model):**
- Spatial: x, y, s, a, dir, o
- Velocity vectors: v_x, v_y
- Acceleration vectors: a_x, a_y
- Orientation vectors: o_x, o_y
- Ball proximity: e_dist_ball_land
- Role indicators: isPasser, isTargeted, isRouteRunner

**Edge Features (21 total - UPDATED to match completion model):**
- Spatial distances: e_dist, x_dist, y_dist
- Relative angles: relative_angle_o, relative_angle_dir
- Ball-related distances: playerA_dist_to_landing, playerB_dist_to_landing, playerA_dist_to_ball_current, playerB_dist_to_ball_current
- **Ball-related angles (NEW):** playerA_angle_to_ball_current, playerB_angle_to_ball_current, playerA_angle_to_ball_landing, playerB_angle_to_ball_landing
- Ball convergence: playerA_ball_convergence, playerB_ball_convergence
- Relative velocity: relative_v_x, relative_v_y, relative_speed
- Team affiliation: same_team
- Temporal: ball_progress, frames_to_landing

**Data Processing:**
1. Filters to relevant players: Passers, route runners, and coverage defenders
2. Filters edges between relevant players only
3. Z-score normalizes features using training set statistics
4. Handles missing comp_yac_epa values by defaulting to 0.0
5. Creates variable-size graphs per frame
6. Uses 80/20 train/validation split with random seed 42

**Training Configuration:**
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam with learning rate 0.001
- Batch size: 1 (variable graph sizes)
- Training time limit: 8 minutes (configurable via MAX_TRAIN_TIME_MINUTES)
- Pilot mode available: 3-game subset for rapid iteration
- Device: GPU if available, otherwise CPU

**Robustness Features:**
- Skips batches with empty edge_index or edge_features
- Transposes edge_index to [2, N_edges] format if needed
- Validates edge_index shape before forward pass
- Handles missing comp_yac_epa values gracefully
- Saves best model based on lowest validation MSE
- Progress logging every 500 batches
- Time-based early stopping

**Key Differences from Completion Model:**
1. **Target variable:** comp_yac_epa (continuous) instead of completion (binary)
2. **Loss function:** MSE instead of BCE
3. **Output head:** Linear output instead of Sigmoid
4. **Training time:** 8 minutes instead of 5 minutes
5. **Edge features:** Updated to include 4 additional angle features (21 total vs previous 17)
6. **Progress logging:** Every 500 batches instead of 100

**Key Implementation Details:**
- Each graph = one frame of one play
- Target = comp_yac_epa from df_b (expected points added from YAC on completed passes)
- Model learns to predict YAC EPA given spatial-temporal state at catch point
- Only relevant for completed passes (though trained on all frames)
- Complements completion model to provide full EPA decomposition: completion probability × YAC EPA

&nbsp;
&nbsp;

# Converting into a Metric

&nbsp;
&nbsp;
# Conclusion & Write Up