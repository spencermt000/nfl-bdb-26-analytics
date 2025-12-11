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

# TO-DO
- rework and clean up the data processing scripts
	- add SumerSports supplementary data to the relevant dataframes
	- retain more features in dataframeB


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

# Raw Data
## Player Tracking Data
### "Inputs" (pre-landing)
**Summary:**  the player tracking (spatio-temporal) data for each player for each frame of each play BEFORE the ball is caught/landed (hence, the "inputs" names)

**File Location and Name:** 
- data/train/input_2023_w01.csv
- one for each week (eg. input_2023_w02.csv)

**Columns:**
- game_id
- play_id
- player_to_predict
- nfl_id
- frame_id
- play_direction
- absolute_yardline_number
- player_name
- player_height
- player_weight
- player_birth_date
- player_position
- player_side
- player_role
- x
- y
- s
- a
- dir
- o
- num_frames_output
- ball_land_x
- ball_land_y
  
### "Outputs" (post-landing)
**Summary:**
Reduced-columns player tracking data AFTER the ball is landed/caught

**File Location and Name:**
- data/train/output_2023_w01.csv
- one for each week (eg. output_2023_w02.csv)

**Columns:** 
- game_id
- play_id
- nfl_id
- frame_id
- x
- y

&nbsp;
## Supplementary Data
###  nflfastR supplementary data
**Summary:**
**File Location and Name:**
- sdv_pbp_raw.parquet
**Columns:** 
- play_id
- game_id
- old_game_id
- home_team
- away_team
- season_type
- week
- posteam
- posteam_type
- defteam
- side_of_field
- yardline_100
- game_date
- quarter_seconds_remaining
- half_seconds_remaining
- game_seconds_remaining
- game_half
- quarter_end
- drive
- sp
- qtr
- down
- goal_to_go
- time
- yrdln
- ydstogo
- ydsnet
- desc
- play_type
- yards_gained
- shotgun
- no_huddle
- qb_dropback
- qb_kneel
- qb_spike
- qb_scramble
- pass_length
- pass_location
- air_yards
- yards_after_catch
- run_location
- run_gap
- field_goal_result
- kick_distance
- extra_point_result
- two_point_conv_result
- home_timeouts_remaining
- away_timeouts_remaining
- timeout
- timeout_team
- td_team
- td_player_name
- td_player_id
- posteam_timeouts_remaining
- defteam_timeouts_remaining
- total_home_score
- total_away_score
- posteam_score
- defteam_score
- score_differential
- posteam_score_post
- defteam_score_post
- score_differential_post
- no_score_prob
- opp_fg_prob
- opp_safety_prob
- opp_td_prob
- fg_prob
- safety_prob
- td_prob
- extra_point_prob
- two_point_conversion_prob
- ep
- epa
- total_home_epa
- total_away_epa
- total_home_rush_epa
- total_away_rush_epa
- total_home_pass_epa
- total_away_pass_epa
- air_epa
- yac_epa
- comp_air_epa
- comp_yac_epa
- total_home_comp_air_epa
- total_away_comp_air_epa
- total_home_comp_yac_epa
- total_away_comp_yac_epa
- total_home_raw_air_epa
- total_away_raw_air_epa
- total_home_raw_yac_epa
- total_away_raw_yac_epa
- wp
- def_wp
- home_wp
- away_wp
- wpa
- vegas_wpa
- vegas_home_wpa
- home_wp_post
- away_wp_post
- vegas_wp
- vegas_home_wp
- total_home_rush_wpa
- total_away_rush_wpa
- total_home_pass_wpa
- total_away_pass_wpa
- air_wpa
- yac_wpa
- comp_air_wpa
- comp_yac_wpa
- total_home_comp_air_wpa
- total_away_comp_air_wpa
- total_home_comp_yac_wpa
- total_away_comp_yac_wpa
- total_home_raw_air_wpa
- total_away_raw_air_wpa
- total_home_raw_yac_wpa
- total_away_raw_yac_wpa
- punt_blocked
- first_down_rush
- first_down_pass
- first_down_penalty
- third_down_converted
- third_down_failed
- fourth_down_converted
- fourth_down_failed
- incomplete_pass
- touchback
- interception
- punt_inside_twenty
- punt_in_endzone
- punt_out_of_bounds
- punt_downed
- punt_fair_catch
- kickoff_inside_twenty
- kickoff_in_endzone
- kickoff_out_of_bounds
- kickoff_downed
- kickoff_fair_catch
- fumble_forced
- fumble_not_forced
- fumble_out_of_bounds
- solo_tackle
- safety
- penalty
- tackled_for_loss
- fumble_lost
- own_kickoff_recovery
- own_kickoff_recovery_td
- qb_hit
- rush_attempt
- pass_attempt
- sack
- touchdown
- pass_touchdown
- rush_touchdown
- return_touchdown
- extra_point_attempt
- two_point_attempt
- field_goal_attempt
- kickoff_attempt
- punt_attempt
- fumble
- complete_pass
- assist_tackle
- lateral_reception
- lateral_rush
- lateral_return
- lateral_recovery
- passer_player_id
- passer_player_name
- passing_yards
- receiver_player_id
- receiver_player_name
- receiving_yards
- rusher_player_id
- rusher_player_name
- rushing_yards
- lateral_receiver_player_id
- lateral_receiver_player_name
- lateral_receiving_yards
- lateral_rusher_player_id
- lateral_rusher_player_name
- lateral_rushing_yards
- lateral_sack_player_id
- lateral_sack_player_name
- interception_player_id
- interception_player_name
- lateral_interception_player_id
- lateral_interception_player_name
- punt_returner_player_id
- punt_returner_player_name
- lateral_punt_returner_player_id
- lateral_punt_returner_player_name
- kickoff_returner_player_name
- kickoff_returner_player_id
- lateral_kickoff_returner_player_id
- lateral_kickoff_returner_player_name
- punter_player_id
- punter_player_name
- kicker_player_name
- kicker_player_id
- own_kickoff_recovery_player_id
- own_kickoff_recovery_player_name
- blocked_player_id
- blocked_player_name
- tackle_for_loss_1_player_id
- tackle_for_loss_1_player_name
- tackle_for_loss_2_player_id
- tackle_for_loss_2_player_name
- qb_hit_1_player_id
- qb_hit_1_player_name
- qb_hit_2_player_id
- qb_hit_2_player_name
- forced_fumble_player_1_team
- forced_fumble_player_1_player_id
- forced_fumble_player_1_player_name
- forced_fumble_player_2_team
- forced_fumble_player_2_player_id
- forced_fumble_player_2_player_name
- solo_tackle_1_team
- solo_tackle_2_team
- solo_tackle_1_player_id
- solo_tackle_2_player_id
- solo_tackle_1_player_name
- solo_tackle_2_player_name
- assist_tackle_1_player_id
- assist_tackle_1_player_name
- assist_tackle_1_team
- assist_tackle_2_player_id
- assist_tackle_2_player_name
- assist_tackle_2_team
- assist_tackle_3_player_id
- assist_tackle_3_player_name
- assist_tackle_3_team
- assist_tackle_4_player_id
- assist_tackle_4_player_name
- assist_tackle_4_team
- tackle_with_assist
- tackle_with_assist_1_player_id
- tackle_with_assist_1_player_name
- tackle_with_assist_1_team
- tackle_with_assist_2_player_id
- tackle_with_assist_2_player_name
- tackle_with_assist_2_team
- pass_defense_1_player_id
- pass_defense_1_player_name
- pass_defense_2_player_id
- pass_defense_2_player_name
- fumbled_1_team
- fumbled_1_player_id
- fumbled_1_player_name
- fumbled_2_player_id
- fumbled_2_player_name
- fumbled_2_team
- fumble_recovery_1_team
- fumble_recovery_1_yards
- fumble_recovery_1_player_id
- fumble_recovery_1_player_name
- fumble_recovery_2_team
- fumble_recovery_2_yards
- fumble_recovery_2_player_id
- fumble_recovery_2_player_name
- sack_player_id
- sack_player_name
- half_sack_1_player_id
- half_sack_1_player_name
- half_sack_2_player_id
- half_sack_2_player_name
- return_team
- return_yards
- penalty_team
- penalty_player_id
- penalty_player_name
- penalty_yards
- replay_or_challenge
- replay_or_challenge_result
- penalty_type
- defensive_two_point_attempt
- defensive_two_point_conv
- defensive_extra_point_attempt
- defensive_extra_point_conv
- safety_player_name
- safety_player_id
- season
- cp
- cpoe
- series
- series_success
- series_result
- order_sequence
- start_time
- time_of_day
- stadium
- weather
- nfl_api_id
- play_clock
- play_deleted
- play_type_nfl
- special_teams_play
- st_play_type
- end_clock_time
- end_yard_line
- fixed_drive
- fixed_drive_result
- drive_real_start_time
- drive_play_count
- drive_time_of_possession
- drive_first_downs
- drive_inside20
- drive_ended_with_score
- drive_quarter_start
- drive_quarter_end
- drive_yards_penalized
- drive_start_transition
- drive_end_transition
- drive_game_clock_start
- drive_game_clock_end
- drive_start_yard_line
- drive_end_yard_line
- drive_play_id_started
- drive_play_id_ended
- away_score
- home_score
- location
- result
- total
- spread_line
- total_line
- div_game
- roof
- surface
- temp
- wind
- home_coach
- away_coach
- stadium_id
- game_stadium
- aborted_play
- success
- passer
- passer_jersey_number
- rusher
- rusher_jersey_number
- receiver
- receiver_jersey_number
- pass
- rush
- first_down
- special
- play
- passer_id
- rusher_id
- receiver_id
- name
- jersey_number
- id
- fantasy_player_name
- fantasy_player_id
- fantasy
- fantasy_id
- out_of_bounds
- home_opening_kickoff
- qb_epa
- xyac_epa
- xyac_mean_yardage
- xyac_median_yardage
- xyac_success
- xyac_fd
- xpass
- pass_oe


### BDB supplementary data
**Summary:**
**File Location and Name:**
**Columns:** 
- game_id
- season
- week
- game_date
- game_time_eastern
- home_team_abbr
- visitor_team_abbr
- play_id
- play_description
- quarter
- game_clock
- down
- yards_to_go
- possession_team
- defensive_team
- yardline_side
- yardline_number
- pre_snap_home_score
- pre_snap_visitor_score
- play_nullified_by_penalty
- pass_result
- pass_length
- offense_formation
- receiver_alignment
- route_of_targeted_receiver
- play_action
- dropback_type
- dropback_distance
- pass_location_type
- defenders_in_the_box
- team_coverage_man_zone
- team_coverage_type
- penalty_yards
- pre_penalty_yards_gained
- yards_gained
- expected_points
- expected_points_added
- pre_snap_home_team_win_probability
- pre_snap_visitor_team_win_probability
- home_team_win_probability_added
- visitor_team_win_probility_added


### SumerSports supplementary data
#### sumer_coverages_frame.parquet
**Summary:** the predicted coverage scheme at each frame/time-step for each play

**Columns:** 
- game_id
- play_id
- frame_id
- coverage_scheme
- coverage_scheme__COVER_0
- coverage_scheme__COVER_1
- coverage_scheme__COVER_2
- coverage_scheme__COVER_2_MAN
- coverage_scheme__COVER_3
- coverage_scheme__COVER_4
- coverage_scheme__COVER_6
- coverage_scheme__MISC
- coverage_scheme__PREVENT
- coverage_scheme__REDZONE
- coverage_scheme__SHORT
  
#### sumer_coverages_player_play.parquet
**Summary:** each defenders's coverage responsibility on an individual play

**Columns:** 
- game_id
- play_id
- nfl_id
- coverage_responsibility
- targeted_defender
- coverage_responsibility_side
- alignment

&nbsp;
## Merged Data
### 2023_inputs_all.parquet
**Summary:** merged dataset of all the weeks of the raw input_2023_wXX.csv files

**Columns:** 
- game_id
- play_id
- player_to_predict
- nfl_id
- frame_id
- play_direction
- absolute_yardline_number
- player_name
- player_height
- player_weight
- player_birth_date
- player_position
- player_side
- player_role
- x
- y
- s
- a
- dir
- o
- num_frames_output
- ball_land_x
- ball_land_y

  
### 2023_inputs_all_standardized.parquet
**Summary:** merged dataset of all the weeks of the raw input_2023_wXX.csv files, but the x, y, dir, and o columns are standardized to be consistent across plays/field positions
**Columns:** 
- game_id
- play_id
- player_to_predict
- nfl_id
- frame_id
- play_direction
- absolute_yardline_number
- player_name
- player_height
- player_weight
- player_birth_date
- player_position
- player_side
- player_role
- x
- y
- s
- a
- dir
- o
- num_frames_output
- ball_land_x
- ball_land_y

  
&nbsp;
## Final Data
### Standardization Process
- uses the supplementary data to get yardline, possession team, and side of field
- standardizes gps data (x, y, direction of movement, direction of player orientation) to make each play offense on the left -> trying to score on the going right (assuming the field is horizontal with an endzone on the left and right sides)

### Node-level information

### Play-level information

### Edge-level relationships

### Temporal information
  
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
general source of commonly used functions to be called across other scripts
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

#### find_nearest_v2
**What it does:** For each player at each frame, finds their n nearest same-team and opposing-team players by Euclidean distance

**Inputs:**
- df: DataFrame with columns game_id, play_id, frame_id, nfl_id, x, y, dir, player_side
- n_same_team: Number of nearest same-team players to find
- n_opp_team: Number of nearest opposing-team players to find

**Outputs:**
- DataFrame with columns for game_id, play_id, frame_id, root_player_id, and nearest_same_1, nearest_same_2, ..., nearest_opp_1, nearest_opp_2, etc.



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
## {TO BE DETERMINED}

## {TO BE DETERMINED}

## {TO BE DETERMINED}

## {TO BE DETERMINED}


&nbsp;
&nbsp;

# Converting into a Metric

&nbsp;
&nbsp;
# Conclusion & Write Up
