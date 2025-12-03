- [STEP 1: PREPROCESS](#step-1-preprocess)
  - [Dataframe A: Node-Level](#dataframe-a-node-level)
    - [Summary](#summary)
    - [Columns](#columns)
    - [Standardization](#standardization)
  - [Dataframe B: Play-Level](#dataframe-b-play-level)
    - [Summary](#summary-1)
    - [Final Columns \& names](#final-columns--names)
  - [Dataframe C: Relationship-level](#dataframe-c-relationship-level)
    - [Summary](#summary-2)
    - [Columns](#columns-1)
  - [Dataframe D: Temporal-level](#dataframe-d-temporal-level)
    - [Summary](#summary-3)
    - [Columns](#columns-2)


# STEP 1: PREPROCESS
## Dataframe A: Node-Level
### Summary

### Columns
- player_height
- player_weight
- player_birth_date
- player_position
- player_side
- player_role
- x (standardized)
- y (standardized)
- s
- a
- dir (standardized)
- o (standardized)
- e_dist_ball_land
- los_dist
- isTargeted
- isPasser
- isRouteRunner

### Standardization 
## Dataframe B: Play-Level
### Summary

### Final Columns & names
- season
- week
- play_id
- quarter
- down
- yards_to_go
- possession_team
- defensive_team
- yardline_side
- yardline_number
- ps_pos_team_score
- ps_def_team_score
- pos_team_wp
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
- pre_penalty_yards_gained
- yards_gained
- expected_points
- expected_points_added
- pos_team_wp
- pos_team_wpa
- def_team_wpa
- quarter_seconds_remaining
- game_seconds_remaining
- half_seconds_remaining
- goal_to_go
- shotgun
- no_huddle
- posteam_timeouts_remaining
- defteam_timeouts_remaining
- air_yards
- yards_after_catch
- air_epa
- comp_air_epa
- comp_yac_epa
- qb_hit
- touchdown
- surface
- roof
- total_line
- spread_line
- series
- play_in_drive
- pbu
- cp
- cpoe
- xpass
- old_game_id
- xyac_epa
- xyac_median_yardage
- xyac_mean_yardage
- xyac_success
  
## Dataframe C: Relationship-level
### Summary

### Columns
- game_id
- play_id
- frame_id
- playerA_id
- playerA_x
- playerA_y
- playerA_s
- playerA_a
- playerA_dir
- playerA_o
- playerA_role
- playerA_side
- playerA_position
- playerA_height
- playerA_weight
- playerB_id
- playerB_x
- playerB_y
- playerB_s
- playerB_a
- playerB_dir
- playerB_o
- playerB_role
- playerB_side
- playerB_position
- playerB_height
- playerB_weight
- x_dist (2d distance between the x coordinates)
- y_dist (2d distance between the y coordinates)
- e_dist (euclidean distance between Player A x/y, and player B x/y)
- relative_angle_o (dif between the 2 angles {0deg to 360d} in the o column)
- relative_angle_dir (dif between the 2 angles {0deg to 360d} in the dir column)
- same_team (1 if playerA_side == playerB_side, 0 if not)
- player_rel_index (playerA_id, "_", playerB_id)

## Dataframe D: Temporal-level
### Summary

### Columns
- game_id
- play_id
- num_frames_output
- frame_id
- n_players_tot
- n_players_off
- n_players_def



