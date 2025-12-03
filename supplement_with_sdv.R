library(tidyverse)
library(nflreadr)
library(nflfastR)
library(arrow)
options(scipen = 999)

sup_file <- paste0('data', '/supplementary_data.csv')
sup_raw <- read_csv(sup_file, show_col_types = FALSE)

# sdv_pbp is data/sdv_raw_pbp.parquet

sup_raw <- sup_raw %>%
  mutate(index = paste0(game_id, "_", play_id))

sdv_pbp <- sdv_pbp %>%
  mutate(index = paste0(old_game_id, "_", play_id))

sdv_pbp <- sdv_pbp %>%
  group_by(game_id, posteam, drive) %>%
  mutate(play_in_drive = row_number()) %>%
  ungroup() %>%
  mutate(
    tfl = if_else(!is.na(tackle_for_loss_1_player_id) & !is.na(tackle_for_loss_2_player_id), 1, 0),
    pbu = if_else((!is.na(pass_defense_1_player_id) & !is.na(pass_defense_2_player_id)) | interception == 1, 1, 0),
    atkl = if_else(!is.na(solo_tackle_1_player_id) & !is.na(solo_tackle_2_player_id), 1, 0),
    stkl = if_else(!is.na(assist_tackle_1_player_id) | !is.na(assist_tackle_2_player_id) | 
                     !is.na(assist_tackle_3_player_id) | !is.na(assist_tackle_4_player_id), 1, 0)
  )
 

sdv_cols <- c(
  "quarter_seconds_remaining", "game_seconds_remaining", "half_seconds_remaining",
  "goal_to_go", "shotgun", "no_huddle", "posteam_timeouts_remaining", "defteam_timeouts_remaining",
  "air_yards", "yards_after_catch", "air_epa", "comp_air_epa", "comp_yac_epa", "qb_hit", "touchdown", 
  "surface", "roof", "total_line", "spread_line", "series", "play_in_drive",
  "tfl", "pbu", "atkl","stkl", "cp", "cpoe", "xpass", "pass_oe", "old_game_id",
  "xyac_epa","xyac_median_yardage","xyac_mean_yardage", "xyac_success", "index"
)

test <- sdv_pbp %>%
  select(all_of(sdv_cols))

master <- sup_raw %>%
  left_join(test, by = 'index') %>%
  select(-penalty_yards) %>%
  mutate(
    yards_after_catch = ifelse(is.na(yards_after_catch) == T, 0, yards_after_catch),
    xyac_epa = ifelse(is.na(xyac_epa), 0, xyac_epa),
    xyac_median_yardage = ifelse(is.na(xyac_median_yardage), 0, xyac_median_yardage),
    xyac_mean_yardage = ifelse(is.na(xyac_mean_yardage), 0, xyac_mean_yardage),
    xyac_success = ifelse(is.na(xyac_success), 0, xyac_success)
  )

master <- master %>%
  mutate(
    air_yards = ifelse(is.na(air_yards), pass_length, air_yards)
  )

# For the single NA in air_epa, you could set it to 0 or leave it
master <- master %>%
  mutate(
    air_epa = ifelse(is.na(air_epa), 0, air_epa))

# Verify the changes
summary(master)


  