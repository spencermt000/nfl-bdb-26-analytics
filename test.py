"""
Quick diagnostic script to check play-level dataframe B
"""

import pandas as pd

# Load play dataframe
df_plays = pd.read_parquet('outputs/dataframe_b/v1.parquet')

# Load edge dataframe to get pilot game's play_ids
df_edges = pd.read_parquet('outputs/dataframe_c/v1.parquet')

# Pilot game ID
pilot_game_id = 2023121602

print("="*70)
print("PLAY-LEVEL DATAFRAME DIAGNOSTIC")
print("="*70)

print(f"\nTotal rows in play df: {len(df_plays):,}")
print(f"Unique play_ids in play df: {df_plays['play_id'].nunique()}")

# Get play_ids from pilot game
pilot_edges = df_edges[df_edges['game_id'] == pilot_game_id]
pilot_play_ids = pilot_edges['play_id'].unique()

print(f"\nPilot game {pilot_game_id}:")
print(f"  Unique play_ids in edge data: {len(pilot_play_ids)}")

# Filter play df to these play_ids
df_plays_pilot = df_plays[df_plays['play_id'].isin(pilot_play_ids)]
print(f"  Rows in play df matching these play_ids: {len(df_plays_pilot)}")

# Check for duplicates
if len(df_plays_pilot) > len(pilot_play_ids):
    print(f"\n⚠️  DUPLICATION DETECTED!")
    print(f"  Expected: {len(pilot_play_ids)} rows (one per play)")
    print(f"  Found: {len(df_plays_pilot)} rows")
    print(f"  Duplication factor: {len(df_plays_pilot) / len(pilot_play_ids):.2f}x")
    
    # Find which play_ids have duplicates
    duplicates = df_plays_pilot[df_plays_pilot.duplicated(subset=['play_id'], keep=False)]
    dup_play_ids = duplicates['play_id'].unique()
    
    print(f"\n  Play_ids with duplicates: {len(dup_play_ids)}")
    print(f"\nExample duplicate play_id: {dup_play_ids[0]}")
    print(df_plays_pilot[df_plays_pilot['play_id'] == dup_play_ids[0]])
else:
    print(f"\n✓ No duplicates - one row per play_id")

# Show distribution of rows per play_id
print("\n" + "="*70)
print("ROWS PER PLAY_ID DISTRIBUTION")
print("="*70)
rows_per_play = df_plays_pilot.groupby('play_id').size()
print(rows_per_play.describe())
print(f"\nMost common rows per play_id:")
print(rows_per_play.value_counts().head(10))

# Now simulate the merge to see what happens
print("\n" + "="*70)
print("SIMULATING THE MERGE")
print("="*70)

print(f"\nBefore merge:")
print(f"  Edge rows (pilot game): {len(pilot_edges):,}")

merged = pilot_edges.merge(df_plays_pilot, on='play_id', how='left')
print(f"\nAfter merge:")
print(f"  Merged rows: {len(merged):,}")
print(f"  Multiplication factor: {len(merged) / len(pilot_edges):.2f}x")

if len(merged) > len(pilot_edges):
    print(f"\n⚠️  THE MERGE CAUSED {len(merged) / len(pilot_edges):.2f}x MULTIPLICATION")