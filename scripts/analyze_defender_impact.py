"""
analyze_defender_impact.py - Defender Impact Analysis via Attention Attribution
================================================================================
Analyzes how defender actions during ball flight affect EPA outcomes by:
  1. Tracking EPA distribution evolution (start → end of play)
  2. Extracting attention weights from the trained model
  3. Attributing EPA shifts to individual defenders
  4. Ranking defenders by their impact

METHODOLOGY:
  - No hypothetical "ideal" paths needed
  - Uses model's learned attention to determine who influenced the outcome
  - Attention-weighted attribution: defenders the model "paid attention to" get credit

INPUTS:
  - model_outputs/epa_distribution/model.pth (trained model)
  - outputs/dataframe_a/v1.parquet (node features)
  - outputs/dataframe_b/v1.parquet (play-level features)
  - outputs/dataframe_c/v2_pilot_3games.parquet (structured edges)

OUTPUTS:
  - model_outputs/defender_impact_analysis/
    ├── defender_impacts.csv (per-defender EPA impact)
    ├── play_summaries.csv (per-play EPA shifts)
    ├── attention_heatmaps/ (visualizations)
    └── summary_report.txt (top defenders)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict

# ============================================================================
# Configuration
# ============================================================================

print("=" * 80)
print("DEFENDER IMPACT ANALYSIS - ATTENTION ATTRIBUTION")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# File paths
MODEL_PATH = 'model_outputs/epa_distribution/model.pth'
INPUT_DF_A = 'outputs/dataframe_a/v1.parquet'
INPUT_DF_B = 'outputs/dataframe_b/v1.parquet'
INPUT_DF_C = 'outputs/dataframe_c/v2_pilot_36games.parquet'

# Output directory
OUTPUT_DIR = 'model_outputs/defender_impact_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'attention_heatmaps'), exist_ok=True)

# Analysis configuration
TOP_N_PLAYS = 10  # Number of plays to visualize
TOP_N_DEFENDERS = 20  # Number of defenders to highlight in report

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}\n")

# ============================================================================
# 1. Load Model Architecture (Must Match Training Script)
# ============================================================================

print("=" * 80)
print("STEP 1: LOADING MODEL ARCHITECTURE")
print("-" * 80)

class MultiHeadGraphAttention(nn.Module):
    """Multi-head graph attention with edge features."""
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, hidden_dim)
        
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        self.edge_attn = nn.Linear(hidden_dim, n_heads)
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_feat, edge_index, edge_feat):
        N = node_feat.shape[0]
        h = F.relu(self.node_encoder(node_feat))
        e = F.relu(self.edge_encoder(edge_feat))
        
        Q = self.W_Q(h).view(N, self.n_heads, self.head_dim)
        K = self.W_K(h).view(N, self.n_heads, self.head_dim)
        V = self.W_V(h).view(N, self.n_heads, self.head_dim)
        
        if edge_index.shape[1] == 0:
            return h
        
        src, dst = edge_index
        scores = (Q[src] * K[dst]).sum(dim=-1) / (self.head_dim ** 0.5)
        scores = scores + self.edge_attn(e)
        
        attn_weights = torch.zeros_like(scores)
        for i in range(N):
            mask = (dst == i)
            if mask.any():
                attn_weights[mask] = F.softmax(scores[mask], dim=0)
        
        # Store attention weights
        self.last_attention_weights = attn_weights
        
        weighted_V = V[src] * attn_weights.unsqueeze(-1)
        out = torch.zeros(N, self.n_heads, self.head_dim, device=node_feat.device)
        out.index_add_(0, dst, weighted_V)
        
        out = self.out_proj(out.view(N, -1))
        return out + h

class EPADistributionModel(nn.Module):
    """Predicts EPA distribution: N(μ, σ²)"""
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, n_heads):
        super().__init__()
        self.gat = MultiHeadGraphAttention(node_in_dim, edge_in_dim, hidden_dim, n_heads)
        self.distribution_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
    def forward(self, node_feat, edge_index, edge_feat, return_attention=False):
        h = self.gat(node_feat, edge_index, edge_feat)
        graph_emb = h.mean(dim=0, keepdim=True).float()
        
        dist_params = self.distribution_head(graph_emb)
        mu = dist_params[:, 0]
        raw_sigma = dist_params[:, 1]
        sigma = F.softplus(raw_sigma) + 0.08
        
        if return_attention:
            attention = self.gat.last_attention_weights if hasattr(self.gat, 'last_attention_weights') else None
            return mu, sigma, attention
        
        return mu, sigma

print("✓ Model architecture defined")

# ============================================================================
# 2. Load Trained Model
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: LOADING TRAINED MODEL")
print("-" * 80)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# Get model configuration from checkpoint
node_features = checkpoint['node_features']
edge_features = checkpoint['edge_features']
node_means = pd.Series(checkpoint['node_means'])
node_stds = pd.Series(checkpoint['node_stds'])
edge_means = pd.Series(checkpoint['edge_means'])
edge_stds = pd.Series(checkpoint['edge_stds'])
edge_type_to_idx = checkpoint['edge_type_to_idx']

print(f"Model configuration:")
print(f"  Node features: {len(node_features)}")
print(f"  Edge features: {len(edge_features)}")
print(f"  Hidden dim: 128")
print(f"  Attention heads: 4")

# Instantiate model
model = EPADistributionModel(
    node_in_dim=len(node_features),
    edge_in_dim=len(edge_features),
    hidden_dim=128,
    n_heads=4
).to(DEVICE)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ Loaded model from: {MODEL_PATH}")
print(f"  Trained for {checkpoint['epoch']} epochs")
print(f"  Best Val NLL: {checkpoint['val_loss']:.4f}")

# ============================================================================
# 3. Load Data
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: LOADING DATA")
print("-" * 80)

print("Loading dataframe_a (node features)...")
df_a = pd.read_parquet(INPUT_DF_A)
print(f"  ✓ Loaded {len(df_a):,} rows")

print("\nLoading dataframe_b (play-level features)...")
df_b = pd.read_parquet(INPUT_DF_B)
print(f"  ✓ Loaded {len(df_b):,} plays")

print("\nLoading dataframe_c v2 (structured edges)...")
df_c = pd.read_parquet(INPUT_DF_C)
print(f"  ✓ Loaded {len(df_c):,} edges")

# Filter to pilot games
pilot_games = df_c['game_id'].unique()
df_a = df_a[df_a['game_id'].isin(pilot_games)].copy()
df_b = df_b[df_b['game_id'].isin(pilot_games)].copy()

# Filter to relevant players
df_a_filtered = df_a[
    (df_a['isPasser'] == 1) | 
    (df_a['isRouteRunner'] == 1) |
    (df_a['coverage_responsibility'].notna())
].copy()

print(f"\nFiltered data:")
print(f"  Nodes: {len(df_a_filtered):,}")
print(f"  Plays: {len(df_b):,}")
print(f"  Edges: {len(df_c):,}")

# ============================================================================
# 4. Helper Functions
# ============================================================================

def build_graph_for_frame(game_id, play_id, frame_id):
    """
    Build graph structure for a single frame.
    
    Returns:
        node_features: Tensor [N, node_dim]
        edge_index: Tensor [2, E]
        edge_features: Tensor [E, edge_dim]
        node_ids: List of NFL IDs
        edge_df: DataFrame of edges with metadata
    """
    # Get nodes
    nodes = df_a_filtered[
        (df_a_filtered['game_id'] == game_id) &
        (df_a_filtered['play_id'] == play_id) &
        (df_a_filtered['frame_id'] == frame_id)
    ].copy()
    
    if len(nodes) == 0:
        return None, None, None, None, None
    
    # Create node ID mapping
    node_ids = nodes['nfl_id'].values
    valid_nfl_ids = set(node_ids)
    node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    
    # Get edges
    edges_raw = df_c[
        (df_c['game_id'] == game_id) &
        (df_c['play_id'] == play_id) &
        (df_c['frame_id'] == frame_id)
    ]
    
    edges = edges_raw[
        edges_raw['playerA_id'].isin(valid_nfl_ids) &
        edges_raw['playerB_id'].isin(valid_nfl_ids)
    ].copy()
    
    if len(edges) == 0:
        return None, None, None, None, None
    
    # Extract and normalize node features
    node_feat = (nodes[node_features].values - node_means.values) / (node_stds.values + 1e-8)
    node_feat = torch.FloatTensor(node_feat).to(DEVICE)
    
    # Extract and normalize edge features
    edge_feat = (edges[edge_features].values - edge_means.values) / (edge_stds.values + 1e-8)
    edge_feat = torch.FloatTensor(edge_feat).to(DEVICE)
    
    # Build edge index
    edge_index = []
    for _, row in edges.iterrows():
        src_idx = node_id_to_idx[row['playerA_id']]
        dst_idx = node_id_to_idx[row['playerB_id']]
        edge_index.append([src_idx, dst_idx])
    
    edge_index = torch.LongTensor(edge_index).t().to(DEVICE)
    
    return node_feat, edge_index, edge_feat, node_ids, edges

def analyze_play(game_id, play_id):
    """
    Analyze a single play: track EPA distribution and attention over time.
    
    Returns:
        results (dict): Play-level analysis results
    """
    # Get all frames for this play
    play_frames = df_c[
        (df_c['game_id'] == game_id) &
        (df_c['play_id'] == play_id)
    ]['frame_id'].unique()
    
    if len(play_frames) == 0:
        return None
    
    play_frames = sorted(play_frames)
    
    # Track EPA and attention over time
    mu_over_time = []
    sigma_over_time = []
    attention_over_time = []
    edge_dfs = []
    node_ids_list = []
    
    with torch.no_grad():
        for frame_id in play_frames:
            node_feat, edge_idx, edge_feat, node_ids, edge_df = build_graph_for_frame(
                game_id, play_id, frame_id
            )
            
            if node_feat is None:
                continue
            
            # Forward pass with attention extraction
            mu, sigma, attention = model(
                node_feat, edge_idx, edge_feat, return_attention=True
            )
            
            mu_over_time.append(mu.item())
            sigma_over_time.append(sigma.item())
            attention_over_time.append(attention.cpu().numpy() if attention is not None else None)
            edge_dfs.append(edge_df)
            node_ids_list.append(node_ids)
    
    if len(mu_over_time) == 0:
        return None
    
    # Calculate EPA shift
    mu_start = mu_over_time[0]
    mu_end = mu_over_time[-1]
    delta_mu = mu_end - mu_start
    
    # Get play info
    play_info = df_b[
        (df_b['game_id'] == game_id) &
        (df_b['play_id'] == play_id)
    ].iloc[0]
    
    actual_epa = play_info['expected_points_added'] if 'expected_points_added' in play_info.index else None
    
    results = {
        'game_id': game_id,
        'play_id': play_id,
        'num_frames': len(play_frames),
        'mu_start': mu_start,
        'mu_end': mu_end,
        'delta_mu': delta_mu,
        'sigma_start': sigma_over_time[0],
        'sigma_end': sigma_over_time[-1],
        'actual_epa': actual_epa,
        'mu_over_time': mu_over_time,
        'sigma_over_time': sigma_over_time,
        'attention_over_time': attention_over_time,
        'edge_dfs': edge_dfs,
        'node_ids_list': node_ids_list,
        'play_frames': play_frames,
    }
    
    return results

def attribute_to_defenders(play_results):
    """
    Attribute EPA shift to individual defenders using attention weights.
    
    Returns:
        defender_attributions (dict): defender_id -> attribution score
    """
    delta_mu = play_results['delta_mu']
    attention_over_time = play_results['attention_over_time']
    edge_dfs = play_results['edge_dfs']
    node_ids_list = play_results['node_ids_list']
    
    # Aggregate attention per player across all frames
    player_attention = defaultdict(float)
    
    for frame_idx, (attn, edge_df, node_ids) in enumerate(zip(
        attention_over_time, edge_dfs, node_ids_list
    )):
        if attn is None:
            continue
        
        # Average attention across heads
        avg_attn = attn.mean(axis=1)  # [num_edges]
        
        # Create node_id to index mapping for this frame
        node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        
        # For each edge, add attention to both players
        for edge_idx, (_, edge_row) in enumerate(edge_df.iterrows()):
            playerA_id = edge_row['playerA_id']
            playerB_id = edge_row['playerB_id']
            
            edge_attn = avg_attn[edge_idx]
            
            player_attention[playerA_id] += edge_attn
            player_attention[playerB_id] += edge_attn
    
    # Normalize attention scores
    total_attention = sum(player_attention.values())
    
    if total_attention == 0:
        return {}
    
    # Attribute EPA shift proportionally
    defender_attributions = {}
    
    for player_id, attn_score in player_attention.items():
        # Get player info
        player_info = df_a_filtered[
            (df_a_filtered['nfl_id'] == player_id) &
            (df_a_filtered['game_id'] == play_results['game_id']) &
            (df_a_filtered['play_id'] == play_results['play_id'])
        ]
        
        if len(player_info) == 0:
            continue
        
        player_info = player_info.iloc[0]
        
        # Only attribute to defenders
        is_defender = pd.notna(player_info.get('coverage_responsibility'))
        
        if is_defender:
            share = attn_score / total_attention
            impact = share * delta_mu
            
            defender_attributions[player_id] = {
                'attention_score': attn_score,
                'attention_share': share,
                'epa_impact': impact,
                'position': player_info.get('player_position'),
                'jersey_number': player_info.get('jerseyNumber'),
            }
    
    return defender_attributions

# ============================================================================
# 5. Analyze All Plays
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: ANALYZING PLAYS")
print("-" * 80)

# Get unique plays
unique_plays = df_c.groupby(['game_id', 'play_id']).size().reset_index()[['game_id', 'play_id']]

print(f"Analyzing {len(unique_plays)} plays...")

play_summaries = []
all_defender_impacts = []

for idx, (_, play_row) in enumerate(unique_plays.iterrows()):
    if (idx + 1) % 10 == 0:
        print(f"  Processed {idx + 1}/{len(unique_plays)} plays...")
    
    game_id = play_row['game_id']
    play_id = play_row['play_id']
    
    # Analyze play
    results = analyze_play(game_id, play_id)
    
    if results is None:
        continue
    
    # Attribute to defenders
    defender_attrs = attribute_to_defenders(results)
    
    # Store play summary
    play_summaries.append({
        'game_id': game_id,
        'play_id': play_id,
        'num_frames': results['num_frames'],
        'mu_start': results['mu_start'],
        'mu_end': results['mu_end'],
        'delta_mu': results['delta_mu'],
        'sigma_start': results['sigma_start'],
        'sigma_end': results['sigma_end'],
        'actual_epa': results['actual_epa'],
        'num_defenders_attributed': len(defender_attrs),
    })
    
    # Store defender impacts
    for defender_id, attrs in defender_attrs.items():
        all_defender_impacts.append({
            'game_id': game_id,
            'play_id': play_id,
            'defender_id': defender_id,
            'position': attrs['position'],
            'jersey_number': attrs['jersey_number'],
            'attention_score': attrs['attention_score'],
            'attention_share': attrs['attention_share'],
            'epa_impact': attrs['epa_impact'],
        })

print(f"✓ Analyzed {len(play_summaries)} plays")
print(f"✓ Generated {len(all_defender_impacts)} defender impact records")

# ============================================================================
# 6. Save Results
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: SAVING RESULTS")
print("-" * 80)

# Play summaries
play_summaries_df = pd.DataFrame(play_summaries)
play_summaries_file = os.path.join(OUTPUT_DIR, 'play_summaries.csv')
play_summaries_df.to_csv(play_summaries_file, index=False)
print(f"✓ Saved play summaries: {play_summaries_file}")
print(f"  {len(play_summaries_df)} plays")

# Defender impacts
defender_impacts_df = pd.DataFrame(all_defender_impacts)
defender_impacts_file = os.path.join(OUTPUT_DIR, 'defender_impacts.csv')
defender_impacts_df.to_csv(defender_impacts_file, index=False)
print(f"✓ Saved defender impacts: {defender_impacts_file}")
print(f"  {len(defender_impacts_df)} records")

# ============================================================================
# 7. Generate Summary Report
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: GENERATING SUMMARY REPORT")
print("-" * 80)

report_lines = []
report_lines.append("=" * 80)
report_lines.append("DEFENDER IMPACT ANALYSIS - SUMMARY REPORT")
report_lines.append("=" * 80)
report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append("")

# Overall statistics
report_lines.append("OVERALL STATISTICS:")
report_lines.append("-" * 80)
report_lines.append(f"Total plays analyzed: {len(play_summaries_df)}")
report_lines.append(f"Total defender impact records: {len(defender_impacts_df)}")
report_lines.append(f"Average EPA shift per play: {play_summaries_df['delta_mu'].mean():.3f}")
report_lines.append(f"EPA shift std dev: {play_summaries_df['delta_mu'].std():.3f}")
report_lines.append("")

# Top defenders by total impact
report_lines.append(f"TOP {TOP_N_DEFENDERS} DEFENDERS BY TOTAL EPA IMPACT:")
report_lines.append("-" * 80)

defender_totals = defender_impacts_df.groupby('defender_id').agg({
    'epa_impact': 'sum',
    'attention_score': 'sum',
    'position': 'first',
    'jersey_number': 'first',
}).reset_index()

defender_totals = defender_totals.sort_values('epa_impact', ascending=True).head(TOP_N_DEFENDERS)

for idx, row in defender_totals.iterrows():
    report_lines.append(
        f"{idx+1:2d}. Defender #{row['jersey_number']} ({row['position']}): "
        f"{row['epa_impact']:+.3f} EPA (attention: {row['attention_score']:.2f})"
    )

report_lines.append("")

# Top plays by EPA shift magnitude
report_lines.append(f"TOP {TOP_N_PLAYS} PLAYS BY EPA SHIFT (Defense Improved Most):")
report_lines.append("-" * 80)

top_defensive_plays = play_summaries_df.sort_values('delta_mu', ascending=True).head(TOP_N_PLAYS)

for idx, row in top_defensive_plays.iterrows():
    report_lines.append(
        f"{idx+1:2d}. Game {row['game_id']}, Play {row['play_id']}: "
        f"{row['delta_mu']:+.3f} EPA shift ({row['mu_start']:.2f} → {row['mu_end']:.2f})"
    )

report_lines.append("")

# Save report
report_file = os.path.join(OUTPUT_DIR, 'summary_report.txt')
with open(report_file, 'w') as f:
    f.write('\n'.join(report_lines))

print(f"✓ Saved summary report: {report_file}")

# Also print to console
print("\n" + "\n".join(report_lines))

# ============================================================================
# Complete
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nOutputs saved to: {OUTPUT_DIR}")
print(f"  - play_summaries.csv")
print(f"  - defender_impacts.csv")
print(f"  - summary_report.txt")
print("\n" + "=" * 80)