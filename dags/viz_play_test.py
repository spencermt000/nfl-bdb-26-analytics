"""
viz_play_test.py - Visualize Attention & Predictions for a Single Play
======================================================================
Generates an animated GIF of a specific play, visualizing:
1. Player positions (Nodes)
2. Attention Weights (Edges) - The "Brain" of the GNN
3. Live Predictions (Completion Probability & YAC EPA)

OUTPUT:
  - play_viz.gif
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import os
import sys

# Append scripts path for utils
sys.path.append('scripts')
from utils import load_parquet_to_df
# ============================================================================
# 1. Model Definitions
# ============================================================================

# --- COMPLETION MODEL ARCHITECTURE ---
class MultiHeadGraphAttention(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        # Completion model used Sequential encoders
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        self.edge_attn = nn.Linear(hidden_dim, n_heads)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, node_feat, edge_index, edge_feat, ball_progress):
        N = node_feat.shape[0]
        h = self.node_encoder(node_feat)
        e = self.edge_encoder(edge_feat)
        
        Q = self.W_Q(h).view(N, self.n_heads, self.head_dim)
        K = self.W_K(h).view(N, self.n_heads, self.head_dim)
        V = self.W_V(h).view(N, self.n_heads, self.head_dim)
        
        src, dst = edge_index
        scores = (Q[src] * K[dst]).sum(dim=-1) / (self.head_dim ** 0.5)
        scores = scores + self.edge_attn(e)
        
        attn_weights = torch.zeros_like(scores)
        for i in range(N):
            mask = (src == i)
            if mask.any():
                attn_weights[mask] = F.softmax(scores[mask], dim=0)
        
        weighted_V = V[src] * attn_weights.unsqueeze(-1)
        out = torch.zeros(N, self.n_heads, self.head_dim, device=node_feat.device)
        out.index_add_(0, dst, weighted_V)
        out = self.out_proj(out.view(N, -1))
        
        return out + h, attn_weights

class AttentionAdjacencyModel(nn.Module):
    def __init__(self, node_in, edge_in, hidden, n_heads):
        super().__init__()
        self.attention = MultiHeadGraphAttention(node_in, edge_in, hidden, n_heads)
        self.completion_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.epa_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1)
        )
    def forward(self, n, ei, ef, bp):
        h, weights = self.attention(n, ei, ef, bp)
        h_graph = h.mean(dim=0).unsqueeze(0).float()
        return self.completion_head(h_graph), weights


# --- YAC MODEL ARCHITECTURE (Specific) ---
class YacGAT(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        # YAC model used simple Linear encoders (Architecture Mismatch Fix)
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, hidden_dim)
        
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        self.edge_attn = nn.Linear(hidden_dim, n_heads)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_feat, edge_index, edge_feat, ball_progress):
        N = node_feat.shape[0]
        # YAC model applied ReLU in forward pass
        h = F.relu(self.node_encoder(node_feat))
        e = F.relu(self.edge_encoder(edge_feat))
        
        Q = self.W_Q(h).view(N, self.n_heads, self.head_dim)
        K = self.W_K(h).view(N, self.n_heads, self.head_dim)
        V = self.W_V(h).view(N, self.n_heads, self.head_dim)
        
        src, dst = edge_index
        scores = (Q[src] * K[dst]).sum(dim=-1) / (self.head_dim ** 0.5)
        scores = scores + self.edge_attn(e)
        
        attn_weights = torch.zeros_like(scores)
        for i in range(N):
            mask = (src == i)
            if mask.any():
                attn_weights[mask] = F.softmax(scores[mask], dim=0)
                
        weighted_V = V[src] * attn_weights.unsqueeze(-1)
        out = torch.zeros(N, self.n_heads, self.head_dim, device=node_feat.device)
        out.index_add_(0, dst, weighted_V)
        out = self.out_proj(out.view(N, -1))
        return out + h, attn_weights

class YacPredictionModel(nn.Module):
    def __init__(self, node_in, edge_in, hidden, n_heads):
        super().__init__()
        self.gat = YacGAT(node_in, edge_in, hidden, n_heads) # Uses YacGAT
        self.yac_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1)
        )
    def forward(self, n, ei, ef, bp):
        h, weights = self.gat(n, ei, ef, bp)
        h_graph = h.mean(dim=0).unsqueeze(0).float()
        return self.yac_head(h_graph)
    
# ============================================================================
# 2. Configuration & Loading
# ============================================================================

print("="*80)
print("VISUALIZING PLAY WITH GNN ATTENTION")
print("="*80)

DEVICE = torch.device('cpu') # Visualization is usually fast enough on CPU
PILOT_GAMES = ['2022091100', '2022091101', '2022091102'] # Adjust if needed

# Load Feature Stats (Needed for Normalization)
import pickle
with open('model_outputs/attention/feature_stats.pkl', 'rb') as f:
    stats = pickle.load(f)

# Load DataFrames
print("Loading data...")
df_a = pd.read_parquet('outputs/dataframe_a/v2.parquet')
df_c = pd.read_parquet('outputs/dataframe_c/v3_pilot_3games.parquet')
df_b = pd.read_parquet('outputs/dataframe_b/v4.parquet')

# Filter to pilot
df_a = df_a[df_a['game_id'].isin(PILOT_GAMES)]
df_c = df_c[df_c['game_id'].isin(PILOT_GAMES)]

# Load Models
print("Loading models...")
comp_model = AttentionAdjacencyModel(
    len(stats['node_features']), len(stats['edge_features']), 128, 4
)
comp_checkpoint = torch.load('model_outputs/attention/best_model.pth', map_location=DEVICE)
comp_model.load_state_dict(comp_checkpoint['model_state_dict'])
comp_model.eval()

yac_model = YacPredictionModel(
    len(stats['node_features']), len(stats['edge_features']), 128, 4
)
yac_checkpoint = torch.load('model_outputs/attention_yac/yac_model.pth', map_location=DEVICE)
yac_model.load_state_dict(yac_checkpoint)
yac_model.eval()

# ============================================================================
# 3. Select a Play
# ============================================================================

# Filter to valid plays in df_b (plays that exist in Pilot)
valid_plays = df_b[df_b['game_id'].isin(PILOT_GAMES)]
if len(valid_plays) == 0:
    print("Error: No plays found in pilot games!")
    sys.exit()

# Pick one random play
selected_play = valid_plays.sample(1).iloc[0]
GAME_ID = str(selected_play['game_id'])
PLAY_ID = int(selected_play['play_id'])

print(f"\nSELECTED PLAY: Game {GAME_ID}, Play {PLAY_ID}")
print(f"Description: {selected_play.get('playDescription', 'N/A')}")

# Get frames for this play
play_frames = df_a[
    (df_a['game_id'] == GAME_ID) & (df_a['play_id'] == PLAY_ID)
]['frame_id'].unique()
play_frames.sort()

# ============================================================================
# 4. Visualization Loop
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 7))

def update(frame_idx):
    ax.clear()
    
    # 1. Setup Field
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.set_facecolor('#f0f0f0') # Light gray turf
    # Draw yard lines
    for x in range(10, 111, 10):
        ax.axvline(x, color='white', linestyle='-', alpha=0.5)
    
    frame_id = play_frames[frame_idx]
    
    # 2. Get Data for Frame
    nodes = df_a[
        (df_a['game_id'] == GAME_ID) & 
        (df_a['play_id'] == PLAY_ID) & 
        (df_a['frame_id'] == frame_id)
    ].copy()
    
    # Filter Edges (The fix we applied earlier)
    valid_nfl_ids = set(nodes['nfl_id'].values)
    edges_raw = df_c[
        (df_c['game_id'] == GAME_ID) & 
        (df_c['play_id'] == PLAY_ID) & 
        (df_c['frame_id'] == frame_id)
    ]
    edges = edges_raw[
        edges_raw['playerA_id'].isin(valid_nfl_ids) & 
        edges_raw['playerB_id'].isin(valid_nfl_ids)
    ].copy()
    
    if len(nodes) == 0: return

    # 3. Prepare Tensors
    node_feat = torch.FloatTensor(
        (nodes[stats['node_features']].values - stats['node_means'].values) / (stats['node_stds'].values + 1e-8)
    )
    edge_feat = torch.FloatTensor(
        (edges[stats['edge_features']].values - stats['edge_means'].values) / (stats['edge_stds'].values + 1e-8)
    )
    
    node_id_to_idx = {nid: i for i, nid in enumerate(nodes['nfl_id'].values)}
    edge_index = []
    for _, row in edges.iterrows():
        edge_index.append([node_id_to_idx[row['playerA_id']], node_id_to_idx[row['playerB_id']]])
    
    edge_index = torch.LongTensor(edge_index)
    if edge_index.ndim == 1 or edge_index.shape[0] != 2:
        edge_index = edge_index.t()
        
    ball_progress = torch.FloatTensor([edges.iloc[0]['ball_progress']]) if not edges.empty else torch.zeros(1)

    # 4. Run Models
    with torch.no_grad():
        # Get Completion Pred + Attention Weights
        comp_prob, attn_weights = comp_model(node_feat, edge_index, edge_feat, ball_progress)
        
        # Get YAC Pred
        yac_pred = yac_model(node_feat, edge_index, edge_feat, ball_progress)
    
    # Process Attention Weights
    # Average across heads: [E, n_heads] -> [E]
    attn_avg = attn_weights.mean(dim=1).numpy()
    
    # 5. Plot Edges (The "Brain")
    # Only draw edges with meaningful attention (> 0.05) to reduce clutter
    for i, (src, dst) in enumerate(edge_index.t().numpy()):
        score = attn_avg[i]
        if score > 0.05: # Threshold
            p1 = nodes.iloc[src]
            p2 = nodes.iloc[dst]
            
            # Opacity based on score
            alpha = min(score * 3, 1.0) # Scale up visibility
            ax.plot([p1['x'], p2['x']], [p1['y'], p2['y']], 
                   color='blue', alpha=alpha, linewidth=1.5, zorder=1)
            
            # Label Edge (Optional - can get messy)
            if score > 0.2:
                mid_x = (p1['x'] + p2['x']) / 2
                mid_y = (p1['y'] + p2['y']) / 2
                ax.text(mid_x, mid_y, f"{score:.2f}", fontsize=6, color='darkblue')

    # 6. Plot Nodes
    # Color mapping
    colors = nodes['team'].map({'home': 'red', 'away': 'blue', 'football': 'brown'}).fillna('grey')
    ax.scatter(nodes['x'], nodes['y'], c=colors, s=100, zorder=2, edgecolors='white')
    
    # Label Nodes (nflId)
    for _, row in nodes.iterrows():
        ax.text(row['x'], row['y']+1, str(int(row['nfl_id']))[-3:], 
               fontsize=7, ha='center', fontweight='bold')

    # 7. Info Box (Predictions)
    info_text = (
        f"Frame: {frame_id}\n"
        f"Pred Completion: {comp_prob.item()*100:.1f}%\n"
        f"Pred YAC EPA: {yac_pred.item():.2f}"
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.set_title(f"Play {PLAY_ID} - Graph Attention Visualization")

print(f"Generating animation for {len(play_frames)} frames...")
ani = animation.FuncAnimation(fig, update, frames=len(play_frames), interval=200)

save_path = 'play_viz.gif'
ani.save(save_path, writer='pillow', fps=5)
print(f"✓ Saved visualization to: {save_path}")
print("You can download this file from the container to view it.")