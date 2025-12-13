"""
viz_play_test.py - Visualize Attention & Predictions (Dynamic Game Detection)
=============================================================================
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys
import pickle

# Append scripts path for utils
sys.path.append('scripts')
# If utils fails, we define simple loader
def load_parquet(path):
    return pd.read_parquet(path)

# ============================================================================
# 1. Model Definitions (Must match Training Scripts)
# ============================================================================

# --- COMPLETION MODEL ARCHITECTURE ---
class MultiHeadGraphAttention(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
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
        
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, hidden_dim)
        
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        self.edge_attn = nn.Linear(hidden_dim, n_heads)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_feat, edge_index, edge_feat, ball_progress):
        N = node_feat.shape[0]
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
        self.gat = YacGAT(node_in, edge_in, hidden, n_heads)
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

DEVICE = torch.device('cpu') 

# Load Feature Stats
with open('model_outputs/attention/feature_stats.pkl', 'rb') as f:
    stats = pickle.load(f)

# Load DataFrames
print("Loading data...")
df_a = load_parquet('outputs/dataframe_a/v2.parquet')
df_c = load_parquet('outputs/dataframe_c/v3_pilot_3games.parquet') # The Pilot Edges
df_b = load_parquet('outputs/dataframe_b/v4.parquet')

# --- FIX: DYNAMICALLY DETECT AVAILABLE GAMES ---
available_games = df_c['game_id'].unique()
print(f"Found {len(available_games)} games in pilot dataset: {available_games}")

# Ensure Type Consistency (Convert everything to string for filtering)
df_a['game_id'] = df_a['game_id'].astype(str)
df_b['game_id'] = df_b['game_id'].astype(str)
df_c['game_id'] = df_c['game_id'].astype(str)
available_games_str = [str(g) for g in available_games]

# Filter df_a and df_b to these games
df_a = df_a[df_a['game_id'].isin(available_games_str)]
df_b = df_b[df_b['game_id'].isin(available_games_str)]

print(f"Filtered Data: {len(df_a)} frames, {len(df_b)} plays.")

if len(df_b) == 0:
    print("CRITICAL ERROR: No matching plays found in df_b for the pilot games.")
    print("Check if df_b/v4.parquet actually contains data for these game IDs.")
    sys.exit()

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

# Pick one random play from the filtered df_b
selected_play = df_b.sample(1).iloc[0]
GAME_ID = str(selected_play['game_id'])
PLAY_ID = int(selected_play['play_id'])

print(f"\nSELECTED PLAY: Game {GAME_ID}, Play {PLAY_ID}")
description = selected_play.get('playDescription', 'N/A')
print(f"Description: {description}")

# Get frames for this play
play_frames = df_a[
    (df_a['game_id'] == GAME_ID) & (df_a['play_id'] == PLAY_ID)
]['frame_id'].unique()
play_frames.sort()

print(f"Visualizing {len(play_frames)} frames...")

# Get ball ending position from the play data
ball_end_x = selected_play.get('ball_end_x', None)
ball_end_y = selected_play.get('ball_end_y', None)

# ============================================================================
# 4. Visualization Loop
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 7))

def update(frame_idx):
    ax.clear()
    
    # 1. Setup Field
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.set_facecolor('#f0f0f0') 
    for x in range(10, 111, 10):
        ax.axvline(x, color='white', linestyle='-', alpha=0.5)
    
    frame_id = play_frames[frame_idx]
    
    # 2. Get Data for Frame
    nodes = df_a[
        (df_a['game_id'] == GAME_ID) & 
        (df_a['play_id'] == PLAY_ID) & 
        (df_a['frame_id'] == frame_id)
    ].copy()
    
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
    if edge_index.shape[0] == 2 and edge_index.shape[1] > 0:
        with torch.no_grad():
            comp_prob, attn_weights = comp_model(node_feat, edge_index, edge_feat, ball_progress)
            yac_pred = yac_model(node_feat, edge_index, edge_feat, ball_progress)
        
        # Draw Edges - Only for Targeted Receiver
        # Find the targeted receiver node index
        targeted_receiver_idx = None
        if 'player_role' in nodes.columns:
            targeted_mask = nodes['player_role'] == 'Targeted Receiver'
            if targeted_mask.any():
                targeted_receiver_idx = nodes[targeted_mask].index[0]
                # Get position in the node list (not DataFrame index)
                targeted_receiver_pos = list(nodes.index).index(targeted_receiver_idx)

        attn_avg = attn_weights.mean(dim=1).numpy()
        for i, (src, dst) in enumerate(edge_index.t().numpy()):
            score = attn_avg[i]
            # Only draw edges connected to the targeted receiver
            if score > 0.10 and targeted_receiver_pos is not None:
                if src == targeted_receiver_pos or dst == targeted_receiver_pos:
                    p1 = nodes.iloc[src]
                    p2 = nodes.iloc[dst]
                    alpha = min(score * 3, 1.0)
                    ax.plot([p1['x'], p2['x']], [p1['y'], p2['y']],
                           color='blue', alpha=alpha, linewidth=1.5, zorder=1)

        # Info Box
        info_text = (
            f"Frame: {frame_id}\n"
            f"Completion %: {comp_prob.item()*100:.1f}%\n"
            f"Exp YAC EPA: {yac_pred.item():.2f}"
        )
    else:
        info_text = "No edges (Pre-snap?)"

    # 5. Plot Nodes - Color by Role, Label by NFL ID
    # ------------------------------------------------------------------
    # Role color mapping (actual values in player_role column)
    role_colors = {
        'Passer': '#FF6B6B',                  # Red
        'Targeted Receiver': '#00FF00',       # Bright Green
        'Other Route Runner': '#FFE66D',      # Yellow
        'Defensive Coverage': '#87CEEB',      # Sky Blue
    }

    # Map colors based on player_role
    if 'player_role' in nodes.columns:
        # FIX: Convert to object (string) type immediately to avoid Categorical errors
        colors = nodes['player_role'].map(role_colors).astype(object).fillna('grey')
        
        # Handle football
        if 'displayName' in nodes.columns:
            mask_football = nodes['displayName'] == 'football'
            # Ensure we can write to this location
            colors.loc[mask_football] = 'brown'
    else:
        colors = 'grey'

    # Plot with bigger markers
    ax.scatter(nodes['x'], nodes['y'], c=colors, s=250, zorder=2, edgecolors='white', linewidths=2)

    # 6. Draw ball ending spot as red square
    if ball_end_x is not None and ball_end_y is not None and pd.notna(ball_end_x) and pd.notna(ball_end_y):
        ax.scatter([ball_end_x], [ball_end_y], 
                  marker='s', s=400, c='red', edgecolors='darkred', 
                  linewidths=3, zorder=3, alpha=0.7, label='Ball End')
        ax.text(ball_end_x, ball_end_y-2, 'BALL END', 
               fontsize=8, ha='center', fontweight='bold', 
               color='darkred', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.set_title(f"Play {PLAY_ID} | Game {GAME_ID}")

ani = animation.FuncAnimation(fig, update, frames=len(play_frames), interval=200)

save_path = 'play_viz.gif'
ani.save(save_path, writer='pillow', fps=5)
print(f"✓ Saved visualization to: {save_path}")