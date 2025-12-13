"""
train_att_adj_yacEPA.py - Train Attention-Based Adjacency Matrix for YAC EPA
============================================================================
Trains a graph attention network to predict Completed YAC EPA (comp_yac_epa).
This is a REGRESSION task (predicting a continuous value).

INPUTS:
  - outputs/dataframe_a/v2.parquet (node features)
  - outputs/dataframe_b/v4.parquet (play context + advanced metrics)
  - outputs/dataframe_c/v3.parquet (edge features)

OUTPUTS:
  - model_outputs/attention_yac/yac_model.pth (trained model)

ARCHITECTURE:
  - Multi-head graph attention (4 heads)
  - Global Pooling
  - Regression Head (Linear -> scalar output)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import time
from datetime import datetime

# Import utils
import sys
sys.path.append('scripts')
from utils import load_parquet_to_df

# ============================================================================
# Configuration
# ============================================================================

print("=" * 80)
print("TRAIN GRAPH ATTENTION FOR YAC EPA (REGRESSION)")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# *** PILOT MODE ***
PILOT_MODE = True  # Set to False for full dataset
PILOT_DF_C = 'outputs/dataframe_c/v3_pilot_3games.parquet'
REGULAR_DF_C = 'outputs/dataframe_c/v3.parquet'

# File paths
INPUT_DF_A = 'outputs/dataframe_a/v2.parquet'
INPUT_DF_B = 'outputs/dataframe_b/v4.parquet' # Ensure this version has comp_yac_epa
INPUT_DF_C = PILOT_DF_C if PILOT_MODE else REGULAR_DF_C

# Output directory for YAC model
OUTPUT_DIR = 'model_outputs/attention_yac'
os.makedirs(OUTPUT_DIR, exist_ok=True)

if PILOT_MODE:
    print("\n" + "⚠" * 40)
    print("PILOT MODE ENABLED".center(80))
    print(f"Using pilot dataset: {PILOT_DF_C}".center(80))
    print("⚠" * 40 + "\n")

# Model hyperparameters
N_ATTENTION_HEADS = 4
NODE_FEATURE_DIM = 64
EDGE_FEATURE_DIM = 32
HIDDEN_DIM = 128
LEARNING_RATE = 0.001
BATCH_SIZE = 1  # Keep 1 for variable graph sizes

# --- TIME-BASED TRAINING CONFIGURATION ---
MAX_TRAIN_TIME_MINUTES = 8
# -----------------------------------------

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}\n")

# ============================================================================
# 1. Load Data
# ============================================================================

print("=" * 80)
print("STEP 1: LOADING DATA")
print("-" * 80)

df_a = load_parquet_to_df(INPUT_DF_A, 'df_a')
df_b = load_parquet_to_df(INPUT_DF_B, 'df_b')
df_c = load_parquet_to_df(INPUT_DF_C, 'df_c')

# Filter to pilot games if needed
if PILOT_MODE:
    pilot_games = df_c['game_id'].unique()
    df_a = df_a[df_a['game_id'].isin(pilot_games)].copy()
    df_b = df_b[df_b['game_id'].isin(pilot_games)].copy()
    print(f"Filtered to {len(pilot_games)} pilot games.")

# ============================================================================
# 2. Filter to Relevant Players
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: FILTERING DATA")
print("-" * 80)

# Filter df_a to relevant players
df_a_filtered = df_a[
    (df_a['isPasser'] == 1) | 
    (df_a['isRouteRunner'] == 1) |
    (df_a['coverage_responsibility'].notna())
].copy()

# Filter df_c to relevant edges
relevant_players = set(df_a_filtered['nfl_id'].unique())
df_c_filtered = df_c[
    df_c['playerA_id'].isin(relevant_players) &
    df_c['playerB_id'].isin(relevant_players)
].copy()

print(f"Nodes: {len(df_a_filtered):,}")
print(f"Edges: {len(df_c_filtered):,}")

# ============================================================================
# 3. Feature Engineering
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: FEATURES")
print("-" * 80)

# Node Features
NODE_FEATURES = [
    'x', 'y', 's', 'a', 'dir', 'o', 'v_x', 'v_y', 'a_x', 'a_y', 'o_x', 'o_y',
    'e_dist_ball_land', 'isPasser', 'isTargeted', 'isRouteRunner',
]
node_features_available = [f for f in NODE_FEATURES if f in df_a_filtered.columns]

# Edge Features
EDGE_FEATURES = [
    'e_dist', 'x_dist', 'y_dist', 'relative_angle_o', 'relative_angle_dir',
    'playerA_dist_to_landing', 'playerB_dist_to_landing',
    'playerA_dist_to_ball_current', 'playerB_dist_to_ball_current',
    'playerA_ball_convergence', 'playerB_ball_convergence',
    'relative_v_x', 'relative_v_y', 'relative_speed', 'same_team',
    'ball_progress', 'frames_to_landing',
]
edge_features_available = [f for f in EDGE_FEATURES if f in df_c_filtered.columns]

# Statistics for normalization
node_feature_means = df_a_filtered[node_features_available].mean()
node_feature_stds = df_a_filtered[node_features_available].std()
edge_feature_means = df_c_filtered[edge_features_available].mean()
edge_feature_stds = df_c_filtered[edge_features_available].std()

# ============================================================================
# 4. PyTorch Dataset
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: DATASET")
print("-" * 80)

class YacGraphDataset(Dataset):
    def __init__(self, df_a, df_b, df_c, node_features, edge_features, 
                 node_means, node_stds, edge_means, edge_stds):
        self.df_a = df_a
        self.df_b = df_b
        self.df_c = df_c
        self.node_features = node_features
        self.edge_features = edge_features
        self.node_means = node_means
        self.node_stds = node_stds
        self.edge_means = edge_means
        self.edge_stds = edge_stds
        
        # Unique frames
        self.frames = df_a.groupby(['game_id', 'play_id', 'frame_id']).size().reset_index()[['game_id', 'play_id', 'frame_id']]
        
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        game_id = self.frames.iloc[idx]['game_id']
        play_id = self.frames.iloc[idx]['play_id']
        frame_id = self.frames.iloc[idx]['frame_id']
        
        # 1. Get nodes
        nodes = self.df_a[
            (self.df_a['game_id'] == game_id) &
            (self.df_a['play_id'] == play_id) &
            (self.df_a['frame_id'] == frame_id)
        ].copy()
        
        # Create mapping of valid NFL IDs
        node_ids = nodes['nfl_id'].values
        valid_nfl_ids = set(node_ids)
        node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        
        # 2. Get edges (AND FILTER THEM IMMEDIATELY)
        edges_raw = self.df_c[
            (self.df_c['game_id'] == game_id) &
            (self.df_c['play_id'] == play_id) &
            (self.df_c['frame_id'] == frame_id)
        ]
        
        # --- FIX: Only keep edges where BOTH players are in the node list ---
        edges = edges_raw[
            edges_raw['playerA_id'].isin(valid_nfl_ids) & 
            edges_raw['playerB_id'].isin(valid_nfl_ids)
        ].copy()
        # -------------------------------------------------------------------
        
        # 3. Extract Features (Now guaranteed to match edge_index count)
        node_feat = torch.FloatTensor((nodes[self.node_features].values - self.node_means.values) / (self.node_stds.values + 1e-8))
        edge_feat = torch.FloatTensor((edges[self.edge_features].values - self.edge_means.values) / (self.edge_stds.values + 1e-8))
        
        # 4. Build Edge Index
        edge_index = []
        for _, row in edges.iterrows():
            # No need to check 'if in node_id_to_idx' because we already filtered df
            src_idx = node_id_to_idx[row['playerA_id']]
            dst_idx = node_id_to_idx[row['playerB_id']]
            edge_index.append([src_idx, dst_idx])
        
        edge_index = torch.LongTensor(edge_index)
        if edge_index.ndim == 1 or edge_index.shape[0] != 2:
            edge_index = edge_index.t()
            
        ball_progress = torch.FloatTensor([edges.iloc[0]['ball_progress']]) if not edges.empty else torch.zeros(1)

        # 5. Target: comp_yac_epa
        play_row = self.df_b[
            (self.df_b['game_id'] == game_id) & 
            (self.df_b['play_id'] == play_id)
        ]
        
        label = 0.0
        if not play_row.empty and 'comp_yac_epa' in play_row.columns:
            val = play_row.iloc[0]['comp_yac_epa']
            if pd.notna(val):
                label = float(val)
        
        target = torch.FloatTensor([[label]])
        
        return {
            'node_features': node_feat,
            'edge_index': edge_index,
            'edge_features': edge_feat,
            'ball_progress': ball_progress,
            'target': target
        }

dataset = YacGraphDataset(
    df_a_filtered, df_b, df_c_filtered,
    node_features_available, edge_features_available,
    node_feature_means, node_feature_stds,
    edge_feature_means, edge_feature_stds
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# ============================================================================
# 5. Model
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: MODEL ARCHITECTURE")
print("-" * 80)

class MultiHeadGraphAttention(nn.Module):
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
        
        src, dst = edge_index
        
        # Attention scores
        scores = (Q[src] * K[dst]).sum(dim=-1) / (self.head_dim ** 0.5)
        scores = scores + self.edge_attn(e)
        
        # Softmax
        attn_weights = torch.zeros_like(scores)
        for i in range(N):
            mask = (src == i)
            if mask.any():
                attn_weights[mask] = F.softmax(scores[mask], dim=0)
                
        # Aggregate
        weighted_V = V[src] * attn_weights.unsqueeze(-1)
        out = torch.zeros(N, self.n_heads, self.head_dim, device=node_feat.device)
        out.index_add_(0, dst, weighted_V) # Safer aggregation
        
        out = self.out_proj(out.view(N, -1))
        return out + h # Residual

class YacPredictionModel(nn.Module):
    def __init__(self, node_in, edge_in, hidden, n_heads):
        super().__init__()
        self.gat = MultiHeadGraphAttention(node_in, edge_in, hidden, n_heads)
        
        # Regression Head: No Sigmoid!
        self.yac_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1) # Linear output for regression
        )
        
    def forward(self, node, edge_idx, edge_feat, bp):
        h = self.gat(node, edge_idx, edge_feat)
        
        # Global pooling
        graph_emb = h.mean(dim=0).unsqueeze(0).float()
        
        # Predict continuous EPA value
        yac_epa = self.yac_head(graph_emb)
        return yac_epa

model = YacPredictionModel(
    len(node_features_available), 
    len(edge_features_available), 
    HIDDEN_DIM, 
    N_ATTENTION_HEADS
).to(DEVICE)

# ============================================================================
# 6. Training Loop
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: TRAINING (REGRESSION)")
print("-" * 80)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss() # <--- Mean Squared Error for Regression

best_val_loss = float('inf')
start_time = time.time()
epoch = 0

while True:
    elapsed = (time.time() - start_time) / 60.0
    if elapsed >= MAX_TRAIN_TIME_MINUTES:
        print(f"🛑 Time limit ({MAX_TRAIN_TIME_MINUTES}m) reached.")
        break
        
    epoch += 1
    model.train()
    train_loss = 0.0
    count = 0
    
    for batch in train_loader:
        # Unpack
        nf = batch['node_features'][0].to(DEVICE)
        ei = batch['edge_index'][0].to(DEVICE)
        ef = batch['edge_features'][0].to(DEVICE)
        bp = batch['ball_progress'][0].to(DEVICE)
        target = batch['target'][0].to(DEVICE)
        
        # Skip bad graphs
        if ei.numel() == 0 or ef.numel() == 0: continue
        if ei.shape[0] != 2: ei = ei.t()
        if ei.shape[0] != 2: continue
            
        # Forward
        pred = model(nf, ei, ef, bp)
        
        # Loss (MSE)
        loss = criterion(pred, target) # No clamping for regression
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        count += 1
        
        if count % 500 == 0:
            print(f"  Epoch {epoch} | Batch {count} | MSE Loss: {loss.item():.4f}")

    avg_train_loss = train_loss / count if count > 0 else 0
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_count = 0
    with torch.no_grad():
        for batch in val_loader:
            nf = batch['node_features'][0].to(DEVICE)
            ei = batch['edge_index'][0].to(DEVICE)
            ef = batch['edge_features'][0].to(DEVICE)
            bp = batch['ball_progress'][0].to(DEVICE)
            target = batch['target'][0].to(DEVICE)
            
            if ei.numel() == 0: continue
            if ei.shape[0] != 2: ei = ei.t()
            
            pred = model(nf, ei, ef, bp)
            val_loss += criterion(pred, target).item()
            val_count += 1
            
    avg_val_loss = val_loss / val_count if val_count > 0 else 0
    
    print(f"Epoch {epoch}: Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'yac_model.pth'))
        print(f"  ✓ Saved best model")

print("\nDone. Model saved to:", OUTPUT_DIR)