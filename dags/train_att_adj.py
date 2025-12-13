"""
train_att_adj.py - Train Attention-Based Adjacency Matrix
==========================================================
Trains a graph attention network to learn dynamic adjacency matrices
for player-player interactions during pass plays.

INPUTS:
  - outputs/dataframe_a/v2.parquet (node features)
  - outputs/dataframe_b/v3.parquet (play context + ball trajectory)
  - outputs/dataframe_c/v3_pilot_3games.parquet (edge features)

OUTPUTS:
  - model_outputs/attention_model.pth (trained model)
  - model_outputs/attention_weights/ (learned adjacency matrices)

ARCHITECTURE:
  - Multi-head graph attention (4 heads)
  - Temporally-dynamic priors based on ball_progress
  - Role-aware attention (defender vs receiver updates)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import time
from datetime import datetime, timedelta

# Import utils
import sys
sys.path.append('scripts')
from utils import load_parquet_to_df

# ============================================================================
# Configuration
# ============================================================================

print("=" * 80)
print("TRAIN ATTENTION-BASED ADJACENCY MATRIX")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# *** PILOT MODE ***
PILOT_MODE = True  # Set to False for full dataset
PILOT_DF_C = 'outputs/dataframe_c/v3_pilot_3games.parquet'
REGULAR_DF_C = 'outputs/dataframe_c/v3.parquet'

# File paths
INPUT_DF_A = 'outputs/dataframe_a/v2.parquet'
INPUT_DF_B = 'outputs/dataframe_b/v4.parquet'
INPUT_DF_C = PILOT_DF_C if PILOT_MODE else REGULAR_DF_C
INPUT_DF_D = 'outputs/dataframe_d/v1.parquet'
OUTPUT_DIR = 'model_outputs/attention'
os.makedirs(OUTPUT_DIR, exist_ok=True)

if PILOT_MODE:
    print("\n" + "⚠" * 40)
    print("PILOT MODE ENABLED".center(80))
    print(f"Using pilot dataset: {PILOT_DF_C}".center(80))
    print("Set PILOT_MODE = False for full training".center(80))
    print("⚠" * 40 + "\n")

# Model hyperparameters
N_ATTENTION_HEADS = 4
NODE_FEATURE_DIM = 64
EDGE_FEATURE_DIM = 32
HIDDEN_DIM = 128
LEARNING_RATE = 0.001
BATCH_SIZE = 32

# --- TIME-BASED TRAINING CONFIGURATION ---
MAX_TRAIN_TIME_MINUTES = 8  # Stop training after approx 60 minutes
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

# -----------------------------------------------------------------------
# Filter to pilot games if PILOT_MODE enabled
# -----------------------------------------------------------------------

if PILOT_MODE:
    print("\n" + "=" * 80)
    print("PILOT MODE: FILTERING TO PILOT GAMES")
    print("-" * 80)
    
    # Get unique games from df_c (pilot)
    pilot_games = df_c['game_id'].unique()
    pilot_plays = df_c[['game_id', 'play_id']].drop_duplicates()
    
    print(f"Pilot dataset:")
    print(f"  Games: {len(pilot_games)}")
    print(f"  Plays: {len(pilot_plays)}")
    
    # Filter df_a
    df_a_size_before = len(df_a)
    df_a = df_a[df_a['game_id'].isin(pilot_games)].copy()
    print(f"\nFiltered df_a: {df_a_size_before:,} → {len(df_a):,} rows ({100*len(df_a)/df_a_size_before:.1f}%)")
    
    # Filter df_b
    df_b_size_before = len(df_b)
    df_b = df_b[df_b['game_id'].isin(pilot_games)].copy()
    print(f"Filtered df_b: {df_b_size_before:,} → {len(df_b):,} rows ({100*len(df_b)/df_b_size_before:.1f}%)")
    
    print("=" * 80)

print("\nDataset summary:")
print(f"  Nodes (df_a): {len(df_a):,} player-frames")
print(f"  Plays (df_b): {len(df_b):,} plays")
print(f"  Edges (df_c): {len(df_c):,} player-player interactions")
print(f"  Unique games: {df_a['game_id'].nunique()}")
print(f"  Unique plays: {df_a['play_id'].nunique()}")

# ============================================================================
# 2. Filter to Relevant Players
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: FILTERING TO RELEVANT PLAYERS")
print("-" * 80)

print("Keeping only: Passer, Route Runners, Coverage Defenders")

# Filter df_a to relevant players
# isPasser, isRouteRunner, or coverage defenders
df_a_filtered = df_a[
    (df_a['isPasser'] == 1) | 
    (df_a['isRouteRunner'] == 1) |
    (df_a['coverage_responsibility'].notna())
].copy()

print(f"  Before: {len(df_a):,} player-frames")
print(f"  After: {len(df_a_filtered):,} player-frames")
print(f"  Reduction: {100*(1 - len(df_a_filtered)/len(df_a)):.1f}%")

# Filter df_c to only edges between relevant players
relevant_players = set(df_a_filtered['nfl_id'].unique())

df_c_filtered = df_c[
    df_c['playerA_id'].isin(relevant_players) &
    df_c['playerB_id'].isin(relevant_players)
].copy()

print(f"\nEdge filtering:")
print(f"  Before: {len(df_c):,} edges")
print(f"  After: {len(df_c_filtered):,} edges")
print(f"  Relevant players: {len(relevant_players):,}")

# ============================================================================
# 3. Feature Engineering
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: FEATURE ENGINEERING")
print("-" * 80)

# -----------------------------------------------------------------------
# 3a. Node Features (from df_a_filtered)
# -----------------------------------------------------------------------

print("\nExtracting node features...")

NODE_FEATURES = [
    # Spatial
    'x', 'y', 's', 'a', 'dir', 'o',
    
    # Velocity/acceleration vectors
    'v_x', 'v_y', 'a_x', 'a_y',
    
    # Orientation vectors
    'o_x', 'o_y',
    
    # Distance to ball
    'e_dist_ball_land',
    
    # Role indicators
    'isPasser', 'isTargeted', 'isRouteRunner',
]

# Filter to available features
node_features_available = [f for f in NODE_FEATURES if f in df_a_filtered.columns]
print(f"  Node features: {len(node_features_available)}")
print(f"    {node_features_available}")

# -----------------------------------------------------------------------
# 3b. Edge Features (from df_c_filtered)
# -----------------------------------------------------------------------

print("\nExtracting edge features...")

EDGE_FEATURES = [
    # Spatial distances
    'e_dist', 'x_dist', 'y_dist',
    
    # Relative angles
    'relative_angle_o', 'relative_angle_dir',
    
    # Ball-related distances
    'playerA_dist_to_landing', 'playerB_dist_to_landing',
    'playerA_dist_to_ball_current', 'playerB_dist_to_ball_current',
    
    # Ball-related angles
    'playerA_angle_to_ball_current', 'playerB_angle_to_ball_current',
    'playerA_angle_to_ball_landing', 'playerB_angle_to_ball_landing',
    
    # Convergence
    'playerA_ball_convergence', 'playerB_ball_convergence',
    
    # Velocity features
    'relative_v_x', 'relative_v_y', 'relative_speed',
    
    # Team indicator
    'same_team',
    
    # Ball trajectory context
    'ball_progress', 'frames_to_landing',
]

edge_features_available = [f for f in EDGE_FEATURES if f in df_c_filtered.columns]
print(f"  Edge features: {len(edge_features_available)}")
print(f"    {edge_features_available}")

# -----------------------------------------------------------------------
# 3c. Play Context Features (from df_b)
# -----------------------------------------------------------------------

print("\nExtracting play context features...")

PLAY_FEATURES = [
    'ball_flight_distance', 'ball_flight_frames',
    'throw_direction', 'throw_type',
    'down', 'yards_to_go',
]

play_features_available = [f for f in PLAY_FEATURES if f in df_b.columns]
print(f"  Play features: {len(play_features_available)}")
print(f"    {play_features_available}")

# -----------------------------------------------------------------------
# 3d. Create feature statistics for normalization
# -----------------------------------------------------------------------

print("\nCalculating feature statistics for normalization...")

# Node feature stats
node_feature_means = df_a_filtered[node_features_available].mean()
node_feature_stds = df_a_filtered[node_features_available].std()

# Edge feature stats  
edge_feature_means = df_c_filtered[edge_features_available].mean()
edge_feature_stds = df_c_filtered[edge_features_available].std()

print(f"  ✓ Feature statistics computed")

# Save for later use
feature_stats = {
    'node_features': node_features_available,
    'edge_features': edge_features_available,
    'play_features': play_features_available,
    'node_means': node_feature_means,
    'node_stds': node_feature_stds,
    'edge_means': edge_feature_means,
    'edge_stds': edge_feature_stds,
}

import pickle
with open(os.path.join(OUTPUT_DIR, 'feature_stats.pkl'), 'wb') as f:
    pickle.dump(feature_stats, f)

print(f"  ✓ Saved feature statistics to {OUTPUT_DIR}/feature_stats.pkl")

# ============================================================================
# 4. Build PyTorch Dataset
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: BUILD PYTORCH DATASET")
print("-" * 80)
class PassPlayGraphDataset(Dataset):
    """
    Dataset for pass play graphs.
    Each sample is one frame of one play with all relevant players and edges.
    """
    
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
        
        # Get unique (game_id, play_id, frame_id) combinations
        self.frames = df_a.groupby(['game_id', 'play_id', 'frame_id']).size().reset_index()[['game_id', 'play_id', 'frame_id']]
        print(f"  Dataset contains {len(self.frames):,} frames")
        
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        """
        Returns graph data for one frame.
        """
        # Get frame identifier
        game_id = self.frames.iloc[idx]['game_id']
        play_id = self.frames.iloc[idx]['play_id']
        frame_id = self.frames.iloc[idx]['frame_id']
        
        # Get nodes for this frame
        nodes = self.df_a[
            (self.df_a['game_id'] == game_id) &
            (self.df_a['play_id'] == play_id) &
            (self.df_a['frame_id'] == frame_id)
        ].copy()
        
        # Get edges for this frame
        edges = self.df_c[
            (self.df_c['game_id'] == game_id) &
            (self.df_c['play_id'] == play_id) &
            (self.df_c['frame_id'] == frame_id)
        ].copy()
        
        # Create node ID to index mapping
        node_ids = nodes['nfl_id'].values
        node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        
        # Extract and normalize node features
        node_feat = nodes[self.node_features].values
        node_feat = (node_feat - self.node_means.values) / (self.node_stds.values + 1e-8)
        node_feat = torch.FloatTensor(node_feat)
        
        # Extract and normalize edge features
        edge_feat = edges[self.edge_features].values
        edge_feat = (edge_feat - self.edge_means.values) / (self.edge_stds.values + 1e-8)
        edge_feat = torch.FloatTensor(edge_feat)
        
        # Build edge index [2, N_edges]
        edge_index = []
        for _, edge_row in edges.iterrows():
            src_id = edge_row['playerA_id']
            dst_id = edge_row['playerB_id']
            
            # Map to local indices
            if src_id in node_id_to_idx and dst_id in node_id_to_idx:
                src_idx = node_id_to_idx[src_id]
                dst_idx = node_id_to_idx[dst_id]
                edge_index.append([src_idx, dst_idx])
        
        edge_index = torch.LongTensor(edge_index)
        if edge_index.ndim == 1 or edge_index.shape[0] != 2:
            edge_index = edge_index.t()  # Transpose if needed
        
        # Get ball progress for temporal prior
        ball_progress = edges.iloc[0]['ball_progress'] if len(edges) > 0 and 'ball_progress' in edges.columns else 0.5
        ball_progress = torch.FloatTensor([ball_progress])

        # --- NEW CODE: Get the Real Target ---
        # 1. Find the play in df_b
        play_row = self.df_b[
            (self.df_b['game_id'] == game_id) & 
            (self.df_b['play_id'] == play_id)
        ]
        
        # 2. Extract the outcome
        # Note: Ensure 'passResult' is the correct column name in your df_b
        if not play_row.empty and 'pass_result' in play_row.columns:
            outcome_str = play_row.iloc[0]['pass_result']
            label = 1.0 if outcome_str == 'C' else 0.0
        else:
            # Fallback if data is missing (should not happen in good data)
            label = 0.0 
            
        target = torch.FloatTensor([[label]])
        # -------------------------------------
        
        return {
            'node_features': node_feat,
            'edge_index': edge_index,
            'edge_features': edge_feat,
            'ball_progress': ball_progress,
            'target': target,  # <--- Added target here
            'n_nodes': len(nodes),
        }
    
# Create dataset
print("\nCreating dataset...")
dataset = PassPlayGraphDataset(
    df_a_filtered, df_b, df_c_filtered,
    node_features_available, edge_features_available,
    node_feature_means, node_feature_stds,
    edge_feature_means, edge_feature_stds
)

print(f"  ✓ Created dataset with {len(dataset):,} frames")

# Create train/val split (80/20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"  Train: {train_size:,} frames")
print(f"  Val: {val_size:,} frames")

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # batch_size=1 for variable graph sizes
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

print(f"  ✓ Created dataloaders")

# ============================================================================
# 5. Define Attention Model
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: DEFINE ATTENTION MODEL")
print("-" * 80)

class MultiHeadGraphAttention(nn.Module):
    """
    Multi-head graph attention with football-specific priors.
    """
    
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // n_heads
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Query, Key, Value projections (per head)
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        
        # Edge attention
        self.edge_attn = nn.Linear(hidden_dim, n_heads)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, node_feat, edge_index, edge_feat, ball_progress, debug=False):
        """
        Args:
            node_feat: [N_nodes, node_in_dim]
            edge_index: [2, N_edges]
            edge_feat: [N_edges, edge_in_dim]
            ball_progress: scalar (0 to 1)
            debug: If True, print NaN detection checkpoints

        Returns:
            node_embeddings: [N_nodes, hidden_dim]
            attention_weights: [N_edges, n_heads]
        """
        N = node_feat.shape[0]
        E = edge_index.shape[1]

        if debug:
            print(f"\n    [ATTN-DEBUG] Graph: {N} nodes, {E} edges")
            print(f"    [ATTN-1] Input node_feat - NaN: {torch.isnan(node_feat).any()}, Inf: {torch.isinf(node_feat).any()}")
            print(f"    [ATTN-1] Input edge_feat - NaN: {torch.isnan(edge_feat).any()}, Inf: {torch.isinf(edge_feat).any()}")

        # Encode nodes and edges
        h = self.node_encoder(node_feat)  # [N, hidden_dim]
        e = self.edge_encoder(edge_feat)  # [E, hidden_dim]

        if debug:
            print(f"    [ATTN-2] After encoding - h NaN: {torch.isnan(h).any()}, e NaN: {torch.isnan(e).any()}")

        # Get Q, K, V
        Q = self.W_Q(h)  # [N, hidden_dim]
        K = self.W_K(h)
        V = self.W_V(h)

        if debug:
            print(f"    [ATTN-3] After QKV - Q NaN: {torch.isnan(Q).any()}, K NaN: {torch.isnan(K).any()}, V NaN: {torch.isnan(V).any()}")

        # Reshape for multi-head: [N, n_heads, head_dim]
        Q = Q.view(N, self.n_heads, self.head_dim)
        K = K.view(N, self.n_heads, self.head_dim)
        V = V.view(N, self.n_heads, self.head_dim)

        # Compute attention scores for each edge
        src_idx = edge_index[0]  # [E]
        dst_idx = edge_index[1]  # [E]

        # Get Q for sources, K for destinations
        Q_src = Q[src_idx]  # [E, n_heads, head_dim]
        K_dst = K[dst_idx]  # [E, n_heads, head_dim]

        # Dot product attention
        attn_scores = (Q_src * K_dst).sum(dim=-1) / (self.head_dim ** 0.5)  # [E, n_heads]

        if debug:
            print(f"    [ATTN-4] Dot product scores - NaN: {torch.isnan(attn_scores).any()}, Inf: {torch.isinf(attn_scores).any()}")
            print(f"    [ATTN-4] Score range: [{attn_scores.min():.4f}, {attn_scores.max():.4f}]")

        # Add edge feature contribution
        edge_contrib = self.edge_attn(e)  # [E, n_heads]
        if debug:
            print(f"    [ATTN-5] Edge contrib - NaN: {torch.isnan(edge_contrib).any()}, Inf: {torch.isinf(edge_contrib).any()}")
            print(f"    [ATTN-5] Edge contrib range: [{edge_contrib.min():.4f}, {edge_contrib.max():.4f}]")
        
        if edge_contrib.shape[0] == attn_scores.shape[0]:
            attn_scores = attn_scores + edge_contrib

        if debug:
            print(f"    [ATTN-6] After edge addition - NaN: {torch.isnan(attn_scores).any()}, Inf: {torch.isinf(attn_scores).any()}")

        # Apply football prior mask (temporal)
        prior_mask = self.compute_prior_mask(edge_feat, ball_progress, E)  # [E, n_heads]
        
        if debug:
            print(f"    [ATTN-7] Prior mask - NaN: {torch.isnan(prior_mask).any()}, Inf: {torch.isinf(prior_mask).any()}")
            print(f"    [ATTN-7] Mask range: [{prior_mask.min():.4f}, {prior_mask.max():.4f}]")
        
        attn_scores = attn_scores * prior_mask

        if debug:
            print(f"    [ATTN-8] After mask multiply - NaN: {torch.isnan(attn_scores).any()}, Inf: {torch.isinf(attn_scores).any()}")
            print(f"    [ATTN-8] Score range: [{attn_scores.min():.4f}, {attn_scores.max():.4f}]")

        # Softmax per source node
        attn_weights = torch.zeros_like(attn_scores)
        for i in range(N):
            mask = (src_idx == i)
            if mask.sum() > 0:
                attn_weights[mask] = F.softmax(attn_scores[mask], dim=0)

        if debug:
            print(f"    [ATTN-9] After softmax - NaN: {torch.isnan(attn_weights).any()}, Inf: {torch.isinf(attn_weights).any()}")
            print(f"    [ATTN-9] Weights range: [{attn_weights.min():.6f}, {attn_weights.max():.6f}]")

        # Aggregate messages
        V_src = V[src_idx]  # [E, n_heads, head_dim]
        weighted_V = V_src * attn_weights.unsqueeze(-1)  # [E, n_heads, head_dim]

        if debug:
            print(f"    [ATTN-10] Weighted V - NaN: {torch.isnan(weighted_V).any()}, Inf: {torch.isinf(weighted_V).any()}")

        # Aggregate to destination nodes
        messages = torch.zeros(N, self.n_heads, self.head_dim, device=node_feat.device)
        for e in range(E):
            dst = dst_idx[e]
            messages[dst] += weighted_V[e]

        if debug:
            print(f"    [ATTN-11] Messages - NaN: {torch.isnan(messages).any()}, Inf: {torch.isinf(messages).any()}")

        # Concatenate heads and project
        messages = messages.view(N, self.hidden_dim)
        output = self.out_proj(messages)

        if debug:
            print(f"    [ATTN-12] After out_proj - NaN: {torch.isnan(output).any()}, Inf: {torch.isinf(output).any()}")

        # Residual connection
        output = h + output

        if debug:
            print(f"    [ATTN-13] Final output (after residual) - NaN: {torch.isnan(output).any()}, Inf: {torch.isinf(output).any()}")
            print(f"    [ATTN-13] Output range: [{output.min():.6f}, {output.max():.6f}]")

        return output, attn_weights
    
    def compute_prior_mask(self, edge_feat, ball_progress, n_edges):
        """
        Compute football-specific prior mask based on ball progress.
        
        Returns: [n_edges, n_heads]
        """
        # Default: all 1.0 (no bias)
        mask = torch.ones(n_edges, self.n_heads, device=edge_feat.device)
        
        # TODO: Implement temporal priors based on ball_progress
        # Early flight: upweight route-running edges
        # Late flight: upweight ball-proximity edges
        
        return mask
    
class AttentionAdjacencyModel(nn.Module):
    """
    Full model: Multi-head attention → Output heads (Completion Probability + EPA)
    """
    
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, n_heads=4):
        super().__init__()
        
        self.attention = MultiHeadGraphAttention(
            node_in_dim, edge_in_dim, hidden_dim, n_heads
        )
        
        # 1. Completion Prediction Head (Binary Classification)
        # Predicts probability of pass completion (0 to 1)
        self.completion_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # Added for regularization
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # 2. EPA Prediction Head (Regression)
        # Predicts Expected Points Added (Continuous value)
        self.epa_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # No activation for regression
        )
        
    def forward(self, node_feat, edge_index, edge_feat, ball_progress, debug=False):
        """
        Forward pass.
        
        Returns:
            predictions: dict with 'completion_prob' and 'epa'
            attention_weights: [N_edges, n_heads]
        """
        if debug:
            print(f"  [MODEL] Starting forward pass")
        
        # Get node embeddings and attention weights
        h, attn_weights = self.attention(node_feat, edge_index, edge_feat, ball_progress, debug=debug)
        
        if debug:
            print(f"  [MODEL] After attention - h NaN: {torch.isnan(h).any()}, Inf: {torch.isinf(h).any()}")
        
        # Global pooling (mean over all nodes) to get graph-level representation
        h_graph = h.mean(dim=0)  # [hidden_dim]
        h_graph = h_graph.unsqueeze(0)  # [1, hidden_dim]

        if debug:
            print(f"  [MODEL] After pooling - h_graph NaN: {torch.isnan(h_graph).any()}, Inf: {torch.isinf(h_graph).any()}")

        # --- CRITICAL FIX for RuntimeError: matmul primitive (originally requested fix) ---
        # Explicitly cast to float32 to prevent CPU matmul errors
        h_graph = h_graph.float()
        # --------------------

        if debug:
            print(f"  [MODEL] After float cast - h_graph NaN: {torch.isnan(h_graph).any()}, Inf: {torch.isinf(h_graph).any()}")

        # Predict completion (Probability)
        completion_prob = self.completion_head(h_graph)  # [1, 1]

        if debug:
            print(f"  [MODEL] completion_prob - NaN: {torch.isnan(completion_prob).any()}, Inf: {torch.isinf(completion_prob).any()}")
            print(f"  [MODEL] completion_prob value: {completion_prob.item():.6f}")

        # Predict EPA (Score)
        epa_pred = self.epa_head(h_graph) # [1, 1]
        
        if debug:
            print(f"  [MODEL] epa_pred - NaN: {torch.isnan(epa_pred).any()}, Inf: {torch.isinf(epa_pred).any()}")
        
        return {
            'completion_prob': completion_prob,
            'epa': epa_pred
        }, attn_weights

# Create model
print("\nInitializing model...")
model = AttentionAdjacencyModel(
    node_in_dim=len(node_features_available),
    edge_in_dim=len(edge_features_available),
    hidden_dim=HIDDEN_DIM,
    n_heads=N_ATTENTION_HEADS
).to(DEVICE)

print(f"  ✓ Model created")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Architecture:")
print(f"    - Node input dim: {len(node_features_available)}")
print(f"    - Edge input dim: {len(edge_features_available)}")
print(f"    - Hidden dim: {HIDDEN_DIM}")
print(f"    - Attention heads: {N_ATTENTION_HEADS}")

# ============================================================================
# 6. Training Loop
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: TRAINING LOOP")
print("-" * 80)

# Optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()  # Binary cross-entropy for completion prediction

print(f"\nTraining configuration:")
print(f"  Optimizer: Adam (lr={LEARNING_RATE})")
print(f"  Loss: BCELoss (completion prediction)")
print(f"  Max Training Time: {MAX_TRAIN_TIME_MINUTES} minutes")
print(f"  Train batches: {len(train_loader):,}")
print(f"  Val batches: {len(val_loader):,}")

# Training loop stats
best_val_loss = float('inf')
train_losses = []
val_losses = []

print("\n" + "=" * 80)
print("Starting training...")
print("=" * 80)

# --- TIME-BASED LOOP CONTROL ---
training_start_time = time.time()
epoch = 0

while True:
    current_time = time.time()
    elapsed_minutes = (current_time - training_start_time) / 60.0
    
    # Check if we should stop (must complete the current epoch if started)
    if elapsed_minutes >= MAX_TRAIN_TIME_MINUTES:
        print(f"\n\n🛑 Time limit reached ({elapsed_minutes:.1f} / {MAX_TRAIN_TIME_MINUTES} mins). Stopping training.")
        break
        
    epoch += 1
    epoch_start_time = time.time()
    
    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    model.train()
    train_loss = 0.0
    train_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Move to device
        node_feat = batch['node_features'][0].to(DEVICE)  # [0] because batch_size=1
        edge_index = batch['edge_index'][0].to(DEVICE)
        edge_feat = batch['edge_features'][0].to(DEVICE)
        ball_progress = batch['ball_progress'][0].to(DEVICE)
        
        # --- FIX: Robustly ensure edge_index is [2, N_edges] and handle empty graphs ---
        
        # 1. Check for empty graph (e.g., shape [0, 2] or empty tensor)
        if edge_index.numel() == 0 or edge_feat.numel() == 0:
            continue
        
        # 2. If shape is [N_edges, 2] (common shape from data processing), transpose to [2, N_edges]
        if edge_index.ndim == 2 and edge_index.shape[1] == 2 and edge_index.shape[0] != 2:
            edge_index = edge_index.t().contiguous()
        
        # 3. Final check to skip any remaining malformed shapes
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            print(f"🔴 Skipped malformed edge_index at batch {batch_idx}: shape={edge_index.shape}")
            continue
        
        # -----------------------------------------------------------------------------

        # Forward pass (enable debug on first batch)
        debug_mode = (epoch == 1 and batch_idx == 0)
        predictions, attn_weights = model(node_feat, edge_index, edge_feat, ball_progress, debug=debug_mode)
        
        # --- FIX: USE REAL TARGET FROM DATA ---
        target = batch['target'][0].to(DEVICE)
        # --------------------------------------
        
        # Check for NaN or Inf in predictions (common cause of out-of-range values)
        if torch.isnan(predictions['completion_prob']).any() or torch.isinf(predictions['completion_prob']).any():
            print(f"\n{'='*80}")
            print(f"🔴 NaN/Inf DETECTED AT BATCH {batch_idx} (Epoch {epoch})")
            print(f"{'='*80}")
            print(f"Graph: {node_feat.shape[0]} nodes, {edge_index.shape[1]} edges")
            print(f"Input checks:")
            print(f"  Node features - NaN: {torch.isnan(node_feat).any()}, Inf: {torch.isinf(node_feat).any()}")
            print(f"  Edge features - NaN: {torch.isnan(edge_feat).any()}, Inf: {torch.isinf(edge_feat).any()}")
            
            # RE-RUN WITH DEBUG MODE ON
            print(f"\n{'*'*80}")
            print(f"🔍 RE-RUNNING BATCH {batch_idx} WITH FULL DEBUG TRACE")
            print(f"{'*'*80}")
            with torch.no_grad():
                predictions_debug, _ = model(node_feat, edge_index, edge_feat, ball_progress, debug=True)
            
            # Save problematic batch data
            print(f"\n💾 Saving problematic batch data...")
            torch.save({
                'batch_idx': batch_idx,
                'epoch': epoch,
                'node_feat': node_feat.cpu(),
                'edge_index': edge_index.cpu(),
                'edge_feat': edge_feat.cpu(),
                'ball_progress': ball_progress.cpu(),
                'predictions': predictions,
            }, os.path.join(OUTPUT_DIR, f'nan_batch_{epoch}_{batch_idx}.pt'))
            print(f"  Saved to: {OUTPUT_DIR}/nan_batch_{epoch}_{batch_idx}.pt")
            print(f"{'='*80}\n")
            
            # Skip this batch
            continue
        
        # Compute loss
        # Clamp BOTH predictions and targets to avoid numerical instability with BCE
        completion_prob_clamped = torch.clamp(predictions['completion_prob'], min=1e-7, max=1-1e-7)
        target_clamped = torch.clamp(target, min=1e-7, max=1-1e-7)
        
        loss = criterion(completion_prob_clamped, target_clamped)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_batches += 1
        
        # Progress update every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
    
    avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
    train_losses.append(avg_train_loss)
    
    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------
    model.eval()
    val_loss = 0.0
    val_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            node_feat = batch['node_features'][0].to(DEVICE)
            edge_index = batch['edge_index'][0].to(DEVICE)
            edge_feat = batch['edge_features'][0].to(DEVICE)
            ball_progress = batch['ball_progress'][0].to(DEVICE)
            target = batch['target'][0].to(DEVICE)  # Load real target in validation too
            
            # --- FIX: Robustly ensure edge_index is [2, N_edges] and handle empty graphs in validation ---
            if edge_index.numel() == 0 or edge_feat.numel() == 0:
                continue
            
            if edge_index.ndim == 2 and edge_index.shape[1] == 2 and edge_index.shape[0] != 2:
                edge_index = edge_index.t().contiguous()
            
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                continue
            # ---------------------------------------------------------------------------------------------
            
            predictions, attn_weights = model(node_feat, edge_index, edge_feat, ball_progress)
            
            # Clamp both predictions and targets
            completion_prob_clamped = torch.clamp(predictions['completion_prob'], min=1e-7, max=1-1e-7)
            target_clamped = torch.clamp(target, min=1e-7, max=1-1e-7)
            
            loss = criterion(completion_prob_clamped, target_clamped)
            val_loss += loss.item()
            val_batches += 1
    
    avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
    val_losses.append(avg_val_loss)
    
    # Time for this epoch
    epoch_duration = time.time() - epoch_start_time
    total_elapsed = (time.time() - training_start_time) / 60.0
    
    # Print epoch summary
    print(f"\nEpoch {epoch}:")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss: {avg_val_loss:.4f}")
    print(f"  Time: {epoch_duration:.1f}s | Total Elapsed: {total_elapsed:.1f}m / {MAX_TRAIN_TIME_MINUTES}m")
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, os.path.join(OUTPUT_DIR, 'best_model.pth'))
        print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")
    
    print("-" * 80)

# Save final model
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses,
}, os.path.join(OUTPUT_DIR, 'final_model.pth'))

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"Total Epochs: {epoch}")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Models saved to: {OUTPUT_DIR}/")
print(f"  - best_model.pth (lowest val loss)")
print(f"  - final_model.pth (last epoch)")

# ============================================================================
# Complete
# ============================================================================

print("\n" + "=" * 80)
print("TRAIN ATTENTION MODEL: SCAFFOLD COMPLETE")
print("=" * 80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nNext steps:")
print("  1. Implement feature engineering")
print("  2. Create PyTorch dataset")
print("  3. Define multi-head attention model")
print("  4. Implement training loop")
print("\n" + "=" * 80)