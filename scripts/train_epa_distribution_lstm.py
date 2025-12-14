"""
train_epa_distribution_lstm.py - Train EPA Distribution Prediction Model (LSTM)
=================================================================================
Trains a GAT+LSTM model to predict EPA distribution over sequential frames.

KEY DIFFERENCES FROM train_epa_distribution.py:
  ✨ Processes ENTIRE PLAY as a sequence (not frame-by-frame)
  ✨ LSTM captures temporal dependencies between frames
  ✨ Uses padding + masking for variable-length plays
  ✨ Learns trajectory patterns and momentum

ARCHITECTURE:
  1. GAT: Extract spatial features from each frame's graph
  2. LSTM: Process sequence of graph embeddings over time  
  3. Distribution Head: Predict (μ, σ) at each timestep

INPUTS:
  - outputs/dataframe_a/v1.parquet (node features)
  - outputs/dataframe_b/v1.parquet (play context + EPA targets)
  - outputs/dataframe_c/v2_pilot_3games.parquet (structured edges)

OUTPUTS:
  - model_outputs/epa_distribution_lstm/model.pth
  - model_outputs/epa_distribution_lstm/training_log.csv
  - model_outputs/epa_distribution_lstm/predictions.csv
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.distributions import Normal
import os
import time
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

print("=" * 80)
print("TRAIN EPA DISTRIBUTION PREDICTION MODEL (LSTM)")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Environment variables
PILOT_MODE = os.getenv('NFL_PILOT_MODE', 'true').lower() == 'true'
PILOT_N_GAMES = int(os.getenv('NFL_PILOT_N_GAMES', '3'))
DATA_VERSION = os.getenv('NFL_DATA_VERSION', 'v1')

# File paths
INPUT_DF_A = f'outputs/dataframe_a/{DATA_VERSION}.parquet'
INPUT_DF_B = f'outputs/dataframe_b/{DATA_VERSION}.parquet'

if PILOT_MODE:
    INPUT_DF_C = f'outputs/dataframe_c/v2_pilot_{PILOT_N_GAMES}games.parquet'
else:
    INPUT_DF_C = 'outputs/dataframe_c/v2.parquet'

# Output directory
OUTPUT_DIR = 'model_outputs/epa_distribution_lstm'
os.makedirs(OUTPUT_DIR, exist_ok=True)

if PILOT_MODE:
    print("\n" + "⚠" * 40)
    print("PILOT MODE ENABLED (LSTM Sequential Processing)".center(80))
    print(f"Using pilot dataset: {INPUT_DF_C}".center(80))
    print("⚠" * 40 + "\n")

# Model hyperparameters
N_ATTENTION_HEADS = int(os.getenv('NFL_ATTENTION_HEADS', '4'))
HIDDEN_DIM = int(os.getenv('NFL_HIDDEN_DIM', '128'))
LSTM_HIDDEN_DIM = int(os.getenv('NFL_LSTM_HIDDEN_DIM', '128'))
LSTM_LAYERS = 2  # Number of stacked LSTM layers
LEARNING_RATE = float(os.getenv('NFL_LEARNING_RATE', '0.001'))
BATCH_SIZE = 8  # Process multiple plays at once

# Training configuration
MAX_TRAIN_TIME_MINUTES = int(os.getenv('NFL_TRAIN_TIME_MINUTES', '10'))
PRINT_EVERY = int(os.getenv('NFL_PRINT_EVERY', '50'))  # Fewer batches now (plays not frames)

print(f"Configuration:")
print(f"  Data version: {DATA_VERSION}")
print(f"  Pilot mode: {PILOT_MODE}")
if PILOT_MODE:
    print(f"  Pilot games: {PILOT_N_GAMES}")
print(f"\nModel hyperparameters:")
print(f"  GAT hidden dim: {HIDDEN_DIM}")
print(f"  LSTM hidden dim: {LSTM_HIDDEN_DIM}")
print(f"  LSTM layers: {LSTM_LAYERS}")
print(f"  Attention heads: {N_ATTENTION_HEADS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Batch size: {BATCH_SIZE} plays")
print(f"\nTraining configuration:")
print(f"  Max time: {MAX_TRAIN_TIME_MINUTES} minutes")
print(f"  Print every: {PRINT_EVERY} batches")
print()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}\n")

# ============================================================================
# 1. Load Data
# ============================================================================

print("=" * 80)
print("STEP 1: LOADING DATA")
print("-" * 80)

print("Loading dataframe_a (node features)...")
df_a = pd.read_parquet(INPUT_DF_A)
print(f"  ✓ Loaded {len(df_a):,} rows")

print("\nLoading dataframe_b (play-level features + EPA targets)...")
df_b = pd.read_parquet(INPUT_DF_B)
print(f"  ✓ Loaded {len(df_b):,} plays")

print("\nLoading dataframe_c v2 (structured edges)...")
df_c = pd.read_parquet(INPUT_DF_C)
print(f"  ✓ Loaded {len(df_c):,} edges")

edge_types = df_c['edge_type'].unique()
print(f"  Edge types: {list(edge_types)}")

if PILOT_MODE:
    pilot_games = df_c['game_id'].unique()
    df_a = df_a[df_a['game_id'].isin(pilot_games)].copy()
    df_b = df_b[df_b['game_id'].isin(pilot_games)].copy()
    print(f"\nFiltered to {len(pilot_games)} pilot games.")

# ============================================================================
# 2. Filter Data
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: FILTERING DATA")
print("-" * 80)

# Keep only relevant players
df_a_filtered = df_a[
    (df_a['isPasser'] == 1) | 
    (df_a['isRouteRunner'] == 1) |
    (df_a['coverage_responsibility'].notna())
].copy()

df_c_filtered = df_c.copy()

print(f"Nodes: {len(df_a_filtered):,}")
print(f"Edges: {len(df_c_filtered):,}")

edge_type_dist = df_c_filtered['edge_type'].value_counts()
print(f"Edge type distribution:")
for et, count in edge_type_dist.items():
    pct = 100 * count / len(df_c_filtered)
    print(f"  {et}: {count:,} ({pct:.1f}%)")

# ============================================================================
# 3. Feature Engineering
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: FEATURE ENGINEERING")
print("-" * 80)

# Node features
NODE_FEATURES = [
    'x', 'y', 's', 'a', 'dir', 'o',
    'v_x', 'v_y', 'a_x', 'a_y',
    'dist_to_ball_carrier', 'dist_to_football',
    'isRouteRunner', 'isPasser', 'isBallCarrier', 'isBlocking'
]

# Edge features
EDGE_FEATURES = [
    'e_dist', 'x_dist', 'y_dist',
    'relative_angle_o', 'relative_angle_dir',
    'playerA_dist_to_landing', 'playerB_dist_to_landing',
    'playerA_dist_to_ball_current', 'playerB_dist_to_ball_current',
    'playerA_angle_to_ball_current', 'playerB_angle_to_ball_current',
    'playerA_angle_to_ball_landing', 'playerB_angle_to_ball_landing',
    'playerA_ball_convergence', 'playerB_ball_convergence',
    'relative_v_x', 'relative_v_y', 'relative_speed',
    'same_team', 'ball_progress', 'frames_to_landing',
    'attention_prior',
    'coverage_scheme_encoded',
]

# Filter to available features
node_features_available = [f for f in NODE_FEATURES if f in df_a_filtered.columns]
edge_features_available = [f for f in EDGE_FEATURES if f in df_c_filtered.columns]

print(f"Node features: {len(node_features_available)}")
print(f"Edge features: {len(edge_features_available)}")

# Compute normalization statistics
node_feature_means = df_a_filtered[node_features_available].mean()
node_feature_stds = df_a_filtered[node_features_available].std()
edge_feature_means = df_c_filtered[edge_features_available].mean()
edge_feature_stds = df_c_filtered[edge_features_available].std()

print(f"\nNormalization computed:")
print(f"  Node features mean range: [{node_feature_means.min():.2f}, {node_feature_means.max():.2f}]")
print(f"  Edge features mean range: [{edge_feature_means.min():.2f}, {edge_feature_means.max():.2f}]")

# Edge type encoding
edge_type_to_idx = {et: i for i, et in enumerate(df_c_filtered['edge_type'].unique())}
print(f"\nEdge type encoding: {edge_type_to_idx}")

# ============================================================================
# 4. PyTorch Dataset (Sequential - Returns Entire Play)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: DATASET CREATION (SEQUENTIAL)")
print("-" * 80)

class SequentialEPADataset(Dataset):
    """
    Dataset that returns entire play sequences (not individual frames).
    Each item is a sequence of graph embeddings for all frames in a play.
    """
    def __init__(self, df_a, df_b, df_c, node_features, edge_features,
                 node_means, node_stds, edge_means, edge_stds, edge_type_to_idx):
        self.df_a = df_a
        self.df_b = df_b
        self.df_c = df_c
        self.node_features = node_features
        self.edge_features = edge_features
        self.node_means = node_means
        self.node_stds = node_stds
        self.edge_means = edge_means
        self.edge_stds = edge_stds
        self.edge_type_to_idx = edge_type_to_idx
        
        # Get unique plays (not frames!)
        self.plays = df_c.groupby(['game_id', 'play_id']).size().reset_index()[
            ['game_id', 'play_id']]
    
    def __len__(self):
        return len(self.plays)
    
    def build_frame_graph(self, game_id, play_id, frame_id):
        """Build graph for a single frame"""
        # Get nodes
        nodes = self.df_a[
            (self.df_a['game_id'] == game_id) &
            (self.df_a['play_id'] == play_id) &
            (self.df_a['frame_id'] == frame_id)
        ].copy()
        
        if len(nodes) == 0:
            return None
        
        # Node mapping
        node_ids = nodes['nfl_id'].values
        valid_nfl_ids = set(node_ids)
        node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        
        # Get edges
        edges_raw = self.df_c[
            (self.df_c['game_id'] == game_id) &
            (self.df_c['play_id'] == play_id) &
            (self.df_c['frame_id'] == frame_id)
        ]
        
        edges = edges_raw[
            edges_raw['playerA_id'].isin(valid_nfl_ids) &
            edges_raw['playerB_id'].isin(valid_nfl_ids)
        ].copy()
        
        if len(edges) == 0:
            return None
        
        # Extract features
        node_feat = (nodes[self.node_features].values - self.node_means.values) / (self.node_stds.values + 1e-8)
        edge_feat = (edges[self.edge_features].values - self.edge_means.values) / (self.edge_stds.values + 1e-8)
        
        # Build edge index
        edge_index = []
        for _, row in edges.iterrows():
            src_idx = node_id_to_idx[row['playerA_id']]
            dst_idx = node_id_to_idx[row['playerB_id']]
            edge_index.append([src_idx, dst_idx])
        
        edge_index = torch.LongTensor(edge_index).t() if len(edge_index) > 0 else torch.zeros((2, 0), dtype=torch.long)
        
        return {
            'node_feat': torch.FloatTensor(node_feat),
            'edge_index': edge_index,
            'edge_feat': torch.FloatTensor(edge_feat),
        }
    
    def __getitem__(self, idx):
        game_id = self.plays.iloc[idx]['game_id']
        play_id = self.plays.iloc[idx]['play_id']
        
        # Get all frames for this play
        play_frames = self.df_c[
            (self.df_c['game_id'] == game_id) &
            (self.df_c['play_id'] == play_id)
        ]['frame_id'].unique()
        
        play_frames = sorted(play_frames)
        
        # Build graph for each frame
        frame_graphs = []
        for frame_id in play_frames:
            graph = self.build_frame_graph(game_id, play_id, frame_id)
            if graph is not None:
                frame_graphs.append(graph)
        
        # Get EPA target
        play_row = self.df_b[
            (self.df_b['game_id'] == game_id) &
            (self.df_b['play_id'] == play_id)
        ]
        
        epa_target = 0.0
        if not play_row.empty and 'expected_points_added' in play_row.columns:
            val = play_row.iloc[0]['expected_points_added']
            if pd.notna(val):
                epa_target = float(val)
        
        return {
            'frame_graphs': frame_graphs,  # List of graphs (variable length)
            'sequence_length': len(frame_graphs),
            'epa_target': torch.FloatTensor([epa_target]),
            'game_id': game_id,
            'play_id': play_id,
        }

def collate_sequential_batch(batch):
    """
    Custom collate function for variable-length play sequences.
    Uses padding to make all sequences same length in batch.
    """
    # Extract data
    sequences = [item['frame_graphs'] for item in batch]
    lengths = torch.tensor([item['sequence_length'] for item in batch])
    targets = torch.stack([item['epa_target'] for item in batch])
    game_ids = [item['game_id'] for item in batch]
    play_ids = [item['play_id'] for item in batch]
    
    # Find max sequence length in this batch
    max_len = max(lengths)
    
    # We'll return list of graphs per timestep (for GAT processing)
    # batch[timestep_i] = list of graphs for timestep i across batch
    batch_size = len(sequences)
    
    return {
        'sequences': sequences,  # List[List[graph_dict]] - [batch][timestep]
        'lengths': lengths,       # [batch]
        'targets': targets,       # [batch, 1]
        'max_length': max_len,
        'game_ids': game_ids,
        'play_ids': play_ids,
    }

# Create dataset
dataset = SequentialEPADataset(
    df_a_filtered, df_b, df_c_filtered,
    node_features_available, edge_features_available,
    node_feature_means, node_feature_stds,
    edge_feature_means, edge_feature_stds,
    edge_type_to_idx
)

# Train/val split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    collate_fn=collate_sequential_batch
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    collate_fn=collate_sequential_batch
)

print(f"Dataset created:")
print(f"  Total plays: {len(dataset):,}")
print(f"  Train: {train_size:,}")
print(f"  Val: {val_size:,}")

# Get sequence length statistics
seq_lengths = [dataset[i]['sequence_length'] for i in range(len(dataset))]
print(f"\nSequence length statistics:")
print(f"  Min: {min(seq_lengths)} frames")
print(f"  Mean: {np.mean(seq_lengths):.1f} frames")
print(f"  Median: {np.median(seq_lengths):.0f} frames")
print(f"  Max: {max(seq_lengths)} frames")
print(f"  Padding overhead with max={max(seq_lengths)}: {(max(seq_lengths) - np.mean(seq_lengths)) / max(seq_lengths) * 100:.1f}%")

# ============================================================================
# 5. Model Architecture (GAT + LSTM)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: MODEL ARCHITECTURE (GAT + LSTM)")
print("-" * 80)

class MultiHeadGraphAttention(nn.Module):
    """Multi-head graph attention with edge features (same as before)"""
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
        
        self.last_attention_weights = attn_weights
        
        weighted_V = V[src] * attn_weights.unsqueeze(-1)
        out = torch.zeros(N, self.n_heads, self.head_dim, device=node_feat.device)
        out.index_add_(0, dst, weighted_V)
        
        out = self.out_proj(out.view(N, -1))
        return out + h

class GAT_LSTM_EPAModel(nn.Module):
    """
    EPA Distribution Prediction with GAT + LSTM
    
    Architecture:
      1. For each frame: GAT extracts spatial graph embedding
      2. LSTM processes sequence of graph embeddings
      3. Distribution head predicts (μ, σ) at each timestep
    """
    def __init__(self, node_in_dim, edge_in_dim, gat_hidden_dim, lstm_hidden_dim, 
                 n_heads, lstm_layers):
        super().__init__()
        
        # Spatial encoder (GAT)
        self.gat = MultiHeadGraphAttention(node_in_dim, edge_in_dim, gat_hidden_dim, n_heads)
        
        # Temporal encoder (LSTM)
        self.lstm = nn.LSTM(
            input_size=gat_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.2 if lstm_layers > 1 else 0.0
        )
        
        # Distribution prediction head
        self.distribution_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # (μ, raw_σ)
        )
    
    def forward(self, sequences, lengths, return_attention=False):
        """
        Args:
            sequences: List[List[graph_dict]] - [batch][timestep]
            lengths: [batch] - actual length of each sequence
            return_attention: bool
        
        Returns:
            mu: [batch, max_len]
            sigma: [batch, max_len]
            mask: [batch, max_len] - True for real frames, False for padding
            attention: Optional[List] - attention weights per timestep
        """
        batch_size = len(sequences)
        max_len = max(lengths)
        
        # Step 1: Process each frame with GAT to get graph embeddings
        # We'll build this frame-by-frame across the batch
        graph_embeddings = []  # Will be [batch, max_len, gat_hidden_dim]
        attention_weights = [] if return_attention else None
        
        for t in range(max_len):
            # Collect graphs at timestep t across batch
            timestep_embeds = []
            timestep_attns = []
            
            for b in range(batch_size):
                if t < lengths[b]:
                    # Real frame - process graph
                    graph = sequences[b][t]
                    node_feat = graph['node_feat'].to(DEVICE)
                    edge_index = graph['edge_index'].to(DEVICE)
                    edge_feat = graph['edge_feat'].to(DEVICE)
                    
                    # GAT forward
                    h = self.gat(node_feat, edge_index, edge_feat)
                    graph_emb = h.mean(dim=0)  # Pool nodes
                    
                    timestep_embeds.append(graph_emb)
                    
                    if return_attention:
                        timestep_attns.append(self.gat.last_attention_weights)
                else:
                    # Padding - zero embedding
                    timestep_embeds.append(torch.zeros(self.gat.out_proj.out_features, device=DEVICE))
                    if return_attention:
                        timestep_attns.append(None)
            
            graph_embeddings.append(torch.stack(timestep_embeds))
            if return_attention:
                attention_weights.append(timestep_attns)
        
        # Stack: [max_len, batch, gat_hidden] → transpose to [batch, max_len, gat_hidden]
        graph_embeddings = torch.stack(graph_embeddings).transpose(0, 1)
        
        # Step 2: LSTM with packing (efficient - skips padding)
        packed = pack_padded_sequence(
            graph_embeddings,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        lstm_out, (h_n, c_n) = self.lstm(packed)
        
        # Unpack: [batch, max_len, lstm_hidden]
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        # Step 3: Predict distribution at each timestep
        predictions = self.distribution_head(lstm_out)  # [batch, max_len, 2]
        
        mu = predictions[:, :, 0]
        raw_sigma = predictions[:, :, 1]
        sigma = F.softplus(raw_sigma) + 0.08
        
        # Create mask
        mask = torch.arange(max_len, device=DEVICE).expand(batch_size, max_len) < lengths.unsqueeze(1)
        
        if return_attention:
            return mu, sigma, mask, attention_weights
        
        return mu, sigma, mask

# Instantiate model
model = GAT_LSTM_EPAModel(
    node_in_dim=len(node_features_available),
    edge_in_dim=len(edge_features_available),
    gat_hidden_dim=HIDDEN_DIM,
    lstm_hidden_dim=LSTM_HIDDEN_DIM,
    n_heads=N_ATTENTION_HEADS,
    lstm_layers=LSTM_LAYERS
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model created:")
print(f"  Node input dim: {len(node_features_available)}")
print(f"  Edge input dim: {len(edge_features_available)}")
print(f"  GAT hidden dim: {HIDDEN_DIM}")
print(f"  LSTM hidden dim: {LSTM_HIDDEN_DIM}")
print(f"  LSTM layers: {LSTM_LAYERS}")
print(f"  Attention heads: {N_ATTENTION_HEADS}")
print(f"  Total parameters: {total_params:,}")

# ============================================================================
# 6. Training
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: TRAINING")
print("-" * 80)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def gaussian_nll_loss(mu, sigma, target, mask):
    """
    Compute Gaussian NLL loss only for non-masked (real) frames.
    
    Args:
        mu: [batch, seq_len]
        sigma: [batch, seq_len]
        target: [batch, 1] - same target for all frames in sequence
        mask: [batch, seq_len] - True for real frames
    
    Returns:
        loss: scalar
    """
    # Expand target to sequence length
    target_expanded = target.expand_as(mu)  # [batch, seq_len]
    
    # NLL = 0.5 * log(2π) + log(σ) + (y - μ)² / (2σ²)
    nll = 0.5 * np.log(2 * np.pi) + torch.log(sigma) + (target_expanded - mu) ** 2 / (2 * sigma ** 2)
    
    # Apply mask - only compute loss on real frames
    masked_nll = nll[mask]
    
    return masked_nll.mean()

print(f"Training for max {MAX_TRAIN_TIME_MINUTES} minutes...")
print(f"Batch size: {BATCH_SIZE} plays")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Loss function: Gaussian NLL (masked)")
print()

training_log = []
start_time = time.time()
best_val_loss = float('inf')
epoch = 0

try:
    while True:
        epoch += 1
        model.train()
        train_losses = []
        train_mu_maes = []
        train_sigma_avgs = []
        
        for batch_idx, batch in enumerate(train_loader):
            sequences = batch['sequences']
            lengths = batch['lengths']
            targets = batch['targets'].to(DEVICE)
            
            # Forward pass
            mu, sigma, mask = model(sequences, lengths)
            
            # Loss
            loss = gaussian_nll_loss(mu, sigma, targets, mask)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics (only on real frames)
            with torch.no_grad():
                masked_mu = mu[mask]
                masked_sigma = sigma[mask]
                masked_targets = targets.expand_as(mu)[mask]
                
                mu_mae = torch.abs(masked_mu - masked_targets).mean().item()
                sigma_avg = masked_sigma.mean().item()
            
            train_losses.append(loss.item())
            train_mu_maes.append(mu_mae)
            train_sigma_avgs.append(sigma_avg)
            
            # Print progress
            if (batch_idx + 1) % PRINT_EVERY == 0:
                elapsed = (time.time() - start_time) / 60
                print(f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | μ MAE: {mu_mae:.3f} | σ: {sigma_avg:.3f}")
        
        # Validation
        model.eval()
        val_losses = []
        val_mu_maes = []
        val_sigma_avgs = []
        
        with torch.no_grad():
            for batch in val_loader:
                sequences = batch['sequences']
                lengths = batch['lengths']
                targets = batch['targets'].to(DEVICE)
                
                mu, sigma, mask = model(sequences, lengths)
                loss = gaussian_nll_loss(mu, sigma, targets, mask)
                
                masked_mu = mu[mask]
                masked_sigma = sigma[mask]
                masked_targets = targets.expand_as(mu)[mask]
                
                mu_mae = torch.abs(masked_mu - masked_targets).mean().item()
                sigma_avg = masked_sigma.mean().item()
                
                val_losses.append(loss.item())
                val_mu_maes.append(mu_mae)
                val_sigma_avgs.append(sigma_avg)
        
        # Epoch summary
        train_nll = np.mean(train_losses)
        val_nll = np.mean(val_losses)
        elapsed = (time.time() - start_time) / 60
        
        print()
        print("=" * 80)
        print(f"Epoch {epoch} Summary:")
        print(f"  Train NLL: {train_nll:.4f}")
        print(f"  Val NLL:   {val_nll:.4f}")
        print(f"  Val μ MAE: {np.mean(val_mu_maes):.4f} EPA")
        print(f"  Val σ avg: {np.mean(val_sigma_avgs):.4f} EPA")
        print(f"  Elapsed:   {elapsed:.1f}m / {MAX_TRAIN_TIME_MINUTES}m")
        print("=" * 80)
        print()
        
        # Save training log
        training_log.append({
            'epoch': epoch,
            'train_nll': train_nll,
            'val_nll': val_nll,
            'val_mu_mae': np.mean(val_mu_maes),
            'val_sigma_avg': np.mean(val_sigma_avgs),
            'elapsed_minutes': elapsed
        })
        
        # Save best model
        if val_nll < best_val_loss:
            best_val_loss = val_nll
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_nll,
                'node_features': node_features_available,
                'edge_features': edge_features_available,
                'node_means': node_feature_means.to_dict(),
                'node_stds': node_feature_stds.to_dict(),
                'edge_means': edge_feature_means.to_dict(),
                'edge_stds': edge_feature_stds.to_dict(),
                'edge_type_to_idx': edge_type_to_idx,
            }, os.path.join(OUTPUT_DIR, 'model.pth'))
            print(f"  ✓ Saved best model (Val NLL: {val_nll:.4f})")
        
        # Check time limit
        if elapsed >= MAX_TRAIN_TIME_MINUTES:
            print(f"\n⏰ Reached time limit ({MAX_TRAIN_TIME_MINUTES} minutes)")
            break

except KeyboardInterrupt:
    print("\n⚠ Training interrupted by user")

# ============================================================================
# 7. Save Results
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: SAVING RESULTS")
print("-" * 80)

# Save training log
log_df = pd.DataFrame(training_log)
log_path = os.path.join(OUTPUT_DIR, 'training_log.csv')
log_df.to_csv(log_path, index=False)
print(f"✓ Saved training log: {log_path}")

# Generate predictions on validation set
print("\nGenerating validation predictions...")
model.eval()
predictions = []

with torch.no_grad():
    for batch in val_loader:
        sequences = batch['sequences']
        lengths = batch['lengths']
        targets = batch['targets'].to(DEVICE)
        game_ids = batch['game_ids']
        play_ids = batch['play_ids']
        
        mu, sigma, mask = model(sequences, lengths)
        
        # Extract predictions for each play
        for b in range(len(sequences)):
            seq_len = lengths[b].item()
            for t in range(seq_len):
                predictions.append({
                    'game_id': game_ids[b],
                    'play_id': play_ids[b],
                    'frame_idx': t,
                    'mu': mu[b, t].item(),
                    'sigma': sigma[b, t].item(),
                    'actual_epa': targets[b].item(),
                })

pred_df = pd.DataFrame(predictions)
pred_path = os.path.join(OUTPUT_DIR, 'predictions.csv')
pred_df.to_csv(pred_path, index=False)
print(f"✓ Saved predictions: {pred_path}")
print(f"  {len(pred_df)} frame-level predictions")

# ============================================================================
# Complete
# ============================================================================

end_time = time.time()
total_time = (end_time - start_time) / 60

print("\n" + "=" * 80)
print("TRAINING COMPLETE (LSTM)")
print("=" * 80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total time: {total_time:.1f} minutes")
print(f"Best validation NLL: {best_val_loss:.4f}")
print(f"\nOutputs saved to: {OUTPUT_DIR}")
print(f"  - model.pth (best checkpoint)")
print(f"  - training_log.csv ({len(training_log)} epochs)")
print(f"  - predictions.csv ({len(pred_df)} predictions)")
print("=" * 80)