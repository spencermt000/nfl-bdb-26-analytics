"""
train_epa_distribution.py - Train EPA Distribution Prediction Model
====================================================================
Trains a graph attention network to predict the DISTRIBUTION of expected_points_added.

KEY IMPROVEMENTS OVER train_yac_epa.py:
  ✨ Predicts DISTRIBUTION (μ, σ) instead of single point estimate
  ✨ Uses structured edges from df_c v2 (edge_type, attention_prior)
  ✨ Enables counterfactual analysis (actual vs. optimal defender actions)
  ✨ Quantifies uncertainty in EPA outcomes

MODEL OUTPUT:
  - μ (mu): Mean EPA (expected value)
  - σ (sigma): Standard deviation (uncertainty)
  
  Distribution: N(μ, σ²)

LOSS FUNCTION:
  Negative Log-Likelihood (NLL) of Gaussian distribution
  
  NLL = 0.5 * log(2π) + log(σ) + ((EPA_actual - μ)² / (2σ²))
  
  This encourages:
    - μ close to actual EPA (accurate mean)
    - σ calibrated to prediction uncertainty (low for certain plays, high for uncertain)

INPUTS:
  - outputs/dataframe_a/v1.parquet (node features)
  - outputs/dataframe_b/v1.parquet (play context + EPA targets)
  - outputs/dataframe_c/v2_pilot_3games.parquet (structured edges with edge_type, attention_prior)

OUTPUTS:
  - model_outputs/epa_distribution/model.pth (trained model)
  - model_outputs/epa_distribution/training_log.csv (loss history)
  - model_outputs/epa_distribution/predictions.csv (validation predictions)

FUTURE USE (Defender Impact Analysis):
  1. Forward pass with actual graph → P_actual(EPA) ~ N(μ_actual, σ_actual)
  2. Forward pass with defender d removed → P_optimal(EPA) ~ N(μ_optimal, σ_optimal)
  3. Defender impact = μ_actual - μ_optimal (EPA cost of defender's actions)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal
import os
import time
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

print("=" * 80)
print("TRAIN EPA DISTRIBUTION PREDICTION MODEL")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# *** PILOT MODE ***
PILOT_MODE = True  # Set to False for full dataset
PILOT_DF_C = 'outputs/dataframe_c/v2_pilot_3games.parquet'
REGULAR_DF_C = 'outputs/dataframe_c/v2.parquet'

# File paths
INPUT_DF_A = 'outputs/dataframe_a/v1.parquet'
INPUT_DF_B = 'outputs/dataframe_b/v1.parquet'
INPUT_DF_C = PILOT_DF_C if PILOT_MODE else REGULAR_DF_C

# Output directory
OUTPUT_DIR = 'model_outputs/epa_distribution'
os.makedirs(OUTPUT_DIR, exist_ok=True)

if PILOT_MODE:
    print("\n" + "⚠" * 40)
    print("PILOT MODE ENABLED".center(80))
    print(f"Using pilot dataset: {PILOT_DF_C}".center(80))
    print("⚠" * 40 + "\n")

# Model hyperparameters
N_ATTENTION_HEADS = 4
HIDDEN_DIM = 128
LEARNING_RATE = 0.001
BATCH_SIZE = 1  # Keep 1 for variable graph sizes

# Training configuration
MAX_TRAIN_TIME_MINUTES = 10
PRINT_EVERY = 500  # Print progress every N batches

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
print(f"  Edge types: {df_c['edge_type'].unique()}")

# Filter to pilot games if needed
if PILOT_MODE:
    pilot_games = df_c['game_id'].unique()
    df_a = df_a[df_a['game_id'].isin(pilot_games)].copy()
    df_b = df_b[df_b['game_id'].isin(pilot_games)].copy()
    print(f"\nFiltered to {len(pilot_games)} pilot games.")

# ============================================================================
# 2. Filter to Relevant Players
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: FILTERING DATA")
print("-" * 80)

# Filter df_a to relevant players (QB, route runners, defenders with coverage)
df_a_filtered = df_a[
    (df_a['isPasser'] == 1) | 
    (df_a['isRouteRunner'] == 1) |
    (df_a['coverage_responsibility'].notna())
].copy()

# Filter df_c to relevant edges (already filtered by v2 script, but double-check)
relevant_players = set(df_a_filtered['nfl_id'].unique())
df_c_filtered = df_c[
    df_c['playerA_id'].isin(relevant_players) &
    df_c['playerB_id'].isin(relevant_players)
].copy()

print(f"Nodes: {len(df_a_filtered):,}")
print(f"Edges: {len(df_c_filtered):,}")
print(f"Edge type distribution:")
for edge_type, count in df_c_filtered['edge_type'].value_counts().items():
    print(f"  {edge_type}: {count:,} ({100*count/len(df_c_filtered):.1f}%)")

# ============================================================================
# 3. Feature Engineering
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: FEATURE ENGINEERING")
print("-" * 80)

# Node Features
NODE_FEATURES = [
    # Spatial & kinematic
    'x', 'y', 's', 'a', 'dir', 'o', 'v_x', 'v_y', 'a_x', 'a_y', 'o_x', 'o_y',
    # Ball proximity
    'e_dist_ball_land',
    # Role indicators
    'isPasser', 'isTargeted', 'isRouteRunner',
]
node_features_available = [f for f in NODE_FEATURES if f in df_a_filtered.columns]

# Edge Features (from df_c v2)
EDGE_FEATURES = [
    # Spatial
    'e_dist', 'x_dist', 'y_dist', 
    # Angular
    'relative_angle_o', 'relative_angle_dir',
    # Ball proximity (both players)
    'playerA_dist_to_landing', 'playerB_dist_to_landing',
    'playerA_dist_to_ball_current', 'playerB_dist_to_ball_current',
    'playerA_angle_to_ball_current', 'playerB_angle_to_ball_current',
    'playerA_angle_to_ball_landing', 'playerB_angle_to_ball_landing',
    'playerA_ball_convergence', 'playerB_ball_convergence',
    # Relative motion
    'relative_v_x', 'relative_v_y', 'relative_speed', 
    # Context
    'same_team', 'ball_progress', 'frames_to_landing',
    # NEW v2: Domain-informed prior
    'attention_prior',
    'coverage_scheme_encoded'
]
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

# Edge type encoding (for future use in attention mechanism)
edge_type_to_idx = {et: i for i, et in enumerate(df_c_filtered['edge_type'].unique())}
print(f"\nEdge type encoding: {edge_type_to_idx}")

# ============================================================================
# 4. PyTorch Dataset
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: DATASET CREATION")
print("-" * 80)

class EPADistributionDataset(Dataset):
    """
    Dataset for EPA distribution prediction.
    Returns graph structure + EPA target for each frame.
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

        # Get unique frames
        self.frames = df_a.groupby(['game_id', 'play_id', 'frame_id']).size().reset_index()[
            ['game_id', 'play_id', 'frame_id']
        ]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        game_id = self.frames.iloc[idx]['game_id']
        play_id = self.frames.iloc[idx]['play_id']
        frame_id = self.frames.iloc[idx]['frame_id']

        # 1. Get nodes for this frame
        nodes = self.df_a[
            (self.df_a['game_id'] == game_id) &
            (self.df_a['play_id'] == play_id) &
            (self.df_a['frame_id'] == frame_id)
        ].copy()

        # Create node ID mapping
        node_ids = nodes['nfl_id'].values
        valid_nfl_ids = set(node_ids)
        node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        # 2. Get edges for this frame
        edges_raw = self.df_c[
            (self.df_c['game_id'] == game_id) &
            (self.df_c['play_id'] == play_id) &
            (self.df_c['frame_id'] == frame_id)
        ]

        # Filter edges to only include nodes in our node list
        edges = edges_raw[
            edges_raw['playerA_id'].isin(valid_nfl_ids) &
            edges_raw['playerB_id'].isin(valid_nfl_ids)
        ].copy()

        # 3. Extract and normalize features
        node_feat = (nodes[self.node_features].values - self.node_means.values) / (self.node_stds.values + 1e-8)
        node_feat = torch.FloatTensor(node_feat)

        if len(edges) > 0:
            edge_feat = (edges[self.edge_features].values - self.edge_means.values) / (self.edge_stds.values + 1e-8)
            edge_feat = torch.FloatTensor(edge_feat)
        else:
            edge_feat = torch.zeros((0, len(self.edge_features)))

        # 4. Build edge index
        edge_index = []
        edge_types = []
        for _, row in edges.iterrows():
            src_idx = node_id_to_idx[row['playerA_id']]
            dst_idx = node_id_to_idx[row['playerB_id']]
            edge_index.append([src_idx, dst_idx])
            edge_types.append(self.edge_type_to_idx.get(row['edge_type'], 0))

        if len(edge_index) > 0:
            edge_index = torch.LongTensor(edge_index).t()  # Shape: [2, num_edges]
            edge_types = torch.LongTensor(edge_types)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_types = torch.zeros(0, dtype=torch.long)

        # 5. Get play-level target (expected_points_added)
        play_row = self.df_b[
            (self.df_b['game_id'] == game_id) &
            (self.df_b['play_id'] == play_id)
        ]

        epa_target = 0.0
        if not play_row.empty and 'expected_points_added' in play_row.columns:
            val = play_row.iloc[0]['expected_points_added']
            if pd.notna(val):
                epa_target = float(val)

        epa_target = torch.FloatTensor([epa_target])

        return {
            'node_features': node_feat,
            'edge_index': edge_index,
            'edge_features': edge_feat,
            'edge_types': edge_types,
            'epa_target': epa_target,
            'game_id': game_id,
            'play_id': play_id,
            'frame_id': frame_id,
        }

dataset = EPADistributionDataset(
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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Dataset created:")
print(f"  Total frames: {len(dataset):,}")
print(f"  Train: {train_size:,}")
print(f"  Val: {val_size:,}")

# ============================================================================
# 5. Model Architecture
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: MODEL ARCHITECTURE")
print("-" * 80)

class MultiHeadGraphAttention(nn.Module):
    """
    Multi-head graph attention with edge features.
    """
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        # Encoders
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, hidden_dim)
        
        # Attention
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        self.edge_attn = nn.Linear(hidden_dim, n_heads)
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_feat, edge_index, edge_feat):
        """
        Args:
            node_feat: [N, node_in_dim]
            edge_index: [2, E] (src, dst indices)
            edge_feat: [E, edge_in_dim]
        Returns:
            out: [N, hidden_dim] (updated node features)
        """
        N = node_feat.shape[0]
        
        # Encode
        h = F.relu(self.node_encoder(node_feat))  # [N, hidden_dim]
        e = F.relu(self.edge_encoder(edge_feat))  # [E, hidden_dim]
        
        # Multi-head projections
        Q = self.W_Q(h).view(N, self.n_heads, self.head_dim)  # [N, heads, head_dim]
        K = self.W_K(h).view(N, self.n_heads, self.head_dim)
        V = self.W_V(h).view(N, self.n_heads, self.head_dim)
        
        if edge_index.shape[1] == 0:
            # No edges - return input
            return h
        
        src, dst = edge_index  # [E], [E]
        
        # Attention scores: Q[src] · K[dst] + edge bias
        scores = (Q[src] * K[dst]).sum(dim=-1) / (self.head_dim ** 0.5)  # [E, heads]
        scores = scores + self.edge_attn(e)  # [E, heads]
        
        # Softmax per destination node
        attn_weights = torch.zeros_like(scores)
        for i in range(N):
            mask = (dst == i)
            if mask.any():
                attn_weights[mask] = F.softmax(scores[mask], dim=0)
        
        # Store attention weights for later extraction
        self.last_attention_weights = attn_weights
        
        # Aggregate: sum weighted values
        weighted_V = V[src] * attn_weights.unsqueeze(-1)  # [E, heads, head_dim]
        out = torch.zeros(N, self.n_heads, self.head_dim, device=node_feat.device)
        out.index_add_(0, dst, weighted_V)
        
        # Project and residual
        out = self.out_proj(out.view(N, -1))  # [N, hidden_dim]
        return out + h  # Residual connection

class EPADistributionModel(nn.Module):
    """
    Predicts distribution of EPA outcomes: N(μ, σ²)
    
    Architecture:
      - Graph Attention Network (GAT) for spatial relationships
      - Graph pooling (mean of node embeddings)
      - MLP head → (μ, σ)
    """
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, n_heads):
        super().__init__()
        
        # Graph attention layer
        self.gat = MultiHeadGraphAttention(node_in_dim, edge_in_dim, hidden_dim, n_heads)
        
        # Distribution prediction head
        self.distribution_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Output: [μ, raw_σ]
        )
        
    def forward(self, node_feat, edge_index, edge_feat, return_attention=False):
        """
        Args:
            node_feat: [N, node_in_dim]
            edge_index: [2, E]
            edge_feat: [E, edge_in_dim]
            return_attention: If True, return attention weights along with mu, sigma
        
        Returns:
            mu: [1] (mean EPA)
            sigma: [1] (std deviation, constrained > 0)
            attention: [E, n_heads] (optional, only if return_attention=True)
        """
        # Graph attention
        h = self.gat(node_feat, edge_index, edge_feat)  # [N, hidden_dim]
        
        # Graph pooling (mean aggregation)
        graph_emb = h.mean(dim=0, keepdim=True)  # [1, hidden_dim]
        
        # Predict distribution parameters
        dist_params = self.distribution_head(graph_emb)  # [1, 2]
        
        mu = dist_params[:, 0]  # [1]
        raw_sigma = dist_params[:, 1]  # [1]
        
        # Ensure σ > 0 using softplus: σ = log(1 + exp(raw_σ))
        # Add small constant for numerical stability
        sigma = F.softplus(raw_sigma) + 0.08  # [1]
        
        if return_attention:
            # Get attention weights from GAT layer
            attention = self.gat.last_attention_weights if hasattr(self.gat, 'last_attention_weights') else None
            return mu, sigma, attention
        
        return mu, sigma

# Instantiate model
model = EPADistributionModel(
    node_in_dim=len(node_features_available),
    edge_in_dim=len(edge_features_available),
    hidden_dim=HIDDEN_DIM,
    n_heads=N_ATTENTION_HEADS
).to(DEVICE)

print(f"Model created:")
print(f"  Node input dim: {len(node_features_available)}")
print(f"  Edge input dim: {len(edge_features_available)}")
print(f"  Hidden dim: {HIDDEN_DIM}")
print(f"  Attention heads: {N_ATTENTION_HEADS}")
print(f"  Output: μ (mean), σ (std deviation)")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# 6. Loss Function
# ============================================================================

def gaussian_nll_loss(mu, sigma, target):
    """
    Negative Log-Likelihood loss for Gaussian distribution.
    
    NLL = 0.5 * log(2π) + log(σ) + (target - μ)² / (2σ²)
    
    Args:
        mu: [batch_size] predicted mean
        sigma: [batch_size] predicted std deviation
        target: [batch_size] actual EPA values
    
    Returns:
        loss: scalar (mean NLL over batch)
    """
    # Create Gaussian distribution
    dist = Normal(mu, sigma)
    
    # Negative log probability
    nll = -dist.log_prob(target)
    
    return nll.mean()

# ============================================================================
# 7. Training Loop
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: TRAINING")
print("-" * 80)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training history
history = {
    'epoch': [],
    'train_loss': [],
    'val_loss': [],
    'val_mu_mae': [],  # Mean absolute error of μ predictions
    'val_sigma_mean': [],  # Average predicted uncertainty
}

best_val_loss = float('inf')
start_time = time.time()
epoch = 0

print(f"Training for max {MAX_TRAIN_TIME_MINUTES} minutes...")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Loss function: Gaussian NLL\n")

while True:
    elapsed = (time.time() - start_time) / 60.0
    if elapsed >= MAX_TRAIN_TIME_MINUTES:
        print(f"\n⏰ Time limit ({MAX_TRAIN_TIME_MINUTES}m) reached.")
        break
    
    epoch += 1
    
    # ========================================
    # TRAIN
    # ========================================
    model.train()
    train_loss = 0.0
    train_count = 0
    
    for batch_idx, batch in enumerate(train_loader):
        node_feat = batch['node_features'][0].to(DEVICE)
        edge_idx = batch['edge_index'][0].to(DEVICE)
        edge_feat = batch['edge_features'][0].to(DEVICE)
        target = batch['epa_target'][0].to(DEVICE)
        
        # Skip if no edges
        if edge_idx.numel() == 0:
            continue
        
        # Forward pass
        mu, sigma = model(node_feat, edge_idx, edge_feat)
        
        # Compute loss
        loss = gaussian_nll_loss(mu, sigma, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_count += 1
        
        # Print progress
        if (batch_idx + 1) % PRINT_EVERY == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f} | μ: {mu.item():.3f} | σ: {sigma.item():.3f}")
    
    avg_train_loss = train_loss / train_count if train_count > 0 else 0
    
    # ========================================
    # VALIDATE
    # ========================================
    model.eval()
    val_loss = 0.0
    val_count = 0
    val_mu_errors = []
    val_sigmas = []
    
    with torch.no_grad():
        for batch in val_loader:
            node_feat = batch['node_features'][0].to(DEVICE)
            edge_idx = batch['edge_index'][0].to(DEVICE)
            edge_feat = batch['edge_features'][0].to(DEVICE)
            target = batch['epa_target'][0].to(DEVICE)
            
            if edge_idx.numel() == 0:
                continue
            
            # Forward pass
            mu, sigma = model(node_feat, edge_idx, edge_feat)
            
            # Compute loss
            loss = gaussian_nll_loss(mu, sigma, target)
            
            val_loss += loss.item()
            val_count += 1
            
            # Track metrics
            val_mu_errors.append(torch.abs(mu - target).item())
            val_sigmas.append(sigma.item())
    
    avg_val_loss = val_loss / val_count if val_count > 0 else 0
    avg_mu_mae = np.mean(val_mu_errors) if val_mu_errors else 0
    avg_sigma = np.mean(val_sigmas) if val_sigmas else 0
    
    # Print epoch summary
    print(f"\n{'='*80}")
    print(f"Epoch {epoch} Summary:")
    print(f"  Train NLL: {avg_train_loss:.4f}")
    print(f"  Val NLL:   {avg_val_loss:.4f}")
    print(f"  Val μ MAE: {avg_mu_mae:.4f} EPA")
    print(f"  Val σ avg: {avg_sigma:.4f} EPA")
    print(f"  Elapsed:   {elapsed:.1f}m / {MAX_TRAIN_TIME_MINUTES}m")
    print(f"{'='*80}\n")
    
    # Save history
    history['epoch'].append(epoch)
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['val_mu_mae'].append(avg_mu_mae)
    history['val_sigma_mean'].append(avg_sigma)
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'node_features': node_features_available,
            'edge_features': edge_features_available,
            'edge_type_to_idx': edge_type_to_idx,
            'node_means': node_feature_means.to_dict(),
            'node_stds': node_feature_stds.to_dict(),
            'edge_means': edge_feature_means.to_dict(),
            'edge_stds': edge_feature_stds.to_dict(),
        }, os.path.join(OUTPUT_DIR, 'model.pth'))
        print(f"  ✓ Saved best model (Val NLL: {avg_val_loss:.4f})")

# ============================================================================
# 8. Save Training History
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: SAVING OUTPUTS")
print("-" * 80)

# Save training log
history_df = pd.DataFrame(history)
history_file = os.path.join(OUTPUT_DIR, 'training_log.csv')
history_df.to_csv(history_file, index=False)
print(f"✓ Saved training log: {history_file}")

# Generate validation predictions
print("\nGenerating validation predictions...")
model.eval()
predictions = []

with torch.no_grad():
    for batch in val_loader:
        node_feat = batch['node_features'][0].to(DEVICE)
        edge_idx = batch['edge_index'][0].to(DEVICE)
        edge_feat = batch['edge_features'][0].to(DEVICE)
        target = batch['epa_target'][0].to(DEVICE)
        game_id = batch['game_id'][0]
        play_id = batch['play_id'][0]
        frame_id = batch['frame_id'][0]
        
        if edge_idx.numel() == 0:
            continue
        
        mu, sigma = model(node_feat, edge_idx, edge_feat)
        
        predictions.append({
            'game_id': game_id,
            'play_id': play_id,
            'frame_id': frame_id,
            'epa_actual': target.item(),
            'epa_mu': mu.item(),
            'epa_sigma': sigma.item(),
            'error': mu.item() - target.item(),
        })

pred_df = pd.DataFrame(predictions)
pred_file = os.path.join(OUTPUT_DIR, 'predictions.csv')
pred_df.to_csv(pred_file, index=False)
print(f"✓ Saved validation predictions: {pred_file}")
print(f"  Total predictions: {len(pred_df):,}")

# ============================================================================
# Complete
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total epochs: {epoch}")
print(f"Best validation NLL: {best_val_loss:.4f}")
print(f"\nOutputs saved to: {OUTPUT_DIR}")
print(f"  - model.pth: Best model checkpoint")
print(f"  - training_log.csv: Loss history")
print(f"  - predictions.csv: Validation predictions")
print("\n" + "=" * 80)