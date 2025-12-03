"""
Adjacency Matrix Training Script
Predict edge weights/existence between players in NFL tracking data
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    f1_score,
    mean_squared_error,
    mean_absolute_error
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Training configuration"""
    
    # Data paths
    DATA_DIR = Path("outputs")
    NODE_DATA = DATA_DIR / "dataframe_a" / "v1.parquet"      # Node-level features
    PLAY_DATA = DATA_DIR / "dataframe_b" / "v1.parquet"      # Play-level features
    EDGE_DATA = DATA_DIR / "dataframe_c" / "v1.parquet"      # Edge-level features (pairwise)
    TEMPORAL_DATA = DATA_DIR / "dataframe_d" / "v1.parquet"  # Temporal features
    
    # Output paths
    OUTPUT_DIR = Path("model_outputs")
    MODEL_SAVE_PATH = OUTPUT_DIR / "adjacency_model.pt"
    SCALER_SAVE_PATH = OUTPUT_DIR / "feature_scaler.pkl"
    LABEL_ENCODERS_PATH = OUTPUT_DIR / "label_encoders.pkl"
    METRICS_SAVE_PATH = OUTPUT_DIR / "training_metrics.json"
    CONFIG_SAVE_PATH = OUTPUT_DIR / "config.json"
    
    # Model hyperparameters
    HIDDEN_DIM = 256
    EMBEDDING_DIM = 128
    NUM_LAYERS = 3
    DROPOUT = 0.3
    
    # Training hyperparameters
    BATCH_SIZE = 1024
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING_PATIENCE = 10
    GRAD_CLIP = 1.0
    
    # Override settings for pilot mode
    PILOT_BATCH_SIZE = 256
    PILOT_NUM_EPOCHS = 10
    PILOT_EARLY_STOPPING_PATIENCE = 5
    
    # Data split
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1  # Of training set
    RANDOM_SEED = 42
    
    # Sampling (for memory management with 55M rows)
    SAMPLE_FRAC = 0.1  # Use 10% of data for initial training
    USE_SAMPLING = True
    
    # Pilot mode - train on single game for quick testing
    PILOT_MODE = True  # Set to False for full training
    PILOT_GAME_ID = None  # Set to specific game_id or None for random
    
    # Target variable
    # Options: 'e_dist' (regression), 'close_proximity' (classification), etc.
    TARGET_TYPE = 'regression'  # or 'classification'
    TARGET_COL = 'e_dist'  # Euclidean distance between players
    
    # For classification, define threshold
    PROXIMITY_THRESHOLD = 5.0  # yards
    
    def __init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    def save(self):
        """Save configuration to JSON"""
        # Get all config attributes (not methods)
        config_dict = {}
        for k, v in self.__class__.__dict__.items():
            # Skip private attributes and methods
            if k.startswith('_') or callable(v):
                continue
            # Convert Path to string
            if isinstance(v, Path):
                config_dict[k] = str(v)
            # Only include JSON-serializable types
            elif isinstance(v, (int, float, str, bool, list, dict, type(None))):
                config_dict[k] = v
        
        with open(self.CONFIG_SAVE_PATH, 'w') as f:
            json.dump(config_dict, f, indent=2)

# ============================================================================
# Data Loading and Processing
# ============================================================================

class DataProcessor:
    """Load and preprocess all dataframes"""
    
    def __init__(self, config):
        self.config = config
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load all dataframes"""
        print("="*70)
        print("LOADING DATA")
        print("="*70)
        
        print("\n1. Loading edge data (pairwise features)...")
        self.df_edges = pd.read_parquet(self.config.EDGE_DATA)
        print(f"   Shape: {self.df_edges.shape}")
        
        # Pilot mode - use single game
        if self.config.PILOT_MODE:
            print("\n   🚀 PILOT MODE ENABLED 🚀")
            if self.config.PILOT_GAME_ID is not None:
                # Use specified game
                pilot_game_id = self.config.PILOT_GAME_ID
                print(f"   Using specified game: {pilot_game_id}")
            else:
                # Pick random game
                pilot_game_id = self.df_edges['game_id'].sample(1, random_state=self.config.RANDOM_SEED).iloc[0]
                print(f"   Randomly selected game: {pilot_game_id}")
            
            self.df_edges = self.df_edges[self.df_edges['game_id'] == pilot_game_id]
            print(f"   Pilot shape: {self.df_edges.shape}")
            
            # Count unique plays in pilot
            n_plays = self.df_edges['play_id'].nunique()
            n_frames = self.df_edges['frame_id'].nunique() 
            print(f"   Plays in game: {n_plays}")
            print(f"   Frames in game: {n_frames}")
        
        # Regular sampling (only if not in pilot mode)
        elif self.config.USE_SAMPLING:
            print(f"   Sampling {self.config.SAMPLE_FRAC*100}% of data...")
            self.df_edges = self.df_edges.sample(
                frac=self.config.SAMPLE_FRAC, 
                random_state=self.config.RANDOM_SEED
            )
            print(f"   Sampled shape: {self.df_edges.shape}")
        
        print("\n2. Loading play-level data...")
        self.df_plays = pd.read_parquet(self.config.PLAY_DATA)
        print(f"   Shape: {self.df_plays.shape}")
        
        print("\n3. Loading temporal data...")
        self.df_temporal = pd.read_parquet(self.config.TEMPORAL_DATA)
        print(f"   Shape: {self.df_temporal.shape}")
        
        # Node data is already embedded in edge data (playerA/playerB features)
        # so we don't need to load it separately
        
        return self
    
    def merge_features(self):
        """Merge play and temporal features into edge data"""
        print("\n" + "="*70)
        print("MERGING FEATURES")
        print("="*70)
        
        df = self.df_edges.copy()
        initial_shape = df.shape
        
        # Merge play-level features
        print("\n1. Merging play-level features...")
        
        # Rename old_game_id to game_id for merging
        if 'old_game_id' in self.df_plays.columns and 'game_id' not in self.df_plays.columns:
            self.df_plays = self.df_plays.rename(columns={'old_game_id': 'game_id'})
            print("   Renamed 'old_game_id' to 'game_id' in play data")
        
        # Ensure game_id has matching data types
        df['game_id'] = df['game_id'].astype(str)
        self.df_plays['game_id'] = self.df_plays['game_id'].astype(str)
        print("   Converted game_id to string for consistent merging")
        
        play_features = [
            'game_id', 'play_id', 'quarter', 'down', 'yards_to_go', 
            'pos_team_wp', 'defenders_in_the_box',
            'quarter_seconds_remaining', 'game_seconds_remaining',
            'air_yards', 'shotgun', 'no_huddle'
        ]
        play_cols = [c for c in play_features if c in self.df_plays.columns]
        df = df.merge(
            self.df_plays[play_cols],
            on=['game_id', 'play_id'],  # merge on both game_id AND play_id
            how='left'
        )
        print(f"   Shape after merge: {df.shape}")
        
        # Merge temporal features
        print("\n2. Merging temporal features...")
        
        # Ensure merge keys have matching data types
        df['game_id'] = df['game_id'].astype(str)
        df['play_id'] = df['play_id'].astype(float)
        df['frame_id'] = df['frame_id'].astype(float)
        
        self.df_temporal['game_id'] = self.df_temporal['game_id'].astype(str)
        self.df_temporal['play_id'] = self.df_temporal['play_id'].astype(float)
        self.df_temporal['frame_id'] = self.df_temporal['frame_id'].astype(float)
        print("   Converted merge keys to matching types")
        
        temporal_features = [
            'game_id', 'play_id', 'frame_id',
            'n_players_tot', 'n_players_off', 'n_players_def'
        ]
        temporal_cols = [c for c in temporal_features if c in self.df_temporal.columns]
        df = df.merge(
            self.df_temporal[temporal_cols],
            on=['game_id', 'play_id', 'frame_id'],
            how='left'
        )
        print(f"   Shape after merge: {df.shape}")
        
        self.df_merged = df
        print(f"\nFinal merged shape: {df.shape}")
        print(f"Added {df.shape[1] - initial_shape[1]} columns")
        
        return self
    
    def prepare_features(self):
        """Prepare features and target variable"""
        print("\n" + "="*70)
        print("PREPARING FEATURES")
        print("="*70)
        
        df = self.df_merged.copy()
        
        # Define target variable
        if self.config.TARGET_TYPE == 'classification':
            # Binary classification: close proximity or not
            self.y = (df[self.config.TARGET_COL] < self.config.PROXIMITY_THRESHOLD).astype(int)
            print(f"\nTarget: Binary classification (proximity < {self.config.PROXIMITY_THRESHOLD} yards)")
            print(f"Positive class (close): {self.y.sum():,} ({100*self.y.mean():.2f}%)")
            print(f"Negative class (far): {(~self.y).sum():,} ({100*(1-self.y.mean()):.2f}%)")
        else:
            # Regression: predict distance
            self.y = df[self.config.TARGET_COL].values
            print(f"\nTarget: Regression ({self.config.TARGET_COL})")
            print(f"Mean: {self.y.mean():.3f}, Std: {self.y.std():.3f}")
            print(f"Min: {self.y.min():.3f}, Max: {self.y.max():.3f}")
        
        # Define feature columns to use
        # Exclude identifiers and target
        exclude_cols = [
            'game_id', 'play_id', 'frame_id', 
            'playerA_id', 'playerB_id',
            'player_rel_index',
            self.config.TARGET_COL
        ]
        
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        print(f"\nProcessing {len(feature_cols)} feature columns...")
        
        # Separate numeric and categorical
        numeric_cols = []
        categorical_cols = []
        
        for col in feature_cols:
            if df[col].dtype in ['float64', 'int64']:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        
        print(f"  Numeric features: {len(numeric_cols)}")
        print(f"  Categorical features: {len(categorical_cols)}")
        
        # Process numeric features
        X_numeric = df[numeric_cols].fillna(0).values
        
        # Process categorical features
        X_categorical = pd.DataFrame()
        for col in categorical_cols:
            le = LabelEncoder()
            X_categorical[col] = le.fit_transform(df[col].fillna('unknown').astype(str))
            self.label_encoders[col] = le
        
        # Combine features
        if len(categorical_cols) > 0:
            self.X = np.hstack([X_numeric, X_categorical.values])
        else:
            self.X = X_numeric
        
        self.feature_names = numeric_cols + categorical_cols
        
        print(f"\nFinal feature matrix shape: {self.X.shape}")
        print(f"Target shape: {self.y.shape}")
        
        # Check for NaN/Inf
        if np.any(np.isnan(self.X)) or np.any(np.isinf(self.X)):
            print("\nWARNING: NaN or Inf values detected, cleaning...")
            self.X = np.nan_to_num(self.X, nan=0.0, posinf=0.0, neginf=0.0)
        
        return self
    
    def split_data(self):
        """Split into train/val/test sets"""
        print("\n" + "="*70)
        print("SPLITTING DATA")
        print("="*70)
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.X, self.y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_SEED,
            stratify=self.y if self.config.TARGET_TYPE == 'classification' else None
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.config.VAL_SIZE,
            random_state=self.config.RANDOM_SEED,
            stratify=y_temp if self.config.TARGET_TYPE == 'classification' else None
        )
        
        print(f"\nTrain set: {X_train.shape[0]:,} samples")
        print(f"Val set:   {X_val.shape[0]:,} samples")
        print(f"Test set:  {X_test.shape[0]:,} samples")
        
        # Scale features
        print("\nScaling features...")
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        
        return self
    
    def save_preprocessors(self):
        """Save scaler and label encoders"""
        print("\nSaving preprocessors...")
        with open(self.config.SCALER_SAVE_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(self.config.LABEL_ENCODERS_PATH, 'wb') as f:
            pickle.dump(self.label_encoders, f)
        print(f"  Saved to {self.config.OUTPUT_DIR}")

# ============================================================================
# Dataset
# ============================================================================

class EdgeDataset(Dataset):
    """PyTorch dataset for edge prediction"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================================
# Model Architecture
# ============================================================================

class AdjacencyPredictor(nn.Module):
    """Neural network for edge prediction"""
    
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers, dropout):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        
        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else embedding_dim
            self.hidden_layers.append(nn.Linear(in_dim, embedding_dim))
            self.batch_norms.append(nn.BatchNorm1d(embedding_dim))
        
        # Output layer
        self.output = nn.Linear(embedding_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Input projection
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Hidden layers
        for hidden, bn in zip(self.hidden_layers, self.batch_norms):
            x = hidden(x)
            x = bn(x)
            x = self.relu(x)
            x = self.dropout(x)
        
        # Output
        x = self.output(x)
        
        return x.squeeze()

# ============================================================================
# Training
# ============================================================================

class Trainer:
    """Handle model training and evaluation"""
    
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Loss and optimizer
        if config.TARGET_TYPE == 'classification':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.MSELoss()
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.GRAD_CLIP
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader, phase='val'):
        """Evaluate on validation/test set"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in tqdm(val_loader, desc=f"Evaluating ({phase})", leave=False):
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item()
                
                # Store predictions
                if self.config.TARGET_TYPE == 'classification':
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    all_preds.extend(probs)
                else:
                    all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        # Calculate metrics
        metrics = {'loss': avg_loss}
        
        if self.config.TARGET_TYPE == 'classification':
            auc = roc_auc_score(all_labels, all_preds)
            ap = average_precision_score(all_labels, all_preds)
            preds_binary = (np.array(all_preds) > 0.5).astype(int)
            f1 = f1_score(all_labels, preds_binary)
            
            metrics.update({
                'auc': auc,
                'ap': ap,
                'f1': f1
            })
        else:
            mse = mean_squared_error(all_labels, all_preds)
            mae = mean_absolute_error(all_labels, all_preds)
            rmse = np.sqrt(mse)
            
            metrics.update({
                'mse': mse,
                'mae': mae,
                'rmse': rmse
            })
        
        return metrics
    
    def train(self, train_loader, val_loader):
        """Full training loop"""
        print("\n" + "="*70)
        print("TRAINING")
        print("="*70)
        
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_metrics = self.evaluate(val_loader, phase='val')
            val_loss = val_metrics['loss']
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_loss:.4f}")
            
            if self.config.TARGET_TYPE == 'classification':
                print(f"Val AUC:    {val_metrics['auc']:.4f}")
                print(f"Val AP:     {val_metrics['ap']:.4f}")
                print(f"Val F1:     {val_metrics['f1']:.4f}")
            else:
                print(f"Val RMSE:   {val_metrics['rmse']:.4f}")
                print(f"Val MAE:    {val_metrics['mae']:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping and model saving
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Convert numpy values to Python types for JSON serialization
                val_metrics_serializable = {}
                for key, value in val_metrics.items():
                    if isinstance(value, np.ndarray):
                        val_metrics_serializable[key] = float(value)
                    elif isinstance(value, (np.integer, np.floating)):
                        val_metrics_serializable[key] = float(value)
                    else:
                        val_metrics_serializable[key] = value
                
                # Save best model
                torch.save({
                    'epoch': int(epoch),
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': float(val_loss),
                    'val_metrics': val_metrics_serializable,
                }, self.config.MODEL_SAVE_PATH)
                print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
                print(f"No improvement for {self.patience_counter} epochs")
                
                if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"\nEarly stopping triggered!")
                    print(f"Best epoch: {self.best_epoch+1} with val_loss: {self.best_val_loss:.4f}")
                    break
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Best model from epoch {self.best_epoch+1}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def plot_training_history(self):
        """Plot training and validation losses"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.train_losses, label='Train Loss', linewidth=2)
        axes[0].plot(self.val_losses, label='Val Loss', linewidth=2)
        axes[0].axvline(self.best_epoch, color='r', linestyle='--', label='Best Model')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training History')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Metrics plot
        if self.config.TARGET_TYPE == 'classification':
            aucs = [m['auc'] for m in self.val_metrics]
            axes[1].plot(aucs, label='AUC', linewidth=2)
            axes[1].set_ylabel('AUC')
            axes[1].set_title('Validation AUC')
        else:
            rmses = [m['rmse'] for m in self.val_metrics]
            axes[1].plot(rmses, label='RMSE', linewidth=2)
            axes[1].set_ylabel('RMSE')
            axes[1].set_title('Validation RMSE')
        
        axes[1].axvline(self.best_epoch, color='r', linestyle='--', label='Best Model')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config.OUTPUT_DIR / 'training_history.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved training history plot to {self.config.OUTPUT_DIR / 'training_history.png'}")
        plt.close()
    
    def save_metrics(self):
        """Save training metrics to JSON"""
        metrics = {
            'best_epoch': int(self.best_epoch),
            'best_val_loss': float(self.best_val_loss),
            'train_losses': [float(x) for x in self.train_losses],
            'val_losses': [float(x) for x in self.val_losses],
            'val_metrics': self.val_metrics,
            'config': {
                'learning_rate': self.config.LEARNING_RATE,
                'batch_size': self.config.BATCH_SIZE,
                'num_epochs': self.config.NUM_EPOCHS,
                'hidden_dim': self.config.HIDDEN_DIM,
                'embedding_dim': self.config.EMBEDDING_DIM,
                'num_layers': self.config.NUM_LAYERS,
                'dropout': self.config.DROPOUT,
            }
        }
        
        with open(self.config.METRICS_SAVE_PATH, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Saved metrics to {self.config.METRICS_SAVE_PATH}")

# ============================================================================
# Main
# ============================================================================

def main():
    """Main training pipeline"""
    
    print("\n" + "="*70)
    print("ADJACENCY MATRIX TRAINING")
    if Config.PILOT_MODE:
        print("🚀 PILOT MODE 🚀")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup
    config = Config()
    
    # Override settings for pilot mode
    if config.PILOT_MODE:
        print("\n⚙️  Using pilot mode settings:")
        print(f"   Batch size: {config.PILOT_BATCH_SIZE}")
        print(f"   Epochs: {config.PILOT_NUM_EPOCHS}")
        print(f"   Early stopping patience: {config.PILOT_EARLY_STOPPING_PATIENCE}")
        config.BATCH_SIZE = config.PILOT_BATCH_SIZE
        config.NUM_EPOCHS = config.PILOT_NUM_EPOCHS
        config.EARLY_STOPPING_PATIENCE = config.PILOT_EARLY_STOPPING_PATIENCE
    
    config.save()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Set random seeds
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # Load and process data
    processor = DataProcessor(config)
    processor.load_data()
    processor.merge_features()
    processor.prepare_features()
    processor.split_data()
    processor.save_preprocessors()
    
    # Create datasets and dataloaders
    print("\n" + "="*70)
    print("CREATING DATALOADERS")
    print("="*70)
    
    train_dataset = EdgeDataset(processor.X_train, processor.y_train)
    val_dataset = EdgeDataset(processor.X_val, processor.y_val)
    test_dataset = EdgeDataset(processor.X_test, processor.y_test)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE * 2,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE * 2,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")
    
    # Create model
    print("\n" + "="*70)
    print("CREATING MODEL")
    print("="*70)
    
    input_dim = processor.X_train.shape[1]
    model = AdjacencyPredictor(
        input_dim=input_dim,
        hidden_dim=config.HIDDEN_DIM,
        embedding_dim=config.EMBEDDING_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    )
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train
    trainer = Trainer(model, config, device)
    trainer.train(train_loader, val_loader)
    
    # Load best model for final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)
    
    checkpoint = torch.load(config.MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = trainer.evaluate(test_loader, phase='test')
    
    print("\nTest Set Results:")
    print("-" * 50)
    for metric, value in test_metrics.items():
        print(f"{metric.upper():10s}: {value:.4f}")
    
    # Save results
    trainer.plot_training_history()
    trainer.save_metrics()
    
    print("\n" + "="*70)
    print("ALL DONE!")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutputs saved to: {config.OUTPUT_DIR}")
    print(f"  - Model: {config.MODEL_SAVE_PATH}")
    print(f"  - Metrics: {config.METRICS_SAVE_PATH}")
    print(f"  - Training plot: {config.OUTPUT_DIR / 'training_history.png'}")

if __name__ == "__main__":
    main()