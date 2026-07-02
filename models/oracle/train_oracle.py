import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import logging

from models.oracle.attention_net import TemporalAttentionOracle
from models.oracle.custom_loss import FocalLoss

logger = logging.getLogger("Oracle_Trainer")

# ==============================================================
# 1. 3D SLIDING WINDOW DATASET
# ==============================================================
class SequenceDataset(Dataset):
    """
    Transforms flat 2D financial time-series into 3D sequential tensors.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 30):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        # We lose the first `seq_len` rows because we need a full buffer to make the first prediction
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        # Extract a 30-period slice
        X_seq = self.X[idx : idx + self.seq_len]
        # The target is the directional outcome associated with the end of this sequence
        y_target = self.y[idx + self.seq_len - 1]
        
        return X_seq, y_target

# ==============================================================
# 2. DATA PIPELINE PREPARATION
# ==============================================================
def prepare_dataloaders(df_train: pd.DataFrame, df_val: pd.DataFrame, seq_len: int, batch_size: int):
    # Ensure standard mathematical variable naming (Ruff Whitelisted)
    feature_cols = [c for c in df_train.columns if c not in ['target', 'time', 'datetime', 'date', 'index'] and not c.startswith("env_")]
    
    X_train = df_train[feature_cols].values
    y_train = df_train['target'].values
    
    X_val = df_val[feature_cols].values
    y_val = df_val['target'].values

    train_dataset = SequenceDataset(X_train, y_train, seq_len=seq_len)
    val_dataset = SequenceDataset(X_val, y_val, seq_len=seq_len)

    # Shuffle training data to break chronological correlation during optimization,
    # but NEVER shuffle validation data (must remain chronological for WFA integrity)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, len(feature_cols)

# ==============================================================
# 3. CORE TRAINING LOOP
# ==============================================================
def train_oracle_model(df_train: pd.DataFrame, df_val: pd.DataFrame, save_path: str, epochs: int = 30, device: torch.device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    seq_len = 30
    batch_size = 256
    
    logger.info(f"Preparing 3D Sequential DataLoaders (Batch Size: {batch_size}, Seq Len: {seq_len})...")
    train_loader, val_loader, input_dim = prepare_dataloaders(df_train, df_val, seq_len, batch_size)

    # Initialize Model
    model = TemporalAttentionOracle(input_dim=input_dim, seq_len=seq_len).to(device)
    
    # Optimizer & Loss
    # AdamW (Weight Decay) is strictly better than Adam for Attention/Transformer architectures
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Gamma=2.0 sharply focuses the network on the minority classes (Long/Short breakouts)
    # Alpha dynamically weights the classes. We assume Hold=0, Long=1, Short=2.
    class_weights = torch.tensor([0.2, 0.8, 0.8]).to(device) 
    criterion = FocalLoss(alpha=class_weights, gamma=2.0, reduction='mean')

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_X)
            
            loss = criterion(logits, batch_y)
            loss.backward()
            
            # Gradient Clipping prevents exploding gradients in RNNs/GRUs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                val_loss += loss.item()
                
                # Accuracy calculation
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)
                
        val_loss /= len(val_loader)
        val_acc = correct / total

        logger.info(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # --- CHECKPOINTING ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logger.info(f"⭐ New Best Model Checkpoint Saved: {save_path}")

    logger.info("✅ PyTorch Phase A Oracle Training Complete.")
    return best_val_loss