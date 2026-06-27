import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from models.oracle.attention_net import TemporalAttentionOracle
from models.oracle.custom_loss import FocalLoss
from sklearn.preprocessing import StandardScaler


def train_oracle_supervised(df: pd.DataFrame, save_path: str, epochs: int = 50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Oracle on {device}...")

    # 1. Use the dataframe directly from the Walk-Forward pipeline
    feature_df = df.copy()

    # 2. Filter out targets, timestamps, and hidden RL environment prices
    target_col = "target"
    exclude_cols = [target_col, "time", "datetime", "date"]
    feature_cols = [
        c
        for c in feature_df.columns
        if c not in exclude_cols and not c.startswith("env_")
    ]

    # 3. Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_df[feature_cols].values)

    # 4. Create the 30-period rolling windows
    seq_len = 30
    X, y = [], []
    targets_raw = feature_df[target_col].values

    for i in range(seq_len, len(scaled_features)):
        X.append(scaled_features[i - seq_len : i])
        y.append(targets_raw[i])

    features_tensor = torch.FloatTensor(np.array(X)).to(device)
    targets_tensor = torch.LongTensor(np.array(y)).to(device)

    # Initialize Model & Optimizer
    input_dim = len(feature_cols)
    model = TemporalAttentionOracle(input_dim=input_dim, seq_len=seq_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = FocalLoss(gamma=2.0)

    # Standard PyTorch Training Loop
    model.train()
    batch_size = 256
    num_batches = max(1, len(features_tensor) // batch_size)

    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        for i in range(0, len(features_tensor), batch_size):
            batch_x = features_tensor[i : i + batch_size]
            batch_y = targets_tensor[i : i + batch_size]

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # --- NEW: Calculate Accuracy for monitoring ---
            _, predicted = torch.max(logits.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        epoch_acc = 100 * correct / total if total > 0 else 0
        print(
            f"Epoch {epoch+1}/{epochs} | Focal Loss: {epoch_loss / num_batches:.4f} | Accuracy: {epoch_acc:.2f}%"
        )

    torch.save(model.state_dict(), save_path)
    print(f"Oracle weights saved to {save_path}")
    return model


# ==========================================
# STANDALONE EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # Define absolute paths relative to your project root
    DATA_PATH = "data/processed/labeled_features_15m.csv"
    SAVE_PATH = "models/oracle/best_oracle.pth"

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Master dataset not found at {DATA_PATH}.")
        print("Please run 'python -m data.build_features' first.")
        exit(1)

    print(f"Loading V3 Master Dataset from {DATA_PATH}...")
    master_df = pd.read_csv(DATA_PATH, parse_dates=True)

    # Ensure the target directory exists before attempting to save
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # Trigger the training sequence
    train_oracle_supervised(df=master_df, save_path=SAVE_PATH, epochs=150)
