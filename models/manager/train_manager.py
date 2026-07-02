import os
import torch
import pandas as pd
import logging
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback

from env.xau_dynamic_env import XAUDynamicEnv

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("SAC_Standalone_Trainer")

def train_standalone_manager(data_path: str, save_dir: str):
    """
    Independent training loop for the Phase B Distributional SAC Agent.
    Allows for hyperparameter tuning of the Episodic Reward logic isolated from the Oracle.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initializing Standalone SAC Training on {device}...")

    # Load Data (Assumes Phase A probabilities are already injected into the parquet file)
    df = pd.read_parquet(data_path)
    
    # Simple Temporal Split (e.g., 80% Train, 20% Validation)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].reset_index()
    df_val = df.iloc[split_idx:].reset_index()

    # Initialize the Master-Slave Environments
    logger.info("Constructing XAU Dynamic Episodic Environments...")
    env_train = XAUDynamicEnv(df=df_train, initial_balance=10000.0)
    env_val = XAUDynamicEnv(df=df_val, initial_balance=10000.0)

    # Validate physical constraints
    check_env(env_train)

    os.makedirs(save_dir, exist_ok=True)

    # Configure the Distributional SAC Agent
    # High batch_size and tau are standard for noisy financial time-series
    model = SAC(
        "MlpPolicy",
        env_train,
        learning_rate=3e-4,
        buffer_size=200000,
        batch_size=512,
        ent_coef='auto',     # Auto-tunes entropy to prevent premature policy convergence
        gamma=0.999,         # Long-term horizon for multi-day swing trades
        tau=0.005,           # Soft update rate for Polyak averaging
        target_update_interval=2,
        device=device,
        verbose=1
    )

    # Evaluation Callback (Grades the agent strictly on unseen chronological validation data)
    eval_callback = EvalCallback(
        env_val,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    logger.info("Commencing SAC Neural Optimization...")
    model.learn(total_timesteps=500_000, callback=eval_callback)
    
    logger.info(f"✅ Standalone SAC Training Complete. Best model saved to {save_dir}")

if __name__ == "__main__":
    # Paths for Google Colab isolated execution
    DATA_PATH = "/content/drive/MyDrive/XAU_RL_V3/data/processed_features_with_oracle.parquet"
    SAVE_DIR = "/content/drive/MyDrive/XAU_RL_V3/models/manager/standalone_tune"
    
    train_standalone_manager(DATA_PATH, SAVE_DIR)