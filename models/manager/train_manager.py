import os
import torch
import pandas as pd
import logging
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

from env.xau_dynamic_env import XAUDynamicEnv

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("SAC_Standalone_Trainer")

class ReplayBufferCheckpointCallback(BaseCallback):
    """
    Periodically saves the replay buffer to prevent catastrophic forgetting
    upon session resumption, maintaining off-policy stability.
    """
    def __init__(self, save_path: str, save_freq: int, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save_replay_buffer(self.save_path)
            if self.verbose > 0:
                logger.info(f"💾 Replay Buffer Checkpoint Saved: {self.save_path}")
        return True

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

    model_path = os.path.join(save_dir, "best_model.zip")
    buffer_path = os.path.join(save_dir, "replay_buffer.pkl")

    # --- STATE RESUMPTION LOGIC ---
    if os.path.exists(model_path):
        logger.info(f"🔄 Existing SAC checkpoint detected at {model_path}. Attempting to load weights...")
        # We explicitly pass the environment and device so the loaded model can continue training natively
        model = SAC.load(
            model_path,
            env=env_train,
            device=device,
            custom_objects={"learning_rate": 3e-4, "buffer_size": 200000}
        )
        
        if os.path.exists(buffer_path):
            logger.info(f"📥 Restoring Off-Policy Memory from {buffer_path}...")
            model.load_replay_buffer(buffer_path)
            logger.info("✅ Replay Buffer Loaded Successfully. Memory continuity preserved.")
        else:
            logger.warning("⚠️ No Replay Buffer found! The agent will start with an empty buffer, which may destabilize the policy temporarily.")
    else:
        logger.info("🆕 No existing checkpoint found. Initializing fresh Distributional SAC Agent...")
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

    # Buffer Checkpoint Callback
    buffer_callback = ReplayBufferCheckpointCallback(
        save_path=buffer_path,
        save_freq=10000, # Synchronized with eval_freq
        verbose=1
    )

    logger.info("Commencing SAC Neural Optimization...")
    # Stable Baselines 3 accepts a list of callbacks
    model.learn(total_timesteps=500_000, callback=[eval_callback, buffer_callback])
    
    logger.info(f"✅ Standalone SAC Training Complete. Best model saved to {save_dir}")
    
    # Final manual save of the buffer ensuring no trailing steps are lost
    model.save_replay_buffer(buffer_path)

if __name__ == "__main__":
    # Paths for Google Colab isolated execution
    DATA_PATH = "/content/drive/MyDrive/XAU_RL_V3/data/processed_features_with_oracle.parquet"
    SAVE_DIR = "/content/drive/MyDrive/XAU_RL_V3/models/manager/standalone_tune"
    
    train_standalone_manager(DATA_PATH, SAVE_DIR)