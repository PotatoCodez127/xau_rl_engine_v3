import os
import gc
import logging
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import timezone
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback

# Internal V3 Imports
from env.xau_dynamic_env import XAUDynamicEnv
from models.oracle.train_oracle import train_oracle_model
from export_onnx import export_models_to_onnx
from models.oracle.attention_net import TemporalAttentionOracle

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S UTC'
)
logger = logging.getLogger("WFA_Orchestrator")

# --- COLAB CONFIGURATION ---
DRIVE_ROOT = "/content/drive/MyDrive/XAU_RL_V3"
DATA_PATH = os.path.join(DRIVE_ROOT, "data", "processed_features.parquet")
MODEL_DIR = os.path.join(DRIVE_ROOT, "models")
DEPLOY_DIR = os.path.join(MODEL_DIR, "deployed")

os.makedirs(os.path.join(MODEL_DIR, "oracle"), exist_ok=True)
os.makedirs(os.path.join(MODEL_DIR, "manager"), exist_ok=True)
os.makedirs(DEPLOY_DIR, exist_ok=True)

from models.oracle.attention_net import TemporalAttentionOracle

def inject_oracle_probs(df: pd.DataFrame, oracle_path: str, device: torch.device) -> pd.DataFrame:
    logger.info("Injecting Oracle Probabilities via Sliding Window...")
    
    # Exclude structural and metadata columns
    exclude_cols = ["target", "time", "datetime", "date", "index"]
    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith("env_")]

    # Initialize and load the frozen Phase A Oracle
    oracle = TemporalAttentionOracle(input_dim=len(feature_cols), seq_len=30).to(device)
    oracle.load_state_dict(torch.load(oracle_path, map_location=device))
    oracle.eval()

    raw_features = df[feature_cols].values
    probs_list = np.zeros((len(df), 3))
    
    # Start at index 30 to allow for the buffer sequence
    with torch.no_grad():
        for i in range(30, len(df)):
            window = raw_features[i - 30 : i]
            window_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)
            logits = oracle(window_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            probs_list[i] = probs

    # Inject the continuous arrays into the DataFrame
    df["prob_hold"] = probs_list[:, 0]
    df["prob_long"] = probs_list[:, 1]
    df["prob_short"] = probs_list[:, 2]

    return df

class WFAOrchestrator:
    def __init__(self, data_path: str, n_splits: int = 5):
        self.data_path = data_path
        self.n_splits = n_splits
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initialized WFA Orchestrator on device: {self.device}")

    def load_and_verify_data(self) -> pd.DataFrame:
        logger.info("Loading feature dataset...")
        df = pd.read_parquet(self.data_path)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
        df = df.sort_index(ascending=True)
        logger.info(f"Dataset Verified: {len(df)} rows. Chronology strictly enforced.")
        return df

    def generate_wfa_splits(self, df: pd.DataFrame):
        total_len = len(df)
        split_size = total_len // (self.n_splits + 1)
        splits = []
        for i in range(1, self.n_splits + 1):
            train_end = i * split_size
            val_end = train_end + split_size
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(train_end, val_end)
            splits.append((train_idx, val_idx))
        return splits

    def run_pipeline(self, start_fold=1):
        """Main WFA Execution Loop with State Continuity."""
        df = self.load_and_verify_data()
        splits = self.generate_wfa_splits(df)
        
        best_calmar = -np.inf
        best_oracle_path = ""
        best_manager_path = ""

        for fold, (train_idx, val_idx) in enumerate(splits, 1):
            if fold < start_fold:
                logger.info(f"⏩ Skipping Fold {fold} (Already Completed).")
                continue

            logger.info(f"\n{'='*40}\n🚀 INITIATING WFA FOLD {fold}/{self.n_splits}\n{'='*40}")
            
            # --- Temporal Slicing ---
            df_train = df.iloc[train_idx].copy().reset_index()
            df_val = df.iloc[val_idx].copy().reset_index()
            
            # ---------------------------------------------------------
            # PHASE A: TRAIN TEMPORAL ATTENTION ORACLE
            # ---------------------------------------------------------
            logger.info("[PHASE A] Training PyTorch Oracle on Fold Data...")
            fold_oracle_path = os.path.join(MODEL_DIR, "oracle", f"oracle_fold_{fold}.pth")
            
            # Check if Oracle for this fold already exists to avoid retraining if interrupted during Phase B
            if os.path.exists(fold_oracle_path) and fold == start_fold:
                logger.info(f"✅ Found existing Oracle for Fold {fold}. Skipping Phase A training...")
            else:
                train_oracle_model(df_train, df_val, save_path=fold_oracle_path, epochs=30, device=self.device)

            # ---------------------------------------------------------
            # PHASE B: TRAIN SAC MANAGER (EPISODIC ENVIRONMENT)
            # ---------------------------------------------------------
            logger.info("[PHASE B] Initializing XAU Dynamic Environment (Episodic Mode)...")
            env_train = XAUDynamicEnv(df=df_train, initial_balance=10000.0)
            env_val = XAUDynamicEnv(df=df_val, initial_balance=10000.0)
            
            check_env(env_train)
            
            fold_manager_dir = os.path.join(MODEL_DIR, "manager", f"fold_{fold}")
            current_model_path = os.path.join(fold_manager_dir, "best_model.zip")
            prev_model_path = os.path.join(MODEL_DIR, "manager", f"fold_{fold-1}", "best_model.zip")

            # --- CONTINUOUS MEMORY INTEGRATION ---
            if os.path.exists(current_model_path):
                logger.info(f"🔄 Resuming from interrupted checkpoint for Fold {fold}...")
                manager_model = SAC.load(current_model_path, env=env_train, device=self.device)
            elif fold > 1 and os.path.exists(prev_model_path):
                logger.info(f"🧠 Inheriting memory: Loading SAC weights from Fold {fold-1}...")
                manager_model = SAC.load(prev_model_path, env=env_train, device=self.device)
            else:
                logger.info("🌱 Initializing blank SAC Manager...")
                manager_model = SAC(
                    "MlpPolicy", 
                    env_train, 
                    learning_rate=3e-4,
                    buffer_size=100000,
                    batch_size=256,
                    ent_coef='auto',
                    gamma=0.99,
                    tau=0.005,
                    device=self.device,
                    verbose=0
                )

            eval_callback = EvalCallback(
                env_val, 
                best_model_save_path=fold_manager_dir,
                log_path=fold_manager_dir, 
                eval_freq=5000,
                deterministic=True, 
                render=False
            )

            logger.info("[PHASE B] Training SAC Manager...")
            # reset_num_timesteps=False prevents SB3 from resetting the learning rate schedule on resume
            manager_model.learn(total_timesteps=150_000, callback=eval_callback, reset_num_timesteps=False)
            
            current_calmar = pd.read_csv(os.path.join(fold_manager_dir, "evaluations.npz"))['results'].mean()
            logger.info(f"🏁 Fold {fold} Complete. Terminal Calmar Proxy: {current_calmar:.2f}")

            if current_calmar > best_calmar:
                best_calmar = current_calmar
                best_oracle_path = fold_oracle_path
                best_manager_path = current_model_path
                
            del env_train, env_val, manager_model
            gc.collect()
            torch.cuda.empty_cache()

        # ---------------------------------------------------------
        # PHASE C: ONNX COMPILATION EXPORT
        # ---------------------------------------------------------
        logger.info(f"\n{'='*40}\n🎉 WFA COMPLETE. Best Calmar: {best_calmar:.2f}\n{'='*40}")
        logger.info("Executing Phase C: Exporting champion models to ONNX for i5 Deployment...")
        
        export_models_to_onnx(best_oracle_path, best_manager_path, DEPLOY_DIR)

if __name__ == "__main__":
    orchestrator = WFAOrchestrator(data_path=DATA_PATH, n_splits=5)
    orchestrator.run_pipeline(start_fold=2)