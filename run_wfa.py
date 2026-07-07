import os
import gc
import logging
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import timezone
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

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

# ==============================================================
# CUSTOM CALLBACK: CONTINUITY MANAGER
# ==============================================================
class ContinuityCallback(BaseCallback):
    """
    Decouples step tracking from volatile evaluation logs and ensures
    the off-policy Replay Buffer is continuously preserved across Colab timeouts.
    """
    def __init__(self, fold_dir: str, save_freq: int, verbose: int = 1):
        super().__init__(verbose)
        self.fold_dir = fold_dir
        self.save_freq = save_freq
        self.buffer_path = os.path.join(fold_dir, "replay_buffer.pkl")
        self.state_path = os.path.join(fold_dir, "training_state.json")
        
        # Initialize internal step counter from persistent state if available
        if os.path.exists(self.state_path):
            with open(self.state_path, 'r') as f:
                self.steps_completed = json.load(f).get("steps_completed", 0)
        else:
            self.steps_completed = 0

    def _on_step(self) -> bool:
        self.steps_completed += 1
        if self.n_calls % self.save_freq == 0:
            self.model.save_replay_buffer(self.buffer_path)
            with open(self.state_path, 'w') as f:
                json.dump({"steps_completed": self.steps_completed}, f)
            if self.verbose > 0:
                logger.info(f"💾 Continuity Checkpoint: {self.steps_completed} steps permanently secured.")
        return True


def inject_oracle_probs(df: pd.DataFrame, oracle_path: str, device: torch.device) -> pd.DataFrame:
    logger.info("Injecting Oracle Probabilities via Sliding Window...")
    
    exclude_cols = ["target", "time", "datetime", "date", "index"]
    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith("env_")]

    oracle = TemporalAttentionOracle(input_dim=len(feature_cols), seq_len=30).to(device)
    
    checkpoint = torch.load(oracle_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        oracle.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Successfully unpacked Oracle weights from comprehensive checkpoint dict.")
    else:
        oracle.load_state_dict(checkpoint)
        logger.warning("Loaded Oracle weights using legacy raw state_dict format.")
    
    oracle.eval()

    raw_features = df[feature_cols].values
    probs_list = np.zeros((len(df), 3))
    
    with torch.no_grad():
        for i in range(30, len(df)):
            window = raw_features[i - 30 : i]
            # Copy added to suppress PyTorch non-writable array warnings
            window_tensor = torch.FloatTensor(window.copy()).unsqueeze(0).to(device)
            logits = oracle(window_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            probs_list[i] = probs

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
            
            df_train = df.iloc[train_idx].copy().reset_index()
            df_val = df.iloc[val_idx].copy().reset_index()
            
            # --- PHASE A ---
            logger.info("[PHASE A] Training PyTorch Oracle on Fold Data...")
            fold_oracle_path = os.path.join(MODEL_DIR, "oracle", f"oracle_fold_{fold}.pth")
            
            if os.path.exists(fold_oracle_path) and fold == start_fold:
                logger.info(f"✅ Found existing Oracle for Fold {fold}. Skipping Phase A training...")
            else:
                train_oracle_model(df_train, df_val, save_path=fold_oracle_path, epochs=30, device=self.device)

            logger.info("[PHASE A] Injecting Oracle Probabilities into Environment State...")
            df_train = inject_oracle_probs(df_train, fold_oracle_path, self.device)
            df_val = inject_oracle_probs(df_val, fold_oracle_path, self.device)

            # --- PHASE B ---
            logger.info("[PHASE B] Initializing XAU Dynamic Environment (Episodic Mode)...")
            env_train = XAUDynamicEnv(df=df_train, initial_balance=10000.0)
            env_val = XAUDynamicEnv(df=df_val, initial_balance=10000.0)
            
            check_env(env_train)
            
            fold_manager_dir = os.path.join(MODEL_DIR, "manager", f"fold_{fold}")
            os.makedirs(fold_manager_dir, exist_ok=True)
            
            current_model_path = os.path.join(fold_manager_dir, "best_model.zip")
            current_buffer_path = os.path.join(fold_manager_dir, "replay_buffer.pkl")
            current_state_path = os.path.join(fold_manager_dir, "training_state.json")
            
            prev_model_path = os.path.join(MODEL_DIR, "manager", f"fold_{fold-1}", "best_model.zip")
            prev_buffer_path = os.path.join(MODEL_DIR, "manager", f"fold_{fold-1}", "replay_buffer.pkl")

            STEPS_PER_FOLD = 150_000
            EVAL_FREQ = 5000
            remaining_steps = STEPS_PER_FOLD

            if os.path.exists(current_model_path):
                logger.info(f"🔄 Resuming from interrupted checkpoint for Fold {fold}...")
                manager_model = SAC.load(current_model_path, env=env_train, device=self.device)
                
                # Load Buffer
                if os.path.exists(current_buffer_path):
                    manager_model.load_replay_buffer(current_buffer_path)
                    logger.info("📥 Fold Replay Buffer Restored.")
                
                # Calculate True Remaining Steps via JSON State
                if os.path.exists(current_state_path):
                    with open(current_state_path, 'r') as f:
                        steps_completed = json.load(f).get("steps_completed", 0)
                        remaining_steps = max(0, STEPS_PER_FOLD - steps_completed)
                        logger.info(f"📊 State Audit: {steps_completed}/{STEPS_PER_FOLD} verified via persistent tracker.")
                        logger.info(f"⏳ Adjusted target: Training for {remaining_steps} remaining steps.")
                else:
                    logger.warning("⚠️ No training_state.json found. Defaulting to full step count to prevent undertraining.")
                        
            elif fold > 1 and os.path.exists(prev_model_path):
                logger.info(f"🧠 Inheriting memory: Loading SAC weights from Fold {fold-1}...")
                manager_model = SAC.load(prev_model_path, env=env_train, device=self.device)
                
                if os.path.exists(prev_buffer_path):
                    manager_model.load_replay_buffer(prev_buffer_path)
                    logger.info("📥 Inherited Replay Buffer from previous fold.")
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

            # Callbacks
            eval_callback = EvalCallback(
                env_val, 
                best_model_save_path=fold_manager_dir,
                log_path=fold_manager_dir, 
                eval_freq=EVAL_FREQ,
                deterministic=True, 
                render=False
            )
            
            continuity_callback = ContinuityCallback(
                fold_dir=fold_manager_dir,
                save_freq=EVAL_FREQ,
                verbose=1
            )

            if remaining_steps > 0:
                logger.info("[PHASE B] Training SAC Manager...")
                manager_model.learn(
                    total_timesteps=remaining_steps, 
                    callback=[eval_callback, continuity_callback], 
                    reset_num_timesteps=False
                )
                # Final memory seal when loop naturally concludes
                manager_model.save_replay_buffer(current_buffer_path)
            else:
                logger.info(f"✅ Fold {fold} already reached {STEPS_PER_FOLD} steps. Skipping training.")
            
            # Metric Parsing (Only attempt if eval exists, gracefully handle missing array on instant-skips)
            try:
                eval_data = np.load(os.path.join(fold_manager_dir, "evaluations.npz"))
                current_calmar = eval_data['results'].mean()
                logger.info(f"🏁 Fold {fold} Complete. Terminal Calmar Proxy: {current_calmar:.2f}")
            except Exception as e:
                current_calmar = -1.0
                logger.warning("No evaluation history available to calculate Calmar Proxy.")

            if current_calmar > best_calmar:
                best_calmar = current_calmar
                best_oracle_path = fold_oracle_path
                best_manager_path = current_model_path
                
            del env_train, env_val, manager_model
            gc.collect()
            torch.cuda.empty_cache()

        logger.info(f"\n{'='*40}\n🎉 WFA COMPLETE. Best Calmar: {best_calmar:.2f}\n{'='*40}")
        logger.info("Executing Phase C: Exporting champion models to ONNX for i5 Deployment...")
        
        export_models_to_onnx(best_oracle_path, best_manager_path, DEPLOY_DIR)

if __name__ == "__main__":
    orchestrator = WFAOrchestrator(data_path=DATA_PATH, n_splits=1)
    orchestrator.run_pipeline(start_fold=1)