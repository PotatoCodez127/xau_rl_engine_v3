import os
import gc
import logging
import json
import torch
import numpy as np
import pandas as pd
from datetime import timezone
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

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

# ==============================================================
# BATCHED ORACLE INFERENCE (GPU OPTIMIZED)
# ==============================================================
def inject_oracle_probs(df: pd.DataFrame, oracle_path: str, device: torch.device) -> pd.DataFrame:
    logger.info("Injecting Oracle Probabilities via Batched Mixed-Precision Inference...")
    
    exclude_cols = ["target", "time", "datetime", "date", "index"]
    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith("env_")]

    oracle = TemporalAttentionOracle(input_dim=len(feature_cols), seq_len=30).to(device)
    
    checkpoint = torch.load(oracle_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        oracle.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Successfully unpacked Oracle weights.")
    else:
        oracle.load_state_dict(checkpoint)
        logger.warning("Loaded Oracle weights using legacy format.")
    
    oracle.eval()

    raw_features = df[feature_cols].values
    probs_list = np.zeros((len(df), 3))
    
    if len(raw_features) >= 30:
        windows = np.lib.stride_tricks.sliding_window_view(raw_features, (30, len(feature_cols)))
        windows = windows.squeeze(1) 
        windows = windows[:-1] 
        
        batch_size = 4096 
        probs_out = []
        
        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                batch = torch.FloatTensor(windows[i:i + batch_size].copy()).to(device)
                if device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        logits = oracle(batch)
                else:
                    logits = oracle(batch)
                
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs_out.append(probs)
                
        if probs_out:
            probs_out = np.vstack(probs_out)
            probs_list[30:30 + len(probs_out)] = probs_out

    df["prob_hold"] = probs_list[:, 0]
    df["prob_long"] = probs_list[:, 1]
    df["prob_short"] = probs_list[:, 2]

    return df

# ==============================================================
# VECTORIZED ENVIRONMENT BUILDER
# ==============================================================
def make_env(df_subset, init_balance):
    def _init():
        return XAUDynamicEnv(df=df_subset, initial_balance=init_balance)
    return _init

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
                continue

            logger.info(f"\n{'='*40}\n🚀 INITIATING WFA FOLD {fold}/{self.n_splits}\n{'='*40}")
            
            df_train = df.iloc[train_idx].copy().reset_index()
            df_val = df.iloc[val_idx].copy().reset_index()
            
            # --- PHASE A ---
            fold_oracle_path = os.path.join(MODEL_DIR, "oracle", f"oracle_fold_{fold}.pth")
            
            if os.path.exists(fold_oracle_path) and fold == start_fold:
                logger.info(f"✅ Found existing Oracle for Fold {fold}.")
            else:
                train_oracle_model(df_train, df_val, save_path=fold_oracle_path, epochs=30, device=self.device)

            df_train = inject_oracle_probs(df_train, fold_oracle_path, self.device)
            df_val = inject_oracle_probs(df_val, fold_oracle_path, self.device)

            # --- PHASE B (True Multi-Core Parallelization) ---
            logger.info("[PHASE B] Initializing Vectorized XAU Dynamic Environments...")
            num_cpu = os.cpu_count() or 2
            
            dummy_check = XAUDynamicEnv(df=df_train, initial_balance=10000.0)
            check_env(dummy_check)
            del dummy_check
            
            # SubprocVecEnv spawns independent OS processes, shattering the Python GIL constraint
            env_train = SubprocVecEnv([make_env(df_train, 10000.0) for _ in range(num_cpu)])
            
            # Validation remains DummyVecEnv to avoid IPC overhead since it requires only 1 env
            env_val = DummyVecEnv([make_env(df_val, 10000.0)]) 
            
            fold_manager_dir = os.path.join(MODEL_DIR, "manager", f"fold_{fold}")
            os.makedirs(fold_manager_dir, exist_ok=True)
            
            current_model_path = os.path.join(fold_manager_dir, "best_model.zip")
            current_buffer_path = os.path.join(fold_manager_dir, "replay_buffer.pkl")
            current_state_path = os.path.join(fold_manager_dir, "training_state.json")
            prev_model_path = os.path.join(MODEL_DIR, "manager", f"fold_{fold-1}", "best_model.zip")
            prev_buffer_path = os.path.join(MODEL_DIR, "manager", f"fold_{fold-1}", "replay_buffer.pkl")

            STEPS_PER_FOLD = 150_000
            EVAL_FREQ = max(1000, 5000 // num_cpu)
            remaining_steps = STEPS_PER_FOLD

            if os.path.exists(current_model_path):
                logger.info(f"🔄 Resuming checkpoint for Fold {fold}...")
                manager_model = SAC.load(current_model_path, env=env_train, device=self.device)
                if os.path.exists(current_buffer_path):
                    manager_model.load_replay_buffer(current_buffer_path)
                
                if os.path.exists(current_state_path):
                    with open(current_state_path, 'r') as f:
                        steps_completed = json.load(f).get("steps_completed", 0)
                        remaining_steps = max(0, STEPS_PER_FOLD - steps_completed)
                        
            elif fold > 1 and os.path.exists(prev_model_path):
                logger.info(f"🧠 Inheriting SAC weights from Fold {fold-1}...")
                manager_model = SAC.load(prev_model_path, env=env_train, device=self.device)
                if os.path.exists(prev_buffer_path):
                    manager_model.load_replay_buffer(prev_buffer_path)
            else:
                logger.info("🌱 Initializing blank SAC Manager...")
                manager_model = SAC(
                    "MlpPolicy", 
                    env_train, 
                    learning_rate=3e-4,
                    buffer_size=100000,
                    batch_size=1024,
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
                eval_freq=EVAL_FREQ,
                deterministic=True, 
                render=False
            )
            
            continuity_callback = ContinuityCallback(
                fold_dir=fold_manager_dir, save_freq=EVAL_FREQ, verbose=1
            )

            if remaining_steps > 0:
                logger.info("[PHASE B] Training SAC Manager...")
                manager_model.learn(
                    total_timesteps=remaining_steps, 
                    callback=[eval_callback, continuity_callback], 
                    reset_num_timesteps=False
                )
                manager_model.save_replay_buffer(current_buffer_path)
            else:
                logger.info(f"✅ Fold {fold} already reached {STEPS_PER_FOLD} steps.")
            
            try:
                eval_data = np.load(os.path.join(fold_manager_dir, "evaluations.npz"))
                current_calmar = eval_data['results'].mean()
                logger.info(f"🏁 Fold {fold} Complete. Terminal Calmar Proxy: {current_calmar:.2f}")
            except Exception:
                current_calmar = -1.0

            if current_calmar > best_calmar:
                best_calmar = current_calmar
                best_oracle_path = fold_oracle_path
                best_manager_path = current_model_path
                
            del env_train, env_val, manager_model
            gc.collect()
            torch.cuda.empty_cache()

        logger.info(f"\n{'='*40}\n🎉 WFA COMPLETE. Best Calmar: {best_calmar:.2f}\n{'='*40}")
        export_models_to_onnx(best_oracle_path, best_manager_path, DEPLOY_DIR)

if __name__ == "__main__":
    orchestrator = WFAOrchestrator(data_path=DATA_PATH, n_splits=5)
    orchestrator.run_pipeline(start_fold=1)