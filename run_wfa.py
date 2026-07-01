import os
import json
import shutil
import multiprocessing as mp
import torch

from data.wfa_pipeline import WalkForwardPipeline
from models.manager.train_manager import ManagerPipeline

# Stop TensorFlow from greedily locking 100% of the GPU VRAM
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Suppress the TensorFlow C++ warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ==========================================
# CLOUD CONTINUITY PROTOCOL (GOOGLE DRIVE)
# ==========================================
DRIVE_ROOT = "/content/drive/MyDrive/XAU_RL_V3_Checkpoints"
STATE_FILE = f"{DRIVE_ROOT}/training_state.json"
DRIVE_MODELS_DIR = f"{DRIVE_ROOT}/models/manager/saved"

def load_training_state():
    """Reads the JSON ledger from Google Drive to determine resume point."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
            print(f"💾 Cloud State Found: Resuming sequence from Split {state['next_split']}")
            return state['next_split'], state['resume_model_path']
    return 0, None

def save_training_state(completed_split, local_model_path):
    """Preserves weights and state to Google Drive to survive Colab timeouts."""
    os.makedirs(DRIVE_MODELS_DIR, exist_ok=True)
    
    # 1. Copy the ephemeral local best_model.zip to persistent Google Drive storage
    drive_model_dir = f"{DRIVE_MODELS_DIR}/wfa_{completed_split}"
    os.makedirs(drive_model_dir, exist_ok=True)
    drive_model_path = f"{drive_model_dir}/best_model.zip"
    
    shutil.copy2(local_model_path, drive_model_path)

    # 2. Update the JSON ledger with the new continuity pointer
    state = {
        "last_completed_split": completed_split,
        "next_split": completed_split + 1,
        "resume_model_path": drive_model_path
    }
    
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=4)
        
    print(f"☁️ State and Weights securely preserved to Google Drive: wfa_{completed_split}")
    
    return drive_model_path


def execute_wfa_training():
    DATA_PATH = "data/processed/labeled_features_15m.csv"
    ORACLE_WEIGHTS = "models/oracle/best_oracle.pth"
    
    if not os.path.exists("/content/drive/MyDrive"):
        print("⚠️ WARNING: Google Drive is not mounted. Checkpoints will be local and destroyed on timeout.")
    
    # --- The Continuity Rule (Automated) ---
    START_SPLIT, RESUME_SAC_PATH = load_training_state()

    print("Initializing Structural Walk-Forward Pipeline...")
    wfa = WalkForwardPipeline(features_path=DATA_PATH, embargo_bars=50)
    master_df = wfa.load_data(holdout_fraction=0.2)
    
    # Configure exact window sizes for the 15m timeframe
    splits = wfa.generate_splits(train_size=15000, test_size=3000, step_size=1500)
    print(f"Total OOS Splits Generated inside the Firewall: {len(splits)}")

    pipeline = ManagerPipeline(
        features_path=DATA_PATH, 
        oracle_weights_path=ORACLE_WEIGHTS
    )
    
    current_model_path = RESUME_SAC_PATH

    for idx in range(START_SPLIT, len(splits)):
        print(f"\n{'='*50}")
        print(f" Executing Parallel WFA Split {idx} / {len(splits)-1} ")
        print(f"{'='*50}")
        
        split_data = splits[idx]
        train_df = split_data['train']
        val_df = split_data['test']
        
        # Pipeline returns the local path to the saved weights
        local_model_path = pipeline.train_wfa_split(
            split_idx=idx,
            train_df=train_df,
            val_df=val_df,
            previous_sac_path=current_model_path
        )
        
        # Immediately push the ephemeral weights and state ledger to Google Drive
        current_model_path = save_training_state(idx, local_model_path)
        
        print(f"Split {idx} Complete. Pipeline ready for next sequence.")

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    execute_wfa_training()