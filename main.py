import os
import json
from data.wfa_pipeline import WalkForwardPipeline
from models.oracle.train_oracle import train_oracle_supervised
from models.manager.train_manager import ManagerPipeline

FEATURES_PATH = "data/processed/labeled_features_15m.csv"
ORACLE_WEIGHTS = "models/oracle/best_oracle.pth"
STATE_FILE = "training_state.json"

# --- DEFAULT CONTROLS (Overridden by state file if it exists) ---
DEFAULT_START_SPLIT = 0
END_SPLIT = 56
DEFAULT_RESUME_SAC_PATH = None


def load_state():
    """Loads the timeline index and neural network path from the last successful checkpoint."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
            print(
                f"🔄 Recovered State: Resuming at Split {state.get('next_split')} using {state.get('resume_sac_path')}"
            )
            return state.get("next_split", DEFAULT_START_SPLIT), state.get(
                "resume_sac_path", DEFAULT_RESUME_SAC_PATH
            )
        except Exception as e:
            print(f"⚠️ Error reading state file: {e}. Starting fresh.")
    return DEFAULT_START_SPLIT, DEFAULT_RESUME_SAC_PATH


def save_state(next_split, resume_sac_path):
    """Saves the timeline index and neural network path after a successful WFA window."""
    state = {"next_split": next_split, "resume_sac_path": resume_sac_path}
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)
    print(f"💾 State saved. Ready for Split {next_split} recovery if interrupted.")


def main():
    print("=== XAU RL Engine V2: Master Brain Initialization ===")

    # Initialize from checkpoint or start fresh at 0
    start_split, current_sac_path = load_state()

    pipeline = WalkForwardPipeline(FEATURES_PATH, embargo_bars=50)
    pipeline.load_data(holdout_fraction=0.2)

    splits = pipeline.generate_splits(train_size=10000, test_size=2500, step_size=2500)
    print(f"Generated {len(splits)} Walk-Forward Splits within the 80% boundary.")

    for idx, split in enumerate(splits):
        if idx < start_split:
            continue

        if idx >= END_SPLIT:
            print(f"Reached END_SPLIT target ({END_SPLIT}). Clean Shutdown.")
            if os.path.exists(STATE_FILE):
                os.remove(STATE_FILE)  # Clean up tracker after full pipeline completion
            break

        print(f"\n--- Processing WFA Window {idx + 1}/{len(splits)} ---")
        train_df = split["train"]
        val_df = split["test"]

        print("PHASE A: Supervised Oracle Training")
        train_oracle_supervised(train_df, save_path=ORACLE_WEIGHTS, epochs=20)

        print("PHASE B: SAC Agent Training")
        manager_pipeline = ManagerPipeline(
            FEATURES_PATH, dxy_path="", oracle_weights_path=ORACLE_WEIGHTS
        )

        current_sac_path = manager_pipeline.train_wfa_split(
            split_idx=idx,
            train_df=train_df,
            val_df=val_df,
            previous_sac_path=current_sac_path,
        )

        # --- AUTOMATED STATE RECOVERY CHECKPOINT ---
        save_state(next_split=idx + 1, resume_sac_path=current_sac_path)


if __name__ == "__main__":
    main()
