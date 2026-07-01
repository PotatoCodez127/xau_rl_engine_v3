import os
from data.wfa_pipeline import WalkForwardPipeline
from models.manager.train_manager import ManagerPipeline

def execute_wfa_training():
    DATA_PATH = "data/processed/labeled_features_15m.csv"
    ORACLE_WEIGHTS = "models/oracle/best_oracle.pth"
    
    # --- The Continuity Rule ---
    # Set to a higher number if a cloud timeout interrupts training
    START_SPLIT = 0 
    
    # If START_SPLIT > 0, set this to the path of the previous split's best_model.zip
    # e.g., "models/manager/saved/wfa_14/best_model.zip"
    RESUME_SAC_PATH = None 

    print("Initializing Structural Walk-Forward Pipeline...")
    wfa = WalkForwardPipeline(features_path=DATA_PATH, embargo_bars=50)
    master_df = wfa.load_data(holdout_fraction=0.2)
    
    # Configure exact window sizes for the 15m timeframe
    # Adjust train_size/test_size if you prefer shorter/longer structural memory
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
        
        # Pipeline returns the path to the saved weights for continuous injection
        current_model_path = pipeline.train_wfa_split(
            split_idx=idx,
            train_df=train_df,
            val_df=val_df,
            previous_sac_path=current_model_path
        )
        
        print(f"Split {idx} Complete. Weights preserved at {current_model_path}")

if __name__ == "__main__":
    execute_wfa_training()