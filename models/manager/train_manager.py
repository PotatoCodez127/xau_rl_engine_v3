import os
import torch
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from sklearn.preprocessing import StandardScaler
import gc
from env.xau_dynamic_env import XAUDynamicEnv
from models.oracle.attention_net import TemporalAttentionOracle

class ManagerPipeline:
    def __init__(
        self, features_path: str, dxy_path: str = "", oracle_weights_path: str = None
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        temp_df = pd.read_csv(features_path, nrows=1)
        exclude_cols = ["target", "time", "datetime", "date"]
        self.feature_cols = [
            c
            for c in temp_df.columns
            if c not in exclude_cols and not c.startswith("env_")
        ]
        input_dim = len(self.feature_cols)

        self.oracle = TemporalAttentionOracle(input_dim=input_dim, seq_len=30).to(
            self.device
        )
        if oracle_weights_path and os.path.exists(oracle_weights_path):
            self.oracle.load_state_dict(
                torch.load(oracle_weights_path, map_location=self.device)
            )
        self.oracle.eval()

    def _precompute_oracle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Pre-computing Oracle pattern recognition...")
        scaler = StandardScaler()
        if len(df) > 0 and len(self.feature_cols) > 0:
            raw_features = scaler.fit_transform(df[self.feature_cols].values)
        else:
            raise ValueError("Dataframe is missing required features.")

        probs_list = np.zeros((len(df), 3))

        with torch.no_grad():
            for i in range(30, len(df)):
                window = raw_features[i - 30 : i]
                window_tensor = torch.FloatTensor(window).unsqueeze(0).to(self.device)
                logits = self.oracle(window_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                probs_list[i] = probs

        df["prob_hold"] = probs_list[:, 0]
        df["prob_long"] = probs_list[:, 1]
        df["prob_short"] = probs_list[:, 2]
        return df

    def train_wfa_split(
        self,
        split_idx: int,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        previous_sac_path: str = None,
    ):
        print(f"[CHECKPOINT 1] Pre-computing Phase A Oracle features for Split {split_idx}...")
        enriched_train = self._precompute_oracle_features(train_df.copy())
        enriched_val = self._precompute_oracle_features(val_df.copy())

        print("[CHECKPOINT 2] Initializing DummyVecEnv (Loop-based Vectorization)...")
        # Structurally safe: Executes environments sequentially to preserve CUDA context
        train_env = make_vec_env(
            XAUDynamicEnv, 
            n_envs=4, 
            env_kwargs={"df": enriched_train}, 
            vec_env_cls=DummyVecEnv
        )
        
        val_env = make_vec_env(
            XAUDynamicEnv, 
            n_envs=1, 
            env_kwargs={"df": enriched_val}, 
            vec_env_cls=DummyVecEnv
        )

        save_dir = f"./models/manager/saved/wfa_{split_idx}/"
        os.makedirs(save_dir, exist_ok=True)

        print("[CHECKPOINT 3] Building SAC Neural Architecture (RAM Optimized)...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = SAC(
            "MlpPolicy",
            train_env,
            gamma=0.9245,
            learning_rate=0.000253,
            batch_size=1024,          
            buffer_size=250000,       
            tau=0.00137,
            train_freq=16,
            ent_coef="auto",
            verbose=1,
            tensorboard_log="logs/",
            device=self.device,
        )

        if previous_sac_path and os.path.exists(previous_sac_path):
            print(f"[CHECKPOINT 4] Injecting Continuous Memory from: {previous_sac_path}")
            model.set_parameters(previous_sac_path, exact_match=False)
        else:
            print("[CHECKPOINT 4] No previous memory found. Initializing blank weights.")

        eval_callback = EvalCallback(
            val_env,
            best_model_save_path=save_dir,
            log_path=f"./logs/wfa_split_{split_idx}/",
            eval_freq=5000,
            deterministic=True,
            render=False,
        )

        print(f"[CHECKPOINT 5] Triggering model.learn() for Split {split_idx}...")
        try:
            model.learn(
                total_timesteps=50000, callback=eval_callback, reset_num_timesteps=False
            )
            print(f"[CHECKPOINT 6] model.learn() successfully completed for Split {split_idx}.")
        except Exception as e:
            print(f"[FATAL ERROR] Crash during model.learn(): {e}")
            raise e
        finally:
            print("[CHECKPOINT 7] Freeing SubprocVecEnv memory buffers...")
            train_env.close()
            val_env.close()
        
        return f"{save_dir}best_model.zip"
