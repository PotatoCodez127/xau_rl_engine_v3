import os
import pandas as pd
import numpy as np
import onnxruntime as ort
from env.xau_dynamic_env import XAUDynamicEnv
import numpy as np
np.random.seed(42) # Locks the stochastic trade resolution
# Silence benign ONNX batch-size shape warnings
ort.set_default_logger_severity(3)

def numpy_softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def run_deep_diagnostics():
    """
    Executes a high-fidelity WFA fold validation run on Local CPU.
    Utilizes ONNX Runtime for massive inference speedups.
    """
    # Standard Local Windows Paths
    DATA_PATH = os.path.join("data", "processed_features.parquet")
    ORACLE_ONNX = os.path.join("models", "deployed", "oracle_v3.onnx")
    MANAGER_ONNX = os.path.join("models", "deployed", "manager_actor_v3.onnx")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Cannot find data at: {DATA_PATH}")
        return

    print("🚀 Initiating Local ONNX Deep Diagnostics...")

    # 1. Load Data
    df = pd.read_parquet(DATA_PATH)
    val_idx = int(len(df) * 0.8)
    
    # FIXED: Removed drop=True so 'datetime' index becomes a column for the Env to use
    df_val = df.iloc[val_idx:].copy().reset_index()

    # 2. Inject ONNX Oracle Probabilities
    print("🧠 Injecting Temporal Attention Contexts (ONNX Batch Mode)...")
    exclude_cols = ["target", "time", "datetime", "date", "index"]
    feature_cols = [c for c in df_val.columns if c not in exclude_cols and not c.startswith("env_")]
    
    oracle_session = ort.InferenceSession(ORACLE_ONNX, providers=['CPUExecutionProvider'])
    oracle_input_name = oracle_session.get_inputs()[0].name

    raw_features = df_val[feature_cols].values.astype(np.float32)
    probs_list = np.zeros((len(df_val), 3), dtype=np.float32)
    
    windows = np.lib.stride_tricks.sliding_window_view(raw_features, (30, len(feature_cols))).squeeze(1)[:-1]
    
    # Process in batches for memory safety
    batch_size = 4096
    for i in range(0, len(windows), batch_size):
        batch = windows[i:i + batch_size].copy()
        logits = oracle_session.run(None, {oracle_input_name: batch})[0]
        probs_list[30 + i : 30 + i + len(batch)] = numpy_softmax(logits)

    df_val["prob_hold"] = probs_list[:, 0]
    df_val["prob_long"] = probs_list[:, 1]
    df_val["prob_short"] = probs_list[:, 2]

    # 3. Instantiate Environment & ONNX Manager
    print("⚙️ Initializing Zero-Overhead Environment & SAC ONNX Policy...")
    env = XAUDynamicEnv(df=df_val, initial_balance=10000.0)
    
    manager_session = ort.InferenceSession(MANAGER_ONNX, providers=['CPUExecutionProvider'])
    manager_input_name = manager_session.get_inputs()[0].name

    # 4. Simulation Loop
    obs, _ = env.reset()
    terminated, truncated = False, False
    
    trades = 0
    wins = 0
    peak = 10000.0
    
    print("📈 Running Fast-Forward ONNX Simulation...")
    while not (terminated or truncated):
        # Shape the observation for the ONNX C++ engine (1, 32)
        onnx_obs = obs.astype(np.float32).reshape(1, -1)
        action = manager_session.run(None, {manager_input_name: onnx_obs})[0][0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if env.bars_since_last_trade == 0:  
            trades += 1
            # Check if balance went up or down compared to previous step
            if reward > 0: 
                wins += 1
                
        peak = max(peak, env.balance)

    # 5. Output Diagnostics
    drawdown = ((peak - env.balance) / peak) * 100 if peak > 0 else 0.0
    win_rate = (wins / trades * 100) if trades > 0 else 0
    roi = ((env.balance - 10000.0) / 10000.0) * 100
    
    print("\n" + "="*40)
    print("📊 LOCAL ONNX DEEP DIAGNOSTICS REPORT")
    print("="*40)
    print(f"Total Steps Simulated : {env.current_step}")
    print(f"Total Trades Executed : {trades}")
    print(f"Win Rate              : {win_rate:.2f}%")
    print(f"Maximum Drawdown      : {drawdown:.2f}%")
    print(f"Final Account Balance : ${env.balance:.2f}")
    print(f"Total ROI             : {roi:.2f}%")
    print("="*40)

if __name__ == "__main__":
    run_deep_diagnostics()