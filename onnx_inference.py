import os
import torch
import numpy as np
from stable_baselines3 import SAC
from models.oracle.attention_net import TemporalAttentionOracle

def export_models_to_onnx(oracle_path: str, manager_path: str, output_dir: str):
    """
    Runs on Colab T4. Compiles the trained PyTorch and SB3 neural architectures 
    into static ONNX computational graphs for your local i5 laptop.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    print(f"🚀 Initiating ONNX Compilation Pipeline on {device}...")

    # --- 1. COMPILE PHASE A (THE ORACLE) ---
    print("\n[1/2] Compiling Temporal Attention Oracle...")
    
    # CORRECTED: Set to match the dynamic feature count from training
    input_dim = 39 
    seq_len = 30
    
    oracle = TemporalAttentionOracle(input_dim=input_dim, seq_len=seq_len).to(device)
    oracle.load_state_dict(torch.load(oracle_path, map_location=device))
    oracle.eval()

    dummy_oracle_input = torch.randn(1, seq_len, input_dim, requires_grad=False).to(device)
    oracle_onnx_path = os.path.join(output_dir, "oracle_v3.onnx")

    torch.onnx.export(
        oracle,
        dummy_oracle_input,
        oracle_onnx_path,
        export_params=True,
        opset_version=15,          
        do_constant_folding=True,  
        input_names=['sequence_features'],
        output_names=['directional_logits'],
        dynamic_axes={'sequence_features': {0: 'batch_size'}, 'directional_logits': {0: 'batch_size'}}
    )
    print(f"✅ Oracle compiled successfully to: {oracle_onnx_path}")

    # --- 2. COMPILE PHASE B (THE SAC MANAGER) ---
    print("\n[2/2] Compiling SAC Manager (Actor Policy)...")
    
    manager = SAC.load(manager_path, device=device)
    actor = manager.policy.actor
    actor.eval()

    # CORRECTED: 39 features + 3 probabilities + 3 state variables = 45
    obs_dim = 45
    dummy_sac_input = torch.randn(1, obs_dim, requires_grad=False).to(device)
    manager_onnx_path = os.path.join(output_dir, "manager_actor_v3.onnx")

    torch.onnx.export(
        actor,
        dummy_sac_input,
        manager_onnx_path,
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=['observation_vector'],
        output_names=['continuous_actions'],
        dynamic_axes={'observation_vector': {0: 'batch_size'}, 'continuous_actions': {0: 'batch_size'}}
    )
    print(f"✅ SAC Actor compiled successfully to: {manager_onnx_path}")
    print("\n🏁 Compilation Complete. Download these files to your local i5 laptop.")

if __name__ == "__main__":
    # Point these to the Fold 4 (best calmar) paths saved in your drive
    # Double check that these string paths perfectly match your Drive directory
    ORACLE_WEIGHTS = "/content/drive/MyDrive/XAU_RL_V3/models/oracle/oracle_fold_4.pth"
    MANAGER_WEIGHTS = "/content/drive/MyDrive/XAU_RL_V3/models/manager/fold_4/best_model.zip"
    OUTPUT_DIRECTORY = "/content/drive/MyDrive/XAU_RL_V3/models/deployed"
    
    export_models_to_onnx(ORACLE_WEIGHTS, MANAGER_WEIGHTS, OUTPUT_DIRECTORY)