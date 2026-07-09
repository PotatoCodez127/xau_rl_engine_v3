import os
import torch
import numpy as np
from stable_baselines3 import SAC
from models.oracle.attention_net import TemporalAttentionOracle

def export_models_to_onnx(oracle_path: str, manager_path: str, output_dir: str):
    """
    Runs on Colab T4. Compiles the trained PyTorch and SB3 neural architectures 
    into static ONNX computational graphs for your local laptop.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    print(f"🚀 Initiating ONNX Compilation Pipeline on {device}...")

    # --- 1. COMPILE PHASE A (THE ORACLE) ---
    print("\n[1/2] Compiling Temporal Attention Oracle...")
    seq_len = 30
    
    # --- DYNAMIC DIMENSION RESOLUTION ---
    checkpoint = torch.load(oracle_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("✅ Successfully unpacked Oracle weights from comprehensive checkpoint dict.")
    else:
        state_dict = checkpoint
        print("⚠️ Loaded Oracle weights using legacy raw state_dict format.")

    # Dynamically infer input_dim from the GRU weight matrix shape: [hidden_size, input_dim]
    inferred_input_dim = state_dict['gru.weight_ih_l0'].shape[1]
    print(f"🔍 Dynamically inferred Oracle input dimension: {inferred_input_dim}")
    
    oracle = TemporalAttentionOracle(input_dim=inferred_input_dim, seq_len=seq_len).to(device)
    oracle.load_state_dict(state_dict)
    oracle.eval()

    # Create dummy tensor matching the shape
    dummy_oracle_input = torch.randn(1, seq_len, inferred_input_dim, requires_grad=False).to(device)
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

    # Dynamically infer obs_dim from the loaded manager's observation space
    inferred_obs_dim = manager.observation_space.shape[0]
    print(f"🔍 Dynamically inferred SAC Manager observation dimension: {inferred_obs_dim}")

    dummy_sac_input = torch.randn(1, inferred_obs_dim, requires_grad=False).to(device)
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
    print("\n🏁 Compilation Complete. Download these files to your local laptop.")

if __name__ == "__main__":
    # Point these to your Google Drive Colab paths
    ORACLE_WEIGHTS = "/content/drive/MyDrive/XAU_RL_V3/models/oracle/oracle_fold_5.pth" # Update to your champion fold path
    MANAGER_WEIGHTS = "/content/drive/MyDrive/XAU_RL_V3/models/manager/fold_5/best_model.zip" # Update to your champion fold path
    OUTPUT_DIRECTORY = "/content/drive/MyDrive/XAU_RL_V3/models/deployed"
    
    export_models_to_onnx(ORACLE_WEIGHTS, MANAGER_WEIGHTS, OUTPUT_DIRECTORY)