%%writefile onnx_inference.py
import os
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import SAC
from models.oracle.attention_net import TemporalAttentionOracle

class PureDeterministicActor(nn.Module):
    """
    A 100% native PyTorch module. Completely severs all ties to SB3's Actor class
    so PyTorch Dynamo doesn't accidentally trace the stochastic distribution methods.
    """
    def __init__(self, features_extractor, latent_pi, mu):
        super(PureDeterministicActor, self).__init__()
        # We only inherit the raw nn.Sequential and nn.Linear layers
        self.features_extractor = features_extractor
        self.latent_pi = latent_pi
        self.mu = mu

    def forward(self, obs):
        features = self.features_extractor(obs)
        latent = self.latent_pi(features)
        mean_actions = self.mu(latent)
        # SAC strictly bounds continuous actions between [-1, 1] using Tanh
        return torch.tanh(mean_actions)

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
    
    input_dim = 29 
    seq_len = 30
    
    oracle = TemporalAttentionOracle(input_dim=input_dim, seq_len=seq_len).to(device)
    
    checkpoint = torch.load(oracle_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        oracle.load_state_dict(checkpoint['model_state_dict'])
    else:
        oracle.load_state_dict(checkpoint)
    
    oracle.eval()

    dummy_oracle_input = torch.randn(1, seq_len, input_dim, requires_grad=False).to(device)
    oracle_onnx_path = os.path.join(output_dir, "oracle_v3.onnx")

    torch.onnx.export(
        oracle,
        dummy_oracle_input,
        oracle_onnx_path,
        export_params=True,
        opset_version=17,          
        do_constant_folding=True,  
        input_names=['sequence_features'],
        output_names=['directional_logits'],
        dynamic_axes={'sequence_features': {0: 'batch_size'}, 'directional_logits': {0: 'batch_size'}}
    )
    print(f"✅ Oracle compiled successfully to: {oracle_onnx_path}")

    # --- 2. COMPILE PHASE B (THE SAC MANAGER) ---
    print("\n[2/2] Compiling SAC Manager (Actor Policy)...")
    
    manager = SAC.load(manager_path, device=device)
    
    # EXTRACT raw PyTorch layers directly out of the SB3 Actor
    raw_extractor = manager.policy.actor.features_extractor
    raw_latent = manager.policy.actor.latent_pi
    raw_mu = manager.policy.actor.mu
    
    # Inject them into our pure native module
    static_actor = PureDeterministicActor(raw_extractor, raw_latent, raw_mu).to(device)
    static_actor.eval()

    obs_dim = 32
    dummy_sac_input = torch.randn(1, obs_dim, requires_grad=False).to(device)
    manager_onnx_path = os.path.join(output_dir, "manager_actor_v3.onnx")

    torch.onnx.export(
        static_actor,
        dummy_sac_input,
        manager_onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['observation_vector'],
        output_names=['continuous_actions'],
        dynamic_axes={'observation_vector': {0: 'batch_size'}, 'continuous_actions': {0: 'batch_size'}}
    )
    print(f"✅ SAC Actor compiled successfully to: {manager_onnx_path}")
    print("\n🏁 Compilation Complete. Download these files to your local i5 laptop.")

if __name__ == "__main__":
    ORACLE_WEIGHTS = "/content/drive/MyDrive/XAU_RL_V3/models/oracle/oracle_fold_5.pth"
    MANAGER_WEIGHTS = "/content/drive/MyDrive/XAU_RL_V3/models/manager/fold_5/best_model.zip"
    OUTPUT_DIRECTORY = "/content/drive/MyDrive/XAU_RL_V3/models/deployed"
    
    export_models_to_onnx(ORACLE_WEIGHTS, MANAGER_WEIGHTS, OUTPUT_DIRECTORY)