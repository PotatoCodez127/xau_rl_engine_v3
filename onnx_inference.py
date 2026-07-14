import os
import torch
import torch.nn as nn
from stable_baselines3 import SAC

# Ensure the local environment path is set if running in Colab
import sys
if '/content/drive/MyDrive/XAU_RL_V3' not in sys.path:
    sys.path.append('/content/drive/MyDrive/XAU_RL_V3')

from models.oracle.attention_net import TemporalAttentionOracle

class BulletproofSACActor(nn.Module):
    """
    A 100% hand-built native PyTorch network. 
    By manually copying the weights, we mathematically sever all ties to Stable-Baselines3.
    This guarantees PyTorch Dynamo will compile the ONNX graph flawlessly.
    """
    def __init__(self, obs_dim=32, action_dim=2):
        super(BulletproofSACActor, self).__init__()
        # SB3 Default SAC Architecture: Flatten -> Linear(256) -> ReLU -> Linear(256) -> ReLU
        self.latent_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        # Final Mean Action Layer
        self.mu = nn.Linear(256, action_dim)

    def forward(self, obs):
        latent = self.latent_net(obs)
        mean_actions = self.mu(latent)
        # SAC strictly bounds continuous actions between [-1.0, 1.0] using Tanh
        return torch.tanh(mean_actions)

def export_models_to_onnx(oracle_path: str, manager_path: str, output_dir: str):
    """
    Compiles the trained PyTorch and SB3 neural architectures 
    into static ONNX computational graphs for local deployment.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(f"🚀 Initiating Hard-Copy ONNX Compilation Pipeline on {device}...")

    # ==========================================
    # 1. COMPILE PHASE A (THE ORACLE)
    # ==========================================
    print("\n[1/2] Compiling Temporal Attention Oracle...")
    
    input_dim = 29  # Adjusted for Priority 4 Macro Leakage Prevention
    seq_len = 30
    
    oracle = TemporalAttentionOracle(input_dim=input_dim, seq_len=seq_len).to(device)
    
    # Safely unpack the dictionary checkpoint
    checkpoint = torch.load(oracle_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        oracle.load_state_dict(checkpoint['model_state_dict'])
        print("Successfully unpacked dictionary checkpoint.")
    else:
        oracle.load_state_dict(checkpoint)
        
    oracle.eval()

    dummy_oracle_input = torch.randn(1, seq_len, input_dim).to(device)
    oracle_onnx_path = os.path.join(output_dir, "oracle_v3.onnx")

    # Disable gradient tracking for safe export
    with torch.no_grad():
        torch.onnx.export(
            oracle, 
            dummy_oracle_input, 
            oracle_onnx_path,
            export_params=True, 
            opset_version=14, 
            do_constant_folding=True,  
            input_names=['sequence_features'], 
            output_names=['directional_logits'],
            dynamic_axes={'sequence_features': {0: 'batch_size'}, 'directional_logits': {0: 'batch_size'}}
        )
    print(f"✅ Oracle compiled successfully to: {oracle_onnx_path}")


    # ==========================================
    # 2. COMPILE PHASE B (THE SAC MANAGER)
    # ==========================================
    print("\n[2/2] Compiling Bulletproof SAC Manager (Actor Policy)...")
    
    manager = SAC.load(manager_path, device=device)
    sb3_actor = manager.policy.actor
    
    # Instantiate our purely native PyTorch Network
    obs_dim = 32
    bulletproof_actor = BulletproofSACActor(obs_dim=obs_dim, action_dim=2).to(device)
    
    # SURGERY: Manually hard-copy the exact weights from SB3 into our Native Network
    bulletproof_actor.latent_net[1].weight.data = sb3_actor.latent_pi[0].weight.data.clone()
    bulletproof_actor.latent_net[1].bias.data = sb3_actor.latent_pi[0].bias.data.clone()
    
    bulletproof_actor.latent_net[3].weight.data = sb3_actor.latent_pi[2].weight.data.clone()
    bulletproof_actor.latent_net[3].bias.data = sb3_actor.latent_pi[2].bias.data.clone()
    
    bulletproof_actor.mu.weight.data = sb3_actor.mu.weight.data.clone()
    bulletproof_actor.mu.bias.data = sb3_actor.mu.bias.data.clone()
    
    bulletproof_actor.eval()
    
    dummy_sac_input = torch.randn(1, obs_dim).to(device)
    manager_onnx_path = os.path.join(output_dir, "manager_actor_v3.onnx")

    with torch.no_grad():
        torch.onnx.export(
            bulletproof_actor, 
            dummy_sac_input, 
            manager_onnx_path,
            export_params=True, 
            opset_version=14, 
            do_constant_folding=True,
            input_names=['observation_vector'], 
            output_names=['continuous_actions'],
            dynamic_axes={'observation_vector': {0: 'batch_size'}, 'continuous_actions': {0: 'batch_size'}}
        )
    print(f"✅ SAC Actor compiled successfully to: {manager_onnx_path}")
    print("\n🏁 Compilation Complete. Download these files to your local machine.")

if __name__ == "__main__":
    ORACLE_WEIGHTS = "/content/drive/MyDrive/XAU_RL_V3/models/oracle/oracle_fold_5.pth"
    MANAGER_WEIGHTS = "/content/drive/MyDrive/XAU_RL_V3/models/manager/fold_5/best_model.zip"
    OUTPUT_DIRECTORY = "/content/drive/MyDrive/XAU_RL_V3/models/deployed"
    
    export_models_to_onnx(ORACLE_WEIGHTS, MANAGER_WEIGHTS, OUTPUT_DIRECTORY)