import torch
import torch.nn as nn

class TemporalAttentionOracle(nn.Module):
    """
    Phase A: Directional Conviction Engine.
    Processes a 30-step sequence buffer to output raw logits for [Hold, Long, Short].
    Designed for seamless ONNX export.
    """
    def __init__(self, input_dim=25, seq_len=30, hidden_dim=64, num_heads=4, num_classes=3):
        super(TemporalAttentionOracle, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        
        # 1. Sequential Feature Extraction
        # Bidirectional allows the network to read the sequence forwards and backwards
        self.gru = nn.GRU(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            batch_first=True, 
            bidirectional=True
        )
        
        # 2. Temporal Self-Attention
        # hidden_dim * 2 because the GRU is bidirectional
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # 3. Non-Linear Classifier Head
        self.fc1 = nn.Linear(hidden_dim * 2, 32)
        self.gelu = nn.GELU() # Smoother gradient flow than standard ReLU
        self.dropout = nn.Dropout(0.3)
        self.fc_out = nn.Linear(32, num_classes) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expects x shape: (batch_size, seq_len, input_dim) -> e.g., (256, 30, 25)
        """
        # Pass through Bidirectional GRU
        gru_out, _ = self.gru(x) # Shape: (batch_size, seq_len, hidden_dim * 2)
        
        # Apply Multi-Head Attention to find relationships between different time steps
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        
        # Pool the sequence by taking the heavily-attended final time step
        # This represents the cumulative "conviction" right now
        context_vector = attn_out[:, -1, :] # Shape: (batch_size, hidden_dim * 2)
        
        # Classification
        out = self.fc1(context_vector)
        out = self.gelu(out)
        out = self.dropout(out)
        
        # Raw Logits (Softmax is applied during ONNX inference or internally by FocalLoss)
        logits = self.fc_out(out) 
        
        return logits