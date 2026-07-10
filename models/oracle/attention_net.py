import torch
import torch.nn as nn

class TemporalAttentionOracle(nn.Module):
    """
    Phase A: Directional Conviction Engine.
    Processes a 30-step sequence buffer to output raw logits for [Hold, Long, Short].
    Refactored: Decoupled Bidirectional Mature Context Extraction.
    """
    def __init__(self, input_dim=25, seq_len=30, hidden_dim=64, num_heads=4, num_classes=3):
        super(TemporalAttentionOracle, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # Preserved for dimensional splitting
        
        # 1. Sequential Feature Extraction
        self.gru = nn.GRU(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            batch_first=True, 
            bidirectional=True
        )
        
        # 2. Temporal Self-Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # 3. Non-Linear Classifier Head
        self.fc1 = nn.Linear(hidden_dim * 2, 32)
        self.gelu = nn.GELU() 
        self.dropout = nn.Dropout(0.3)
        self.fc_out = nn.Linear(32, num_classes) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expects x shape: (batch_size, seq_len, input_dim)
        """
        # 1. Pass through Bidirectional GRU
        gru_out, _ = self.gru(x) # Shape: (batch_size, seq_len, hidden_dim * 2)
        
        # 2. Decouple Directional Passes
        # PyTorch concatenates forward/backward states along the last dimension
        forward_states = gru_out[:, :, :self.hidden_dim]
        backward_states = gru_out[:, :, self.hidden_dim:]
        
        # 3. Extract Mathematically Mature Contexts
        # Forward pass matures at the end (-1), Backward pass matures at the beginning (0)
        mature_forward = forward_states[:, -1, :] 
        mature_backward = backward_states[:, 0, :] 
        
        # 4. Re-concatenate into the True Context Vector
        # Shape: (batch_size, hidden_dim * 2)
        true_context = torch.cat([mature_forward, mature_backward], dim=-1)
        
        # 5. Targeted Temporal Attention
        # Unsqueeze context to shape (batch_size, 1, hidden_dim * 2) to act as the Query.
        # The full sequence (gru_out) acts as the Keys and Values.
        query = true_context.unsqueeze(1)
        attn_out, _ = self.attention(query, gru_out, gru_out)
        
        # 6. Squeeze back to (batch_size, hidden_dim * 2) for the classification head
        context_vector = attn_out.squeeze(1)
        
        # 7. Classification
        out = self.fc1(context_vector)
        out = self.gelu(out)
        out = self.dropout(out)
        logits = self.fc_out(out) 
        
        return logits