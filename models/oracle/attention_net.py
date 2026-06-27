import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of the
    tokens in the sequence. Attention is permutation-invariant, so it needs
    this to understand time.
    """

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return x


class TemporalAttentionOracle(nn.Module):
    def __init__(self, input_dim, seq_len=30, num_heads=4, hidden_dim=64):
        super(TemporalAttentionOracle, self).__init__()

        # --- NEW: Input Projection Layer ---
        # Decouples raw CSV feature count from Transformer dimensions
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # --- NEW: Learnable [CLS] Token ---
        # Sized to match the projected hidden_dim space
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Positional Encoding to ensure chronology is respected
        self.pos_encoder = PositionalEncoding(d_model=hidden_dim, max_len=seq_len + 1)

        # Attention now operates purely on hidden_dim (64), making it perfectly divisible by num_heads (4)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # The classifier now only takes the extracted output of the [CLS] token
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 3),  # 3 Classes: Hold, Long, Short
        )

    def forward(self, x):
        # Raw x shape: (batch_size, seq_len, input_dim)
        batch_size = x.shape[0]

        # 1. Project raw features to standard Transformer dimensional space
        # New shape: (batch_size, seq_len, hidden_dim)
        x = self.input_proj(x)

        # 2. Expand and prepend the [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat(
            (cls_tokens, x), dim=1
        )  # Shape: (batch_size, seq_len + 1, hidden_dim)

        # 3. Apply Positional Encoding
        x = self.pos_encoder(x)

        # 4. Pass through Temporal Attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # 5. Feed-Forward Network
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        # 6. Extraction: Pull ONLY the state of the [CLS] token (index 0)
        cls_state = x[:, 0, :]

        # 7. Final Classification
        logits = self.classifier(cls_state)
        return logits
