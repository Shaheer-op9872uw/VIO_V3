import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_length, hidden_size)
        
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x)
        x = residual + self.dropout1(attn_output)

        residual = x
        x = self.norm2(x)
        x = residual + self.dropout2(self.ff(x))

        return x
