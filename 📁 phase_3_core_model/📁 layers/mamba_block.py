import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlock(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(MambaBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.mamba_linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.mamba_linear2 = nn.Linear(hidden_size * 4, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-norm input
        residual = x
        x = self.norm1(x)

        # Mamba-style non-linearity (can be upgraded later)
        x = self.mamba_linear1(x)
        x = F.silu(x)  # Swish/SILU activation
        x = self.dropout(x)
        x = self.mamba_linear2(x)

        x = x + residual  # Residual connection

        # Optional second norm (post-layer output)
        x = self.norm2(x)

        return x
