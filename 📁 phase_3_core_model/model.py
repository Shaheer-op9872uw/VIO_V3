import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config
from mamba_block import MambaBlock
from transformer_block import TransformerBlock


class VIOModel(nn.Module):
    def __init__(self, vocab_size):
        super(VIOModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = config.get("embedding_dim", 512)
        self.embedding = self._build_embedding_layer()
        self.layers = self._build_layers()
        self.norm = nn.LayerNorm(self.embedding_dim)
        self.output = self._build_output_head()

    def _build_embedding_layer(self):
        return nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim
        )

    def _build_layers(self):
        layer_type = config.get("block_type", "mamba")
        num_layers = config.get("num_layers", 6)
        layers = nn.ModuleList()

        for i in range(num_layers):
            if layer_type == "mamba":
                layers.append(MambaBlock(
                    d_model=self.embedding_dim,
                    d_state=config.get("mamba_d_state", 16),
                    d_conv=config.get("mamba_d_conv", 4)
                ))
            elif layer_type == "transformer":
                layers.append(TransformerBlock(
                    d_model=self.embedding_dim,
                    num_heads=config.get("num_heads", 8),
                    dim_feedforward=config.get("dim_feedforward", 2048)
                ))
            else:
                raise ValueError(f"Unknown block_type '{layer_type}' in config.")
        return layers

    def _build_output_head(self):
        return nn.Linear(self.embedding_dim, self.vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    def generate(self, prompt_tensor, max_length=50, temperature=1.0, top_k=0):
        self.eval()
        generated = prompt_tensor.clone()

        for _ in range(max_length):
            with torch.no_grad():
                logits = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature

                if top_k > 0:
                    values, indices = torch.topk(next_token_logits, top_k)
                    probs = torch.zeros_like(next_token_logits).scatter_(
                        dim=-1, index=indices, src=F.softmax(values, dim=-1)
                    )
                else:
                    probs = F.softmax(next_token_logits, dim=-1)

                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)

        return generated

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)

    def log_model_info(self):
        print(f"Model Type: {'Mamba' if isinstance(self.layers[0], MambaBlock) else 'Transformer'}")
        print(f"Number of layers: {len(self.layers)}")
        print(f"Embedding dim: {self.embedding_dim}")
        print(f"Total parameters: {self.count_parameters():,}")


# Integration utilities for Phase 4: train.py and evaluate.py
def get_model_and_optimizer(vocab_size):
    model = VIOModel(vocab_size)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 3e-4),
        weight_decay=config.get("weight_decay", 0.01)
    )
    return model, optimizer


def get_scheduler(optimizer, total_steps):
    warmup_steps = config.get("warmup_steps", 100)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min((step + 1) / warmup_steps, 1.0)
    )
    return scheduler


# Compatibility check
def sanity_check_model(model, tokenizer):
    test_input = torch.randint(0, tokenizer.vocab_size, (2, 16))
    try:
        output = model(test_input)
        print(f"Sanity check passed: output shape = {output.shape}")
    except Exception as e:
        print(f"Sanity check failed: {e}")
