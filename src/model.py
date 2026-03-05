"""
S&P 500 Regime Detection — Expert-Level MLP Model
==================================================
Feed-Forward Neural Network for binary regime classification.

Architecture features:
- Residual connections (skip connections) for gradient flow
- BatchNorm after each layer for training stability
- GELU activation (smoother than ReLU, used in modern NLP/finance models)
- Dropout for regularization
- Configurable depth and width
"""

import torch
import torch.nn as nn
import numpy as np


class ResidualBlock(nn.Module):
    """
    Residual block with BatchNorm → Linear → GELU → Dropout → Linear → Add.
    Skip connections allow gradients to flow directly, enabling deeper networks.
    """
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.activation(x + self.block(x)))


class RegimeClassifier(nn.Module):
    """
    Expert-level Feed-Forward Neural Network for regime detection.
    
    Architecture:
        Input → BatchNorm → Linear(n_features, hidden) → GELU → Dropout
        → [ResidualBlock × n_blocks]
        → Linear(hidden, hidden//2) → GELU → Dropout
        → Linear(hidden//2, 1) → Sigmoid
    
    The residual blocks allow training deeper networks without vanishing
    gradients, while BatchNorm stabilizes training across different feature scales.
    """
    def __init__(self, n_features: int, hidden_dim: int = 128,
                 n_blocks: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.input_norm = nn.BatchNorm1d(n_features)
        self.input_layer = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),  # Less dropout near output
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        x = self.input_norm(x)
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.output_head(x).squeeze(-1)
    
    def get_architecture_string(self) -> str:
        """Returns a string description for the leaderboard."""
        layers = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                layers.append(f"{module.in_features}→{module.out_features}")
        return " | ".join(layers)


class EarlyStopping:
    """Early stopping with patience and best model restoration."""
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.best_state = None
        self.should_stop = False
    
    def step(self, val_loss: float, model: nn.Module):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
    
    def restore(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
