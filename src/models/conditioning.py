"""
Regime conditioning  (CLAUDE.md §5.2).

FiLMLayer is exactly the sketch in §5.2. ConditionEncoder turns the Phase-1
regime signal — one-hot regime class (3) + stratification index (1) + month
sin/cos (2) — into the conditioning vector.

The conditioning vector is derived ONLY from Phase-1 regime features, never from
the satellite channels being used for prediction (that separation is the whole
point — see src/regime/regime_labels.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """Feature-wise linear modulation:  y = x * (1 + gamma(c)) + beta(c)."""

    def __init__(self, feature_dim: int, condition_dim: int):
        super().__init__()
        self.gamma = nn.Linear(condition_dim, feature_dim)
        self.beta = nn.Linear(condition_dim, feature_dim)
        # start as identity so an untrained FiLM layer doesn't wreck the backbone
        nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight); nn.init.zeros_(self.beta.bias)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(condition)
        beta = self.beta(condition)
        if features.dim() == 4:                       # [B, C, H, W]
            gamma = gamma[:, :, None, None]
            beta = beta[:, :, None, None]
        return features * (1.0 + gamma) + beta


class ConditionEncoder(nn.Module):
    """(regime_onehot[3], strat_index[1], month_sincos[2]) -> condition_dim."""

    RAW_DIM = 6

    def __init__(self, condition_dim: int = 32, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.RAW_DIM, hidden),
            nn.GELU(),
            nn.Linear(hidden, condition_dim),
        )
        self.condition_dim = condition_dim

    def forward(self, regime_onehot, strat_index, month_sincos) -> torch.Tensor:
        x = torch.cat([regime_onehot, strat_index, month_sincos], dim=-1)
        return self.net(x)
