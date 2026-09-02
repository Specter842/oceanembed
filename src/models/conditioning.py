"""
FiLM-style regime-conditioning adapter (CLAUDE.md §5.2).

condition_vector = stratification index and/or one-hot regime class from
src/regime/regime_labels.py  —  NOT raw satellite SSS.

Reference implementation (to be activated in Phase 2, needs torch):

    class FiLMLayer(nn.Module):
        def __init__(self, feature_dim, condition_dim):
            super().__init__()
            self.gamma = nn.Linear(condition_dim, feature_dim)
            self.beta  = nn.Linear(condition_dim, feature_dim)
        def forward(self, features, condition_vector):
            gamma = self.gamma(condition_vector)
            beta  = self.beta(condition_vector)
            return features * (1 + gamma) + beta

Status: STUB — implemented in Phase 2.
"""

from __future__ import annotations
