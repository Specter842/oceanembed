"""
OceanEmbed model assembly (CLAUDE.md §5.1).

    satellite channel stack [SST, SSH, SSS, ocean_color] @ (lat, lon, week)
        -> frozen pretrained backbone
        -> FiLM conditioning (regime label / stratification index from Phase 1)
        -> decoder (small MLP or transposed-conv) -> per-depth (T, S)
        -> uncertainty head (log-variance per level, or MC-dropout at inference)

Baseline variant (CLAUDE.md §5.5): same architecture, NO FiLM, NO physics loss,
single global model, same data split. Must exist before Phase 3.

Status: STUB — implemented in Phase 2.
"""

from __future__ import annotations


def build_oceanembed(use_film: bool = True):
    raise NotImplementedError("Phase 2 — see CLAUDE.md §5.1")


def build_baseline():
    """Ablation baseline: no FiLM, no physics loss (CLAUDE.md §5.5)."""
    raise NotImplementedError("Phase 2 — see CLAUDE.md §5.5")
