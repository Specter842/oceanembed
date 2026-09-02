"""
Physics-consistency loss (CLAUDE.md §5.3).

Uses the `gsw` library (TEOS-10) — do NOT hand-roll the equation of state.
Penalises:
  1. density inversions (density should be non-decreasing with depth)
  2. deviation from TEOS-10 T-S-density relation vs an independent density
     estimate, where available.

Reference (activate in Phase 2, needs torch + gsw):

    def physics_consistency_loss(pred_T, pred_S, pressure_levels):
        density = gsw.rho(pred_S, pred_T, pressure_levels)
        depth_diffs = density[:, 1:] - density[:, :-1]
        inversion_penalty = torch.relu(-depth_diffs).mean()
        return inversion_penalty

Status: STUB — implemented in Phase 2.
"""

from __future__ import annotations
