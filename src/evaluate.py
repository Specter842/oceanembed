"""
Validation & evidence (CLAUDE.md §6).

For BOTH models, on BOTH test_spatial.npz and test_temporal.npz:
  - RMSE per depth level (T and S separately)
  - RMSE by regime_class  <-- the single most important output
  - Pearson r between predicted and observed profiles

Writes outputs/metrics/per_regime_rmse.csv with columns:
  model, test_set, regime_class, depth_level, rmse_T, rmse_S, r_T, r_S

Honesty rule (§6.2): if OceanEmbed does NOT beat baseline in
barrier_layer_stratified rows, report it — do not retrain until it wins.

Status: STUB — implemented in Phase 3.
"""

from __future__ import annotations


def main():
    raise NotImplementedError("Phase 3 — see CLAUDE.md §6")


if __name__ == "__main__":
    main()
