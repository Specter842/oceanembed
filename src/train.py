"""
Training entrypoint (CLAUDE.md §5.4).

    total_loss = data_loss (MSE vs Argo ground truth)
               + lambda_physics * physics_consistency_loss     # start 0.1

Trains BOTH: OceanEmbed (FiLM + physics) and the baseline (§5.5), same split.
Logs to plain CSV under outputs/metrics/ — no W&B / MLflow.
Checkpoints -> outputs/checkpoints/.

Status: STUB — implemented in Phase 2.
"""

from __future__ import annotations


def main():
    raise NotImplementedError("Phase 2 — see CLAUDE.md §5")


if __name__ == "__main__":
    main()
