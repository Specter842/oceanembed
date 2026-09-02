"""Training / model configuration (CLAUDE.md §5)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict


@dataclass
class Config:
    # model
    backbone: str = "resnet50"          # resnet50 | resnet18 | prithvi
    pretrained: bool = True
    freeze_backbone: bool = True
    use_film: bool = True
    condition_dim: int = 32
    decoder_hidden: int = 512
    dropout: float = 0.2
    in_channels: int = 3

    # loss
    lambda_physics: float = 0.1         # §5.4 start value
    physics_stability_weight: float = 0.0

    # optim
    epochs: int = 40
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    val_frac: float = 0.1
    seed: int = 42
    num_workers: int = 0                # Windows-safe default
    amp: bool = True                    # used only on cuda

    # bookkeeping
    name: str = "oceanembed"
    augment: bool = True

    @staticmethod
    def add_args(ap: argparse.ArgumentParser):
        for f, default in asdict(Config()).items():
            if isinstance(default, bool):
                ap.add_argument(f"--{f.replace('_','-')}", dest=f,
                                type=lambda s: s.lower() in ("1", "true", "yes"),
                                default=default)
            else:
                ap.add_argument(f"--{f.replace('_','-')}", dest=f,
                                type=type(default), default=default)

    @classmethod
    def from_args(cls, ns: argparse.Namespace) -> "Config":
        return cls(**{k: getattr(ns, k) for k in asdict(cls())})


def baseline_of(cfg: Config) -> Config:
    """§5.5 baseline: identical, but no FiLM and no physics term."""
    d = asdict(cfg)
    d.update(use_film=False, lambda_physics=0.0, physics_stability_weight=0.0,
             name="baseline")
    return Config(**d)


SMOKE = dict(backbone="resnet18", epochs=2, batch_size=32, augment=False,
             name="smoke")
