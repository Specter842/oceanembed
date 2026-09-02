"""
OceanEmbed model  (CLAUDE.md §5.1).

    patch[B,3,32,32]                        regime_onehot[3], strat[1], month_sincos[2]
          |                                              |
     pretrained backbone                          ConditionEncoder
          |  feats[B, F]                                 |  cond[B, D]
          +------------------  FiLMLayer  <--------------+     (skipped if use_film=False)
                                  |
                          decoder MLP (dropout)
                                  |
             +--------------------+--------------------+
          T[B,L]                S[B,L]           log_var[B, 2, L]   (heteroscedastic)

The baseline model (§5.5) is exactly this with use_film=False and, at train
time, lambda_physics=0 — nothing else differs, so the comparison isolates the
two treatments.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import build_backbone
from .conditioning import ConditionEncoder, FiLMLayer


class OceanEmbedModel(nn.Module):
    def __init__(self, n_levels: int, in_channels: int = 3,
                 backbone: str = "resnet50", pretrained: bool = True,
                 freeze_backbone: bool = True, use_film: bool = True,
                 condition_dim: int = 32, decoder_hidden: int = 512,
                 dropout: float = 0.2):
        super().__init__()
        self.n_levels = n_levels
        self.use_film = use_film

        self.backbone = build_backbone(backbone, in_channels=in_channels,
                                       pretrained=pretrained, freeze=freeze_backbone)
        f = self.backbone.feat_dim

        self.cond_encoder = ConditionEncoder(condition_dim=condition_dim)
        self.film = FiLMLayer(f, condition_dim) if use_film else None

        self.decoder = nn.Sequential(
            nn.Linear(f, decoder_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(decoder_hidden, decoder_hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.head_T = nn.Linear(decoder_hidden, n_levels)
        self.head_S = nn.Linear(decoder_hidden, n_levels)
        self.head_logvar = nn.Linear(decoder_hidden, 2 * n_levels)

    def forward(self, patch, regime_onehot, strat_index, month_sincos):
        feats = self.backbone(patch)
        if self.film is not None:
            cond = self.cond_encoder(regime_onehot, strat_index, month_sincos)
            feats = self.film(feats, cond)
        h = self.decoder(feats)
        logvar = self.head_logvar(h).view(-1, 2, self.n_levels)
        return {
            "T": self.head_T(h),
            "S": self.head_S(h),
            "logvar_T": logvar[:, 0].clamp(-8.0, 8.0),
            "logvar_S": logvar[:, 1].clamp(-8.0, 8.0),
        }

    @torch.no_grad()
    def predict_mc(self, patch, regime_onehot, strat_index, month_sincos, n=20):
        """MC-dropout inference: returns (mean_T, std_T, mean_S, std_S)."""
        was_training = self.training
        self.train()                                     # dropout on
        for m in self.modules():                          # but frozen BN stays eval
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.eval()
        Ts, Ss = [], []
        for _ in range(n):
            o = self.forward(patch, regime_onehot, strat_index, month_sincos)
            Ts.append(o["T"]); Ss.append(o["S"])
        self.train(was_training)
        Ts, Ss = torch.stack(Ts), torch.stack(Ss)
        return Ts.mean(0), Ts.std(0), Ss.mean(0), Ss.std(0)


def build_model(cfg, n_levels: int) -> OceanEmbedModel:
    return OceanEmbedModel(
        n_levels=n_levels,
        in_channels=cfg.in_channels,
        backbone=cfg.backbone,
        pretrained=cfg.pretrained,
        freeze_backbone=cfg.freeze_backbone,
        use_film=cfg.use_film,
        condition_dim=cfg.condition_dim,
        decoder_hidden=cfg.decoder_hidden,
        dropout=cfg.dropout,
    )
