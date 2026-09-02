"""
Pretrained backbone  (CLAUDE.md §5.1).

Registry:
  resnet50   torchvision ResNet-50, ImageNet-pretrained  (default; always works)
  resnet18   smaller, for fast CPU smoke tests
  prithvi    Prithvi-EO-2.0 geospatial foundation model via `terratorch`
             (transfer learning as in the abstract). Needs `pip install
             terratorch` + a weights download; falls back with a logged reason
             if unavailable.

All backbones expose:  .forward(x[B,C,H,W]) -> features[B, feat_dim]
and a `.feat_dim` attribute. `freeze` keeps the pretrained weights fixed and
trains only the adapter/decoder (the honest scoping choice on limited compute —
and how the FiLM adapter is meant to work, §5.2).
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn


class _ChannelAdapter(nn.Module):
    """Learnable 1x1 conv mapping our z-scored ocean channels into the input
    distribution a frozen ImageNet stem expects. Always trainable (even 3->3):
    with a frozen backbone this is the only place the raw channels can be
    re-scaled/mixed. Initialised near identity."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, 3, kernel_size=1)
        with torch.no_grad():
            self.proj.weight.zero_()
            k = min(in_channels, 3)
            for i in range(k):
                self.proj.weight[i, i, 0, 0] = 1.0
            self.proj.bias.zero_()

    def forward(self, x):
        return self.proj(x)


class ResNetBackbone(nn.Module):
    def __init__(self, in_channels=3, arch="resnet50", pretrained=True, freeze=True):
        super().__init__()
        from torchvision import models

        self.adapter = _ChannelAdapter(in_channels)
        if arch == "resnet50":
            w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            net = models.resnet50(weights=w)
            self.feat_dim = 2048
        elif arch == "resnet18":
            w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.resnet18(weights=w)
            self.feat_dim = 512
        else:
            raise ValueError(arch)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.frozen = freeze
        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)
            for p in self.adapter.parameters():          # adapter stays trainable
                p.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:                                  # keep frozen BN in eval
            for m in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
                m.eval()
        return self

    def forward(self, x):
        # NB: no torch.no_grad() here — the backbone params are frozen
        # (requires_grad=False) but activations must keep grad so the learnable
        # channel adapter (and, if unfrozen, the FiLM layer) receive a signal.
        x = self.adapter(x)
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.pool(x).flatten(1)


class PrithviBackbone(nn.Module):
    """Prithvi-EO-2.0 encoder via terratorch. Transfer learning per the abstract."""

    def __init__(self, in_channels=3, variant="prithvi_eo_v2_300", freeze=True):
        super().__init__()
        try:
            from terratorch.registry import BACKBONE_REGISTRY
        except Exception as e:  # noqa: BLE001
            raise ImportError(
                "Prithvi backbone needs terratorch:  pip install terratorch\n"
                f"(import failed: {e})  — use --backbone resnet50 otherwise."
            ) from e

        # Prithvi expects 6 HLS bands; adapt our 3 ocean channels to its stem.
        self.adapter = nn.Conv2d(in_channels, 6, kernel_size=1)
        self.encoder = BACKBONE_REGISTRY.build(
            variant, pretrained=True, num_frames=1,
        )
        self.feat_dim = getattr(self.encoder, "embed_dim", 1024)
        self.frozen = freeze
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        x = self.adapter(x)
        if x.shape[-1] < 224:                              # Prithvi ViT wants >=224
            x = nn.functional.interpolate(x, size=224, mode="bilinear",
                                          align_corners=False)
        feats = self.encoder(x)
        tok = feats[-1] if isinstance(feats, (list, tuple)) else feats
        if tok.dim() == 3:                                 # [B, N, D] -> [B, D]
            tok = tok.mean(dim=1)
        return tok


def build_backbone(name: str, in_channels: int = 3, pretrained: bool = True,
                   freeze: bool = True) -> nn.Module:
    name = name.lower()
    if name in ("resnet50", "resnet18"):
        return ResNetBackbone(in_channels, arch=name, pretrained=pretrained, freeze=freeze)
    if name == "prithvi":
        try:
            return PrithviBackbone(in_channels, freeze=freeze)
        except ImportError as e:
            warnings.warn(f"{e}\nFalling back to resnet50.", stacklevel=2)
            return ResNetBackbone(in_channels, arch="resnet50",
                                  pretrained=pretrained, freeze=freeze)
    raise ValueError(f"unknown backbone {name!r}")
