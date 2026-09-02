"""
Pretrained geospatial backbone (CLAUDE.md §5.1).

Primary attempt: ibm-nasa-geospatial/Prithvi-EO-2.0 encoder, used as a FROZEN
feature extractor (this machine is CPU-only, Intel Iris Xe — full fine-tuning is
not feasible; see build notes).
Fallback: torchvision ResNet-50 pretrained, satellite channels stacked as
pseudo-RGB. On CPU this is the realistic primary path.

Transfer learning here = frozen backbone + trained adapter/decoder. This still
matches the abstract ("adapter without retraining the full network").

Status: STUB — implemented in Phase 2. Requires Tier-2 deps (torch, torchvision,
huggingface_hub).
"""

from __future__ import annotations


def build_backbone(kind: str = "resnet50", freeze: bool = True):
    raise NotImplementedError("Phase 2 — see CLAUDE.md §5.1")
