"""
Torch datasets over the Phase-1 npz files + normalisation from norm_stats.json.

Each item:
  patch          [C,P,P]  channel-normalised, invalid pixels set to 0
  regime_onehot  [3]
  strat_index    [1]
  month_sincos   [2]
  target_T,S     [L]      per-level-normalised
  target_mask    [L]
  raw_T, raw_S   [L]      physical units (for physics loss + metrics)
  pressure       [L]      dbar (~depth)
  lat, lon       scalars
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


class NormStats:
    def __init__(self, path: Path | None = None):
        d = json.loads((path or PROCESSED / "norm_stats.json").read_text())
        self.channels = d["channels"]
        self.ch_mean = np.array(d["channel_mean"], "float32")
        self.ch_std = np.array(d["channel_std"], "float32")
        self.depth_levels = np.array(d["depth_levels"], "float32")
        self.T_mean = np.array(d["target_T_mean"], "float32")
        self.T_std = np.array(d["target_T_std"], "float32")
        self.S_mean = np.array(d["target_S_mean"], "float32")
        self.S_std = np.array(d["target_S_std"], "float32")

    def denorm_T(self, x):
        m = torch.as_tensor(self.T_mean, device=x.device)
        s = torch.as_tensor(self.T_std, device=x.device)
        return x * s + m

    def denorm_S(self, x):
        m = torch.as_tensor(self.S_mean, device=x.device)
        s = torch.as_tensor(self.S_std, device=x.device)
        return x * s + m


class OceanProfileDataset(Dataset):
    def __init__(self, split: str, stats: NormStats | None = None,
                 augment: bool = False):
        p = PROCESSED / f"{split}.npz"
        if not p.exists():
            raise FileNotFoundError(f"{p} — run `python -m src.data_pipeline` first")
        z = np.load(p, allow_pickle=True)
        self.n = len(z["profile_id"])
        self.stats = stats or NormStats()
        self.augment = augment

        cm = self.stats.ch_mean[:, None, None]
        cs = self.stats.ch_std[:, None, None]
        self.patch = ((z["patch"].astype("float32") - cm) / cs)
        self.patch_mask = z["patch_mask"].astype("float32")
        self.patch *= self.patch_mask[:, None]                # keep invalid px at 0

        self.regime_onehot = z["regime_onehot"].astype("float32")
        self.strat_index = z["strat_index"].astype("float32")
        self.month_sincos = z["month_sincos"].astype("float32")

        self.raw_T = z["target_T"].astype("float32")
        self.raw_S = z["target_S"].astype("float32")
        self.mask = z["target_mask"].astype("float32")
        self.T = (self.raw_T - self.stats.T_mean) / self.stats.T_std
        self.S = (self.raw_S - self.stats.S_mean) / self.stats.S_std
        self.T = np.nan_to_num(self.T); self.S = np.nan_to_num(self.S)
        self.raw_T = np.nan_to_num(self.raw_T); self.raw_S = np.nan_to_num(self.raw_S)

        self.lat = z["lat"].astype("float32")
        self.lon = z["lon"].astype("float32")
        self.regime_class = z["regime_class"]
        self.pressure = self.stats.depth_levels.copy()        # ~dbar
        self.depth_levels = self.stats.depth_levels.copy()

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        patch = self.patch[i]
        if self.augment:
            if np.random.rand() < 0.5:
                patch = patch[:, :, ::-1].copy()
            if np.random.rand() < 0.5:
                patch = patch[:, ::-1, :].copy()
        return {
            "patch": torch.from_numpy(np.ascontiguousarray(patch)),
            "regime_onehot": torch.from_numpy(self.regime_onehot[i]),
            "strat_index": torch.from_numpy(self.strat_index[i]),
            "month_sincos": torch.from_numpy(self.month_sincos[i]),
            "target_T": torch.from_numpy(self.T[i]),
            "target_S": torch.from_numpy(self.S[i]),
            "target_mask": torch.from_numpy(self.mask[i]),
            "raw_T": torch.from_numpy(self.raw_T[i]),
            "raw_S": torch.from_numpy(self.raw_S[i]),
            "pressure": torch.from_numpy(self.pressure),
            "lat": torch.tensor(self.lat[i]),
            "lon": torch.tensor(self.lon[i]),
            "regime_idx": torch.tensor(int(np.argmax(self.regime_onehot[i]))),
        }
