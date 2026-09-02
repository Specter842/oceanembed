"""
Train OceanEmbed and the baseline  (CLAUDE.md §5.4, §5.5).

Device-aware: uses CUDA + AMP when available (e.g. the Lenovo LOQ), CPU otherwise.

    python -m src.train --models both                 # full run
    python -m src.train --smoke                        # 2-epoch CPU sanity check
    python -m src.train --backbone prithvi --models oceanembed

Outputs:
    outputs/checkpoints/<name>.pt
    outputs/metrics/train_log_<name>.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config, baseline_of, SMOKE
from .datasets import OceanProfileDataset, NormStats
from .models.oceanembed_model import build_model
from .models.physics_loss import physics_consistency_loss

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "outputs" / "checkpoints"
METRICS = ROOT / "outputs" / "metrics"


def masked_gaussian_nll(pred, target, logvar, mask):
    inv = torch.exp(-logvar)
    nll = 0.5 * (inv * (pred - target) ** 2 + logvar)
    return (nll * mask).sum() / mask.sum().clamp_min(1.0)


def masked_rmse(pred, target, mask):
    se = ((pred - target) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)
    return se.sqrt()


def run_epoch(model, loader, stats, cfg, device, optim=None, scaler=None):
    train = optim is not None
    model.train(train)
    tot = {"loss": 0.0, "nll": 0.0, "phys": 0.0, "rmseT": 0.0, "rmseS": 0.0, "n": 0}
    Tm = torch.as_tensor(stats.T_mean, device=device)
    Ts = torch.as_tensor(stats.T_std, device=device)
    Sm = torch.as_tensor(stats.S_mean, device=device)
    Ss = torch.as_tensor(stats.S_std, device=device)

    for b in loader:
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        bs = b["patch"].shape[0]
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type,
                                enabled=(cfg.amp and device.type == "cuda")):
                out = model(b["patch"], b["regime_onehot"],
                            b["strat_index"], b["month_sincos"])
                m = b["target_mask"]
                nll = (masked_gaussian_nll(out["T"], b["target_T"], out["logvar_T"], m)
                       + masked_gaussian_nll(out["S"], b["target_S"], out["logvar_S"], m))

                phys = torch.tensor(0.0, device=device)
                if cfg.lambda_physics > 0:
                    T_phys = out["T"].float() * Ts + Tm
                    S_phys = out["S"].float() * Ss + Sm
                    phys = physics_consistency_loss(
                        T_phys, S_phys, b["pressure"].float(), m,
                        stability_weight=cfg.physics_stability_weight)
                loss = nll + cfg.lambda_physics * phys

        if train:
            optim.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 5.0)
                scaler.step(optim); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 5.0)
                optim.step()

        with torch.no_grad():
            rt = masked_rmse(out["T"].float() * Ts + Tm, b["raw_T"], m)
            rs = masked_rmse(out["S"].float() * Ss + Sm, b["raw_S"], m)
        for k, v in (("loss", loss), ("nll", nll), ("phys", phys),
                     ("rmseT", rt), ("rmseS", rs)):
            tot[k] += float(v.detach()) * bs
        tot["n"] += bs

    n = max(tot["n"], 1)
    return {k: tot[k] / n for k in ("loss", "nll", "phys", "rmseT", "rmseS")}


def train_one(cfg: Config):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)

    stats = NormStats()
    full_aug = OceanProfileDataset("train", stats=stats, augment=cfg.augment)
    full_plain = OceanProfileDataset("train", stats=stats, augment=False)
    n_val = max(1, int(len(full_aug) * cfg.val_frac))
    perm = torch.randperm(len(full_aug),
                          generator=torch.Generator().manual_seed(cfg.seed)).tolist()
    va_idx, tr_idx = perm[:n_val], perm[n_val:]
    tr = torch.utils.data.Subset(full_aug, tr_idx)
    va = torch.utils.data.Subset(full_plain, va_idx)

    dl_tr = DataLoader(tr, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=cfg.num_workers, drop_last=True)
    dl_va = DataLoader(va, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=cfg.num_workers)

    n_levels = len(stats.depth_levels)
    model = build_model(cfg, n_levels).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    optim = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)
    use_scaler = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if use_scaler else None

    print(f"[{cfg.name}] device={device}  backbone={cfg.backbone}  "
          f"use_film={cfg.use_film}  lambda_physics={cfg.lambda_physics}  "
          f"trainable params={n_params:,}  train/val={len(tr)}/{len(va)}")

    log_path = METRICS / f"train_log_{cfg.name}.csv"
    with log_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "split", "loss", "nll", "phys", "rmseT", "rmseS", "lr", "sec"])
        best = float("inf")
        for ep in range(1, cfg.epochs + 1):
            t0 = time.time()
            tr_m = run_epoch(model, dl_tr, stats, cfg, device, optim, scaler)
            va_m = run_epoch(model, dl_va, stats, cfg, device)
            sched.step()
            dt = time.time() - t0
            for split, mm in (("train", tr_m), ("val", va_m)):
                w.writerow([ep, split, *[f"{mm[k]:.5f}" for k in
                            ("loss", "nll", "phys", "rmseT", "rmseS")],
                            f"{optim.param_groups[0]['lr']:.2e}", f"{dt:.1f}"])
            fh.flush()
            flag = ""
            if va_m["loss"] < best:
                best = va_m["loss"]
                torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                            "n_levels": n_levels, "epoch": ep,
                            "val": va_m}, CKPT / f"{cfg.name}.pt")
                flag = "  *"
            print(f"  ep{ep:>3}/{cfg.epochs}  "
                  f"train loss {tr_m['loss']:.4f} rmseT {tr_m['rmseT']:.3f}  |  "
                  f"val loss {va_m['loss']:.4f} rmseT {va_m['rmseT']:.3f} "
                  f"rmseS {va_m['rmseS']:.3f}  ({dt:.0f}s){flag}")
            if not np.isfinite(tr_m["loss"]):
                raise SystemExit("NaN/inf loss — aborting")

    print(f"  -> {CKPT / f'{cfg.name}.pt'}   best val loss {best:.4f}")
    return CKPT / f"{cfg.name}.pt"


def main():
    ap = argparse.ArgumentParser()
    Config.add_args(ap)
    ap.add_argument("--models", choices=["oceanembed", "baseline", "both"],
                    default="oceanembed")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny 2-epoch CPU sanity run")
    ns = ap.parse_args()
    cfg = Config.from_args(ns)
    if ns.smoke:
        for k, v in SMOKE.items():
            setattr(cfg, k, v)

    targets = []
    if ns.models in ("oceanembed", "both"):
        targets.append(cfg)
    if ns.models in ("baseline", "both"):
        b = baseline_of(cfg)
        if ns.smoke:
            b.name = "smoke_baseline"
        targets.append(b)

    for c in targets:
        train_one(c)


if __name__ == "__main__":
    main()
