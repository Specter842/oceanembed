"""
Phase 3 — validation & evidence  (CLAUDE.md §6).

For each checkpoint x each holdout, compute per-depth-level, per-regime RMSE (T
and S) and Pearson r, and write the single most important artefact of the
project:

    outputs/metrics/per_regime_rmse.csv
        model, test_set, regime_class, depth_level, rmse_T, rmse_S, r_T, r_S

(depth_level = -1 rows are the all-level pooled summary; regime_class = "all"
rows pool every regime.)

Also writes outputs/metrics/predictions_<model>_<set>.npz for the plots, and
outputs/metrics/physics_diagnostics.csv (TEOS-10 density-inversion rate via gsw).

    python -m src.evaluate                         # both models, both holdouts
    python -m src.evaluate --checkpoints oceanembed --test-sets spatial
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config
from .datasets import OceanProfileDataset, NormStats
from .models.oceanembed_model import build_model
from .models.physics_loss import physics_diagnostics_gsw

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "outputs" / "checkpoints"
METRICS = ROOT / "outputs" / "metrics"
TEST_SETS = {"spatial": "test_spatial", "temporal": "test_temporal"}


def load_model(name: str):
    path = CKPT / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} — train it first (see RUN.md)")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config(**blob["cfg"])
    model = build_model(cfg, blob["n_levels"])
    model.load_state_dict(blob["model"])
    model.eval()
    return model, cfg, blob


@torch.no_grad()
def predict(model, ds, mc_dropout=0):
    from torch.utils.data import DataLoader

    stats = ds.stats
    dl = DataLoader(ds, batch_size=128, shuffle=False)
    out = {k: [] for k in ("T_pred", "S_pred", "T_std", "S_std",
                           "T_obs", "S_obs", "mask", "lat", "lon", "regime")}
    for b in dl:
        args = (b["patch"], b["regime_onehot"], b["strat_index"], b["month_sincos"])
        if mc_dropout:
            mT, sT, mS, sS = model.predict_mc(*args, n=mc_dropout)
            std_T = sT * torch.as_tensor(stats.T_std)      # normalised std -> physical
            std_S = sS * torch.as_tensor(stats.S_std)
            pT, pS = mT, mS
        else:
            o = model(*args)
            pT, pS = o["T"], o["S"]
            std_T = torch.exp(0.5 * o["logvar_T"]) * torch.as_tensor(stats.T_std)
            std_S = torch.exp(0.5 * o["logvar_S"]) * torch.as_tensor(stats.S_std)
        out["T_pred"].append((stats.denorm_T(pT)).numpy())
        out["S_pred"].append((stats.denorm_S(pS)).numpy())
        out["T_std"].append(std_T.numpy())
        out["S_std"].append(std_S.numpy())
        out["T_obs"].append(b["raw_T"].numpy())
        out["S_obs"].append(b["raw_S"].numpy())
        out["mask"].append(b["target_mask"].numpy())
        out["lat"].append(b["lat"].numpy())
        out["lon"].append(b["lon"].numpy())
        out["regime"].append(b["regime_idx"].numpy())
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["depth_levels"] = ds.depth_levels
    return res


def _rmse_r(pred, obs, m):
    m = m > 0
    if m.sum() < 3:
        return np.nan, np.nan
    p, o = pred[m], obs[m]
    rmse = float(np.sqrt(np.mean((p - o) ** 2)))
    r = float(np.corrcoef(p, o)[0, 1]) if p.std() > 1e-9 and o.std() > 1e-9 else np.nan
    return rmse, r


def metrics_for(res, model_name, test_set, regime_classes):
    levels = res["depth_levels"]
    rows = []
    regime_names = list(regime_classes) + ["all"]
    for rn in regime_names:
        if rn == "all":
            sel = np.ones(len(res["regime"]), bool)
        else:
            sel = res["regime"] == regime_classes.index(rn)
        if sel.sum() == 0:
            continue
        Tp, To = res["T_pred"][sel], res["T_obs"][sel]
        Sp, So = res["S_pred"][sel], res["S_obs"][sel]
        Mk = res["mask"][sel]
        for li, z in enumerate(levels):
            rt, rrt = _rmse_r(Tp[:, li], To[:, li], Mk[:, li])
            rs, rrs = _rmse_r(Sp[:, li], So[:, li], Mk[:, li])
            rows.append(dict(model=model_name, test_set=test_set, regime_class=rn,
                             depth_level=float(z), rmse_T=rt, rmse_S=rs,
                             r_T=rrt, r_S=rrs, n_profiles=int(sel.sum())))
        # pooled over all levels
        rt, rrt = _rmse_r(Tp, To, Mk)
        rs, rrs = _rmse_r(Sp, So, Mk)
        rows.append(dict(model=model_name, test_set=test_set, regime_class=rn,
                         depth_level=-1.0, rmse_T=rt, rmse_S=rs, r_T=rrt, r_S=rrs,
                         n_profiles=int(sel.sum())))
    return rows


_Z = {0.68: 0.9945, 0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600}


def calibration_for(res, model_name, test_set, regime_classes):
    """Interval-calibration rows: does a nominal-c predicted-std interval
    (mean +/- z(c)*std) actually contain the truth c of the time?"""
    import math
    rows = []
    zs = np.linspace(0.15, 2.7, 20)
    for rn in list(regime_classes) + ["all"]:
        if rn == "all":
            sel = np.ones(len(res["regime"]), bool)
        else:
            sel = res["regime"] == regime_classes.index(rn)
        if sel.sum() == 0:
            continue
        for var in ("T", "S"):
            m = res["mask"][sel] > 0
            err = np.abs(res[f"{var}_pred"][sel][m] - res[f"{var}_obs"][sel][m])
            sd = np.maximum(res[f"{var}_std"][sel][m], 1e-9)
            if err.size < 20:
                continue
            nominal = np.array([math.erf(z / math.sqrt(2)) for z in zs])
            empirical = np.array([float(np.mean(err <= z * sd)) for z in zs])
            ece = float(np.mean(np.abs(empirical - nominal)))
            row = dict(model=model_name, test_set=test_set, regime_class=rn,
                       variable=var, n_points=int(err.size), calib_error=ece)
            for c, z in _Z.items():
                row[f"cover_{int(c*100)}"] = float(np.mean(err <= z * sd))
                row[f"width_{int(c*100)}"] = float(np.mean(2.0 * z * sd))
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", default=["oceanembed", "baseline"])
    ap.add_argument("--test-sets", nargs="+", default=["spatial", "temporal"])
    ap.add_argument("--mc-dropout", type=int, default=0,
                    help="if >0, use N-sample MC-dropout for uncertainty instead of the logvar head")
    args = ap.parse_args()
    METRICS.mkdir(parents=True, exist_ok=True)

    stats = NormStats()
    all_rows, phys_rows, calib_rows = [], [], []
    for name in args.checkpoints:
        model, cfg, blob = load_model(name)
        for ts in args.test_sets:
            ds = OceanProfileDataset(TEST_SETS[ts], stats=stats, augment=False)
            regime_classes = ["well_mixed", "barrier_layer_stratified", "upwelling"]
            res = predict(model, ds, mc_dropout=args.mc_dropout)

            np.savez_compressed(METRICS / f"predictions_{name}_{ts}.npz", **res)
            all_rows += metrics_for(res, name, ts, regime_classes)
            calib_rows += calibration_for(res, name, ts, regime_classes)

            pd_ = physics_diagnostics_gsw(res["T_pred"], res["S_pred"],
                                          res["depth_levels"], res["lon"], res["lat"],
                                          res["mask"])
            phys_rows.append(dict(model=name, test_set=ts, **pd_))
            pooled = [r for r in all_rows if r["model"] == name and r["test_set"] == ts
                      and r["regime_class"] == "barrier_layer_stratified"
                      and r["depth_level"] == -1.0]
            bl = pooled[0] if pooled else {}
            print(f"  {name:10s} {ts:9s}  barrier-layer pooled  "
                  f"RMSE_T={bl.get('rmse_T', float('nan')):.3f}  "
                  f"RMSE_S={bl.get('rmse_S', float('nan')):.3f}  "
                  f"inv={pd_['inversion_fraction']:.3f}")

    df = pd.DataFrame(all_rows)
    df.to_csv(METRICS / "per_regime_rmse.csv", index=False)
    pd.DataFrame(phys_rows).to_csv(METRICS / "physics_diagnostics.csv", index=False)
    pd.DataFrame(calib_rows).to_csv(METRICS / "calibration.csv", index=False)
    print(f"\n  -> {METRICS / 'per_regime_rmse.csv'}  ({len(df)} rows)")
    print(f"  -> {METRICS / 'calibration.csv'}  ({len(calib_rows)} rows)")

    # headline comparison
    head = (df[(df.depth_level == -1)]
            .pivot_table(index=["test_set", "regime_class"], columns="model",
                         values="rmse_T"))
    print("\n  RMSE_T (pooled over depth), by test set x regime:\n")
    print("    " + head.round(3).to_string().replace("\n", "\n    "))


if __name__ == "__main__":
    main()
