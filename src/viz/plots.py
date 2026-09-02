"""
Phase 3 figures  (CLAUDE.md §6.3).

Reads what src/evaluate.py wrote:
    outputs/metrics/per_regime_rmse.csv
    outputs/metrics/predictions_<model>_<set>.npz

Writes:
    outputs/figures/rmse_by_regime.png        bar chart, OceanEmbed vs baseline
    outputs/figures/profiles_barrier_layer.png  pred vs Argo, 3 BoB locations + band
    outputs/figures/uncertainty_map_bob.png   spatial predictive uncertainty, BoB

    python -m src.viz.plots
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "outputs" / "metrics"
FIGS = ROOT / "outputs" / "figures"
REGIMES = ["well_mixed", "barrier_layer_stratified", "upwelling"]


def _load_pred(model, ts):
    p = METRICS / f"predictions_{model}_{ts}.npz"
    return np.load(p, allow_pickle=True) if p.exists() else None


def fig_rmse_by_regime(df):
    sub = df[df.depth_level == -1]
    sets = sorted(sub.test_set.unique())
    models = sorted(sub.model.unique())
    fig, axes = plt.subplots(1, len(sets), figsize=(5.2 * len(sets), 4.2), squeeze=False)
    x = np.arange(len(REGIMES)); w = 0.8 / max(len(models), 1)
    for ax, ts in zip(axes[0], sets):
        for j, mdl in enumerate(models):
            vals = [sub[(sub.test_set == ts) & (sub.model == mdl) &
                        (sub.regime_class == r)]["rmse_T"].mean() for r in REGIMES]
            ax.bar(x + j * w, vals, w, label=mdl)
        ax.set_xticks(x + w * (len(models) - 1) / 2)
        ax.set_xticklabels([r.replace("_", "\n") for r in REGIMES], fontsize=9)
        ax.set_title(f"{ts} holdout")
        ax.set_ylabel("RMSE  T  (deg C)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    fig.suptitle("Temperature RMSE by regime — OceanEmbed vs baseline")
    fig.tight_layout()
    fig.savefig(FIGS / "rmse_by_regime.png", dpi=130)
    plt.close(fig)


def fig_profiles(model="oceanembed", ts="spatial", n=3):
    d = _load_pred(model, ts)
    if d is None:
        return
    lv = d["depth_levels"]
    bl = np.where(d["regime"] == REGIMES.index("barrier_layer_stratified"))[0]
    if len(bl) == 0:
        return
    # pick profiles with the deepest coverage, spread across the basin
    cover = d["mask"][bl].sum(1)
    pick = bl[np.argsort(-cover)[:max(n * 4, n)]]
    pick = sorted(pick, key=lambda i: d["lon"][i])[:: max(1, len(pick) // n)][:n]

    fig, axes = plt.subplots(1, len(pick), figsize=(3.4 * len(pick), 5.4), squeeze=False)
    for ax, i in zip(axes[0], pick):
        m = d["mask"][i] > 0
        ax.plot(d["T_obs"][i][m], lv[m], "k-o", ms=3, label="Argo")
        ax.plot(d["T_pred"][i][m], lv[m], "C0-", label="OceanEmbed")
        lo = d["T_pred"][i] - 2 * d["T_std"][i]
        hi = d["T_pred"][i] + 2 * d["T_std"][i]
        ax.fill_betweenx(lv[m], lo[m], hi[m], color="C0", alpha=0.2, label="±2σ")
        ax.invert_yaxis()
        ax.set_title(f"{d['lat'][i]:.1f}°N {d['lon'][i]:.1f}°E", fontsize=10)
        ax.set_xlabel("T (°C)"); ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("depth (m)")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"Barrier-layer profiles — {model}, {ts} holdout")
    fig.tight_layout()
    fig.savefig(FIGS / "profiles_barrier_layer.png", dpi=130)
    plt.close(fig)


def fig_uncertainty_map(model="oceanembed", ts="spatial"):
    d = _load_pred(model, ts)
    if d is None:
        return
    lv = d["depth_levels"]
    k = int(np.argmin(np.abs(lv - 100)))            # uncertainty at ~100 m
    unc = d["T_std"][:, k]
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sc = ax.scatter(d["lon"], d["lat"], c=unc, s=18, cmap="viridis")
    fig.colorbar(sc, label=f"predictive σ(T) at {lv[k]:.0f} m  (°C)")
    ax.set_xlabel("lon (°E)"); ax.set_ylabel("lat (°N)")
    ax.set_title(f"Predictive uncertainty — {model}, {ts} holdout")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "uncertainty_map_bob.png", dpi=130)
    plt.close(fig)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    csv = METRICS / "per_regime_rmse.csv"
    if not csv.exists():
        raise SystemExit("run `python -m src.evaluate` first")
    df = pd.read_csv(csv)
    fig_rmse_by_regime(df)
    fig_profiles()
    fig_uncertainty_map()
    made = sorted(p.name for p in FIGS.glob("*.png"))
    print("  wrote:", ", ".join(made))


if __name__ == "__main__":
    main()
