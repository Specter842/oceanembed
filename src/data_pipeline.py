"""
Phase 1 matching + holdout splits  (CLAUDE.md §4.3).

Builds, from
    data/interim/argo_profiles.parquet      (ground truth)
    data/interim/satellite_weekly.nc        (SST/SSH/SSS weekly, 0.25 deg)
    src/regime/regime_labels.py             (regime conditioning)

the training-ready tensors:
    data/processed/train.npz
    data/processed/test_spatial.npz         (Bay of Bengal: lat>=5 AND lon>=85)
    data/processed/test_temporal.npz        (one full monsoon: JJAS 2022)
    data/processed/norm_stats.json

Each row =
    patch          float32 [C=3, P, P]   SST,SSH,SSS window centred on the profile
    patch_mask     float32 [P, P]        1 where satellite data is valid
    regime_onehot  float32 [3]
    strat_index    float32 [1]
    month_sincos   float32 [2]
    target_T       float32 [L]           Argo T on standard depth levels
    target_S       float32 [L]
    target_mask    float32 [L]           1 where the profile actually covers that level
  + metadata: lat, lon, week_start, profile_id, regime_class

Splits are made by physically partitioning profiles BEFORE writing — never a
random split (that is the whole point of the holdouts).

CLI:  python -m src.data_pipeline --patch 32
"""

from __future__ import annotations

import argparse
import json

import sys

import numpy as np
import pandas as pd
import xarray as xr

try:                                    # Windows console is cp1252 by default
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                       # noqa: BLE001
    pass

from .data_access.common import PROCESSED, INTERIM, STD_DEPTHS, in_bob_holdout
from .data_access.fetch_discharge import discharge_for_month
from .regime.regime_labels import compute_regime_label, regime_onehot, REGIME_CLASSES

ARGO = INTERIM / "argo_profiles.parquet"
SAT = INTERIM / "satellite_weekly.nc"

MIN_TOP_LEVELS = 12          # profile must cover at least the top 12 std levels (->300 m)
MATCH_DDEG = 0.5             # spatial tolerance (deg) for "co-located"
TEMPORAL_HOLDOUT = (2022, (6, 7, 8, 9))     # JJAS 2022
CHANNELS = ("sst", "ssh", "sss")

# physically valid ranges — values outside become NaN (coastal / RFI / tide artefacts)
CH_VALID = {"sst": (10.0, 36.0), "ssh": (-1.5, 1.5), "sss": (20.0, 41.0)}


# --------------------------------------------------------------------------- #
# profile -> standard-level T/S
# --------------------------------------------------------------------------- #

def _profile_to_levels(pres, temp, psal, levels):
    """Linear-interp one profile onto `levels`; mask = coverage."""
    order = np.argsort(pres)
    p, t, s = pres[order], temp[order], psal[order]
    keep = np.isfinite(p) & np.isfinite(t) & np.isfinite(s)
    p, t, s = p[keep], t[keep], s[keep]
    if p.size < 5:
        return None
    # collapse duplicate pressures
    p, idx = np.unique(p, return_index=True)
    t, s = t[idx], s[idx]
    T = np.full(len(levels), np.nan, "float32")
    S = np.full(len(levels), np.nan, "float32")
    M = np.zeros(len(levels), "float32")
    for i, z in enumerate(levels):
        if z < p[0] - 5 or z > p[-1] + 10:
            continue
        T[i] = np.interp(z, p, t)
        S[i] = np.interp(z, p, s)
        M[i] = 1.0
    return T, S, M


# --------------------------------------------------------------------------- #
# main build
# --------------------------------------------------------------------------- #

def build(patch: int = 32):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    levels = np.array(STD_DEPTHS, "float32")

    sat = xr.open_dataset(SAT)
    sat_weeks = pd.to_datetime(sat["week_start"].values)
    slat = sat["lat"].values
    slon = sat["lon"].values
    dlat = float(np.abs(np.diff(slat)).mean())
    dlon = float(np.abs(np.diff(slon)).mean())
    half = patch // 2

    # reflect-pad every channel once so all patches are in-bounds
    padded = {}
    for ch in CHANNELS:
        if ch not in sat:
            raise SystemExit(f"satellite_weekly.nc missing channel '{ch}'")
        arr = sat[ch].values.astype("float32")            # [week, lat, lon]
        lo, hi = CH_VALID[ch]
        arr = np.where((arr >= lo) & (arr <= hi), arr, np.nan)
        padded[ch] = np.pad(arr, ((0, 0), (half, half), (half, half)),
                            mode="reflect")

    prof = pd.read_parquet(ARGO)
    if not np.issubdtype(prof["time"].dtype, np.datetime64):
        prof["time"] = pd.to_datetime(prof["time"], cache=False)

    # collapse to one metadata row + numpy level-arrays per profile (much lighter
    # than holding 7 M-row groupby state while we build patches)
    meta = (prof.groupby("profile_id", sort=False)
            .agg(lat=("lat", "first"), lon=("lon", "first"), time=("time", "first")))
    lv = {pid: (g["pres"].to_numpy("float64"),
                g["temp"].to_numpy("float64"),
                g["psal"].to_numpy("float64"))
          for pid, g in prof.groupby("profile_id", sort=False)}
    del prof
    print(f"{len(meta):,} Argo profiles; {len(sat_weeks)} satellite weeks")

    wk_mid = sat_weeks + pd.Timedelta(days=3)          # week-centre timestamps
    rows = []
    n_no_depth = n_no_week = n_no_sat = 0

    for pid, m in meta.iterrows():
        lat = float(m["lat"])
        lon = float(m["lon"])
        t = m["time"]
        pres_a, temp_a, psal_a = lv[pid]

        res = _profile_to_levels(pres_a, temp_a, psal_a, levels)
        if res is None or res[2][:MIN_TOP_LEVELS].sum() < MIN_TOP_LEVELS:
            n_no_depth += 1
            continue
        T, S, M = res

        # nearest satellite week (within 4 days of a week centre => same 7-day bin)
        k = int(np.argmin(np.abs(wk_mid - t)))
        if abs((wk_mid[k] - t).total_seconds()) > 4 * 86400:
            n_no_week += 1
            continue

        iy = int(np.argmin(np.abs(slat - lat)))
        ix = int(np.argmin(np.abs(slon - lon)))
        if abs(slat[iy] - lat) > MATCH_DDEG or abs(slon[ix] - lon) > MATCH_DDEG:
            n_no_sat += 1
            continue

        patch_stack = np.stack([
            padded[ch][k, iy:iy + patch, ix:ix + patch] for ch in CHANNELS
        ])                                                # [C, P, P]
        pmask = np.isfinite(patch_stack).all(axis=0).astype("float32")
        if pmask.mean() < 0.30:                           # mostly land/no-data
            n_no_sat += 1
            continue
        # centre pixel must be valid
        if not np.isfinite(patch_stack[:, half, half]).all():
            n_no_sat += 1
            continue

        month = int(t.month)
        lab = compute_regime_label(lat, lon, month, discharge_for_month(month))

        rows.append(dict(
            profile_id=pid, lat=lat, lon=lon,
            week_start=sat_weeks[k], year=int(t.year), month=month,
            regime_class=lab["regime_class"],
            strat_index=np.float32(lab["stratification_index"]),
            patch=np.nan_to_num(patch_stack, nan=0.0).astype("float32"),
            patch_mask=pmask,
            target_T=T, target_S=S, target_mask=M,
        ))

    sat.close()
    print(f"  kept {len(rows):,}   (dropped: depth {n_no_depth:,}, "
          f"week {n_no_week:,}, satellite {n_no_sat:,})")
    if not rows:
        raise SystemExit("no matched rows — cannot continue")

    df = pd.DataFrame(rows)

    # ---- physical splits (profiles, not random) ----
    is_spatial = in_bob_holdout(df["lat"].to_numpy(), df["lon"].to_numpy())
    ty, tmonths = TEMPORAL_HOLDOUT
    is_temporal = (~is_spatial) & (df["year"].to_numpy() == ty) & \
                  (df["month"].isin(tmonths).to_numpy())
    is_train = ~is_spatial & ~is_temporal

    split_of = np.where(is_spatial, "spatial",
                        np.where(is_temporal, "temporal", "train"))
    df["split"] = split_of

    _write("train.npz", df[is_train], levels)
    _write("test_spatial.npz", df[is_spatial], levels)
    _write("test_temporal.npz", df[is_temporal], levels)

    _assert_disjoint(df)
    _write_norm_stats(df[is_train], levels)
    _summary(df)
    return df


def _stack(sub: pd.DataFrame):
    return dict(
        patch=np.stack(sub["patch"].to_list()).astype("float32"),
        patch_mask=np.stack(sub["patch_mask"].to_list()).astype("float32"),
        regime_onehot=np.stack([regime_onehot(c) for c in sub["regime_class"]]),
        strat_index=sub["strat_index"].to_numpy("float32")[:, None],
        month_sincos=np.stack([
            [np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)]
            for m in sub["month"]
        ]).astype("float32"),
        target_T=np.stack(sub["target_T"].to_list()).astype("float32"),
        target_S=np.stack(sub["target_S"].to_list()).astype("float32"),
        target_mask=np.stack(sub["target_mask"].to_list()).astype("float32"),
        lat=sub["lat"].to_numpy("float32"),
        lon=sub["lon"].to_numpy("float32"),
        month=sub["month"].to_numpy("int16"),
        year=sub["year"].to_numpy("int16"),
        week_start=sub["week_start"].astype("int64").to_numpy(),
        regime_class=sub["regime_class"].to_numpy(dtype=object),
        profile_id=sub["profile_id"].to_numpy(dtype=object),
    )


def _write(name: str, sub: pd.DataFrame, levels):
    path = PROCESSED / name
    if len(sub) == 0:
        print(f"  !! {name}: 0 rows")
    payload = _stack(sub) if len(sub) else {}
    np.savez_compressed(path, depth_levels=levels, regime_classes=np.array(REGIME_CLASSES),
                        **payload)
    print(f"  -> {name}: {len(sub):,} rows")


def _assert_disjoint(df: pd.DataFrame):
    sets = {s: set(df.loc[df["split"] == s, "profile_id"]) for s in
            ("train", "spatial", "temporal")}
    for a in sets:
        for b in sets:
            if a < b:
                inter = sets[a] & sets[b]
                assert not inter, f"OVERLAP {a} & {b}: {len(inter)} profiles"
    print("  [ok] splits are profile-disjoint")


def _write_norm_stats(train: pd.DataFrame, levels):
    patches = np.stack(train["patch"].to_list())          # [N,C,P,P]
    masks = np.stack(train["patch_mask"].to_list())[:, None]
    ch_mean, ch_std = [], []
    for c in range(patches.shape[1]):
        vals = patches[:, c][masks[:, 0] > 0]
        ch_mean.append(float(vals.mean()))
        ch_std.append(float(vals.std() + 1e-6))
    Mk = np.stack(train["target_mask"].to_list()) > 0
    Tm = np.ma.masked_array(np.stack(train["target_T"].to_list()), mask=~Mk)
    Sm = np.ma.masked_array(np.stack(train["target_S"].to_list()), mask=~Mk)
    t_mean = Tm.mean(axis=0).filled(0.0)
    s_mean = Sm.mean(axis=0).filled(0.0)
    t_std = Tm.std(axis=0).filled(1.0) + 1e-6
    s_std = Sm.std(axis=0).filled(1.0) + 1e-6
    stats = dict(
        channels=list(CHANNELS),
        channel_mean=ch_mean, channel_std=ch_std,
        depth_levels=[float(z) for z in levels],
        target_T_mean=[float(x) for x in t_mean], target_T_std=[float(x) for x in t_std],
        target_S_mean=[float(x) for x in s_mean], target_S_std=[float(x) for x in s_std],
        n_train=int(len(train)),
    )
    (PROCESSED / "norm_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"  -> norm_stats.json  (channel_mean={[round(x,2) for x in ch_mean]})")


def _summary(df: pd.DataFrame):
    print("\n  split x regime:")
    tab = df.groupby(["split", "regime_class"]).size().unstack(fill_value=0)
    print("    " + tab.to_string().replace("\n", "\n    "))
    print(f"\n  year span: {df['year'].min()}-{df['year'].max()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", type=int, default=32)
    build(patch=ap.parse_args().patch)


if __name__ == "__main__":
    main()
