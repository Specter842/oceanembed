"""
Build the regime-conditioning stratification climatology (CLAUDE.md §4.2).

From WOA23 monthly T/S (data/interim/woa23_nio_monthly.nc) compute, per
(month, lat, lon) 1-deg cell:

    mld_m    mixed-layer depth      (de Boyer Montegut: sigma0 threshold 0.03
                                     kg/m^3 referenced to 10 m)
    ild_m    isothermal-layer depth (T(10 m) - 0.5 C threshold)
    blt_m    barrier-layer thickness = max(ild - mld, 0)
    strat    upper-ocean stratification index in [0, 1]
             (normalised sigma0 contrast between 10 m and 60 m)

Output: data/interim/regime_climatology.nc

This is an INDEPENDENT climatology. It never sees satellite SST/SSS/SSH — it is
the fix for the circularity flaw. src/regime/regime_labels.py reads only this
file + the month + the river-discharge climatology.

CLI:  python -m src.regime.build_climatology
"""

from __future__ import annotations

import numpy as np
import xarray as xr
import gsw

from ..data_access.common import INTERIM

WOA = INTERIM / "woa23_nio_monthly.nc"
OUT = INTERIM / "regime_climatology.nc"

REF_DEPTH = 10.0          # reference depth for MLD/ILD criteria
SIGMA_THRESH = 0.03       # kg/m^3
TEMP_THRESH = 0.5         # deg C
STRAT_TOP, STRAT_BOT = 10.0, 60.0
STRAT_SIGMA_SCALE = 2.0   # kg/m^3 contrast that maps to strat = 1.0


def _interp_to(depth_src, arr_1d, target):
    """Linear interp of a 1-D profile (may contain NaNs) to a target depth."""
    m = np.isfinite(arr_1d)
    if m.sum() < 2:
        return np.nan
    return float(np.interp(target, depth_src[m], arr_1d[m],
                           left=np.nan, right=np.nan))


def _layer_depth(depth, value, surf_value, threshold, sign):
    """First depth where sign*(value - surf_value) exceeds threshold (interp).

    Returns NaN if the threshold is never reached within the analysed column
    (i.e. layer is deeper than we can resolve) — the caller treats that as
    'no barrier layer confirmed' rather than inventing a floor value.
    """
    dev = sign * (value - surf_value)
    for i in range(1, len(depth)):
        if not np.isfinite(dev[i]):
            continue
        if dev[i] >= threshold:
            d0, d1 = depth[i - 1], depth[i]
            v0, v1 = dev[i - 1], dev[i]
            if not np.isfinite(v0) or v1 == v0:
                return float(d1)
            return float(d0 + (threshold - v0) * (d1 - d0) / (v1 - v0))
    return np.nan


def build():
    ds = xr.open_dataset(WOA, decode_times=False)
    depth = ds["depth"].values.astype("float64")
    lats = ds["lat"].values.astype("float64")
    lons = ds["lon"].values.astype("float64")
    months = ds["month"].values

    shape = (len(months), len(lats), len(lons))
    mld = np.full(shape, np.nan)
    ild = np.full(shape, np.nan)
    blt = np.full(shape, np.nan)
    strat = np.full(shape, np.nan)

    t_all = ds["t_an"].transpose("month", "lat", "lon", "depth").values
    s_all = ds["s_an"].transpose("month", "lat", "lon", "depth").values

    for mi in range(len(months)):
        for la in range(len(lats)):
            lat = lats[la]
            p = gsw.p_from_z(-depth, lat)
            for lo in range(len(lons)):
                t = t_all[mi, la, lo].astype("float64")
                s = s_all[mi, la, lo].astype("float64")
                if np.isfinite(t).sum() < 3 or np.isfinite(s).sum() < 3:
                    continue
                SA = gsw.SA_from_SP(s, p, lons[lo], lat)
                CT = gsw.CT_from_t(SA, t, p)
                sig0 = gsw.sigma0(SA, CT)

                sig_ref = _interp_to(depth, sig0, REF_DEPTH)
                t_ref = _interp_to(depth, t, REF_DEPTH)
                if not (np.isfinite(sig_ref) and np.isfinite(t_ref)):
                    continue

                mld[mi, la, lo] = _layer_depth(depth, sig0, sig_ref, SIGMA_THRESH, +1.0)
                ild[mi, la, lo] = _layer_depth(depth, t, t_ref, TEMP_THRESH, -1.0)

                s_top = _interp_to(depth, sig0, STRAT_TOP)
                s_bot = _interp_to(depth, sig0, STRAT_BOT)
                if np.isfinite(s_top) and np.isfinite(s_bot):
                    strat[mi, la, lo] = np.clip(
                        (s_bot - s_top) / STRAT_SIGMA_SCALE, 0.0, 1.0)

    # barrier layer where both layers resolved; NaN where no profile at all.
    with np.errstate(invalid="ignore"):
        blt = np.clip(ild - mld, 0.0, 60.0)
    no_profile = ~np.isfinite(mld) & ~np.isfinite(ild)
    blt = np.where(no_profile, np.nan, np.where(np.isfinite(blt), blt, 0.0))

    out = xr.Dataset(
        {
            "mld_m": (("month", "lat", "lon"), mld),
            "ild_m": (("month", "lat", "lon"), ild),
            "blt_m": (("month", "lat", "lon"), blt),
            "strat": (("month", "lat", "lon"), strat),
        },
        coords={"month": months, "lat": lats, "lon": lons},
    )
    out.attrs["source"] = "derived from WOA23 monthly T/S (decav, 1deg)"
    out.attrs["criteria"] = (f"MLD sigma0>{SIGMA_THRESH} ref {REF_DEPTH}m; "
                             f"ILD dT>{TEMP_THRESH} ref {REF_DEPTH}m; "
                             f"strat = d sigma0 ({STRAT_TOP}->{STRAT_BOT} m)/{STRAT_SIGMA_SCALE}")
    out.attrs["circularity"] = "independent of all satellite prediction inputs"
    out.to_netcdf(OUT)

    print(f"  -> {OUT}")
    print(f"  cells with MLD resolved: {np.isfinite(mld).sum()}/{mld.size}")
    print(f"  BLT  mean {np.nanmean(blt):.1f} m   p90 {np.nanpercentile(blt,90):.1f} m   "
          f"max {np.nanmax(blt):.1f} m")
    print(f"  MLD  mean {np.nanmean(mld):.1f} m   ILD mean {np.nanmean(ild):.1f} m")
    print(f"  strat mean {np.nanmean(strat):.2f}   frac>0.5 {(strat>0.5).mean():.2f}")
    ds.close()
    return out


if __name__ == "__main__":
    build()
