"""
Bake the NetCDF inputs the dashboard needs into one numpy bundle, so the
dashboard itself depends only on numpy / pandas / plotly / streamlit — no
netCDF4, xarray, gsw or torch. Run once after the pipeline:

    python -m dashboard.prep_assets

Writes dashboard/assets.npz  (a few MB).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
SAT = ROOT / "data" / "interim" / "satellite_weekly.nc"
CLIM = ROOT / "data" / "interim" / "regime_climatology.nc"
OUT = ROOT / "dashboard" / "assets.npz"
COARSEN = 2                     # spatial downsample factor for the map panels


def main():
    d = {}

    ds = xr.open_dataset(SAT).coarsen(lat=COARSEN, lon=COARSEN, boundary="trim").mean()
    d["sat_weeks"] = ds["week_start"].values.astype("datetime64[D]").astype(str)
    d["sat_lat"] = ds["lat"].values.astype("float32")
    d["sat_lon"] = ds["lon"].values.astype("float32")
    for v in ("sst", "ssh", "sss"):
        d[f"sat_{v}"] = ds[v].values.astype("float32")
    ds.close()

    c = xr.open_dataset(CLIM, decode_times=False)
    d["clim_month"] = c["month"].values.astype("int16")
    d["clim_lat"] = c["lat"].values.astype("float32")
    d["clim_lon"] = c["lon"].values.astype("float32")
    for v in ("mld_m", "ild_m", "blt_m", "strat"):
        d[f"clim_{v}"] = c[v].values.astype("float32")
    c.close()

    from src.regime.regime_labels import regime_grid, REGIME_CLASSES
    grids = [regime_grid(m, step=1.0) for m in range(1, 13)]
    d["regime_classes"] = np.array(REGIME_CLASSES)
    d["regime_grid"] = np.stack([g.values for g in grids]).astype("int8")   # (12,lat,lon)
    d["regime_grid_lat"] = grids[0].lat.values.astype("float32")
    d["regime_grid_lon"] = grids[0].lon.values.astype("float32")

    np.savez_compressed(OUT, **d)
    mb = OUT.stat().st_size / 1e6
    print(f"  -> {OUT}  ({mb:.1f} MB)  "
          f"sat {d['sat_sst'].shape}  regime_grid {d['regime_grid'].shape}")


if __name__ == "__main__":
    main()
