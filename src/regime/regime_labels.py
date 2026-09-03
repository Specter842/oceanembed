"""
Regime label pipeline  (CLAUDE.md §4.2).

compute_regime_label(lat, lon, month, river_discharge) -> dict

Inputs used — and ONLY these:
  * static stratification climatology  (data/interim/regime_climatology.nc,
    derived from WOA23 T/S — see src/regime/build_climatology.py)
  * month  (1-12)  and a fixed monsoon-upwelling geography
  * river_discharge  (m^3/s, combined Ganga-Brahmaputra; a STATIC monthly
    climatology from src/data_access/fetch_discharge.py)

This module MUST NOT import or reference any satellite field (fetch_satellite,
SST/SSS/SSH/ADT/ocean_color) or the Argo ground truth (fetch_argo). That is the
fix for the circularity flaw and is enforced by
tests/test_regime_independence.py — a static import check, not this docstring.

Regime classes: "barrier_layer_stratified", "well_mixed", "upwelling".
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from ..data_access.fetch_discharge import discharge_for_month, monthly_climatology

_INTERIM = Path(__file__).resolve().parents[2] / "data" / "interim"
_CLIM_PATH = _INTERIM / "regime_climatology.nc"

# combined Ganga-Brahmaputra annual-mean discharge (m^3/s) — normalising reference
_GB_ANNUAL_MEAN = float(np.mean(list(monthly_climatology().values())))

# classification thresholds
BLT_MIN_M = 12.0          # barrier layer must be at least this thick
STRAT_MIN = 0.35          # and upper ocean at least this stratified
MONSOON_MONTHS = (6, 7, 8, 9)

# fixed summer-monsoon upwelling geography (lat_min, lat_max, lon_min, lon_max)
_UPWELLING_BOXES = (
    ("somali", 2.0, 12.0, 45.0, 55.0),
    ("west_arabian_sea_oman", 8.0, 20.0, 52.0, 62.0),
    ("sw_india_sri_lanka_dome", 4.0, 10.0, 76.0, 88.0),
)


@functools.lru_cache(maxsize=1)
def _clim():
    import xarray as xr

    if not _CLIM_PATH.exists():
        raise FileNotFoundError(
            f"{_CLIM_PATH} missing — run `python -m src.regime.build_climatology` first"
        )
    return xr.open_dataset(_CLIM_PATH, decode_times=False).load()


def _in_upwelling(lat: float, lon: float) -> bool:
    for _name, la0, la1, lo0, lo1 in _UPWELLING_BOXES:
        if la0 <= lat <= la1 and lo0 <= lon <= lo1:
            return True
    return False


def compute_regime_label(lat: float, lon: float, month: int,
                         river_discharge: float | None = None) -> dict:
    """Return {"regime_class", "stratification_index", "blt_m", "mld_m",
    "discharge_anom"} for one space-time point.

    river_discharge: combined Ganga-Brahmaputra discharge in m^3/s. If None, the
    static monthly climatology value for `month` is used.
    """
    month = int(month)
    if not 1 <= month <= 12:
        raise ValueError(f"month must be 1-12, got {month}")
    if river_discharge is None:
        river_discharge = discharge_for_month(month)
    discharge_anom = float(river_discharge) / _GB_ANNUAL_MEAN - 1.0

    c = _clim().sel(month=month,
                    lat=float(lat), lon=float(lon), method="nearest")
    blt = float(c["blt_m"].values)
    mld = float(c["mld_m"].values)
    strat = float(c["strat"].values)
    if not np.isfinite(blt):
        blt = 0.0
    if not np.isfinite(strat):
        strat = 0.0

    # river discharge sharpens stratification in the northern/eastern Bay of Bengal
    strat_index = strat
    if lat >= 8.0 and lon >= 82.0 and discharge_anom > 0:
        strat_index = float(np.clip(strat + 0.15 * discharge_anom, 0.0, 1.0))

    # ---- classify ----
    if month in MONSOON_MONTHS and _in_upwelling(lat, lon) and mld <= 40.0:
        regime = "upwelling"
    elif blt >= BLT_MIN_M and strat_index >= STRAT_MIN:
        regime = "barrier_layer_stratified"
    else:
        regime = "well_mixed"

    return {
        "regime_class": regime,
        "stratification_index": round(strat_index, 4),
        "blt_m": round(blt, 2),
        "mld_m": round(mld, 2) if np.isfinite(mld) else None,
        "discharge_anom": round(discharge_anom, 3),
    }


REGIME_CLASSES = ("well_mixed", "barrier_layer_stratified", "upwelling")


def regime_onehot(regime_class: str) -> np.ndarray:
    v = np.zeros(len(REGIME_CLASSES), dtype="float32")
    v[REGIME_CLASSES.index(regime_class)] = 1.0
    return v


def regime_grid(month: int, step: float = 1.0):
    """lat/lon grid of regime-class indices for a month (for maps). Uses the
    climatology's own extent. Returns an xarray.DataArray of ints (-1 = no data)."""
    import xarray as xr

    c = _clim()
    lats = np.arange(float(c.lat.min()), float(c.lat.max()) + step, step)
    lons = np.arange(float(c.lon.min()), float(c.lon.max()) + step, step)
    disch = discharge_for_month(month)
    g = np.full((len(lats), len(lons)), -1, "int8")
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            cell = c.sel(month=month, lat=la, lon=lo, method="nearest")
            if not np.isfinite(float(cell["mld_m"].values)):
                continue
            lab = compute_regime_label(la, lo, month, disch)
            g[i, j] = REGIME_CLASSES.index(lab["regime_class"])
    return xr.DataArray(g, coords={"lat": lats, "lon": lons}, dims=("lat", "lon"),
                        name="regime_class_idx")


if __name__ == "__main__":
    # quick smoke: N Bay of Bengal in Sep (expect barrier_layer), N Arabian Sea in
    # Feb (expect well_mixed), Somali coast in Aug (expect upwelling).
    for name, la, lo, mo in [
        ("N Bay of Bengal, Sep", 18.0, 89.0, 9),
        ("Central BoB, Nov", 13.0, 87.0, 11),
        ("N Arabian Sea, Feb", 20.0, 64.0, 2),
        ("Somali coast, Aug", 7.0, 51.0, 8),
        ("Equatorial IO, Apr", 0.0, 80.0, 4),
    ]:
        print(f"  {name:24s} -> {compute_regime_label(la, lo, mo)}")
