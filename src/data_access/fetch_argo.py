"""
Fetch Argo T/S/pressure profiles for the North Indian Ocean (CLAUDE.md §3.1, §4).

Source: AOML ERDDAP tabledap `argo_float_indian_<era>` datasets
(erddap.aoml.noaa.gov). The Ifremer GDAC / Argo ERDDAP are unreachable from this
machine — see src/data_access/common.py.

Ground truth for training + validation, upper ~2000 m.

CLI:
    python -m src.data_access.fetch_argo --start 2021-06 --end 2023-06
"""

from __future__ import annotations

import argparse
import json
from datetime import date

import numpy as np
import pandas as pd

from .common import NIO_BOX, RAW, INTERIM, FetchError, http_get, open_netcdf_bytes

ERDDAP = "https://erddap.aoml.noaa.gov/hdb/erddap/tabledap"
VARS = ("time,latitude,longitude,PLATFORM_NUMBER,"
        "PRES,PRES_QC,PRES_ADJUSTED,PRES_ADJUSTED_QC,"
        "TEMP,TEMP_QC,TEMP_ADJUSTED,TEMP_ADJUSTED_QC,"
        "PSAL,PSAL_QC,PSAL_ADJUSTED,PSAL_ADJUSTED_QC")
GOOD_QC = {"1", "2"}          # good / probably good (Argo reference table 2)

# era dataset -> (first year, last year inclusive; 9999 = present)
ERAS = [
    ("argo_float_indian_1998_2011", 1998, 2011),
    ("argo_float_indian_2012_2016", 2012, 2016),
    ("argo_float_indian_2017_2020", 2017, 2020),
    ("argo_float_indian_2021_2024", 2021, 2024),
    ("argo_float_indian_2025_present", 2025, 9999),
]

# hard physical clamp AFTER QC filtering (North Indian Ocean)
T_RANGE = (-2.0, 36.0)
S_RANGE = (3.0, 41.5)
P_RANGE = (0.0, 2100.0)


def _months(start: str, end: str):
    y0, m0 = map(int, start.split("-"))
    y1, m1 = map(int, end.split("-"))
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def _era_for(year: int) -> str:
    for name, lo, hi in ERAS:
        if lo <= year <= hi:
            return name
    raise ValueError(f"no Argo era dataset for year {year}")


def _month_bounds(y: int, m: int) -> tuple[str, str]:
    start = date(y, m, 1)
    end = date(y + (m == 12), (m % 12) + 1, 1)
    return start.isoformat(), end.isoformat()


def fetch_month(y: int, m: int, box=NIO_BOX, cache=True) -> pd.DataFrame:
    """Pull one month of profiles as a tidy long DataFrame (one row per level)."""
    raw_dir = RAW / "argo"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_nc = raw_dir / f"argo_v2_{y}{m:02d}.nc"     # v2 = includes QC + ADJUSTED

    t0, t1 = _month_bounds(y, m)
    if cache and cache_nc.exists() and cache_nc.stat().st_size > 0:
        body = cache_nc.read_bytes()
    else:
        ds_name = _era_for(y)
        query = (
            f"{VARS}"
            f"&time>={t0}T00:00:00Z&time<{t1}T00:00:00Z"
            f"&latitude>={box['lat_min']}&latitude<={box['lat_max']}"
            f"&longitude>={box['lon_min']}&longitude<={box['lon_max']}"
        )
        url = f"{ERDDAP}/{ds_name}.nc?{query}"
        try:
            body = http_get(url, read=120, tries=3, expect_netcdf=True)
        except FetchError as e:
            if "produced no matching results" in str(e):
                return pd.DataFrame()
            raise
        cache_nc.write_bytes(body)

    ds = open_netcdf_bytes(body, f"argo_v2_{y}{m:02d}.nc")
    n = ds.sizes["row"]
    good = np.array(sorted(int(x) for x in GOOD_QC))

    def qc_ok(name):
        """QC flag in {1,2}. ERDDAP returns these as float (1.0, 2.0, ...) or char."""
        if name not in ds:
            return np.zeros(n, bool)
        v = ds[name].values
        try:
            q = np.asarray(v, "float64")
        except (ValueError, TypeError):
            q = np.array([float(str(x).strip() or "nan")
                          if str(x).strip() not in ("", "b''") else np.nan for x in v])
        return np.isin(np.floor(np.nan_to_num(q, nan=0.0)).astype(int), good)

    def best(raw_name, adj_name):
        """prefer ADJUSTED value where its QC is good, else raw where its QC is good."""
        raw = np.asarray(ds[raw_name].values, "float64")
        out = np.where(qc_ok(raw_name + "_QC"), raw, np.nan)
        if adj_name in ds:
            adj = np.asarray(ds[adj_name].values, "float64")
            use_adj = qc_ok(adj_name + "_QC") & np.isfinite(adj)
            out = np.where(use_adj, adj, out)
        return out

    df = pd.DataFrame({
        "time": pd.to_datetime(ds["time"].values),
        "lat": np.asarray(ds["latitude"].values, "float64"),
        "lon": np.asarray(ds["longitude"].values, "float64"),
        "platform": np.asarray(ds["PLATFORM_NUMBER"].values).astype(str),
        "pres": best("PRES", "PRES_ADJUSTED"),
        "temp": best("TEMP", "TEMP_ADJUSTED"),
        "psal": best("PSAL", "PSAL_ADJUSTED"),
    })
    ds.close()

    df = df.dropna(subset=["pres", "temp", "psal"])
    df = df[
        df["temp"].between(*T_RANGE)
        & df["psal"].between(*S_RANGE)
        & df["pres"].between(*P_RANGE)
    ]
    # profile id = platform + rounded timestamp + rounded position
    df["profile_id"] = (
        df["platform"].str.strip()
        + "_" + df["time"].dt.strftime("%Y%m%dT%H%M")
        + "_" + df["lat"].round(3).astype(str)
        + "_" + df["lon"].round(3).astype(str)
    )
    return df


def fetch_range(start: str, end: str, box=NIO_BOX) -> pd.DataFrame:
    frames, manifest = [], []
    for y, m in _months(start, end):
        try:
            dfm = fetch_month(y, m, box=box)
        except FetchError as e:
            print(f"  {y}-{m:02d}: FETCH FAILED — {e}")
            manifest.append({"month": f"{y}-{m:02d}", "status": "failed", "rows": 0})
            continue
        n_prof = dfm["profile_id"].nunique() if len(dfm) else 0
        print(f"  {y}-{m:02d}: {len(dfm):>7} levels  {n_prof:>4} profiles")
        manifest.append({"month": f"{y}-{m:02d}", "status": "ok",
                         "levels": int(len(dfm)), "profiles": int(n_prof)})
        if len(dfm):
            frames.append(dfm)

    INTERIM.mkdir(parents=True, exist_ok=True)
    (INTERIM / "argo_fetch_manifest.json").write_text(json.dumps(manifest, indent=2))

    if not frames:
        raise FetchError("Argo fetch produced zero profiles across the whole range — "
                         "this is the CLAUDE.md hard-stop condition (no ground truth).")
    out = pd.concat(frames, ignore_index=True)
    out_path = INTERIM / "argo_profiles.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\n  -> {out_path}  ({len(out):,} levels, "
          f"{out['profile_id'].nunique():,} profiles)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-06", help="YYYY-MM inclusive")
    ap.add_argument("--end", default="2023-06", help="YYYY-MM inclusive")
    args = ap.parse_args()
    fetch_range(args.start, args.end)


if __name__ == "__main__":
    main()
