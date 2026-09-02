"""
Fetch satellite surface fields for the North Indian Ocean (CLAUDE.md §3.1, §4).

PREDICTION INPUTS. `src/regime/` must never import this module.

Reachable sources (Phase 1 diagnosis — see common.py):
    SST : OISST v2.1 daily files, www.ncei.noaa.gov  (download + local subset)
    SSH : noaacwBLENDEDsshDaily  var 'sla'  (CoastWatch hub ERDDAP, weekly mean)
    SSS : noaacwSMAPsssDaily     var 'sss'  (CoastWatch hub ERDDAP, weekly mean)

ocean_color: no historical-coverage source was reachable from this machine
    (CoastWatch VIIRS chl-a is NRT-only ~275 days; pfeg MODIS unreachable).
    Omitted from v1. SSS itself is a strong Bay-of-Bengal river-plume proxy, so
    the [SST, SSH, SSS] stack still captures the freshwater-lens signal. Adding a
    true ocean-colour channel (OC-CCI) is future work — noted in README.

CLI:
    python -m src.data_access.fetch_satellite --start 2021-06 --end 2023-06
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta

import numpy as np
import xarray as xr

from .common import NIO_BOX, RAW, INTERIM, FetchError, http_get, open_netcdf_bytes

OISST_BASE = ("https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-"
              "interpolation/v2.1/access/avhrr")
CW = "https://coastwatch.noaa.gov/erddap/griddap"

LAT0, LAT1 = NIO_BOX["lat_min"], NIO_BOX["lat_max"]
LON0, LON1 = NIO_BOX["lon_min"], NIO_BOX["lon_max"]


# --------------------------------------------------------------------------- #
# week grid
# --------------------------------------------------------------------------- #

def week_starts(start: str, end: str):
    """Yield Monday date objects for 7-day bins covering [start, end] months."""
    y0, m0 = map(int, start.split("-"))
    y1, m1 = map(int, end.split("-"))
    d = date(y0, m0, 1)
    d -= timedelta(days=d.weekday())            # back up to Monday
    last = date(y1 + (m1 == 12), (m1 % 12) + 1, 1)
    while d < last:
        yield d
        d += timedelta(days=7)


# --------------------------------------------------------------------------- #
# SST — OISST v2.1 daily file, midweek snapshot (documented simplification)
# --------------------------------------------------------------------------- #

def _oisst_day(d: date) -> xr.Dataset | None:
    cache = RAW / "oisst" / f"{d:%Y%m}" / f"oisst-avhrr-v02r01.{d:%Y%m%d}.nc"
    cache.parent.mkdir(parents=True, exist_ok=True)

    def _download():
        for suffix in ("", "_preliminary"):
            url = f"{OISST_BASE}/{d:%Y%m}/oisst-avhrr-v02r01{suffix}.{d:%Y%m%d}.nc"
            try:
                return http_get(url, read=90, tries=2, expect_netcdf=True)
            except FetchError:
                continue
        return None

    if cache.exists() and cache.stat().st_size > 5000 and cache.read_bytes()[:3] in (b"CDF", b"\x89HD"):
        body = cache.read_bytes()
    else:
        body = _download()
        if body is None:
            return None
        cache.write_bytes(body)
    try:
        return open_netcdf_bytes(body, f"oisst_{d:%Y%m%d}.nc")
    except FetchError:                       # corrupt cache -> one clean re-download
        body = _download()
        if body is None:
            return None
        cache.write_bytes(body)
        return open_netcdf_bytes(body, f"oisst_{d:%Y%m%d}.nc")


def sst_week(monday: date) -> xr.DataArray | None:
    """Midweek (Wed, fallback Tue/Thu) OISST SST subset for the NIO box."""
    for off in (2, 1, 3, 0, 4):
        ds = _oisst_day(monday + timedelta(days=off))
        if ds is None:
            continue
        da = (ds["sst"].sel(lat=slice(LAT0, LAT1), lon=slice(LON0, LON1))
              .squeeze(drop=True))
        ds.close()
        da = da.rename("sst")
        da.attrs["oceanembed_note"] = f"OISST v2.1 midweek snapshot {monday+timedelta(days=off)}"
        return da
    return None


# --------------------------------------------------------------------------- #
# ERDDAP griddap weekly mean (SSH, SSS)
# --------------------------------------------------------------------------- #

def _erddap_week(dsid: str, var: str, monday: date, has_altitude: bool) -> xr.DataArray | None:
    t0 = f"{monday:%Y-%m-%d}T00:00:00Z"
    t1 = f"{monday + timedelta(days=6):%Y-%m-%d}T23:59:59Z"
    alt = "%5B(0.0)%5D" if has_altitude else ""
    q = (f"{CW}/{dsid}.nc?{var}"
         f"%5B({t0}):({t1})%5D{alt}"
         f"%5B({LAT0}):({LAT1})%5D%5B({LON0}):({LON1})%5D")
    cache = RAW / dsid / f"{dsid}_{monday:%Y%m%d}.nc"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and cache.stat().st_size > 0:
        body = cache.read_bytes()
    else:
        try:
            body = http_get(q, read=90, tries=3, expect_netcdf=True)
        except FetchError as e:
            if "no matching results" in str(e) or "HTTP 404" in str(e):
                return None
            raise
        cache.write_bytes(body)
    ds = open_netcdf_bytes(body, f"{dsid}_{monday:%Y%m%d}.nc")
    if var not in ds.variables:
        ds.close()
        return None
    da = ds[var].squeeze(drop=True)
    if "time" in da.dims:
        da = da.mean("time", keep_attrs=True)
    da = da.rename(var).load()
    ds.close()
    return da


def ssh_week(monday: date):
    return _erddap_week("noaacwBLENDEDsshDaily", "sla", monday, has_altitude=False)


def sss_week(monday: date):
    return _erddap_week("noaacwSMAPsssDaily", "sss", monday, has_altitude=True)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def fetch_range(start: str, end: str):
    INTERIM.mkdir(parents=True, exist_ok=True)
    weeks = list(week_starts(start, end))
    print(f"Satellite fetch: {len(weeks)} weeks {weeks[0]} .. {weeks[-1]}")

    rows = []
    per_week = []
    for i, mon in enumerate(weeks):
        rec = {"week_start": mon.isoformat()}
        try:
            sst = sst_week(mon)
            ssh = ssh_week(mon)
            sss = sss_week(mon)
        except Exception as e:  # noqa: BLE001 — one bad week must not abort the run
            print(f"  [{i + 1:>3}/{len(weeks)}] {mon}  ERROR {type(e).__name__}: {str(e)[:90]}")
            rec.update(sst=False, ssh=False, sss=False, error=str(e)[:200])
            per_week.append(rec)
            continue
        rec["sst"] = sst is not None
        rec["ssh"] = ssh is not None
        rec["sss"] = sss is not None
        per_week.append(rec)
        got = "".join(lbl if rec[k] else "-"
                      for k, lbl in (("sst", "T"), ("ssh", "H"), ("sss", "S")))
        print(f"  [{i + 1:>3}/{len(weeks)}] {mon}  {got}")
        if sst is None:
            continue
        ds = xr.Dataset({"sst": sst})
        # regrid SSH/SSS onto the SST 0.25deg grid by nearest — cheap, adequate for matching
        if ssh is not None:
            ds["ssh"] = ssh.interp(latitude=sst["lat"], longitude=sst["lon"],
                                   method="nearest").drop_vars(
                                       [c for c in ("latitude", "longitude") if c in ssh.coords],
                                       errors="ignore")
        if sss is not None:
            ds["sss"] = sss.interp(latitude=sst["lat"], longitude=sst["lon"],
                                   method="nearest").drop_vars(
                                       [c for c in ("latitude", "longitude") if c in sss.coords],
                                       errors="ignore")
        ds = ds.expand_dims(week_start=[np.datetime64(mon)])
        rows.append(ds)

    if not rows:
        raise FetchError("no satellite weeks assembled — SST source unreachable")

    combined = xr.concat(rows, dim="week_start")
    out = INTERIM / "satellite_weekly.nc"
    combined.to_netcdf(out)
    (INTERIM / "satellite_fetch_manifest.json").write_text(json.dumps(per_week, indent=2))
    print(f"\n  -> {out}  vars={list(combined.data_vars)}  "
          f"weeks={combined.sizes['week_start']}")
    return combined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-06")
    ap.add_argument("--end", default="2023-06")
    args = ap.parse_args()
    fetch_range(args.start, args.end)


if __name__ == "__main__":
    main()
