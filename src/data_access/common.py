"""
Shared HTTP + NetCDF helpers for the data-access layer.

Reachability note (from Phase 0 / Phase 1 diagnosis, 2026-09):
    From this machine, *.ifremer.fr, seanoe.org, and coastwatch.pfeg.noaa.gov
    time out or fail TLS. The pipeline therefore uses only endpoints that were
    empirically reachable:
      - Argo   : erddap.aoml.noaa.gov  (argo_float_indian_* tabledap)
      - SST    : www.ncei.noaa.gov     (OISST v2.1 daily THREDDS files)
      - SSH    : coastwatch.noaa.gov   (noaacwBLENDEDsshDaily, var 'sla')
      - SSS    : coastwatch.noaa.gov   (noaacwSMAPsssDaily, var 'sss')
      - WOA    : www.ncei.noaa.gov     (woa23 T/S monthly climatology)
"""

from __future__ import annotations

import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

# North Indian Ocean fetch box (wider than the BoB — training needs Arabian Sea
# + equatorial IO; the BoB is the spatial holdout).  CLAUDE.md §4.3 / abstract §3.3
NIO_BOX = dict(lat_min=-5.0, lat_max=25.0, lon_min=45.0, lon_max=100.0)

# Bay of Bengal spatial-holdout definition
BOB_HOLDOUT = dict(lat_min=5.0, lon_min=85.0)   # lat >= 5 AND lon >= 85  -> holdout

USER_AGENT = "OceanEmbed/0.1 (SIH research build; offline batch pipeline)"
_HEADERS = {"User-Agent": USER_AGENT}


class FetchError(RuntimeError):
    pass


def http_get(url: str, *, params=None, connect=10, read=60, tries=3,
             backoff=2.0, expect_netcdf=False) -> bytes:
    """GET with retry. Returns response bytes or raises FetchError."""
    last = None
    for k in range(tries):
        t0 = time.monotonic()
        try:
            r = requests.get(url, params=params, headers=_HEADERS,
                             timeout=(connect, read))
            if r.status_code == 200:
                body = r.content
                if expect_netcdf and body[:3] not in (b"CDF", b"\x89HD"):
                    snippet = body[:200].decode("utf-8", "replace")
                    raise FetchError(f"expected NetCDF, got {r.headers.get('Content-Type')}"
                                     f" / {snippet!r}")
                return body
            last = f"HTTP {r.status_code}: {r.content[:200].decode('utf-8', 'replace')}"
        except FetchError:
            raise
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        dt = time.monotonic() - t0
        if k < tries - 1:
            time.sleep(backoff * (k + 1))
        last = f"{last} (attempt {k + 1}, {dt:.1f}s)"
    raise FetchError(f"GET failed: {url}\n  {last}")


def open_netcdf_bytes(body: bytes, tmp_name: str, *, decode_times=True):
    """Write bytes to data/raw/_tmp and open with xarray (file-based, netcdf4)."""
    import xarray as xr

    tmpdir = RAW / "_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    p = tmpdir / tmp_name
    p.write_bytes(body)
    last = None
    for engine in ("netcdf4", "h5netcdf", "scipy"):
        try:
            return xr.open_dataset(p, engine=engine, decode_times=decode_times)
        except Exception as e:  # noqa: BLE001
            last = f"{engine}: {type(e).__name__}: {e}"
    raise FetchError(f"could not open NetCDF {tmp_name}: {last}")


def in_bob_holdout(lat, lon):
    """Boolean mask: True where a point falls in the Bay of Bengal spatial holdout."""
    import numpy as np

    lat = np.asarray(lat)
    lon = np.asarray(lon)
    return (lat >= BOB_HOLDOUT["lat_min"]) & (lon >= BOB_HOLDOUT["lon_min"])


# Standard output depth levels (CLAUDE.md §4.1) — trimmed at runtime to what Argo reaches
STD_DEPTHS = [0, 10, 20, 30, 50, 75, 100, 125, 150, 200,
              250, 300, 400, 500, 700, 1000, 1500, 2000]
