"""
Fetch WOA23 monthly T/S climatology for the North Indian Ocean.

Used ONLY to build the regime-conditioning stratification climatology
(src/regime/build_climatology.py). This is an independent climatology, not a
prediction input — keeps the regime pipeline non-circular (CLAUDE.md §4.2).

Source: NCEI THREDDS OPeNDAP (server-side subset — a few MB, not the full 1.4 GB).
    https://www.ncei.noaa.gov/thredds-ocean/dodsC/woa23/DATA/<var>/netcdf/decav/1.00/

CLI:  python -m src.data_access.fetch_woa
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from .common import NIO_BOX, INTERIM, FetchError

DODS = ("https://www.ncei.noaa.gov/thredds-ocean/dodsC/woa23/DATA/"
        "{long}/netcdf/decav/1.00/woa23_decav_{code}{mm:02d}_01.nc")
N_DEPTH = 25          # 0..~250 m, 5-10 m spacing — enough for MLD / barrier layer
OUT = INTERIM / "woa23_nio_monthly.nc"


def _load_var(long_name: str, code: str) -> xr.DataArray:
    slabs = []
    for mm in range(1, 13):
        url = DODS.format(long=long_name, code=code, mm=mm)
        last = None
        for attempt in range(3):
            try:
                ds = xr.open_dataset(url, decode_times=False)
                da = (ds[f"{code}_an"].isel(time=0, drop=True)
                      .sel(lat=slice(NIO_BOX["lat_min"], NIO_BOX["lat_max"]),
                           lon=slice(NIO_BOX["lon_min"], NIO_BOX["lon_max"]))
                      .isel(depth=slice(0, N_DEPTH))
                      .load())
                ds.close()
                da = da.reset_coords(drop=True)
                for bad in ("time", "climatology_bounds"):
                    da.attrs.pop(bad, None)
                slabs.append(da.assign_coords(month=mm))
                print(f"  {code}{mm:02d}: {tuple(da.sizes.values())}")
                last = None
                break
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
        if last:
            raise FetchError(f"WOA {code}{mm:02d} failed after retries: {last}")
    out = xr.concat(slabs, dim="month", coords="minimal", compat="override")
    return out.rename(f"{code}_an")


def build():
    INTERIM.mkdir(parents=True, exist_ok=True)
    print("WOA23 temperature ...")
    t = _load_var("temperature", "t")
    print("WOA23 salinity ...")
    s = _load_var("salinity", "s")
    ds = xr.Dataset({"t_an": t, "s_an": s})
    ds.attrs["source"] = "WOA23 decav 1.00-deg monthly objectively-analysed mean"
    ds.attrs["purpose"] = "regime-conditioning stratification climatology only (non-circular)"
    ds.to_netcdf(OUT)
    print(f"\n  -> {OUT}  dims={dict(ds.sizes)}")
    return ds


if __name__ == "__main__":
    build()
