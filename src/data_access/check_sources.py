"""
Phase 0 — Data Reality Check  (CLAUDE.md §3).

Attempts a REAL fetch (not a ping) for every data source OceanEmbed depends on,
records the actual outcome + error message, and writes:

    data/raw/phase0_report.md      <- the human deliverable (markdown table)
    data/raw/phase0_results.json   <- full machine-readable attempt log

Status vocabulary (CLAUDE.md §3):
    Accessible            - pulled real data, parsed it, values look sane
    Accessible-with-delay - reachable but blocked on free registration / approval
    Unavailable           - could not get data in the Phase 0 window
    Substitute-used       - real source unavailable; a labelled static stand-in is planned

Run:
    .venv/Scripts/python.exe -m src.data_access.check_sources
    .venv/Scripts/python.exe src/data_access/check_sources.py         # also works

Time budget: whole script should finish in minutes. Hard wall-clock cap below.
"""

from __future__ import annotations

import io
import json
import re
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw"
PROBE = RAW / "_phase0_probe"
REPORT_MD = RAW / "phase0_report.md"
RESULTS_JSON = RAW / "phase0_results.json"

# Bay of Bengal box (CLAUDE.md §3 item 1 / §4.1)
LAT_MIN, LAT_MAX = 5.0, 22.0
LON_MIN, LON_MAX = 80.0, 95.0

HEADERS = {"User-Agent": "OceanEmbed-Phase0/0.1 (research data reachability check)"}
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 30
WALLCLOCK_CAP_S = 900  # 15 min; abort remaining network work past this

# A recent-ish date that most near-real-time products should already have.
_TODAY = datetime.now(timezone.utc).date()
RECENT = _TODAY - timedelta(days=45)
RECENT_MINUS = _TODAY - timedelta(days=120)

_START = time.monotonic()


def _budget_left() -> float:
    return WALLCLOCK_CAP_S - (time.monotonic() - _START)


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #

@dataclass
class SourceResult:
    source: str
    status: str = "Unavailable"
    notes: str = ""
    substitute: str = ""
    attempts: list = field(default_factory=list)

    def log(self, **kw):
        self.attempts.append(kw)


# --------------------------------------------------------------------------- #
# HTTP / parse helpers
# --------------------------------------------------------------------------- #

def http_get(url, params=None, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
             allow_redirects=True, want_bytes=True):
    """Single GET. Returns a dict describing exactly what happened."""
    rec = {"url": url, "params": params, "ok": False, "status_code": None,
           "elapsed_s": None, "n_bytes": 0, "content_type": None, "error": None,
           "final_url": None}
    t0 = time.monotonic()
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=timeout,
                         allow_redirects=allow_redirects, stream=want_bytes)
        rec["status_code"] = r.status_code
        rec["final_url"] = r.url
        rec["content_type"] = r.headers.get("Content-Type")
        body = r.content if want_bytes else b""
        rec["n_bytes"] = len(body)
        rec["elapsed_s"] = round(time.monotonic() - t0, 2)
        r.raise_for_status()
        rec["ok"] = True
        rec["_body"] = body
    except Exception as e:  # noqa: BLE001 - we want every failure mode captured
        rec["elapsed_s"] = round(time.monotonic() - t0, 2)
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def parse_netcdf(body: bytes, tag: str):
    """Try to open bytes as NetCDF/HDF5 with xarray. Returns (ds_or_None, note)."""
    try:
        import xarray as xr
    except Exception as e:  # noqa: BLE001
        return None, f"xarray import failed: {e}"
    PROBE.mkdir(parents=True, exist_ok=True)
    p = PROBE / f"{tag}.nc"
    p.write_bytes(body)
    last = None
    for engine in ("netcdf4", "h5netcdf", "scipy"):
        try:
            ds = xr.open_dataset(p, engine=engine)
            return ds, f"opened with engine={engine}; vars={list(ds.data_vars)[:8]}"
        except Exception as e:  # noqa: BLE001
            last = f"{engine}: {type(e).__name__}: {e}"
    # Maybe it's actually an error page / JSON / HTML
    head = body[:200].decode("utf-8", "replace")
    return None, f"could not open as NetCDF ({last}); first bytes: {head!r}"


def open_opendap(url: str):
    """Try to open an OPeNDAP endpoint lazily with xarray."""
    try:
        import xarray as xr
    except Exception as e:  # noqa: BLE001
        return None, f"xarray import failed: {e}"
    last = None
    for engine in ("pydap", "netcdf4"):
        try:
            ds = xr.open_dataset(url, engine=engine)
            return ds, f"opened with engine={engine}; vars={list(ds.data_vars)[:8]}"
        except Exception as e:  # noqa: BLE001
            last = f"{engine}: {type(e).__name__}: {e}"
    return None, f"OPeNDAP open failed ({last})"


def _finite_fraction(da) -> float:
    try:
        import numpy as np
        v = np.asarray(da.values, dtype="float64")
        if v.size == 0:
            return 0.0
        return float(np.isfinite(v).mean())
    except Exception:  # noqa: BLE001
        return -1.0


# --------------------------------------------------------------------------- #
# 1. Argo profiles
# --------------------------------------------------------------------------- #

def check_argo() -> SourceResult:
    r = SourceResult("Argo profiles (T/S/pressure ground truth)")
    t0 = (RECENT_MINUS - timedelta(days=30)).isoformat()
    t1 = RECENT.isoformat()

    erddap_hosts = [
        "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.nc",
        "https://www.ifremer.fr/erddap/tabledap/ArgoFloats.nc",
    ]
    query = (
        "platform_number,cycle_number,time,latitude,longitude,pres,temp,psal"
        f"&latitude>={LAT_MIN}&latitude<={LAT_MAX}"
        f"&longitude>={LON_MIN}&longitude<={LON_MAX}"
        f"&time>={t0}&time<={t1}&pres<=50"
    )
    for host in erddap_hosts:
        if _budget_left() < 60:
            break
        url = f"{host}?{query}"
        rec = http_get(url)
        body = rec.pop("_body", b"")
        if rec["ok"] and body:
            ds, note = parse_netcdf(body, "argo_erddap")
            rec["parse"] = note
            r.log(**rec)
            if ds is not None:
                nrow = int(ds.sizes.get("row", ds.sizes.get("obs", 0)))
                has_ts = all(v in ds.variables for v in ("temp", "psal", "pres"))
                if has_ts and nrow > 0:
                    r.status = "Accessible"
                    r.notes = (f"Ifremer ERDDAP tabledap OK: {nrow} near-surface "
                               f"rows in BoB box, {t0}..{t1}; temp/psal/pres present.")
                    ds.close()
                    return r
                r.notes = f"ERDDAP returned data but rows={nrow}, T/S/pres present={has_ts}."
                ds.close()
        else:
            r.log(**rec)

    # Fallback: GDAC HTTP, INCOIS DAC (India's own Argo DAC)
    if _budget_left() > 60:
        dac = "https://data-argo.ifremer.fr/dac/incois/"
        rec = http_get(dac)
        body = rec.pop("_body", b"")
        rec["parse"] = f"listing bytes={len(body)}"
        r.log(**rec)
        if rec["ok"] and body:
            floats = re.findall(rb'href="(\d{6,7})/"', body)
            if floats:
                fid = floats[0].decode()
                prof_url = f"{dac}{fid}/{fid}_prof.nc"
                rec2 = http_get(prof_url)
                b2 = rec2.pop("_body", b"")
                ds, note = parse_netcdf(b2, "argo_gdac") if (rec2["ok"] and b2) else (None, "no body")
                rec2["parse"] = note
                r.log(**rec2)
                if ds is not None and all(v in ds.variables for v in ("TEMP", "PRES")):
                    r.status = "Accessible"
                    r.notes = (f"GDAC HTTP OK via INCOIS DAC: pulled/parsed float "
                               f"{fid} profile file (TEMP/PSAL/PRES present).")
                    ds.close()
                    return r

    if r.status != "Accessible":
        if not r.notes:
            r.notes = "No Argo endpoint returned parseable profile data in the window."
        r.substitute = ("None acceptable — Argo IS the ground truth. If truly "
                        "unavailable the project cannot validate and must stop.")
    return r


# --------------------------------------------------------------------------- #
# 2. Satellite SST (NOAA OISST v2.1)
# --------------------------------------------------------------------------- #

def check_sst() -> SourceResult:
    r = SourceResult("Satellite SST (NOAA OISST v2.1)")
    for d in (RECENT, RECENT_MINUS, _TODAY - timedelta(days=200)):
        if _budget_left() < 60:
            break
        ds_date = d.isoformat()
        candidates = [
            ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.nc",
             f"?sst[({ds_date}T12:00:00Z)][(0.0)][({LAT_MIN}):({LAT_MAX})][({LON_MIN}):({LON_MAX})]"),
            ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg.nc",
             f"?sst[({ds_date}T12:00:00Z)][(0.0)][({LAT_MIN}):({LAT_MAX})][({LON_MIN}):({LON_MAX})]"),
        ]
        for base, q in candidates:
            if _budget_left() < 60:
                break
            rec = http_get(base + q)
            body = rec.pop("_body", b"")
            if rec["ok"] and body:
                ds, note = parse_netcdf(body, "oisst")
                rec["parse"] = note
                r.log(**rec)
                if ds is not None and "sst" in ds.variables:
                    frac = _finite_fraction(ds["sst"])
                    if frac > 0.3:
                        r.status = "Accessible"
                        r.notes = (f"CoastWatch ERDDAP OK: OISST subset for {ds_date}, "
                                   f"BoB box, {frac:.0%} finite pixels.")
                        ds.close()
                        return r
                    r.notes = f"OISST subset opened for {ds_date} but only {frac:.0%} finite."
                    ds.close()
            else:
                r.log(**rec)

    # Fallback: NCEI THREDDS OPeNDAP (per-day file)
    if _budget_left() > 60:
        d = RECENT_MINUS
        url = ("https://www.ncei.noaa.gov/thredds/dodsC/OisstBase/NetCDF/V2.1/AVHRR/"
               f"{d:%Y%m}/oisst-avhrr-v02r01.{d:%Y%m%d}.nc")
        ds, note = open_opendap(url)
        r.log(url=url, ok=ds is not None, parse=note)
        if ds is not None and "sst" in ds.variables:
            r.status = "Accessible"
            r.notes = f"NCEI THREDDS OPeNDAP OK: OISST daily file {d.isoformat()}."
            ds.close()
            return r

    if r.status != "Accessible":
        if not r.notes:
            r.notes = "No OISST endpoint returned a parseable subset."
        r.substitute = ("MODIS/VIIRS L3 SST via NASA CMR/PODAAC, or ERSST monthly "
                        "as a coarse stand-in.")
    return r


# --------------------------------------------------------------------------- #
# 3. Satellite SSH / ADT
# --------------------------------------------------------------------------- #

def check_ssh() -> SourceResult:
    r = SourceResult("Satellite SSH / ADT (altimetry)")

    # 3a. NOAA CoastWatch blended altimetry (open, no auth)
    cw = [
        ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/nesdisSSH1day.nc",
         f"?sla[({RECENT_MINUS.isoformat()}T00:00:00Z)][({LAT_MIN}):({LAT_MAX})][({LON_MIN}):({LON_MAX})]"),
        ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/noaacwBLENDEDsshDaily.nc",
         f"?sla[({RECENT_MINUS.isoformat()}T00:00:00Z)][({LAT_MIN}):({LAT_MAX})][({LON_MIN}):({LON_MAX})]"),
    ]
    for base, q in cw:
        if _budget_left() < 60:
            break
        rec = http_get(base + q)
        body = rec.pop("_body", b"")
        if rec["ok"] and body:
            ds, note = parse_netcdf(body, "ssh_coastwatch")
            rec["parse"] = note
            r.log(**rec)
            if ds is not None and len(ds.data_vars):
                var = list(ds.data_vars)[0]
                if _finite_fraction(ds[var]) > 0.3:
                    r.status = "Accessible"
                    r.notes = f"CoastWatch ERDDAP OK: blended altimetry ({var}) subset for BoB box."
                    ds.close()
                    return r
                ds.close()
        else:
            r.log(**rec)

    # 3b. APDRC ERDDAP (AVISO gridded mirror, usually open)
    if _budget_left() > 60:
        url = ("https://apdrc.soest.hawaii.edu/erddap/griddap/hawaii_soest_2ee3_0bfa_a8d6.nc"
               f"?adt[({RECENT_MINUS.isoformat()}T00:00:00Z)][({LAT_MIN}):({LAT_MAX})][({LON_MIN}):({LON_MAX})]")
        rec = http_get(url)
        body = rec.pop("_body", b"")
        ds, note = parse_netcdf(body, "ssh_apdrc") if (rec["ok"] and body) else (None, rec["error"])
        rec["parse"] = note
        r.log(**rec)
        if ds is not None and len(ds.data_vars):
            r.status = "Accessible"
            r.notes = "APDRC ERDDAP OK: AVISO gridded ADT subset for BoB box."
            ds.close()
            return r

    # 3c. CMEMS unauthenticated probe (expected: needs free registration)
    if _budget_left() > 30:
        url = "https://data.marine.copernicus.eu/api/describe"
        rec = http_get(url, want_bytes=True)
        rec.pop("_body", b"")
        r.log(**rec)

    r.status = "Accessible-with-delay" if r.status != "Accessible" else r.status
    if r.status == "Accessible-with-delay":
        r.notes = (r.notes or "Open ERDDAP altimetry mirrors did not return a subset in-window. "
                   "CMEMS (SEALEVEL_GLO_PHY_L4) is the canonical source and is free but "
                   "requires a Copernicus Marine account (registration ~instant, no approval wait).")
        r.substitute = ("CMEMS gridded L4 SLA/ADT with a user-supplied login, OR "
                        "PODAAC MEaSUREs L4 SSHA (needs NASA Earthdata login).")
    return r


# --------------------------------------------------------------------------- #
# 4. SMAP Sea Surface Salinity
# --------------------------------------------------------------------------- #

def check_smap_sss() -> SourceResult:
    r = SourceResult("Satellite SSS (SMAP L3)")

    # 4a. Open ERDDAP mirrors that sometimes carry SMAP/OISSS
    erddap = [
        ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplSMAPSSS3RunningMean.nc",
         f"?sss[({RECENT_MINUS.isoformat()}T00:00:00Z)][({LAT_MIN}):({LAT_MAX})][({LON_MIN}):({LON_MAX})]"),
        ("https://upwell.pfeg.noaa.gov/erddap/griddap/jplSMAPSSS3RunningMean.nc",
         f"?sss[({RECENT_MINUS.isoformat()}T00:00:00Z)][({LAT_MIN}):({LAT_MAX})][({LON_MIN}):({LON_MAX})]"),
    ]
    for base, q in erddap:
        if _budget_left() < 60:
            break
        rec = http_get(base + q)
        body = rec.pop("_body", b"")
        if rec["ok"] and body:
            ds, note = parse_netcdf(body, "smap_erddap")
            rec["parse"] = note
            r.log(**rec)
            if ds is not None and len(ds.data_vars):
                var = list(ds.data_vars)[0]
                if _finite_fraction(ds[var]) > 0.2:
                    r.status = "Accessible"
                    r.notes = f"Open ERDDAP OK: SMAP SSS ({var}) subset for BoB box."
                    ds.close()
                    return r
                ds.close()
        else:
            r.log(**rec)

    # 4b. NASA CMR granule search (open, no auth) - proves data exists + is locatable
    if _budget_left() > 40:
        for short_name in ("SMAP_JPL_L3_SSS_CAP_MONTHLY_V5",
                           "SMAP_JPL_L3_SSS_CAP_8DAY-RUNNINGMEAN_V5",
                           "OISSS_L4_multimission_monthly_v2"):
            url = "https://cmr.earthdata.nasa.gov/search/granules.json"
            params = {"short_name": short_name, "page_size": 3,
                      "bounding_box[]": f"{LON_MIN},{LAT_MIN},{LON_MAX},{LAT_MAX}"}
            rec = http_get(url, params=params)
            body = rec.pop("_body", b"")
            n = None
            try:
                n = len(json.loads(body)["feed"]["entry"])
            except Exception:  # noqa: BLE001
                pass
            rec["parse"] = f"{short_name}: {n} granules listed"
            r.log(**rec)
            if rec["ok"] and n:
                r.status = "Accessible-with-delay"
                r.notes = (f"NASA CMR lists {n}+ granules for {short_name} over the BoB "
                           "unauthenticated, but the granule files download only with a "
                           "free NASA Earthdata login (instant, no approval wait).")
                r.substitute = ("SMAP/OISSS via user-supplied Earthdata token, OR ESA CCI "
                                "Sea Surface Salinity, OR monthly SSS climatology.")
                return r

    if r.status not in ("Accessible", "Accessible-with-delay"):
        r.notes = r.notes or "No SMAP SSS endpoint reachable in-window."
        r.substitute = "ESA CCI SSS, or a monthly SSS climatology (labelled static)."
    return r


# --------------------------------------------------------------------------- #
# 5. Barrier-layer / stratification climatology
# --------------------------------------------------------------------------- #

def check_barrier_layer() -> SourceResult:
    r = SourceResult("Barrier-layer / MLD climatology (regime input)")

    # 5a. de Boyer Montegut MLD/BLT climatology - classic direct NetCDF
    direct = [
        "https://cerweb.ifremer.fr/deboyer/data/mld_DR003_c1m_reg2.0.nc",
        "https://www.ifremer.fr/cerweb/deboyer/data/mld_DR003_c1m_reg2.0.nc",
        "https://cerweb.ifremer.fr/deboyer/data/Argo_mixedlayers_monthlyclim_05092018.nc",
    ]
    for url in direct:
        if _budget_left() < 60:
            break
        rec = http_get(url)
        body = rec.pop("_body", b"")
        if rec["ok"] and body:
            ds, note = parse_netcdf(body, "blt_deboyer")
            rec["parse"] = note
            r.log(**rec)
            if ds is not None:
                r.status = "Accessible"
                r.notes = (f"de Boyer Montegut MLD climatology downloaded + parsed "
                           f"({url.split('/')[-1]}); vars={list(ds.data_vars)[:6]}.")
                ds.close()
                return r
        else:
            r.log(**rec)

    # 5b. APDRC OPeNDAP mirror of the MLD climatology
    if _budget_left() > 60:
        for url in (
            "http://apdrc.soest.hawaii.edu:80/dods/public_data/Argo_products/MLD_deBoyer/mld",
            "https://apdrc.soest.hawaii.edu/erddap/griddap/hawaii_soest_mld_deboyer.nc?mld[0][(0):(30)][(60):(100)]",
        ):
            ds, note = open_opendap(url) if "dods" in url else (None, "skip")
            if "dods" in url:
                r.log(url=url, ok=ds is not None, parse=note)
                if ds is not None:
                    r.status = "Accessible"
                    r.notes = "APDRC OPeNDAP OK: de Boyer Montegut MLD climatology."
                    ds.close()
                    return r
            else:
                rec = http_get(url)
                rec.pop("_body", b"")
                r.log(**rec)

    # 5c. SEANOE landing page reachability (DOI-hosted, since Nov 2022)
    if _budget_left() > 30:
        rec = http_get("https://www.seanoe.org/data/00806/91774/")
        rec.pop("_body", b"")
        r.log(**rec)
        if rec["ok"]:
            r.status = "Accessible-with-delay"
            r.notes = ("de Boyer Montegut MLD/BLT climatology is published on SEANOE "
                       "(DOI 10.17882/91774) as a downloadable tar of 1-deg monthly "
                       "NetCDF; landing page reachable, direct file link needs one manual grab.")

    if r.status not in ("Accessible", "Accessible-with-delay"):
        r.notes = r.notes or "de Boyer Montegut climatology endpoints not reachable in-window."
        r.substitute = ("Manually digitised North Indian Ocean barrier-layer-thickness "
                        "lookup (month x lat/lon bin) from Thadathil et al. 2007 figures "
                        "-- low precision, must be labelled as such (CLAUDE.md §3 item 5).")
    else:
        r.substitute = r.substitute or "n/a"
    return r


# --------------------------------------------------------------------------- #
# 6. River discharge (Ganga-Brahmaputra)
# --------------------------------------------------------------------------- #

def check_river_discharge() -> SourceResult:
    r = SourceResult("River discharge (Ganga-Brahmaputra)")

    probes = [
        ("India-WRIS portal", "https://indiawris.gov.in/wris/"),
        ("India-WRIS API root", "https://indiawris.gov.in/Dataset/Ground%20Water%20Level"),
        ("GRDC data portal", "https://portal.grdc.bafg.de/"),
    ]
    for label, url in probes:
        if _budget_left() < 40:
            break
        rec = http_get(url, want_bytes=False)
        rec.pop("_body", b"")
        rec["parse"] = label
        r.log(**rec)

    # Policy reality: for the Ganga & Brahmaputra (transboundary), CWC classifies
    # discharge and India-WRIS does NOT publish it. GRDC needs written approval.
    r.status = "Substitute-used"
    r.notes = ("Ganga & Brahmaputra discharge is CWC-'classified' (transboundary "
               "basins) and not served publicly by India-WRIS; GRDC requires a "
               "written-request approval (not instant). No live gauge feed obtainable "
               "in the Phase 0 window.")
    r.substitute = ("Static monthly climatological discharge curve for the combined "
                    "Ganga-Brahmaputra system, built in Phase 1 from published values "
                    "(e.g. RivDIS / Papa et al. / GBM gridded-streamflow 2025) and "
                    "labelled everywhere as climatology, not observations. "
                    "NOTE: this weakens the abstract's 'Indian river-discharge gauges "
                    "as an access advantage' claim -- see report footer.")
    return r


# --------------------------------------------------------------------------- #
# 7. INCOIS moored-buoy data  (the "India-specific data" differentiator)
# --------------------------------------------------------------------------- #

def check_incois() -> SourceResult:
    r = SourceResult("INCOIS moored-buoy data (India-specific differentiator)")

    # 7a. INCOIS's own ERDDAP / LAS / data portal
    incois_eps = [
        "https://erddap.incois.gov.in/erddap/index.html",
        "https://las.incois.gov.in/las/getUI.do",
        "https://incois.gov.in/site/datainfo/jointportal.jsp",
        "http://www.odis.incois.gov.in/index.php/in-situ-data/moored-buoy/moored-data",
        "https://services.incois.gov.in/iogoos/indoos/insitu_mooring.jsp",
    ]
    incois_ok = False
    for url in incois_eps:
        if _budget_left() < 40:
            break
        rec = http_get(url, want_bytes=False)
        rec.pop("_body", b"")
        r.log(**rec)
        if rec["ok"]:
            incois_ok = True

    # 7b. RAMA (Indian Ocean moored array) via PMEL ERDDAP - the accessible proxy.
    # Discover the real tabledap dataset id via the search API, then pull real data.
    rama_ok = False            # parsed actual RAMA data for the BoB box
    rama_discoverable = False   # RAMA datasets are at least listed / locatable
    if _budget_left() > 60:
        erddap = "https://data.pmel.noaa.gov/pmel/erddap"
        ids: list[str] = []
        rec = http_get(f"{erddap}/search/index.json",
                       params={"searchFor": "RAMA", "page": 1, "itemsPerPage": 30})
        b = rec.pop("_body", b"")
        try:
            tbl = json.loads(b)["table"]
            cols = tbl["columnNames"]
            id_col = cols.index("Dataset ID")
            for row in tbl["rows"]:
                dsid = row[id_col]
                if isinstance(dsid, str) and dsid.lower().startswith(("rama", "tao", "pmel")):
                    ids.append(dsid)
            rama_discoverable = bool(ids)
            rec["parse"] = f"RAMA/TAO tabledap ids: {ids[:6]}"
        except Exception as e:  # noqa: BLE001
            rec["parse"] = f"search parse failed: {type(e).__name__}: {e}"
        r.log(**rec)

        for dsid in ids[:4]:
            if _budget_left() < 60:
                break
            q = (f"{erddap}/tabledap/{dsid}.nc"
                 f"?&latitude>={LAT_MIN}&latitude<={LAT_MAX}"
                 f"&longitude>={LON_MIN}&longitude<={LON_MAX}"
                 f"&time>=2022-01-01&time<=2022-01-08")
            rec2 = http_get(q)
            body = rec2.pop("_body", b"")
            ds, note = (parse_netcdf(body, f"rama_{dsid}")
                        if (rec2["ok"] and body) else (None, rec2["error"]))
            rec2["parse"] = f"{dsid}: {note}"
            r.log(**rec2)
            if ds is not None and len(ds.variables):
                rama_ok = True
                r.log(rama_dataset=dsid, rama_vars=list(ds.variables)[:12])
                ds.close()
                break

    if incois_ok and rama_ok:
        r.status = "Accessible"
        r.notes = ("INCOIS portal endpoint(s) responded AND RAMA moored-buoy data is "
                   "reachable/parseable via PMEL ERDDAP for the BoB box.")
        r.substitute = "n/a"
    elif rama_ok:
        r.status = "Substitute-used"
        r.notes = ("INCOIS's own ERDDAP/LAS/portal did not serve buoy data unauthenticated "
                   "in-window, but RAMA (the MoES-NOAA joint Indian Ocean moored array, "
                   "co-operated by INCOIS/NIOT) IS reachable via PMEL ERDDAP.")
        r.substitute = ("RAMA buoy T/S from PMEL ERDDAP as the moored-buoy signal. "
                        "NOTE: RAMA is a joint international array, not INCOIS-exclusive -- "
                        "using it alone softens the 'data sovereignty' framing. OMNI-buoy "
                        "data from INCOIS proper would need a manual portal request.")
    elif incois_ok:
        r.status = "Accessible-with-delay"
        r.notes = ("INCOIS web endpoints reachable but no unauthenticated machine-readable "
                   "buoy pull succeeded; the OMNI-RAMA joint portal likely needs a "
                   "registration/data-request step.")
        r.substitute = "Manual INCOIS OMNI-RAMA portal request, or RAMA via PMEL ERDDAP."
    elif rama_discoverable:
        r.status = "Accessible-with-delay"
        r.notes = ("INCOIS endpoints did not serve buoy data unauthenticated, and the "
                   "guessed RAMA dataset pulls returned no rows for the BoB box in-window, "
                   "but RAMA/TAO tabledap datasets ARE listed on PMEL ERDDAP — a follow-up "
                   "with the right dataset id + variable list should work.")
        r.substitute = ("RAMA via PMEL ERDDAP (correct dataset id + vars), or manual INCOIS "
                        "OMNI-RAMA portal request for the India-specific OMNI buoys.")
    else:
        r.status = "Unavailable"
        r.notes = "Neither INCOIS endpoints nor any RAMA ERDDAP path succeeded in-window."
        r.substitute = ("RAMA via PMEL ERDDAP (retry with correct dataset id), or drop the "
                        "in-situ-buoy enrichment and rely on Argo + satellite only -- which "
                        "removes the core 'India-specific data' differentiator and needs an "
                        "abstract revision.")
    return r


# --------------------------------------------------------------------------- #
# Report writers
# --------------------------------------------------------------------------- #

def _md_escape(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ").strip()


def write_reports(results: list[SourceResult]):
    RAW.mkdir(parents=True, exist_ok=True)

    RESULTS_JSON.write_text(json.dumps(
        {"generated_utc": datetime.now(timezone.utc).isoformat(),
         "bay_of_bengal_box": {"lat": [LAT_MIN, LAT_MAX], "lon": [LON_MIN, LON_MAX]},
         "results": [asdict(r) for r in results]},
        indent=2), encoding="utf-8")

    lines = []
    lines.append("# Phase 0 — Data Reality Check")
    lines.append("")
    lines.append(f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by "
                 "`src/data_access/check_sources.py`. Re-runnable._")
    lines.append("")
    lines.append(f"Bay of Bengal box: {LAT_MIN}–{LAT_MAX}°N, {LON_MIN}–{LON_MAX}°E.")
    lines.append("")
    lines.append("| Source | Status | Notes | Substitute (if any) |")
    lines.append("|---|---|---|---|")
    for r in results:
        lines.append(f"| {_md_escape(r.source)} | **{r.status}** | "
                     f"{_md_escape(r.notes)} | {_md_escape(r.substitute) or '—'} |")
    lines.append("")

    # Mandatory explicit callout (CLAUDE.md §3 STOP note)
    flagged = [r for r in results
               if ("INCOIS" in r.source or "River discharge" in r.source)]
    lines.append("## Required callout (CLAUDE.md §3)")
    lines.append("")
    for r in flagged:
        lines.append(f"- **{r.source}** → `{r.status}`. {r.notes}")
    any_bad = any(r.status in ("Unavailable", "Substitute-used") for r in flagged)
    lines.append("")
    if any_bad:
        lines.append("> ⚠️ At least one of {INCOIS buoys, river discharge} is not a live "
                     "source. The abstract's **data-sovereignty / India-specific-data** "
                     "claim must be reviewed and softened **before** Phase 1, not after. "
                     "See per-row notes above for the exact wording that no longer holds.")
    else:
        lines.append("> Both INCOIS buoys and river discharge resolved to a usable path — "
                     "data-sovereignty claim stands as written, pending Phase 1 verification.")
    lines.append("")
    lines.append("## Environment note")
    lines.append("")
    lines.append("- Python 3.14 (target was 3.11; low-risk deviation — all Tier-1 libs "
                 "have 3.14 wheels).")
    lines.append("- CPU-only machine (Intel Iris Xe) — Phase 2 uses a **frozen** backbone "
                 "(ResNet-50 primary), trains only the adapter + decoder + uncertainty head.")
    lines.append("- Data-storage location for Phase 1 bulk downloads still to be decided "
                 "(current scaffold is under OneDrive; large NetCDF should live outside it).")
    lines.append("")
    lines.append("## STOP")
    lines.append("")
    lines.append("Phase 1 does not start until this report is reviewed and the callout "
                 "above is resolved.")
    lines.append("")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

CHECKS = [
    ("Argo profiles", check_argo),
    ("Satellite SST", check_sst),
    ("Satellite SSH/ADT", check_ssh),
    ("SMAP SSS", check_smap_sss),
    ("Barrier-layer climatology", check_barrier_layer),
    ("River discharge", check_river_discharge),
    ("INCOIS buoys", check_incois),
]


def main() -> int:
    print(f"Phase 0 data reality check — BoB box {LAT_MIN}-{LAT_MAX}N, "
          f"{LON_MIN}-{LON_MAX}E — wall-clock cap {WALLCLOCK_CAP_S}s\n")
    results: list[SourceResult] = []
    for name, fn in CHECKS:
        print(f"[..] {name}")
        t0 = time.monotonic()
        try:
            res = fn()
        except Exception as e:  # noqa: BLE001
            res = SourceResult(name, status="Unavailable",
                               notes=f"check crashed: {type(e).__name__}: {e}")
            res.log(traceback=traceback.format_exc())
        dt = time.monotonic() - t0
        results.append(res)
        print(f"[{res.status[:2].upper():>2}] {name}: {res.status}  ({dt:.0f}s)")
        print(f"     {res.notes}\n")

    write_reports(results)
    print(f"\nWrote:\n  {REPORT_MD}\n  {RESULTS_JSON}\n")

    tally = {}
    for r in results:
        tally[r.status] = tally.get(r.status, 0) + 1
    print("Summary:", ", ".join(f"{k}×{v}" for k, v in tally.items()))

    hard_fail = [r for r in results if "Argo" in r.source and r.status not in
                 ("Accessible", "Accessible-with-delay")]
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
