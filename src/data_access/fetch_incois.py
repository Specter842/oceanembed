"""
Moored-buoy data for the North Indian Ocean  (CLAUDE.md §3 item 7).

ABSTRACT CLAIM AT STAKE: the abstract lists "INCOIS moored buoy data" as the
India-specific differentiator. Phase 0/1 reality:

  * INCOIS OMNI / OGD portal (incois.gov.in, las.incois.gov.in): NOT
    machine-accessible from this build — the Live Access Server is unreachable
    (TLS failure) and bulk data needs a registered ESSO-INCOIS data request.
    => the "integrates India's own INCOIS buoy assets" claim is NOT substantiated
       by this build and must be softened in the abstract/README to future work.

  * RAMA moored array (PMEL ERDDAP, data.pmel.noaa.gov) IS reachable. RAMA is an
    international array (PMEL-led; several NIO sites serviced by India's NIOT).
    NIO sites present: 15n65e (Arabian Sea), 8n90e (southern Bay of Bengal),
    plus southern-hemisphere IO sites. Indian Ocean RAMA coverage has degraded
    badly since ~2018, so recent data is sparse. Used here as an *independent
    moored validation* source where data exists — NOT as an INCOIS substitute
    and NOT described as one.

Output: data/raw/moorings/rama_<site>_<var>.csv  + data/raw/moorings/status.json

CLI:  python -m src.data_access.fetch_incois
"""

from __future__ import annotations

import json
import io

import pandas as pd

from .common import RAW, http_get, FetchError

PMEL = "https://data.pmel.noaa.gov/pmel/erddap/tabledap"
NIO_SITES = ("15n65e", "8n90e", "12n90e", "4n90e", "18n89e")   # try; not all exist
RAMA = {"rama_hourly_temp": "TEMP", "rama_hourly_psal": "PSAL"}

INCOIS_PROBES = (
    "https://las.incois.gov.in/las/getUI.do",
    "https://incois.gov.in/portal/datainfo/buoys.jsp",
    "https://incois.gov.in/OON/index.jsp",
)


def _probe_incois() -> list[dict]:
    out = []
    for url in INCOIS_PROBES:
        try:
            body = http_get(url, connect=8, read=15, tries=1)
            ok = b"buoy" in body.lower() or b"moored" in body.lower()
            out.append({"url": url, "status": "reachable" if ok else "reachable_no_data",
                        "bytes": len(body)})
        except FetchError as e:
            out.append({"url": url, "status": "unreachable", "error": str(e)[:160]})
    return out


def _fetch_rama_site(dataset: str, var: str, site: str) -> pd.DataFrame | None:
    q = (f"{PMEL}/{dataset}.csv?site_code,time,latitude,longitude,depth,{var},{var}_QC"
         f'&site_code="{site}"&time>=2015-01-01T00:00:00Z')
    try:
        body = http_get(q, connect=8, read=60, tries=2)
    except FetchError as e:
        if "nRows = 0" in str(e) or "Not Found" in str(e):
            return None
        if "Unrecognized" in str(e) or "Bad Request" in str(e):
            return None
        raise
    df = pd.read_csv(io.StringIO(body.decode("utf-8", "replace")), skiprows=[1])
    if df.empty:
        return None
    df = df[df[var].notna()]
    return df if len(df) else None


def fetch_moorings() -> dict:
    outdir = RAW / "moorings"
    outdir.mkdir(parents=True, exist_ok=True)
    status = {"incois": _probe_incois(), "rama": []}

    any_rama = False
    for dataset, var in RAMA.items():
        for site in NIO_SITES:
            try:
                df = _fetch_rama_site(dataset, var, site)
            except FetchError as e:
                status["rama"].append({"site": site, "var": var, "status": "error",
                                       "detail": str(e)[:160]})
                continue
            if df is None:
                status["rama"].append({"site": site, "var": var, "status": "no_data"})
                continue
            any_rama = True
            path = outdir / f"rama_{site}_{var.lower()}.csv"
            df.to_csv(path, index=False)
            status["rama"].append({
                "site": site, "var": var, "status": "ok", "rows": int(len(df)),
                "time_min": str(df["time"].min()), "time_max": str(df["time"].max()),
                "depths": sorted(set(round(float(d), 1) for d in df["depth"].dropna()))[:20],
                "file": path.name,
            })
            print(f"  RAMA {site} {var}: {len(df):,} rows -> {path.name}")

    # "reachable" here means an HTML info page loaded — NOT that buoy *data* is
    # machine-retrievable. No INCOIS endpoint returns parseable buoy time series.
    incois_data_accessible = False
    rama_time_max = max((s.get("time_max", "") for s in status["rama"]
                         if s["status"] == "ok"), default="")
    status["summary"] = {
        "incois_buoy_DATA_accessible": incois_data_accessible,
        "incois_note": "only an HTML 'Ocean Observation Networks' info page loads; "
                       "OMNI/OGD buoy data needs a registered ESSO-INCOIS request; LAS is down",
        "rama_data_obtained": any_rama,
        "rama_latest_observation": rama_time_max,
        "rama_covers_2021_2023_window": rama_time_max >= "2021",
        "abstract_note": ("INCOIS buoy integration is NOT delivered in this build; "
                          "abstract 'data sovereignty' claim must be scoped to future "
                          "work. RAMA data exists for 2 NIO sites but ends 2020 — it "
                          "does NOT overlap a 2021-2023 evaluation window."),
    }
    (outdir / "status.json").write_text(json.dumps(status, indent=2))
    s = status["summary"]
    print(f"\n  INCOIS buoy DATA accessible: {s['incois_buoy_DATA_accessible']}")
    print(f"  RAMA data obtained:          {s['rama_data_obtained']}  "
          f"(latest obs {s['rama_latest_observation'][:10] or 'n/a'})")
    print(f"  RAMA covers 2021-2023:       {s['rama_covers_2021_2023_window']}")
    print(f"  -> {outdir / 'status.json'}")
    return status


if __name__ == "__main__":
    fetch_moorings()
