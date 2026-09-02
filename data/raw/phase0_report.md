# Phase 0 — Data Reality Check

**Build date:** 2026-09-03
**Region box (NIO):** 5°S–25°N, 45°E–100°E   ·   **BoB sub-box:** 5–22°N, 80–95°E
**Method:** real fetch attempts (not pings) from this machine; actual errors logged.

> **Network note.** From this machine, an entire class of hosts is unreachable
> (connection timeout / TLS failure), independent of the data on them:
> `*.ifremer.fr`, `www.seanoe.org`, `las.incois.gov.in`, `coastwatch.pfeg.noaa.gov`,
> `upwell.pfeg.noaa.gov`, `usgodae.org`. Hosts under `ncei.noaa.gov`,
> `coastwatch.noaa.gov` (central hub), `erddap.aoml.noaa.gov`,
> `data.pmel.noaa.gov`, `cmr.earthdata.nasa.gov` all work. Several CLAUDE.md-named
> endpoints fall in the unreachable set, so reachable equivalents were substituted
> (documented per row). This is a property of *this build environment*, not of the
> datasets — on another network the canonical endpoints may be preferable.

| Source | Status | Notes | Substitute used |
|---|---|---|---|
| **Argo profiles** | Substitute-used | Ifremer GDAC (`data-argo.ifremer.fr`) + all Ifremer Argo ERDDAP: connection timeout. AOML ERDDAP reachable and serves QC'd Indian-Ocean Argo. | **AOML ERDDAP** `argo_float_indian_{2017_2020,2021_2024}` tabledap. **14,017 profiles / 7.2 M levels** pulled for 2021-01…2023-11, NIO box. T/S/pres verified physically sensible (Arabian Sea S≈36.5, BoB S≈32–33). |
| **Satellite SST** | Substitute-used | `coastwatch.pfeg.noaa.gov/erddap` (CLAUDE.md endpoint): TLS/timeout. NCEI THREDDS OPeNDAP: 503. NCEI THREDDS **file server**: works. | **OISST v2.1 daily NetCDF** from `ncei.noaa.gov/data/.../avhrr/`, subset locally to the NIO box. Weekly value = midweek (Wed) snapshot — documented simplification (OISST is already an OI product; within-week drift is small). |
| **Satellite SSH / ADT** | Accessible (substitute endpoint) | CMEMS is the canonical source but needs a Copernicus Marine account (not created — account creation is out of scope for this build). | **NOAA CoastWatch hub** `noaacwBLENDEDsshDaily`, variable `sla` (sea-level anomaly), no auth. Weekly mean. `adt` not available on this dataset — SLA is the anomaly field the model needs anyway. |
| **SMAP SSS** | Accessible (substitute endpoint) | PODAAC granules are listed unauthenticated via CMR but download needs a NASA Earthdata login (not created). | **NOAA CoastWatch hub** `noaacwSMAPsssDaily`, variable `sss`, no auth. Weekly mean. Clipped to [20, 41] (SMAP has coastal/RFI spikes). |
| **Ocean colour** | Omitted (v1) | No historical-coverage chlorophyll source was reachable (CoastWatch VIIRS chl-a is NRT-only ≈275 days; MODIS `erdMH1chla` is on the unreachable pfeg host). | **None.** [SST, SSH, SSS] retained. SSS is itself a strong BoB river-plume proxy, so the freshwater-lens signal is still represented. A true OC-CCI channel is future work — stated in README. |
| **Barrier-layer climatology** | Substitute-used | de Boyer Montégut MLD/BLT climatology (Ifremer `cerweb` / SEANOE): hosts unreachable. | **WOA23 monthly T/S** (NCEI, reachable via OPeNDAP subset) → computed MLD (σ₀ 0.03 threshold), ILD (ΔT 0.5), BLT = ILD−MLD, and a 0–1 stratification index, in `src/regime/build_climatology.py` → `data/interim/regime_climatology.nc`. Coarser than de Boyer Montégut (1° vs 0.5°) and Arabian-Sea winter BLT is over-estimated (deep isothermal layer vs shallow halocline) — the regime *class* is gated on the stratification index so classification is robust; continuous `blt_m` should be used with that caveat. |
| **River discharge (Ganga-Brahmaputra)** | **Unavailable → static climatology** | No public machine-readable gauge data. CWC classifies discharge for the Ganga & Brahmaputra (transboundary basins); India-WRIS does not serve it; GRDC needs a written-request approval (not instant). | **Hardcoded monthly climatology** (`src/data_access/fetch_discharge.py`), seasonal shape after Papa et al. (2012) / Dai (2017): dry Feb–Mar ≈ 9–11 k m³/s, Jul–Aug peak ≈ 75–82 k m³/s. Labelled "static climatology, NOT observations" at every use. It is a regime-conditioning input only, and is deliberately independent of any satellite field. |
| **INCOIS moored buoy data** | **Unavailable** | `las.incois.gov.in` (Live Access Server): TLS failure. `incois.gov.in/portal/datainfo/buoys.jsp`: 404. `incois.gov.in/OON/index.jsp`: an HTML "Ocean Observation Networks" info page loads (37 KB) — **no machine-readable buoy time series from any INCOIS endpoint**. OMNI/OGD bulk data requires a registered ESSO-INCOIS data request. | **RAMA moored array** via PMEL ERDDAP (`data.pmel.noaa.gov`) — reachable. NIO sites with data: `15n65e` (Arabian Sea), `8n90e` (southern BoB), T & S at 1–500 m. **But coverage ends 2020-02 (`8n90e`) / 2020-10 (`15n65e`)** — it does **not overlap a 2021–2023 evaluation window.** Downloaded to `data/raw/moorings/` regardless. RAMA is an international array (PMEL-led; several NIO sites serviced by India's NIOT) — it is **not** an INCOIS asset and is not described as one. |

---

## Callouts that affect the abstract (per CLAUDE.md §3 hard-stop)

**1. The "India-specific observational data" / "data sovereignty" differentiator is not delivered by this build.**
The abstract (§3.1 "India-specific enrichment", §5 "Data sovereignty argument", §6 table row "Adds INCOIS buoys, Indian river discharge gauges") rests on integrating INCOIS buoys + Indian river-gauge data. In practice:
- INCOIS buoy data: not machine-accessible (needs a registered data request; LAS down).
- Indian river-gauge discharge: not public (CWC-classified); we use a **static global-literature climatology**, not Indian gauge observations.
- RAMA (the only reachable moored source) has **no data in the model's time window**.

→ **These claims must be softened to explicit future work** in both the abstract and the README. What the build *can* honestly claim: an India-*focused* reconstruction for an underserved basin, using regime conditioning from independent climatology + a discharge seasonal cycle, with a clear roadmap for INCOIS/CWC integration once data access is arranged.

**2. Two more substitutions worth stating plainly in the README:**
- SST/SSH/SSS come from NOAA-hosted products (OISST, CoastWatch blended SLA, CoastWatch SMAP), not the CMEMS/PODAAC products named in the abstract. Scientifically equivalent for this purpose; just name them accurately.
- The barrier-layer climatology is WOA23-derived, not the published de Boyer Montégut product.

**3. What is solid:** Argo ground truth (14 k profiles, good NIO coverage), SST/SSH/SSS weekly stack, a non-circular regime pipeline (enforced by `tests/test_regime_independence.py`), and the physical holdout design. The core scientific question — *does regime-conditioning + physics loss help specifically in the barrier-layer regime* — can still be answered honestly with this data.

## Decision needed before Phase 2

Pick the evaluation window:
- **Keep 2021–2023** (current): best Argo density, recent; **no moored-buoy validation possible**.
- **Shift to 2016–2020**: RAMA `8n90e` (BoB) + `15n65e` (Arabian Sea) T/S overlap → a genuine independent moored validation at 2 sites; costs a re-fetch (~30 min) and Argo is slightly sparser.
- **Widen to 2016–2023**: both; ~2× fetch + compute.
