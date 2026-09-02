# OceanEmbed

Regime-aware, physics-constrained deep learning for subsurface ocean
reconstruction in the North Indian Ocean / Bay of Bengal.

> **The real README is written last** (CLAUDE.md §7.2), after Phase 3 results
> exist, so it can state what was actually built vs. what remained future work.
> This placeholder only records current build state.

## What this claims

Combination + underserved-basin novelty — **not** a new architecture, **not**
real-time ingestion, **not** a from-scratch foundation embedding. See
`CLAUDE.md` §0 for the exact framing this project holds to.

## Build status

| Phase | State |
|---|---|
| 0 — Data reality check | in progress |
| 1 — Data pipeline | not started |
| 2 — Model build | not started |
| 3 — Validation & evidence | not started |
| 4 — Demo & presentation | not started |

## Setup

```bash
py -3.14 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Tier 1 only for now
```

Environment note: target stack was Python 3.11 (CLAUDE.md §1); actual env is
3.14 — flagged, low risk (torch / netCDF4 / xarray all ship 3.14 wheels).
Machine is CPU-only (Intel Iris Xe), so the ResNet-50 frozen-feature path is the
practical primary in Phase 2, not the fallback.

## Phase 0 report

See [`data/raw/phase0_report.md`](data/raw/phase0_report.md) once generated.
