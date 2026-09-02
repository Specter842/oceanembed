"""
Ganga-Brahmaputra river discharge  (CLAUDE.md §3 item 6).

Phase 0 finding: live gauge data is NOT publicly obtainable. CWC classifies
discharge for the Ganga & Brahmaputra (transboundary basins) and India-WRIS does
not serve it; GRDC requires written-request approval. So this module provides a
STATIC MONTHLY CLIMATOLOGY, not observations, and every downstream use is
labelled as such.

Values: approximate monthly-mean combined Ganga+Brahmaputra freshwater discharge
(m^3 s^-1), reflecting the well-established seasonal cycle — dry Feb-Mar minimum,
Jul-Aug monsoon peak ~2.5x annual mean (annual mean ~3.3e4 m^3 s^-1, ~1050 km^3
yr^-1). Shape after Papa et al. (2012, JGR-Oceans) and Dai (2017) global runoff;
magnitudes are indicative and should be replaced with a digitised figure trace if
precision matters. This is a regime-conditioning input only — it is deliberately
independent of any satellite field (no circularity).

This module must stay importable by src/regime/ ; it must NOT import satellite code.
"""

from __future__ import annotations

import numpy as np

# month (1-12) -> combined Ganga-Brahmaputra discharge, m^3/s  (STATIC CLIMATOLOGY)
_GB_MONTHLY_CLIM = {
    1: 16_000, 2: 11_000, 3: 9_000, 4: 10_000, 5: 17_000, 6: 42_000,
    7: 75_000, 8: 82_000, 9: 62_000, 10: 43_000, 11: 25_000, 12: 18_000,
}

_ANNUAL_MEAN = float(np.mean(list(_GB_MONTHLY_CLIM.values())))

IS_CLIMATOLOGY = True
SOURCE_NOTE = ("static monthly climatology (Papa et al. 2012 / Dai 2017 shape); "
               "NOT observed discharge — live CWC/India-WRIS data is classified")


def discharge_for_month(month: int) -> float:
    """Combined Ganga-Brahmaputra discharge (m^3/s) for a calendar month, 1-12."""
    if not 1 <= int(month) <= 12:
        raise ValueError(f"month must be 1-12, got {month}")
    return float(_GB_MONTHLY_CLIM[int(month)])


def discharge_anomaly(month: int) -> float:
    """Discharge as a fraction above/below annual mean (dimensionless).
    ~ -0.7 in the dry season, ~ +1.5 at the monsoon peak."""
    return discharge_for_month(month) / _ANNUAL_MEAN - 1.0


def monthly_climatology() -> dict[int, float]:
    return dict(_GB_MONTHLY_CLIM)


def fetch_river_discharge(*_, **__):
    """No live source exists (see module docstring). Kept for API symmetry."""
    raise NotImplementedError(
        "Live Ganga-Brahmaputra discharge is not publicly available "
        "(CWC-classified). Use discharge_for_month() / monthly_climatology()."
    )


if __name__ == "__main__":
    for mth in range(1, 13):
        print(f"  {mth:2d}  {discharge_for_month(mth):>7.0f} m3/s   "
              f"anom {discharge_anomaly(mth):+.2f}")
    print(f"\n  annual mean ~ {_ANNUAL_MEAN:.0f} m3/s   ({SOURCE_NOTE})")
