"""
OceanEmbed demo dashboard (Streamlit) — CLAUDE.md §7.1.

Sections, in order:
  1. Map view: input satellite fields for a selected week
  2. Reconstructed subsurface profile viewer (click a location) vs Argo truth,
     with uncertainty band
  3. Regime overlay: map coloured by regime_class
  4. Comparison panel: OceanEmbed vs baseline RMSE bar chart, filter by regime
  5. "Data & Scope" footer — verbatim:
     "This demo runs on historical downloaded satellite and Argo data. Real-time
      ingestion and the float-deployment/cyclone-flagging features described in
      the roadmap are not implemented in this build."

Status: STUB — implemented in Phase 4. Run: streamlit run dashboard/app.py
"""

from __future__ import annotations


def main():
    raise NotImplementedError("Phase 4 — see CLAUDE.md §7.1")


if __name__ == "__main__":
    main()
