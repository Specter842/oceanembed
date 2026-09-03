# Abstract revisions — India-specific-data / data-sovereignty claim

**Why:** Phase 0 established that the abstract's differentiator on "integrating
India's own observational assets" cannot be substantiated by this build:

- **INCOIS moored-buoy data** is not machine-accessible — the OGD portal needs a
  registered ESSO-INCOIS data request and the Live Access Server is unreachable.
- **Indian river-gauge discharge** (CWC / India-WRIS) is not public — the
  Ganga & Brahmaputra are transboundary basins and discharge is classified.
  The build uses a *static monthly climatology* (Papa et al. 2012 seasonal
  shape), not gauge observations.
- **RAMA** (the only reachable moored array with a Bay of Bengal site) has no
  data after 2020 — no overlap with the 2021–2023 evaluation window.

The honest position: this is an India-*focused* reconstruction for an underserved
basin, with a **roadmap** for integrating India's observational assets once data
access is arranged. Edits below make the abstract match that.

---

## §2 — Proposed Solution (2nd paragraph)

**Before:**
> Each piece has precedent elsewhere; their joint application to the Bay of
> Bengal / North Indian Ocean, using India-specific observational data that
> international research groups rarely integrate, is the novel contribution.

**After:**
> Each piece has precedent elsewhere; their joint application to the Bay of
> Bengal / North Indian Ocean — a basin the international ocean-ML community has
> largely passed over in favour of data-rich, easily-validated regions — is the
> contribution. The framework is designed to ingest India's own subsurface
> observational assets (INCOIS moored buoys, CWC river gauges) as they become
> accessible; the current build runs on globally-available satellite and Argo
> data and treats that India-specific integration as a defined next step
> (Section 4).

---

## §3.1 — Data Sources

**Before:**
> - **Regime-conditioning inputs (independent of the predicted variable, to avoid
>   circularity):** published barrier-layer thickness climatology for the North
>   Indian Ocean, river discharge gauge data (Ganga-Brahmaputra-Irrawaddy
>   systems), and seasonal/monsoon phase indicators
> - **India-specific enrichment (differentiator):** INCOIS moored buoy data and
>   coastal tide-gauge networks, integrated as additional training/validation
>   signal not commonly used by international groups publishing on this problem

**After:**
> - **Regime-conditioning inputs (independent of the predicted variable, to avoid
>   circularity):** an upper-ocean stratification / barrier-layer climatology
>   derived from WOA23 monthly T/S, a monthly climatology of combined
>   Ganga-Brahmaputra freshwater discharge (seasonal cycle after Papa et al.
>   2012 — a climatological curve, not live gauge data), and monsoon-phase
>   indicators
> - **Planned India-specific enrichment (not in the current build):** INCOIS
>   moored-buoy profiles and CWC river-gauge discharge, to be added as
>   additional training/validation signal once data-access agreements are in
>   place. These are not publicly downloadable today (INCOIS bulk data requires a
>   registered ESSO request; CWC discharge for the Ganga/Brahmaputra is
>   classified), so the current build substitutes the climatological inputs above.

---

## §4 — Feasibility and Scope Honesty (add a sentence)

Append to the first paragraph, or add as a third scoped-out item:

> (c) integration of India's own subsurface observational assets — INCOIS moored
> buoys and CWC river-gauge discharge — which are not currently available for
> programmatic download. Until that access is arranged the framework uses a
> WOA23-derived stratification climatology and a climatological discharge curve
> in their place; the pipeline is built so these sources can be swapped in
> without architectural change.

---

## §5 — Impact

**Before:**
> - **Data sovereignty argument:** Demonstrates that integrating India's own
>   observational assets (INCOIS, IMD, CWC data) into globally-inspired ML
>   architectures closes a gap that international research, optimizing for
>   data-rich basins, has not prioritized.

**After:**
> - **Data sovereignty roadmap:** The framework is structured so that India's own
>   observational assets (INCOIS buoys, IMD, CWC gauges) can be integrated into a
>   globally-inspired ML architecture as access is arranged — closing a gap that
>   international research, optimising for data-rich basins, has not prioritised.
>   The current build demonstrates the architecture on open data; the
>   India-specific integration is the next milestone, not a completed result.

---

## §6 — Key Differentiators table

**Before (Data sources row):**
> | Data sources | Global satellite products only | Adds INCOIS buoys, Indian
> river discharge gauges — an access advantage specific to an Indian team |

**After:**
> | Data sources | Global satellite products only | Same open satellite + Argo
> data; regime conditioning from an independent WOA23-derived climatology + a
> Ganga-Brahmaputra discharge climatology. Architecture is built to ingest INCOIS
> buoy + CWC gauge data as a planned next step (not in this build). |

Also soften the **Regional focus** row's "OceanEmbed" cell if it implies
completed India-data integration — keep it to "North Indian Ocean / Bay of Bengal
specifically", which is fully supported.

---

## What stays exactly as written (fully supported by the build)

- Regional focus on the Bay of Bengal / North Indian Ocean.
- Non-circular regime conditioning (physically-independent climatology + discharge
  curve, not inferred from the satellite channels) — enforced by a test.
- Physics-informed loss (equation-of-state consistency).
- Per-regime RMSE reporting with an explicit barrier-layer breakdown.
- Spatial-block + temporal (monsoon-season) holdouts.
- The offline-batch scope statement in §4.
