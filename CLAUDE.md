# CLAUDE.md — OceanEmbed Build Context (Detailed)

Read this in full before writing any code. Re-read the relevant phase section immediately before starting that phase — do not rely on memory of it from earlier in the session. Do not proceed past a phase's verification checklist until every box is genuinely checked, not assumed.

If at any point you (Claude Code) are about to write a comment, README line, or pitch sentence containing words like "novel architecture," "real-time," "first-ever," or "proprietary embedding," stop and surface it to me before writing it. These are the exact overclaims we have deliberately removed from this project's framing.

---

## 0. Project Identity & Non-Negotiable Framing

**Name:** OceanEmbed
**Problem Statement:** Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations (MoES, SIH)

**What we are claiming (memorize this, it governs every design choice below):**
We combine three independently-proven techniques — (1) transfer learning from a pretrained geospatial foundation model, (2) regime-conditioning derived from data sources independent of the prediction inputs, (3) a physics-consistency loss in the style of OSnet — and apply this combination to the Bay of Bengal / North Indian Ocean, a basin underserved by existing published work (which concentrates on the Gulf Stream, South China Sea, Kuroshio Extension, and global averages), using India-specific observational data that international groups rarely integrate.

**What we are explicitly NOT claiming:**
- Not a new architecture family or algorithm
- Not real-time ingestion (the demo runs on downloaded historical files)
- Not a from-scratch-trained foundation embedding
- Not guaranteed outperformance in every regime — if the barrier-layer regime doesn't show a clear win in validation, that is reported honestly, not massaged

---

## 1. Tech Stack (fixed — do not substitute without flagging)

- **Language:** Python 3.11
- **Data handling:** `xarray`, `netCDF4`, `pandas`, `numpy`
- **Geophysics helper:** `gsw` (TEOS-10 seawater equation of state library — use this for the physics-consistency loss, do not hand-roll the equation of state)
- **Deep learning:** PyTorch (not TensorFlow — foundation model checkpoints for Prithvi-EO are PyTorch-native)
- **Foundation model:** attempt `ibm-nasa-geospatial/Prithvi-EO-2.0` from HuggingFace; fallback is a `torchvision` pretrained ResNet-50 encoder on satellite channels stacked as pseudo-RGB if Prithvi weights are inaccessible or too large to fine-tune in the time available
- **Regridding/interpolation:** `xesmf` if installable (requires ESMF, can be a pain — if install fails within 20 minutes, fall back to `scipy.interpolate.griddata`)
- **Experiment tracking:** plain CSV logs + `matplotlib` — do not spend time setting up W&B or MLflow, it's not worth the setup cost for a 36-hour build
- **Dashboard:** `streamlit` (fastest path to a working demo UI)
- **Environment:** single `requirements.txt`, one virtualenv, no Docker — containerization costs time you don't have

```
requirements.txt (starting point, adjust as needed):
xarray
netCDF4
pandas
numpy
scipy
gsw
torch
torchvision
huggingface_hub
streamlit
matplotlib
plotly
requests
```

---

## 2. Repository Structure (create this exact layout in Phase 0)

```
oceanembed/
├── CLAUDE.md                    (this file)
├── requirements.txt
├── data/
│   ├── raw/                     (downloaded, untouched source files)
│   ├── interim/                 (regridded/matched intermediate files)
│   └── processed/               (final training-ready tensors, e.g. .npz or .zarr)
├── src/
│   ├── data_access/
│   │   ├── check_sources.py     (Phase 0 — tests reachability of every data source)
│   │   ├── fetch_argo.py
│   │   ├── fetch_satellite.py
│   │   ├── fetch_incois.py
│   │   └── fetch_discharge.py
│   ├── regime/
│   │   └── regime_labels.py     (builds regime index — MUST NOT import satellite SST/SSS/SSH modules)
│   ├── models/
│   │   ├── backbone.py
│   │   ├── conditioning.py      (FiLM-style adapter)
│   │   ├── physics_loss.py      (gsw-based EOS consistency term)
│   │   └── oceanembed_model.py
│   ├── train.py
│   ├── evaluate.py
│   └── viz/
│       └── plots.py
├── notebooks/                    (throwaway exploration only — nothing here should be load-bearing for the demo)
├── dashboard/
│   └── app.py                   (streamlit)
├── outputs/
│   ├── metrics/
│   │   └── per_regime_rmse.csv
│   └── figures/
└── README.md                     (written last, after Phase 3 results exist)
```

---

## 3. Phase 0 — Data Reality Check (hard stop before any modeling code)

**Time budget: no more than 3 hours.** If a source isn't confirmed reachable in this window, treat it as unavailable and move on — don't let one stubborn API eat your whole day.

For each source below, write a small standalone script in `src/data_access/check_sources.py` that attempts a real fetch (not just a ping) and logs success/failure with the actual error message.

1. **Argo profiles** — try `https://data-argo.ifremer.fr` GDAC index files, or an ERDDAP endpoint (e.g. `https://www.ifremer.fr/erddap/`). Confirm you can pull at least one NetCDF profile file for the Bay of Bengal region (roughly 5°N–22°N, 80°E–95°E) and parse temperature/salinity/pressure arrays out of it.
2. **Satellite SST** — try NOAA OISST via ERDDAP (`https://coastwatch.pfeg.noaa.gov/erddap/`) or PODAAC. Confirm a subset download for the same bounding box works and opens cleanly in `xarray`.
3. **Satellite SSH/ADT** — try CMEMS. Note: CMEMS typically requires account registration; if that approval isn't instant, log it as blocked and fall back to a static sample SSH file (check if PODAAC or a public Zenodo/OpenDAP mirror has a usable substitute).
4. **SMAP SSS** — try PODAAC's SMAP L3 SSS product.
5. **Barrier-layer climatology** — search for a machine-readable version of published North Indian Ocean barrier-layer thickness climatology (Thadathil et al. or similar). If none is downloadable as data (only as figures in a paper), this becomes a **manually digitized static lookup table** keyed by month × lat/lon bin, built by reading approximate values off published figures. Flag this explicitly — it is a legitimate but low-precision substitute, and must be described accurately (not implied to be a live dataset) in any documentation.
6. **River discharge (Ganga-Brahmaputra)** — check India-WRIS, CWC public portals, or global river discharge datasets (e.g. GRDC) for a station near the delta. If inaccessible in time, fall back to a static seasonal climatological discharge curve (monthly average, not real observations) and label it as such everywhere it's used.
7. **INCOIS buoy data** — check INCOIS's public data portal / OGD platform for moored buoy time series. This is the highest-risk item and the core of your "India-specific data" differentiator claim.

**Output required from this phase:** a markdown table in `data/raw/phase0_report.md` with columns: Source | Status (Accessible / Accessible-with-delay / Unavailable / Substitute-used) | Notes | Substitute-if-any.

**STOP. Do not proceed to Phase 1 until I have reviewed `phase0_report.md`.** If INCOIS or river discharge came back Unavailable, tell me explicitly — we need to revise the abstract's "data sovereignty" claim before continuing, not after the demo is built around a claim we can't support.

---

## 4. Phase 1 — Data Pipeline

**Time budget: ~6 hours.**

### 4.1 Grid and matching spec
- Target spatial resolution: 0.25° × 0.25° (matches most satellite products at native or easily-downsampled resolution; don't over-engineer to finer resolution than your Argo density supports)
- Target depth grid for output profiles: standard levels at 0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 700, 1000, 1500, 2000 m (or whatever subset your Argo data actually reaches — don't predict depths you have no ground truth for)
- Temporal resolution: weekly composites (daily satellite data is noisier and Argo profiles aren't daily-dense enough to justify daily granularity)

### 4.2 Regime label pipeline (`src/regime/regime_labels.py`)
This function's signature should look like:
```python
def compute_regime_label(lat: float, lon: float, month: int, river_discharge: float) -> dict:
    """
    Returns e.g. {"regime_class": "barrier_layer_stratified", "stratification_index": 0.73}
    Uses ONLY: static climatology lookup + river_discharge + month.
    MUST NOT import or reference satellite SST/SSS/SSH modules.
    """
```
Write a unit test (`tests/test_regime_independence.py`) that statically checks this module's imports contain no reference to `fetch_satellite`. This is not paranoia — it's the actual fix for the circularity flaw identified earlier in this project, and it needs to be enforced by a test, not just a promise in a docstring.

### 4.3 Matching and splits
- Build a matched dataset: for each (lat, lon, week), a row containing satellite input channels, the regime label, and the Argo-derived ground-truth profile (where available — most cells will NOT have a co-located Argo profile; only keep rows where a profile exists within a reasonable spatiotemporal tolerance, e.g. ±3 days, ±0.5°).
- **Spatial holdout:** physically exclude all Bay of Bengal grid cells (e.g. lon > 85°E within the northern Indian Ocean box) from the training file; put them in a separate `test_spatial.npz`.
- **Temporal holdout:** from the remaining training data, exclude one full monsoon season (June–September of one year) into `test_temporal.npz`.
- Do this split at the file-writing stage, not via a random `train_test_split` call — the whole point is that these are non-random, physically separated holdouts.

**Verification checklist before Phase 2:**
- [ ] `tests/test_regime_independence.py` passes
- [ ] At least one season of matched rows exists in `data/processed/train.npz`
- [ ] `test_spatial.npz` and `test_temporal.npz` exist and contain zero overlapping rows with `train.npz` (write a quick assertion script for this, don't eyeball it)

---

## 5. Phase 2 — Model Build

**Time budget: ~10 hours.**

### 5.1 Architecture sketch
```
Input: satellite channel stack [SST, SSH, SSS, ocean_color] at (lat, lon, week)
   │
   ▼
Pretrained backbone (Prithvi-EO encoder, or ResNet-50 fallback)
   │
   ▼  (backbone features)
FiLM conditioning layer ── modulated by regime_label / stratification_index from Phase 1
   │
   ▼
Decoder (small MLP or transposed-conv stack) → per-depth-level (T, S) predictions
   │
   ▼
Uncertainty head (predict log-variance per depth level, or use MC-dropout at inference — pick MC-dropout if time is short, it requires zero extra training complexity)
```

### 5.2 FiLM conditioning (`src/models/conditioning.py`)
```python
class FiLMLayer(nn.Module):
    def __init__(self, feature_dim, condition_dim):
        super().__init__()
        self.gamma = nn.Linear(condition_dim, feature_dim)
        self.beta = nn.Linear(condition_dim, feature_dim)

    def forward(self, features, condition_vector):
        gamma = self.gamma(condition_vector)
        beta = self.beta(condition_vector)
        return features * (1 + gamma) + beta
```
`condition_vector` = the stratification index (and/or one-hot regime class) from Phase 1, NOT raw satellite SSS.

### 5.3 Physics-consistency loss (`src/models/physics_loss.py`)
Use the `gsw` library to compute density from predicted (T, S, pressure) at each depth level, and penalize:
1. Density inversions (density should be non-decreasing with depth in a stable water column) — penalize any negative vertical density gradient.
2. Deviation from the TEOS-10 equation of state relationship between predicted T, S, and an independently-estimated density profile, if available from reanalysis.

```python
def physics_consistency_loss(pred_T, pred_S, pressure_levels):
    density = gsw.rho(pred_S, pred_T, pressure_levels)  # per depth
    depth_diffs = density[:, 1:] - density[:, :-1]
    inversion_penalty = torch.relu(-depth_diffs).mean()
    return inversion_penalty
```

### 5.4 Total loss
```
total_loss = data_loss (MSE against Argo ground truth)
           + lambda_physics * physics_consistency_loss
```
Start with `lambda_physics = 0.1` and treat it as a tunable hyperparameter — log results at a couple of values (e.g. 0.05, 0.1, 0.3) if time permits, but don't spend more than an hour on this sweep.

### 5.5 Baseline model
Train an identical architecture WITHOUT the FiLM conditioning layer and WITHOUT the physics loss term (data loss only, single global model) on the exact same data split. This is your comparison baseline for Phase 3 — it must exist before Phase 3 begins.

**Verification checklist before Phase 3:**
- [ ] Both models (OceanEmbed and baseline) train for at least a few epochs without NaN loss
- [ ] Checkpoints for both are saved to `outputs/checkpoints/`

---

## 6. Phase 3 — Validation & Evidence

**Time budget: ~6 hours.**

### 6.1 Metrics
For both models, on both `test_spatial.npz` and `test_temporal.npz`, compute:
- RMSE per depth level (T and S separately)
- RMSE broken down by `regime_class` (barrier_layer_stratified vs. well_mixed vs. upwelling) — **this table is the single most important output of the entire project**
- Correlation coefficient (Pearson r) between predicted and observed profiles

Write this to `outputs/metrics/per_regime_rmse.csv` with columns: `model, test_set, regime_class, depth_level, rmse_T, rmse_S, r_T, r_S`.

### 6.2 Honesty check
If `per_regime_rmse.csv` shows the OceanEmbed model does NOT beat baseline in `barrier_layer_stratified` rows specifically, **do not retrain repeatedly until it does** — report it, and write one paragraph in the README discussing likely causes (e.g. insufficient barrier-layer training examples, climatology resolution too coarse, lambda_physics miscalibrated). A judge will find an honest negative result more credible than a suspiciously clean win, and it protects you from being caught overfitting the story to the number.

### 6.3 Plots (`src/viz/plots.py`)
- Depth-profile comparison plot: predicted vs. Argo ground truth, for 2–3 illustrative barrier-layer-regime locations
- Uncertainty map: spatial plot of predictive uncertainty across the Bay of Bengal
- Bar chart: RMSE by regime, OceanEmbed vs. baseline, side by side

**Verification checklist before Phase 4:**
- [ ] `per_regime_rmse.csv` exists and is generated by a script (re-runnable), not manually typed
- [ ] At least one plot exists showing the regime-level comparison

---

## 7. Phase 4 — Demo & Presentation

**Time budget: remaining hours.**

### 7.1 Dashboard (`dashboard/app.py`, Streamlit)
Sections, in order:
1. Map view: input satellite fields for a selected week
2. Reconstructed subsurface profile viewer: click a location, see predicted T/S profile vs. Argo ground truth (for holdout points) with uncertainty band
3. Regime overlay: map colored by regime_class
4. Comparison panel: OceanEmbed vs. baseline RMSE bar chart, filterable by regime
5. A clearly labeled "Data & Scope" footer stating: "This demo runs on historical downloaded satellite and Argo data. Real-time ingestion and the float-deployment/cyclone-flagging features described in the roadmap are not implemented in this build."

### 7.2 README
Write this last. Structure:
- Problem & why this basin (barrier layer, underserved region — from the abstract)
- What we built vs. what's future work (be explicit and match Phase 0's actual findings — if INCOIS data didn't work out, the README must say so, not silently drop the claim)
- Architecture diagram (can be the ASCII sketch from section 5.1, cleaned up)
- Results table (per-regime RMSE)
- Honest limitations section

---

## 8. Standing Rules (apply throughout, not just at the phase they're mentioned in)

- Never claim real-time ingestion when the demo runs on static files.
- Never claim a "novel algorithm" — the claim is application/combination novelty to an underserved basin, stated explicitly, with the OSnet/OG-PINN/foundation-model precedents credited.
- Never let the regime-label pipeline silently import satellite SST/SSS/SSH — this is enforced by `tests/test_regime_independence.py`, which must exist and must pass.
- Never tune results until the barrier-layer regime shows an improvement — report what validation actually produces.
- Never fabricate or assume a data source "would probably work" instead of testing it in Phase 0.
- If a phase's time budget is exceeded by more than ~50%, stop and tell me rather than silently cutting corners on the verification checklist to catch up.
