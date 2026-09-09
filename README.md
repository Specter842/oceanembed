# OceanEmbed

Regime-conditioned, physics-constrained deep learning for reconstructing
subsurface **temperature and salinity profiles** from satellite surface fields,
built for the **North Indian Ocean / Bay of Bengal**.

Smart India Hackathon · Ministry of Earth Sciences · Space Technology theme.

---

## What this is — and isn't

**The contribution is a *combination* applied to an *underserved basin*, not a new
algorithm.** Three independently-proven techniques —
1. transfer learning from a pretrained geospatial backbone,
2. regime conditioning from data *independent of the prediction inputs*,
3. a physics-consistency (equation-of-state) loss —

applied together to the Bay of Bengal, using a non-circular regime pipeline and
physically-separated holdouts.

**Not claimed:** a novel architecture, real-time ingestion (this is an offline
batch demo on downloaded historical files), a from-scratch foundation embedding,
or a guaranteed win in every regime. Where OceanEmbed does *not* beat the baseline,
this README says so (see [Results](#results)).

---

## Why this basin

The Bay of Bengal is the basin most responsible for the Indian subcontinent's
deadliest cyclones, and the one where subsurface reconstruction is hardest to get
right. The Ganga–Brahmaputra–Irrawaddy rivers pour freshwater onto the surface,
forming a low-salinity lens over denser, saltier water — a **salinity-driven
barrier layer** that suppresses vertical mixing and decouples the surface
thermal/haline signal from the subsurface structure. Global ML reconstructions,
trained mostly on the Atlantic, Pacific and global averages, assume a coupling
the Bay of Bengal breaks — and systematically misplace the thermocline and
upper-ocean heat content there, the exact quantities that drive tropical-cyclone
rapid intensification.

---

## Architecture

```
   SST      SSH(SLA)      SSS            regime one-hot [3]
  (OISST)  (CoastWatch)  (SMAP)          stratification index [1]      <- from an
     \        |          /               month sin/cos [2]                INDEPENDENT
      \       |         /                        |                        climatology,
   32x32 x 3-channel patch                 ConditionEncoder               never the
              |                                  |  cond [D]              satellite
     pretrained backbone  (frozen)               |                        channels
     ResNet-50  /  ResNet-18  /  Prithvi-EO      |
              |  features [F]                    |
              +------------  FiLM  <-------------+     (baseline: FiLM removed)
                             |
                     decoder MLP (dropout)
                             |
         +-------------------+-------------------+
      T [18 levels]      S [18 levels]      log-variance [2 x 18]
                                            (or MC-dropout at inference)

   loss = masked Gaussian NLL(T) + NLL(S)  +  lambda_physics * inversion_penalty
                                               (EOS-80 density, torch; validated
                                                against TEOS-10 / gsw)
```

Depth levels: 0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 700,
1000, 1500, 2000 m. The regime pipeline (`src/regime/`) is import-isolated from
the satellite fetchers — enforced by `tests/test_regime_independence.py`, not a
comment.

---

## Data

| Role | Product | Source used | Note |
|---|---|---|---|
| Ground truth | Argo T/S profiles (QC 1–2, delayed-mode preferred) | **AOML ERDDAP** | Ifremer GDAC unreachable from the build machine |
| Input — SST | OISST v2.1 daily | **NCEI** | coastwatch.pfeg unreachable |
| Input — SSH | Blended sea-level anomaly | **NOAA CoastWatch** | CMEMS needs an account |
| Input — SSS | SMAP L3 | **NOAA CoastWatch** | PODAAC needs Earthdata login |
| Regime climatology | MLD / barrier-layer / stratification | **derived from WOA23** monthly T/S | de Boyer Montégut product unreachable |
| Regime input | Ganga–Brahmaputra discharge | **static monthly climatology** | not gauge data — see below |

Window: 2021-01 … 2023-12, weekly composites, 0.25°, box 5°S–25°N / 45–100°E.
**10,962** QC'd profiles → **8,340** matched to satellite → split:

| Split | Profiles | barrier-layer | well-mixed | upwelling |
|---|--:|--:|--:|--:|
| `train` (Arabian Sea + equatorial IO) | 6,656 | 1,719 | 4,601 | 336 |
| `test_spatial` — **Bay of Bengal** (lat ≥ 5 & lon ≥ 85) | 1,044 | 789 | 220 | 35 |
| `test_temporal` — **JJAS 2022** (non-BoB) | 640 | 82 | 505 | 53 |

Holdouts are physical, not random: the entire Bay of Bengal is withheld
spatially, and one full monsoon season temporally. Full reachability account:
[`data/raw/phase0_report.md`](data/raw/phase0_report.md).

---

## Results

**Preliminary** — ResNet-18 backbone, 20 epochs, CPU. The definitive runs
(ResNet-50 / Prithvi-EO, 40+ epochs, GPU) are set up for a separate machine in
[`RUN.md`](RUN.md). Regenerate everything with `python -m src.evaluate && python -m src.viz.plots`.

**Temperature RMSE (°C), pooled over depth:**

| Holdout | Regime | Baseline | OceanEmbed |
|---|---|--:|--:|
| spatial (Bay of Bengal) | **barrier-layer** | **0.962** | 0.985 |
| spatial | well-mixed | **0.935** | 0.949 |
| spatial | upwelling | 1.079 | **1.052** |
| temporal (JJAS 2022) | **barrier-layer** | 1.063 | **1.049** |
| temporal | well-mixed | 1.209 | **1.187** |
| temporal | upwelling | **1.376** | 1.397 |

Salinity RMSE and Pearson r are within noise between the two models everywhere
(profile-shape r ≈ 0.99 for both). Full per-depth table:
`outputs/metrics/per_regime_rmse.csv`.

**Uncertainty is calibrated on real held-out Argo.** Using the model's predicted
variance, an 80 % temperature interval covers **81 %** of Bay-of-Bengal Argo
observations (82 % in the barrier-layer regime), a 95 % interval covers **95 %**,
mean calibration error ≈ **1 pt**. On the temporal (monsoon) holdout the
temperature intervals are mildly over-confident (80 % nominal → 76 % actual,
≈ 2.6 pt); salinity calibrates ≈ 2–3 pt. OceanEmbed and the baseline calibrate
near-identically. Regenerated into `outputs/metrics/calibration.csv` by
`python -m src.evaluate`.

### Honest read (CLAUDE.md §6.2)

**On the primary test — the Bay of Bengal, barrier-layer regime — OceanEmbed does
not beat the baseline** (0.985 vs 0.962 °C, ~2% worse). On the temporal holdout it
is ~1% better in that regime, with a clearer local gain (−0.15 °C at 75 m). Every
difference is within run-to-run noise at this model size. **We did not retrain to
change this.** Likely causes:

- The conditioning signal is weakest exactly on the spatial holdout: held-out BoB
  cells take their regime label from the 1° WOA23 climatology, which is nearly
  uniform across that region — so FiLM has little to modulate on. *Most likely
  fixable cause* (a finer barrier-layer climatology).
- By design, no BoB barrier layers are in training — only SE-Arabian-Sea /
  equatorial ones, which may not transfer to the fresher BoB lens.
- [SST, SSH, SSS] under-constrains the sub-surface halocline.
- `lambda_physics` ∈ {0.05, 0.1, 0.3} makes **no measurable difference** — both
  models already produce density-stable profiles (TEOS-10 inversion fraction
  ≈ 0.00), so the physics term is idle at this data scale.

Details: [`outputs/metrics/PHASE3_FINDINGS.md`](outputs/metrics/PHASE3_FINDINGS.md).

---

## Limitations & what is future work

- **India-specific observational data / "data sovereignty" — NOT delivered.**
  INCOIS moored-buoy data is not machine-accessible (portal needs a registered
  ESSO request; the Live Access Server is unreachable). River discharge is a
  literature climatology (Papa et al. 2012 shape), **not** Indian gauge data
  (CWC-classified). RAMA — the one reachable moored array with a BoB site — has no
  data after 2020. The abstract's differentiator on this point is a roadmap item.
- **Substituted products.** SST/SSH/SSS are NOAA-hosted, not the CMEMS/PODAAC
  products in the abstract (scientifically equivalent; just named accurately).
  The barrier-layer climatology is WOA23-derived, not the published de Boyer
  Montégut product. Ocean-colour channel omitted (no reachable historical source).
- **Preliminary training.** CPU, ResNet-18, 20 epochs. Prithvi-EO is wired
  (`--backbone prithvi`, via `terratorch`) but the tested path is ResNet.
- **Not implemented:** real-time ingestion; active-learning Argo-deployment
  recommendations; cyclone rapid-intensification flagging. All roadmap only.
- WOA23-derived MLD is too shallow in the winter Arabian Sea (shallow halocline
  vs deep isothermal layer) — the regime *class* is robust to this via the
  stratification gate, but continuous `blt_m` should be used with that caveat.

---

## Run it

```bash
# dashboard only (what's deployed):
pip install -r requirements.txt
streamlit run dashboard/app.py

# full pipeline (rebuild data / retrain / re-evaluate):
pip install -r requirements-pipeline.txt
python -m src.train --models both --backbone resnet50 --epochs 40   # on a GPU box
python -m src.evaluate
python -m src.viz.plots
```

`pytest -q` runs the regime-independence and split-integrity checks.
Full instructions, including the Lenovo LOQ / CUDA path and the Prithvi-EO
opt-in: [`RUN.md`](RUN.md).

## Repo layout

```
src/data_access/   fetch_argo, fetch_satellite, fetch_woa, fetch_discharge,
                   fetch_incois (INCOIS probe + RAMA), common (reachable endpoints)
src/regime/        build_climatology (WOA23 -> MLD/BLT/strat), regime_labels
src/models/        backbone, conditioning (FiLM), physics_loss (EOS-80), oceanembed_model
src/               data_pipeline (matching + holdouts), datasets, train, evaluate, config
src/viz/           plots
dashboard/app.py   Streamlit demo (about + 6 views; reads pre-computed artefacts, no torch)
data/raw/phase0_report.md      data reality check
outputs/metrics/PHASE3_FINDINGS.md   results analysis
```
