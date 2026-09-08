# Running OceanEmbed

Two machines are involved:
- **this PC** (no CUDA GPU) — data pipeline + code, smoke-tested only
- **Lenovo LOQ** (CUDA GPU) — the real training + evaluation runs

Everything is CPU/GPU-agnostic; `src/train.py` auto-detects CUDA and turns on AMP.

---

## 0. Environment

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate on Linux
pip install -r requirements.txt
```

On the LOQ, install the CUDA build of torch instead of the CPU one:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Optional (for the Prithvi-EO backbone — see step 3):

```bash
pip install terratorch
```

Python 3.14 works. `argopy` does **not** install on 3.14 and is not needed.

---

## 1. Data pipeline  (already run on this PC; re-run only to rebuild)

```bash
python -m src.data_access.fetch_argo       --start 2021-01 --end 2023-12
python -m src.data_access.fetch_satellite  --start 2021-01 --end 2023-12
python -m src.data_access.fetch_woa
python -m src.regime.build_climatology
python -m src.data_access.fetch_incois            # INCOIS probe + RAMA download
python -m src.data_pipeline --patch 32            # -> data/processed/*.npz
pytest -q                                         # regime independence + split integrity
```

Reachable data sources are pinned in `src/data_access/common.py` (this network
cannot reach ifremer / seanoe / pfeg — NOAA-hosted equivalents are used).
`data/raw/phase0_report.md` has the full account.

Outputs: `data/processed/{train,test_spatial,test_temporal}.npz` + `norm_stats.json`.

---

## 2. Training  (run on the LOQ)

Full run — OceanEmbed **and** the baseline (§5.5), same split, same seed:

```bash
python -m src.train --models both --backbone resnet50 --epochs 40 --batch-size 128
```

- `--models both` writes `outputs/checkpoints/oceanembed.pt` and `baseline.pt`.
- The baseline is auto-derived: identical config, `use_film=False`,
  `lambda_physics=0`.
- CUDA + AMP kick in automatically; on CPU it falls back (slower, still works).
- `--freeze-backbone true` (default) trains only the adapter + FiLM + decoder
  (~0.6 M params). Set `false` to fine-tune the whole backbone if VRAM allows.

`lambda_physics` sweep (§5.4 — a couple of values, ~1 h budget):

```bash
python -m src.train --models oceanembed --lambda-physics 0.05 --name oe_lp05 --epochs 40 --backbone resnet50
python -m src.train --models oceanembed --lambda-physics 0.30 --name oe_lp30 --epochs 40 --backbone resnet50
```

Smoke test (any machine, ~1 min): `python -m src.train --smoke --models both`

---

## 3. Prithvi-EO backbone  (optional, LOQ)

```bash
pip install terratorch
python -m src.train --models both --backbone prithvi --epochs 40 --batch-size 32 --freeze-backbone true
```

`src/models/backbone.py` adapts the 3 ocean channels to Prithvi's 6-band stem
and upsamples patches to 224 px. If `terratorch` or the weights are missing it
logs a warning and falls back to ResNet-50, so the command never hard-fails.

---

## 4. Evaluation + figures  (run wherever the checkpoints are)

```bash
python -m src.evaluate                    # both checkpoints x both holdouts
python -m src.viz.plots
```

Writes:
- `outputs/metrics/per_regime_rmse.csv`  — the key table (§6.1)
- `outputs/metrics/physics_diagnostics.csv`  — TEOS-10 inversion rate (gsw)
- `outputs/metrics/predictions_<model>_<set>.npz`
- `outputs/figures/{rmse_by_regime,profiles_barrier_layer,uncertainty_map_bob}.png`

`--mc-dropout 20` uses MC-dropout for uncertainty instead of the log-variance head.

Then update `outputs/metrics/PHASE3_FINDINGS.md` / the README results section with
the real numbers. **Do not tune to make OceanEmbed win the barrier-layer regime**
(CLAUDE.md §6.2 / §8) — report what the run produces.

---

## 5. Dashboard  (Phase 4)

Pure-black Streamlit dashboard, 9 sections. It reads a baked numpy bundle
(`dashboard/assets.npz`) plus the `outputs/metrics/` CSV/NPZ files — **no torch,
netCDF4, xarray or gsw at runtime**, so it starts fast and runs anywhere.

```bash
python -m dashboard.prep_assets        # once, after the pipeline — bakes assets.npz (~7 MB)
python -m streamlit run dashboard/app.py --client.toolbarMode minimal
```

`assets.npz` is committed, so a fresh clone can run the dashboard straight away;
re-run `prep_assets` only if the satellite stack or the regime climatology change.
After a new training run, re-run `python -m src.evaluate` so the comparison and
skill panels pick up the new checkpoints.
