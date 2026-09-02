# Running OceanEmbed

Interim run notes. Folds into README at Phase 4.

## Environment

```
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
# GPU box (Lenovo LOQ): install the CUDA build of torch instead of the CPU one
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Dev box here: Python 3.14, **no CUDA** (Intel Iris Xe) — everything runs CPU, slowly.
`argopy` will not install on 3.14 (no aiohttp wheel); not needed.

## Data pipeline (Phase 0-1) — already run, cached under data/

```
python -m src.data_access.fetch_argo      --start 2021-01 --end 2023-12
python -m src.data_access.fetch_satellite --start 2021-01 --end 2023-12
python -m src.data_access.fetch_woa
python -m src.data_access.fetch_incois
python -m src.regime.build_climatology
python -m src.data_pipeline --patch 32
pytest -q
```

Produces `data/processed/{train,test_spatial,test_temporal}.npz` + `norm_stats.json`.

## Training (Phase 2)

```
# quick sanity (CPU, ~1 min)
python -m src.train --smoke --models both

# full run — do this on the Lenovo LOQ (auto-uses CUDA + AMP)
python -m src.train --models both --backbone resnet50 --epochs 40 --batch-size 128

# geospatial foundation model (needs: pip install terratorch)
python -m src.train --models oceanembed --backbone prithvi --epochs 40
python -m src.train --models baseline   --backbone prithvi --epochs 40
```

`--models both` trains OceanEmbed (FiLM regime-conditioning + physics loss) and the
baseline (identical, no FiLM, no physics term) on the same split — §5.5.

Outputs: `outputs/checkpoints/{oceanembed,baseline}.pt`,
`outputs/metrics/train_log_*.csv`.

Rough CPU timing here: resnet18 ~10 s/epoch. resnet50 on the LOQ GPU should be
a few seconds/epoch at batch 128.

### lambda_physics sweep (§5.4, optional, <1 h)

```
for L in 0.05 0.1 0.3; do
  python -m src.train --models oceanembed --backbone resnet50 --epochs 30 \
      --lambda-physics $L --name oceanembed_lp$L
done
```
