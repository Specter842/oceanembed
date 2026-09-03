# Phase 3 — findings (preliminary)

**Models evaluated:** ResNet-18 backbone (frozen), 20 epochs, CPU. These are
*preliminary* models — the definitive runs (ResNet-50 / Prithvi-EO, 40+ epochs,
GPU + AMP, lambda_physics sweep) happen on the Lenovo LOQ. Numbers below will be
regenerated from those.

Source of truth: `outputs/metrics/per_regime_rmse.csv` (re-run `python -m src.evaluate`).

## Headline: temperature RMSE (°C), pooled over depth

| test set | regime | baseline | OceanEmbed | Δ (OE − base) |
|---|---|---:|---:|---:|
| **spatial (Bay of Bengal)** | **barrier_layer_stratified** | **0.962** | 0.985 | **+0.023 (worse)** |
| spatial | well_mixed | 0.935 | 0.949 | +0.014 |
| spatial | upwelling | 1.079 | 1.052 | −0.027 |
| temporal (JJAS 2022) | barrier_layer_stratified | 1.063 | **1.049** | −0.014 (better) |
| temporal | well_mixed | 1.209 | 1.187 | −0.022 |
| temporal | upwelling | 1.376 | 1.397 | +0.021 |

Salinity, BoB barrier-layer: baseline 0.443 vs OceanEmbed 0.450 PSU — also a wash.
Density-inversion fraction (TEOS-10, gsw): ~0.00 for both models → the physics
term has little to correct here.

### lambda_physics sweep (§5.4)

lambda in {0.05, 0.10, 0.30} gives **identical** RMSE to 3 decimals on every
regime and both holdouts (`oceanembed` = 0.10, `oceanembed_lp05`, `oceanembed_lp30`
in `per_regime_rmse.csv`). The physics-consistency term is **inactive for this
problem**: both models already produce density-stable profiles (inversion
fraction 0.000) with or without it, so the penalty has nothing to push against.
It is not harmful — just not doing work at this data scale. Worth re-checking on
the full LOQ runs, but don't expect it to move the needle.

## Honest assessment (CLAUDE.md §6.2 / §8)

**On the primary holdout — the Bay of Bengal, barrier-layer regime — OceanEmbed
does not beat the baseline.** It is ~2% worse on T and S. On the temporal holdout
it is ~1% better in that regime, with a clearer local gain (−0.15 °C at 75 m, in
the barrier-layer depth band). All differences are within run-to-run noise for
models this small and short. **We are not retraining to force a win** — this is
what the validation produced.

### Likely causes

1. **Preliminary models.** ResNet-18 / 20 epochs / CPU. No lambda_physics sweep,
   no ResNet-50 or Prithvi run yet. The comparison is not yet at full strength.
2. **Weak conditioning signal on the spatial holdout specifically.** The held-out
   BoB cells take their regime label + stratification index from the WOA23 1°
   climatology, which is coarse in the Bay of Bengal. Across the holdout region
   that signal is nearly uniform, so the FiLM layer has little to modulate on —
   exactly where regime conditioning should help most, it has the least to work
   with.
3. **No BoB barrier layers in training, by design.** The spatial holdout removes
   the entire Bay of Bengal, so the only barrier-layer examples the model trains
   on are from the SE Arabian Sea and equatorial Indian Ocean. Those may not
   transfer to the much fresher, river-dominated BoB lens.
4. **Surface inputs under-constrain the halocline.** BoB barrier-layer salinity
   RMSE is ~0.45 PSU against a surface-S spread of ~1.2 — the [SST, SSH, SSS]
   stack does not strongly determine the sub-surface salinity structure, which is
   the crux of the barrier-layer reconstruction problem.
5. **Thermocline dominates the error.** Both models peak at ~1.6–2.1 °C RMSE at
   75–150 m; that band drives the pooled number and is where surface fields carry
   least information about thermocline depth.

### What would change the verdict (to try on the LOQ, without tuning to the metric)

- Full ResNet-50 / Prithvi runs, 40+ epochs.
- lambda_physics ∈ {0.05, 0.1, 0.3} (§5.4).
- A finer barrier-layer climatology than WOA23 1° for the conditioning input
  (this is the item most likely to matter — see cause #2).

## Figures (`outputs/figures/`)

- `rmse_by_regime.png` — the comparison bar chart (§6.3)
- `profiles_barrier_layer.png` — 3 BoB profiles, prediction vs Argo + ±2σ band;
  reconstruction tracks ground truth well, band widens correctly in the thermocline
- `uncertainty_map_bob.png` — predictive σ(T) at 100 m; higher in the dynamic
  southern BoB, lower in the north
