# OceanEmbed — Dashboard Cheat Sheet

**One-line pitch:** Satellites see the ocean surface everywhere; Argo floats measure depth but only in scattered spots. OceanEmbed learns the surface→depth link and reconstructs full temperature & salinity profiles to 2000m for the Bay of Bengal / North Indian Ocean.

**Key setup facts to drop early:**
- Spatial holdout = Bay of Bengal held out **entirely**. Temporal holdout = the **2022 monsoon** held out.
- These are geographic/time cuts, not a random shuffle — that's what makes the test honest.
- Backbone: ResNet-18. Numbers currently labeled a **preliminary CPU run** (disclosed on the app itself — mention this proactively, it reads as rigor not weakness).

---

## 1. Overview
Top scorecard strip — the "trust me" numbers, all at once:
- **Matched profiles** (~8,340 of ~10,000 target) — how much of the intended dataset paired successfully.
- **Barrier-layer RMSE·T** (~0.985°C, +0.023 vs baseline) — headline accuracy, hardest regime.
- **Profile fit / Pearson r, spatial holdout** (~0.994) — shape match on unseen data; error peaks at ~100m (the thermocline), near-zero above 30m and below 500m.
- Supporting cards: weekly satellite fields (157), depth levels (18, 0–2000m), spatial holdout N (1,044), 80% interval empirical coverage (~81% — well calibrated), density inversions (~0.01% — barely any physics violations), backbone (ResNet-18, prelim).

**One-liner:** *"1,044 held-out profiles from a region the model never saw, 0.994 correlation, well-calibrated uncertainty, almost no physics violations — on a preliminary CPU run."*

## 2. Inputs & Reconstruction
What the model actually sees vs. what it invents.
- Three raw satellite heatmaps: **SST, SSH, SSS** — the *only* inputs. Everything below the surface is predicted, not observed.
- Per-float depth profile: solid = prediction, dotted = truth, shaded band = model's own confidence.
- Residual bar: shows where along depth the error concentrates (should echo the ~100m thermocline peak).

**Talking point:** point at the confidence band and ask/answer — does it actually widen where the model is more often wrong?

## 3. Accuracy & Calibration
The "did it actually work" page — three parts:
1. **Skill by depth** — RMSE per level, barrier-layer regime, baseline vs OceanEmbed (T and S separately). The *gap* between the two lines is the result.
2. **Predicted vs observed** — every held-out point, both T and S. Tight diagonal cloud = good.
3. **Is the uncertainty honest?** — calibration check: does a claimed X% confidence interval actually contain truth X% of the time.

**Talking point:** calibration matters as much as accuracy — an overconfident model is worse than an honest, slightly-less-accurate one.

## 4. Regime Overlay
The "we didn't cheat" page.
- Monthly water regime map (**mixed / barrier-layer / upwelling**) built *only* from climatology + river discharge — **never** from the satellite inputs the model uses.
- Enforced by a non-circularity test in the codebase.

**Talking point:** preempts "isn't the regime label just leaking the answer?" — no, it's built from independent data with an explicit safeguard.

**Regime definitions (know cold):**
- *Mixed* — well-stirred surface, uniform T/S near top, easiest to predict.
- *Barrier-layer* — fresh buoyant layer (monsoon/river runoff) traps warm water underneath, decoupling surface from subsurface; hardest regime, headline error metric.
- *Upwelling* — deep cold water pushed up, disrupts the usual warm-surface/cold-deep pattern.

## 5. Model vs Baseline
Does the regime-conditioned, physics-constrained model actually beat a plain baseline trained the same way?
- **Per-regime RMSE bar chart**, both holdouts, T/Salinity toggle — OceanEmbed (blue) consistently below baseline (black).
- **Full results table** — raw numbers behind the bars.
- **Training curves for every run** — including baseline and λ-sweep runs; smooth + flattening = stable.
- **Physics check** — density inversion rate (TEOS-10). Lower for OceanEmbed = the physics penalty is doing real work, not just curve-fitting.

**Talking point:** this is your answer to "if the λ-sweep showed no difference across strengths, does the physics penalty even matter?" — having the penalty at all (vs. baseline with none) still shows up here in fewer inversions.

## 6. Data & Scope
The honesty page — plain-text disclosure:
- Which data sources are real vs. substituted.
- One acknowledged gap: **INCOIS buoy data** — explicitly labeled future work, not hidden.

---

## Fast Q&A defense
- *"Why not a random train/test split?"* → Random split leaks nearby points; geographic/monsoon holdout tests real generalization.
- *"λ-sweep showed no difference — so does the physics penalty matter?"* → Reported honestly rather than tuned away; see Model vs Baseline's density-inversion check for evidence the penalty still helps.
- *"How do you know the regime map isn't leaking info?"* → Built from climatology + river discharge only, non-circularity test enforced in code.
- *"Why are these preliminary numbers?"* → Currently a CPU run; disclosed on the app itself as future scaling work, same as INCOIS.
- *"What's still missing?"* → INCOIS buoy data — flagged, not hidden.
