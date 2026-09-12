# OceanEmbed Dashboard — Every Graph, Page by Page

Read straight from `dashboard/app.py`, so this matches exactly what's on screen —
plain language, with the real term in parentheses where it matters.

Two controls sit above every page and change what all the charts below show:
**MODEL** (which trained version you're looking at) and **HOLDOUT** — **spatial**
(the whole Bay of Bengal, held out entirely from training) vs. **temporal** (the
2022 monsoon season, held out entirely from training). Neither test set was ever
seen during training — that's what makes the numbers honest.

---

## Landing page

No charts — just the pitch and the **Run the pipeline** button (see the earlier
brief for exactly what that replays).

---

## Overview ("home")

**3 big scorecard tiles**: how many satellite-to-float pairs were matched
(8,340); the error (**RMSE**, root-mean-square error — roughly "how far off, on
average, in °C") in the hardest region (the barrier layer) compared with the
baseline; and how closely the *shape* of the predicted water column follows the
real one (a correlation score, **Pearson r** — 1.0 would mean a perfect match).

**6 small stat tiles**: how many weekly satellite images went in; the depth range
predicted (18 levels, 0–2000 m); how many holdout profiles the current test uses;
what fraction of the model's "80% confident" predictions were actually right;
what fraction of predicted profiles were physically impossible (a **density
inversion** — heavier water sitting above lighter water); which AI backbone ran.

**Headline table**: temperature error for every regime (well-mixed, barrier-layer,
upwelling) side by side, baseline vs. the model — the single most important table
in the whole app, and where the honest "doesn't win yet in the barrier layer" result
lives in plain numbers.

**Graph 1 — "OceanEmbed vs baseline · RMSE by regime"** (bar chart): x-axis is
water-regime type, y-axis is temperature error in °C, two bars per group (baseline
vs. the model), split into two side-by-side panels for the two holdouts. Shorter
bar wins that regime.

**Graph 2 — "error vs depth · barrier-layer regime"** (line chart): x-axis is
error, y-axis is depth (flipped so the seafloor direction points down); shows
exactly where in the water column the model struggles most. It peaks around
75–150 m — the **thermocline**, the layer where temperature changes fastest and is
hardest to guess from surface clues alone.

**Graph 3 — "predicted vs observed temperature"** (density scatter/contour):
every held-out prediction plotted as one point — x is what the float actually
measured, y is what the model predicted. A dashed diagonal line marks "perfect."
A tight cloud hugging that line means accurate; scattered far from it means not.

**Graph 4 — "training — validation RMSE per epoch"** (line chart): x-axis is
training epoch (one full pass through the training data), y-axis is error on a
held-back slice of the training data itself (not the honest holdout — this is the
number used to pick when to stop training). A smoothly falling, flattening line
means stable learning; a jagged or rising one would mean trouble.

---

## Inputs & reconstruction ("s1")

**3 heatmaps** — literally what the satellite photographed that week: sea-surface
temperature, sea-surface height (**SLA**, a stand-in for currents and swirling
eddies), and sea-surface saltiness. This is the *entire* input the model gets —
everything below the surface on this page is predicted, not observed.

**Float map** (scatter, clickable): every held-out Argo float location, coloured
by which water regime it's in. Click one to load it below.

**Reconstructed profile** (line chart): for the clicked float, depth runs down the
page; solid lines are the model's prediction for temperature and salinity, dotted
lines are what the float actually measured, and a shaded band around the solid
line is the model's own stated **uncertainty** (a ±2 standard-deviation range — its
built-in confidence estimate, not a guess bolted on afterward). If the dotted line
stays inside the shaded band, the model's confidence is trustworthy for that
profile.

**Residual bar** (bar chart): for that same float, prediction-minus-truth error at
every depth. Near zero at the surface and at depth, largest around the
thermocline — the same story as Overview's error-vs-depth chart, but for one real
location instead of the average.

---

## Accuracy & calibration ("s2")

**Skill by depth** (2 line charts, temperature and salinity): same idea as
Overview's error-vs-depth chart, restricted to the barrier-layer regime
specifically — the hardest, most important case — baseline vs. the model.

**Predicted vs observed** (2 density scatters, temperature and salinity): same
idea as Overview's scatter, shown separately for each variable with the
correlation number printed on the chart.

**Coverage tiles**: for a claimed 80% and 95% confidence level, what fraction of
real measurements actually fell inside that predicted range. A number close to
the claim (e.g. 81% actual for an 80% claim) means the model's confidence is
close to honest — neither over- nor under-confident.

**Reliability curves** (2 line charts, temperature and salinity) — this is the
**calibration** check: x-axis is the *claimed* confidence level (0–100%), y-axis
is the *actual* fraction of truths that landed inside a band at that level. A
perfectly honest model traces the diagonal exactly. The chart also prints the
average gap between claim and reality (**calibration error**) — about 1 point for
temperature.

---

## Regime overlay ("s3")

**Regime map** (heatmap): for a chosen month, colours every location by which
water regime it's in — well-mixed, barrier-layer, or upwelling. Built *only* from
an independent ocean-climate average and a river-flow estimate — never from the
satellite images the model predicts from, which is what keeps this "hint" from
being a cheat (checked automatically by the code, as covered earlier).

**Bay of Bengal averages** (small bar chart): average barrier-layer thickness,
mixed-layer depth, and stratification for the Bay of Bengal that month — shows the
barrier layer thickening in monsoon months.

**Holdout regime mix** (horizontal bar chart): how many of the current test-set
floats fall into each regime category.

---

## Model vs baseline ("s4")

**Per-regime comparison** (bar chart, toggle between temperature/salinity):
repeats the Overview-style bar chart with a toggle, plus an expandable full data
table (every number, every regime, every holdout, baseline vs. the model) — the
literal source of every claim in the pitch.

**Training dynamics** (2 line charts, temperature and salinity): the same
training-curve idea as Overview, but now showing *every* trained variant on one
chart — baseline, the full model, and the two extra physics-strength runs from
the **λ-sweep** (the physics-penalty-strength test).

**Physics consistency** (bar chart): the fraction of predicted depth-pairs that
are **inverted** (denser water sitting on top of lighter water — physically
impossible in a stable water column), for each model on each holdout. Both sit
near 0% — an honest near-non-result: profiles were already physically stable, so
the physics penalty had little left to correct at this data scale, whether it's
switched on or not.

---

## Data & scope ("s5")

No charts — a plain-text disclosure card listing exactly which data sources are
real, which are substituted and why (river discharge, INCOIS), what's explicitly
out of scope (real-time ingestion, cyclone-flagging), plus links to the Phase 0
and Phase 3 write-ups and the same **Re-run the pipeline** replay button as the
landing page.
