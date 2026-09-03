"""
OceanEmbed demo dashboard (Streamlit) — CLAUDE.md §7.1.

Reads only pre-computed artefacts (no torch):
  data/interim/satellite_weekly.nc
  data/interim/regime_climatology.nc
  outputs/metrics/per_regime_rmse.csv
  outputs/metrics/predictions_<model>_<set>.npz   (from `python -m src.evaluate`)

Run:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAT = ROOT / "data" / "interim" / "satellite_weekly.nc"
CLIM = ROOT / "data" / "interim" / "regime_climatology.nc"
METRICS = ROOT / "outputs" / "metrics"
REGIMES = ["well_mixed", "barrier_layer_stratified", "upwelling"]
REGIME_COLORS = {"well_mixed": "#4C78A8", "barrier_layer_stratified": "#E45756",
                 "upwelling": "#54A24B", "no data": "#DDDDDD"}

st.set_page_config(page_title="OceanEmbed", layout="wide", page_icon="🌊")


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def load_sat():
    ds = xr.open_dataset(SAT)
    weeks = pd.to_datetime(ds["week_start"].values)
    return ds.load(), weeks


@st.cache_data(show_spinner=False)
def load_metrics():
    p = METRICS / "per_regime_rmse.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def available_models():
    got = set()
    for f in METRICS.glob("predictions_*_spatial.npz"):
        got.add(f.name[len("predictions_"):-len("_spatial.npz")])
    return sorted(got) or ["oceanembed"]


@st.cache_data(show_spinner=False)
def load_pred(model, ts):
    p = METRICS / f"predictions_{model}_{ts}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.files}


@st.cache_data(show_spinner=True)
def regime_map(month: int):
    from src.regime.regime_labels import regime_grid
    return regime_grid(month, step=1.0)


# --------------------------------------------------------------------------- #
# header
# --------------------------------------------------------------------------- #

st.title("OceanEmbed — subsurface reconstruction, North Indian Ocean")
st.caption("Regime-conditioned, physics-constrained reconstruction of T/S profiles "
           "from satellite surface fields. Offline batch demo on historical data.")

sat, weeks = load_sat()
metrics = load_metrics()
models = available_models()

c1, c2 = st.sidebar.columns(2)
model = c1.selectbox("Model", models, index=models.index("oceanembed")
                     if "oceanembed" in models else 0)
holdout = c2.radio("Holdout", ["spatial", "temporal"], horizontal=True,
                   help="spatial = Bay of Bengal block; temporal = JJAS 2022")
pred = load_pred(model, holdout)

LAT = sat["lat"].values
LON = sat["lon"].values


# --------------------------------------------------------------------------- #
# 1 — input satellite fields
# --------------------------------------------------------------------------- #

st.header("1 · Satellite input fields")
wk_labels = [d.strftime("%Y-%m-%d") for d in weeks]
wk = st.select_slider("Week", options=wk_labels, value=wk_labels[len(wk_labels) // 2])
wi = wk_labels.index(wk)

cols = st.columns(3)
for col, (var, label, cmap) in zip(cols, [
    ("sst", "SST (°C)", "thermal"), ("ssh", "Sea-level anomaly (m)", "balance"),
    ("sss", "Sea-surface salinity (PSU)", "haline")]):
    if var not in sat:
        col.info(f"{var} not available")
        continue
    arr = sat[var].isel(week_start=wi).values
    fig = px.imshow(arr, x=LON, y=LAT, origin="lower", aspect="auto",
                    color_continuous_scale=cmap, labels={"color": ""})
    fig.update_layout(title=label, margin=dict(l=0, r=0, t=30, b=0), height=300,
                      coloraxis_colorbar=dict(thickness=10))
    col.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# 2 — reconstructed profile viewer
# --------------------------------------------------------------------------- #

st.header("2 · Reconstructed subsurface profile")
if pred is None:
    st.warning(f"No predictions for **{model} / {holdout}**. "
               f"Run `python -m src.evaluate` first.")
else:
    lv = pred["depth_levels"]
    reg_names = np.array(REGIMES)[pred["regime"].astype(int)]
    pts = pd.DataFrame({"lat": pred["lat"], "lon": pred["lon"], "regime": reg_names,
                        "i": np.arange(len(pred["lat"]))})
    left, right = st.columns([1, 1.3])

    with left:
        st.caption("Click a profile location")
        m = px.scatter(pts, x="lon", y="lat", color="regime",
                       color_discrete_map=REGIME_COLORS, height=430)
        m.update_traces(marker=dict(size=7))
        m.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                        legend=dict(orientation="h", y=-0.15))
        ev = st.plotly_chart(m, use_container_width=True, on_select="rerun",
                             key="map2")
        sel = ev.get("selection", {}).get("points", []) if ev else []
        idx = int(pts.iloc[sel[0]["point_index"]]["i"]) if sel else 0

    with right:
        p = {k: pred[k][idx] for k in ("T_pred", "S_pred", "T_std", "S_std",
                                       "T_obs", "S_obs", "mask")}
        mk = p["mask"] > 0
        row = pts.iloc[idx]
        st.caption(f"#{idx} · {row['lat']:.2f}°N {row['lon']:.2f}°E · **{row['regime']}**")
        fig = go.Figure()
        for var, pv, sv, ov, cc in [("T", "T_pred", "T_std", "T_obs", "#E45756"),
                                    ("S", "S_pred", "S_std", "S_obs", "#4C78A8")]:
            xa = "x1" if var == "T" else "x2"
            fig.add_trace(go.Scatter(
                x=np.r_[p[pv][mk] - 2 * p[sv][mk], p[pv][mk][::-1] + 2 * p[sv][mk][::-1]],
                y=np.r_[lv[mk], lv[mk][::-1]], fill="toself", fillcolor=cc,
                opacity=0.15, line=dict(width=0), xaxis=xa, showlegend=False,
                hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=p[pv][mk], y=lv[mk], name=f"{var} pred",
                                     line=dict(color=cc), xaxis=xa))
            fig.add_trace(go.Scatter(x=p[ov][mk], y=lv[mk], name=f"{var} Argo",
                                     line=dict(color=cc, dash="dot"),
                                     mode="lines+markers", marker=dict(size=4), xaxis=xa))
        fig.update_layout(
            height=430, margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(title="depth (m)", autorange="reversed"),
            xaxis=dict(title="T (°C)", domain=[0, 1], side="bottom"),
            xaxis2=dict(title="S (PSU)", overlaying="x", side="top"),
            legend=dict(orientation="h", y=-0.18))
        st.plotly_chart(fig, use_container_width=True)
        rt = float(np.sqrt(np.mean((p["T_pred"][mk] - p["T_obs"][mk]) ** 2)))
        rs = float(np.sqrt(np.mean((p["S_pred"][mk] - p["S_obs"][mk]) ** 2)))
        st.caption(f"this profile — RMSE  T {rt:.2f} °C · S {rs:.2f} PSU")


# --------------------------------------------------------------------------- #
# 3 — regime overlay
# --------------------------------------------------------------------------- #

st.header("3 · Regime overlay")
st.caption("Regime class from the independent climatology + river-discharge "
           "seasonal cycle — never from the satellite channels used for prediction.")
mo = st.slider("Month", 1, 12, 8, help="1 = Jan")
try:
    g = regime_map(mo)
    z = g.values.astype(float)
    z[z < 0] = np.nan
    fig = px.imshow(z, x=g.lon.values, y=g.lat.values, origin="lower", aspect="auto",
                    color_continuous_scale=[REGIME_COLORS[r] for r in REGIMES],
                    range_color=[-0.5, 2.5])
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                      coloraxis_colorbar=dict(
                          tickvals=[0, 1, 2],
                          ticktext=["well-mixed", "barrier layer", "upwelling"]))
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:  # noqa: BLE001
    st.warning(f"regime map unavailable: {e}")


# --------------------------------------------------------------------------- #
# 4 — comparison panel
# --------------------------------------------------------------------------- #

st.header("4 · OceanEmbed vs baseline")
if metrics is None:
    st.warning("run `python -m src.evaluate` to produce per_regime_rmse.csv")
else:
    var = st.radio("Variable", ["rmse_T", "rmse_S"], horizontal=True,
                   format_func=lambda s: "Temperature" if s.endswith("T") else "Salinity")
    pooled = metrics[metrics.depth_level == -1]
    keep = [m for m in ["baseline", "oceanembed", model] if m in pooled.model.unique()]
    sub = pooled[pooled.model.isin(keep) & pooled.regime_class.isin(REGIMES)]
    fig = px.bar(sub, x="regime_class", y=var, color="model", barmode="group",
                 facet_col="test_set", height=420,
                 labels={var: var.replace("rmse_", "RMSE ") +
                         ("  (°C)" if var.endswith("T") else "  (PSU)")})
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Pooled over depth. Full per-depth table: `outputs/metrics/per_regime_rmse.csv`")
    with st.expander("headline table"):
        st.dataframe(
            pooled[pooled.regime_class.isin(REGIMES + ["all"])]
            .pivot_table(index=["test_set", "regime_class"], columns="model",
                         values=var).round(3))
    phys = METRICS / "physics_diagnostics.csv"
    if phys.exists():
        st.caption("TEOS-10 density-inversion fraction (gsw):")
        st.dataframe(pd.read_csv(phys).round(4), hide_index=True)


# --------------------------------------------------------------------------- #
# 5 — data & scope footer
# --------------------------------------------------------------------------- #

st.divider()
st.subheader("Data & Scope")
st.markdown(
    "> This demo runs on historical downloaded satellite and Argo data. Real-time "
    "ingestion and the float-deployment / cyclone-flagging features described in the "
    "roadmap are **not** implemented in this build.\n\n"
    "**Additional scope notes for this build:**\n"
    "- Satellite inputs are NOAA-hosted products (OISST v2.1, CoastWatch blended SLA, "
    "CoastWatch SMAP SSS), substituted for CMEMS/PODAAC which need accounts. "
    "Ocean-colour channel omitted (no reachable historical source).\n"
    "- **INCOIS moored-buoy data could not be accessed** — the abstract's "
    "India-specific-data / data-sovereignty differentiator is future work, not "
    "delivered here. River discharge is a literature climatology, not Indian gauge data.\n"
    "- Barrier-layer climatology is WOA23-derived, not the published de Boyer Montégut product.\n"
    "- Backbone shown here is ResNet-18 / 20 epochs on CPU — preliminary. See `RUN.md`.")
st.caption("Full account: `data/raw/phase0_report.md` · findings: `outputs/metrics/PHASE3_FINDINGS.md`")
