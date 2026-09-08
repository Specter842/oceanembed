"""
OceanEmbed demo dashboard (Streamlit) — CLAUDE.md §7.1.

Reads only pre-computed artefacts (no torch):
  data/interim/satellite_weekly.nc
  data/interim/regime_climatology.nc
  outputs/metrics/per_regime_rmse.csv, physics_diagnostics.csv, train_log_*.csv
  outputs/metrics/predictions_<model>_<set>.npz   (from `python -m src.evaluate`)

Run:  .venv/Scripts/python.exe -m streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ASSETS = ROOT / "dashboard" / "assets.npz"          # baked by dashboard/prep_assets.py
METRICS = ROOT / "outputs" / "metrics"

REGIMES = ["well_mixed", "barrier_layer_stratified", "upwelling"]
RC = {"well_mixed": "#38BDF8", "barrier_layer_stratified": "#FB3D7A",
      "upwelling": "#4ADE80", "no data": "#242424"}
MC = {"oceanembed": "#22D3EE", "baseline": "#8A8A8A",
      "oceanembed_lp05": "#A3E635", "oceanembed_lp30": "#FB3D7A"}
CYAN, ROSE, LIME, AMBER = "#22D3EE", "#FB3D7A", "#A3E635", "#FBBF24"

st.set_page_config(page_title="OceanEmbed", layout="wide", page_icon="🛰️",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background:#000 !important; }
[data-testid="stHeader"] { height:0; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display:none !important; }
.block-container { padding:1.4rem 2.2rem 4rem !important; max-width:1500px; }
html, body, [class*="css"], .stMarkdown, p, span, div, label { font-family:'Space Grotesk','Inter',system-ui,sans-serif; }
[data-testid="stMetricValue"], .mono { font-family:'JetBrains Mono',monospace !important; }

h1.oe-title { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:2.05rem;
  letter-spacing:-0.5px; margin:0 0 .1rem; color:#FFF; }
.oe-sub { color:#6E6E6E; font-size:.82rem; letter-spacing:.3px; margin-bottom:1.2rem; }

.oe-sec { display:flex; align-items:center; gap:.7rem; margin:2.1rem 0 .7rem;
  border-bottom:1px solid #1A1A1A; padding-bottom:.45rem; }
.oe-sec .n { font-family:'JetBrains Mono',monospace; font-size:.72rem; color:#000;
  background:#22D3EE; padding:.12rem .42rem; border-radius:3px; font-weight:700; }
.oe-sec .t { font-size:.92rem; letter-spacing:2.4px; text-transform:uppercase;
  color:#D8D8D8; font-weight:500; }
.oe-sec .h { flex:1; }
.oe-sec .hint { font-size:.7rem; color:#5A5A5A; letter-spacing:.4px; }

[data-testid="stMetric"] { background:#080808; border:1px solid #1C1C1C; border-radius:8px;
  padding:.7rem .9rem .55rem; }
[data-testid="stMetricLabel"] p { font-size:.62rem !important; letter-spacing:1.6px;
  text-transform:uppercase; color:#6A6A6A !important; }
[data-testid="stMetricValue"] { font-size:1.5rem !important; color:#22D3EE !important;
  font-weight:600 !important; }
[data-testid="stMetricDelta"] { font-size:.68rem !important; }

[data-testid="stVerticalBlock"] { gap:.55rem; }
.stPlotlyChart { background:#060606; border:1px solid #171717; border-radius:8px; padding:.3rem; }
.stRadio label, .stSlider label, .stSelectbox label { color:#8A8A8A !important;
  font-size:.66rem !important; letter-spacing:1.4px; text-transform:uppercase; }
[data-baseweb="tab-list"] { background:transparent; }
hr { border-color:#161616; }
.stCaption, [data-testid="stCaptionContainer"] { color:#5C5C5C !important; }
.oe-foot { border:1px solid #1C1C1C; border-left:2px solid #22D3EE; background:#070707;
  border-radius:6px; padding:1rem 1.2rem; color:#9A9A9A; font-size:.8rem; line-height:1.55; }
.oe-foot b { color:#D8D8D8; }
[data-testid="stExpander"] { border:1px solid #1A1A1A; background:#070707; border-radius:6px; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
def sec(n, title, hint=""):
    st.markdown(f'<div class="oe-sec"><span class="n">{n}</span>'
                f'<span class="t">{title}</span><span class="h"></span>'
                f'<span class="hint">{hint}</span></div>', unsafe_allow_html=True)


def style_fig(fig, h=270, legend_top=True):
    fig.update_layout(
        height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", size=10, color="#9A9A9A"),
        margin=dict(l=6, r=6, t=8, b=6),
        colorway=[CYAN, ROSE, LIME, AMBER, "#38BDF8"],
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9),
                    orientation="h", y=1.13 if legend_top else -0.2, x=0),
        hoverlabel=dict(bgcolor="#0A0A0A", font_family="JetBrains Mono"),
    )
    fig.update_xaxes(gridcolor="#131313", zerolinecolor="#1E1E1E", linecolor="#1E1E1E",
                     tickfont=dict(size=9))
    fig.update_yaxes(gridcolor="#131313", zerolinecolor="#1E1E1E", linecolor="#1E1E1E",
                     tickfont=dict(size=9))
    return fig


def heatmap(arr, x, y, scale, label):
    fig = px.imshow(arr, x=x, y=y, origin="lower", aspect="auto",
                    color_continuous_scale=scale)
    fig.update_layout(coloraxis_colorbar=dict(
        title=dict(text=label, font=dict(size=9)), thickness=8, len=0.9,
        tickfont=dict(size=8), outlinewidth=0))
    return style_fig(fig, h=250)


# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_assets():
    if not ASSETS.exists():
        st.error("dashboard/assets.npz missing — run  `python -m dashboard.prep_assets`")
        st.stop()
    z = np.load(ASSETS, allow_pickle=True)
    return {k: z[k] for k in z.files}


@st.cache_data(show_spinner=False)
def load_metrics():
    p = METRICS / "per_regime_rmse.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_phys():
    p = METRICS / "physics_diagnostics.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_logs():
    out = {}
    for f in METRICS.glob("train_log_*.csv"):
        name = f.stem[len("train_log_"):]
        if name.startswith("smoke"):
            continue
        out[name] = pd.read_csv(f)
    return out


@st.cache_data(show_spinner=False)
def models_avail():
    return sorted({f.name[len("predictions_"):-len("_spatial.npz")]
                   for f in METRICS.glob("predictions_*_spatial.npz")}) or ["oceanembed"]


@st.cache_data(show_spinner=False)
def load_pred(model, ts):
    p = METRICS / f"predictions_{model}_{ts}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.files}


def regime_map(A, month):
    """(grid, lat, lon) for a month from the baked bundle; -1 -> NaN."""
    g = A["regime_grid"][month - 1].astype(float)
    g[g < 0] = np.nan
    return g, A["regime_grid_lat"], A["regime_grid_lon"]


# --------------------------------------------------------------------------- #
st.markdown('<h1 class="oe-title">OceanEmbed</h1>', unsafe_allow_html=True)
st.markdown('<div class="oe-sub">REGIME-CONDITIONED · PHYSICS-CONSTRAINED · '
            'SUBSURFACE T/S RECONSTRUCTION FROM SATELLITE SURFACE FIELDS · '
            'NORTH INDIAN OCEAN · OFFLINE BATCH DEMO</div>', unsafe_allow_html=True)

A = load_assets()
metrics = load_metrics()
phys = load_phys()
logs = load_logs()
models = models_avail()
weeks = pd.to_datetime(A["sat_weeks"])
LAT, LON = A["sat_lat"], A["sat_lon"]

c1, c2, c3 = st.columns([2, 2, 6])
model = c1.radio("model", models, horizontal=True,
                 index=models.index("oceanembed") if "oceanembed" in models else 0)
holdout = c2.radio("holdout", ["spatial", "temporal"], horizontal=True,
                   help="spatial = Bay of Bengal block · temporal = JJAS 2022")
pred = load_pred(model, holdout)


# ---- KPI strip -------------------------------------------------------------- #
def pooled(m, ts, reg, col):
    if metrics is None:
        return np.nan
    r = metrics[(metrics.model == m) & (metrics.test_set == ts)
                & (metrics.regime_class == reg) & (metrics.depth_level == -1)]
    return float(r[col].iloc[0]) if len(r) else np.nan


k = st.columns(6)
k[0].metric("matched profiles", "8,340")
k[1].metric("depth range", "0–2000 m")
k[2].metric("weekly fields", f"{len(weeks)}")
k[3].metric(f"{holdout} · n", f"{len(pred['lat']) if pred else 0}")
rt = pooled(model, holdout, "barrier_layer_stratified", "rmse_T")
rb = pooled("baseline", holdout, "barrier_layer_stratified", "rmse_T")
k[4].metric("RMSE·T barrier-layer", f"{rt:.3f} °C" if rt == rt else "—",
            delta=f"{rt - rb:+.3f} vs baseline" if rt == rt and rb == rb else None,
            delta_color="inverse")
rr = pooled(model, holdout, "barrier_layer_stratified", "r_T")
k[5].metric("profile r·T", f"{rr:.3f}" if rr == rr else "—")


# ---- 1 · satellite input fields ------------------------------------------- #
sec("01", "Satellite input fields", "surface observations — the only model input")
wk = st.select_slider("week", [d.strftime("%Y-%m-%d") for d in weeks],
                      value=weeks[min(len(weeks) - 1, 86)].strftime("%Y-%m-%d"))
wi = [d.strftime("%Y-%m-%d") for d in weeks].index(wk)
sst_sc = [[0, "#04070F"], [.35, "#3B1D63"], [.65, "#F0399A"], [1, "#FBBF24"]]
ssh_sc = [[0, "#22D3EE"], [.5, "#0A0A0A"], [1, "#FB3D7A"]]
sss_sc = [[0, "#08131A"], [.5, "#22D3EE"], [1, "#A3E635"]]
g = st.columns(3)
for col, var, lab, sc in zip(g, ("sst", "ssh", "sss"),
                             ("SST °C", "SLA m", "SSS PSU"), (sst_sc, ssh_sc, sss_sc)):
    col.plotly_chart(heatmap(A[f"sat_{var}"][wi], LON, LAT, sc, lab),
                     use_container_width=True)


# ---- 2 · profile reconstruction ----------------------------------------- #
sec("02", "Subsurface reconstruction", "click a float location")
if pred is None:
    st.warning(f"no predictions for {model}/{holdout} — run `python -m src.evaluate`")
    st.stop()
lv = pred["depth_levels"]
regnames = np.array(REGIMES)[pred["regime"].astype(int)]
pts = pd.DataFrame({"lat": pred["lat"], "lon": pred["lon"], "regime": regnames,
                    "i": np.arange(len(pred["lat"]))})
L, R = st.columns([1, 1.25])
with L:
    m = px.scatter(pts, x="lon", y="lat", color="regime", color_discrete_map=RC,
                   custom_data=["i"])
    m.update_traces(marker=dict(size=6, line=dict(width=0)))
    ev = st.plotly_chart(style_fig(m, h=380), use_container_width=True,
                         on_select="rerun", key="pmap")
    picks = ev.get("selection", {}).get("points", []) if ev else []
    idx = int(picks[0]["customdata"][0]) if picks else 176 if len(pts) > 176 else 0
with R:
    P = {kk: pred[kk][idx] for kk in ("T_pred", "S_pred", "T_std", "S_std",
                                      "T_obs", "S_obs", "mask")}
    mk = P["mask"] > 0
    row = pts.iloc[idx]
    st.markdown(f'<span class="mono" style="color:#7A7A7A;font-size:.75rem">'
                f'#{idx} · {row.lat:.2f}°N {row.lon:.2f}°E · '
                f'<span style="color:{RC[row.regime]}">{row.regime}</span></span>',
                unsafe_allow_html=True)
    fig = go.Figure()
    for var, pv, sv, ov, cc in [("T", "T_pred", "T_std", "T_obs", ROSE),
                                ("S", "S_pred", "S_std", "S_obs", CYAN)]:
        xa = "x" if var == "T" else "x2"
        fig.add_trace(go.Scatter(
            x=np.r_[P[pv][mk] - 2 * P[sv][mk], (P[pv][mk] + 2 * P[sv][mk])[::-1]],
            y=np.r_[lv[mk], lv[mk][::-1]], fill="toself", fillcolor=cc,
            opacity=0.13, line=dict(width=0), xaxis=xa, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=P[pv][mk], y=lv[mk], name=f"{var} pred",
                                 line=dict(color=cc, width=2), xaxis=xa))
        fig.add_trace(go.Scatter(x=P[ov][mk], y=lv[mk], name=f"{var} Argo",
                                 line=dict(color=cc, width=1, dash="dot"),
                                 mode="lines+markers", marker=dict(size=3), xaxis=xa))
    fig.update_layout(
        height=380, yaxis=dict(title="depth m", autorange="reversed"),
        xaxis=dict(title="T °C", domain=[0, 1]),
        xaxis2=dict(title="S PSU", overlaying="x", side="top"))
    st.plotly_chart(style_fig(fig, h=380), use_container_width=True)
rt_i = float(np.sqrt(np.mean((P["T_pred"][mk] - P["T_obs"][mk]) ** 2)))
rs_i = float(np.sqrt(np.mean((P["S_pred"][mk] - P["S_obs"][mk]) ** 2)))
resid = (P["T_pred"] - P["T_obs"])[mk]
rf = go.Figure(go.Bar(x=lv[mk], y=resid,
                      marker_color=np.where(resid >= 0, ROSE, CYAN)))
rf.update_layout(height=140, xaxis_title="depth m", yaxis_title="T resid °C",
                 bargap=0.5)
cc1, cc2 = st.columns([3, 1])
cc1.plotly_chart(style_fig(rf, h=140, legend_top=False), use_container_width=True)
cc2.metric("this profile RMSE·T", f"{rt_i:.2f} °C")
cc2.metric("RMSE·S", f"{rs_i:.2f} PSU")


# ---- 3 · skill by depth  (NEW) ---------------------------------------------- #
sec("03", "Skill by depth", f"{holdout} holdout · per standard level")
if metrics is not None:
    d = metrics[(metrics.test_set == holdout) & (metrics.depth_level >= 0)
                & (metrics.regime_class == "barrier_layer_stratified")
                & (metrics.model.isin(["baseline", model]))]
    g2 = st.columns(2)
    for col, yv, lab in zip(g2, ("rmse_T", "rmse_S"), ("RMSE T °C", "RMSE S PSU")):
        f = go.Figure()
        for mdl, color in [("baseline", MC["baseline"]), (model, MC.get(model, CYAN))]:
            s = d[d.model == mdl].sort_values("depth_level")
            f.add_trace(go.Scatter(x=s[yv], y=s.depth_level, name=mdl,
                                   line=dict(color=color, width=2),
                                   mode="lines+markers", marker=dict(size=4)))
        f.update_layout(height=300, xaxis_title=lab,
                        yaxis=dict(title="depth m", autorange="reversed"))
        col.plotly_chart(style_fig(f, h=300), use_container_width=True)


# ---- 4 · predicted vs observed  (NEW) ------------------------------------- #
sec("04", "Predicted vs observed", f"all {len(pts)} holdout profiles · {model}")
g3 = st.columns(2)
for col, pk, ok, lab, cc in [(g3[0], "T_pred", "T_obs", "temperature °C", ROSE),
                             (g3[1], "S_pred", "S_obs", "salinity PSU", CYAN)]:
    mm = pred["mask"] > 0
    pv, ov = pred[pk][mm], pred[ok][mm]
    r = np.corrcoef(pv, ov)[0, 1]
    f = go.Figure()
    f.add_trace(go.Histogram2dContour(x=ov, y=pv, colorscale=[[0, "#000"], [1, cc]],
                                      showscale=False, ncontours=14, line=dict(width=0)))
    lim = [min(pv.min(), ov.min()), max(pv.max(), ov.max())]
    f.add_trace(go.Scatter(x=lim, y=lim, line=dict(color="#4A4A4A", dash="dash", width=1),
                           showlegend=False, hoverinfo="skip"))
    f.update_layout(height=300, xaxis_title=f"observed {lab}",
                    yaxis_title=f"predicted {lab}",
                    annotations=[dict(x=0.05, y=0.93, xref="paper", yref="paper",
                                      text=f"r = {r:.3f}", showarrow=False,
                                      font=dict(color=cc, size=13, family="JetBrains Mono"))])
    col.plotly_chart(style_fig(f, h=300, legend_top=False), use_container_width=True)


# ---- 5 · regime overlay --------------------------------------------------- #
sec("05", "Regime overlay",
    "class from independent climatology + discharge — never the satellite channels")
mo = st.slider("month", 1, 12, 8)
L2, R2 = st.columns([1.4, 1])
with L2:
    gg, glat, glon = regime_map(A, mo)
    f = px.imshow(gg, x=glon, y=glat, origin="lower", aspect="auto",
                  range_color=[-0.5, 2.5],
                  color_continuous_scale=[RC[r] for r in REGIMES])
    f.update_layout(coloraxis_colorbar=dict(
        tickvals=[0, 1, 2], ticktext=["mixed", "barrier", "upwell"],
        thickness=8, len=0.8, tickfont=dict(size=8)))
    st.plotly_chart(style_fig(f, h=330), use_container_width=True)
with R2:
    clat, clon = A["clim_lat"], A["clim_lon"]
    my = (clat >= 5) & (clat <= 22)
    mx = (clon >= 85) & (clon <= 99)
    mi = mo - 1
    blt = np.nanmean(A["clim_blt_m"][mi][np.ix_(my, mx)])
    mld = np.nanmean(A["clim_mld_m"][mi][np.ix_(my, mx)])
    strat = np.nanmean(A["clim_strat"][mi][np.ix_(my, mx)])
    f = go.Figure(go.Bar(x=["BLT m", "MLD m", "strat×50"],
                         y=[blt, mld, strat * 50], marker_color=[ROSE, CYAN, LIME]))
    f.update_layout(height=150, yaxis_title="Bay of Bengal mean")
    st.plotly_chart(style_fig(f, h=150, legend_top=False), use_container_width=True)
    vc = pd.Series(regnames).value_counts().reindex(REGIMES, fill_value=0)
    f = go.Figure(go.Bar(y=vc.index, x=vc.values, orientation="h",
                         marker_color=[RC[r] for r in vc.index]))
    f.update_layout(height=150, xaxis_title=f"{holdout} holdout profiles")
    st.plotly_chart(style_fig(f, h=150, legend_top=False), use_container_width=True)


# ---- 6 · OceanEmbed vs baseline ----------------------------------------- #
sec("06", "OceanEmbed vs baseline", "per-regime RMSE, pooled over depth")
if metrics is not None:
    var = st.radio("variable", ["rmse_T", "rmse_S"], horizontal=True,
                   format_func=lambda s: "Temperature" if s.endswith("T") else "Salinity")
    pool = metrics[metrics.depth_level == -1]
    keep = [m for m in ["baseline", "oceanembed", model] if m in pool.model.unique()]
    sub = pool[pool.model.isin(keep) & pool.regime_class.isin(REGIMES)]
    f = px.bar(sub, x="regime_class", y=var, color="model", barmode="group",
               facet_col="test_set", color_discrete_map=MC)
    f.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].upper(),
                                             font=dict(size=10, color="#8A8A8A")))
    f.update_layout(height=330)
    st.plotly_chart(style_fig(f, h=330), use_container_width=True)
    with st.expander("full table — per test set × regime"):
        st.dataframe(pool[pool.regime_class.isin(REGIMES + ["all"])]
                     .pivot_table(index=["test_set", "regime_class"], columns="model",
                                  values=var).round(3), use_container_width=True)


# ---- 7 · training dynamics  (NEW) --------------------------------------- #
sec("07", "Training dynamics", "validation RMSE per epoch")
if logs:
    g4 = st.columns(2)
    for col, yv, lab in zip(g4, ("rmseT", "rmseS"), ("val RMSE T °C", "val RMSE S PSU")):
        f = go.Figure()
        for name, dfl in logs.items():
            v = dfl[dfl.split == "val"]
            f.add_trace(go.Scatter(x=v.epoch, y=v[yv], name=name,
                                   line=dict(color=MC.get(name, "#8A8A8A"), width=1.8)))
        f.update_layout(height=260, xaxis_title="epoch", yaxis_title=lab)
        col.plotly_chart(style_fig(f, h=260), use_container_width=True)


# ---- 8 · physics diagnostics  (NEW) ----------------------------------- #
sec("08", "Physics consistency", "TEOS-10 density-inversion fraction (lower = better)")
if phys is not None:
    f = go.Figure()
    for ts in ["spatial", "temporal"]:
        s = phys[phys.test_set == ts]
        f.add_trace(go.Bar(name=ts, x=s.model, y=s.inversion_fraction * 100,
                           marker_color=CYAN if ts == "spatial" else ROSE))
    f.update_layout(height=230, barmode="group", yaxis_title="% inverted level pairs")
    st.plotly_chart(style_fig(f, h=230), use_container_width=True)
    st.caption("≈ 0 for every model — profiles are already density-stable, so the "
               "physics loss term has little to correct at this data scale.")


# ---- 9 · data & scope --------------------------------------------------- #
sec("09", "Data & scope")
st.markdown(
    '<div class="oe-foot">'
    'This demo runs on <b>historical downloaded satellite and Argo data</b>. Real-time '
    'ingestion and the float-deployment / cyclone-flagging features in the roadmap are '
    '<b>not implemented</b> in this build.<br><br>'
    '· Satellite inputs are NOAA-hosted (OISST v2.1, CoastWatch blended SLA, CoastWatch '
    'SMAP SSS), substituted for CMEMS / PODAAC which need accounts. No ocean-colour channel.<br>'
    '· <b>INCOIS moored-buoy data could not be accessed</b> — the India-specific-data / '
    'data-sovereignty differentiator is future work. River discharge is a literature '
    'climatology, not Indian gauge data.<br>'
    '· Barrier-layer climatology is WOA23-derived, not the published de Boyer Montégut product.<br>'
    '· Backbone shown: ResNet-18 / 20 epochs / CPU — preliminary. See RUN.md for the full runs.'
    '</div>', unsafe_allow_html=True)
st.caption("phase0_report.md · outputs/metrics/PHASE3_FINDINGS.md · github.com/Specter842/oceanembed")
