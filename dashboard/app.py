"""
OceanEmbed demo dashboard (Streamlit) — CLAUDE.md §7.1.

Reads only pre-computed artefacts (no torch / netCDF4 / xarray / gsw):
  dashboard/assets.npz                     (baked by dashboard/prep_assets.py)
  outputs/metrics/per_regime_rmse.csv, physics_diagnostics.csv, train_log_*.csv
  outputs/metrics/predictions_<model>_<set>.npz   (from `python -m src.evaluate`)

Run:  python -m streamlit run dashboard/app.py --client.toolbarMode minimal
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
ASSETS = ROOT / "dashboard" / "assets.npz"
METRICS = ROOT / "outputs" / "metrics"

# ---- palette ------------------------------------------------------------- #
FOG = "#8E9FA9"          # muted slate backdrop
CREAM = "#F5F4EF"
INK = "#0D0D0D"
AQUA = "#2E90FA"         # ocean-blue accent
MUTE = "#8B8E86"
HAIR = "#E4E3DB"

REGIMES = ["well_mixed", "barrier_layer_stratified", "upwelling"]
RC = {"well_mixed": AQUA, "barrier_layer_stratified": INK,
      "upwelling": "#B0A98F", "no data": "#DEDDD4"}
MC = {"oceanembed": AQUA, "baseline": INK,
      "oceanembed_lp05": "#9FB6C4", "oceanembed_lp30": "#C4C3BB"}

st.set_page_config(page_title="OceanEmbed", layout="wide", page_icon="🛰️",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0&display=swap');
.material-symbols-rounded { font-family:'Material Symbols Rounded'; font-weight:400; font-style:normal;
  line-height:1; letter-spacing:normal; text-transform:none; display:inline-block; white-space:nowrap;
  direction:ltr; -webkit-font-feature-settings:'liga'; font-feature-settings:'liga'; }

.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background:#8E9FA9 !important; }
[data-testid="stHeader"] { background:transparent !important; height:0; }
[data-testid="stMainBlockContainer"] {
  background:#F5F4EF; border-radius:30px; max-width:1440px; position:relative;
  padding:2rem 2.3rem 3rem 6.2rem !important; margin:1.1rem auto 1.4rem;
  box-shadow:0 24px 60px -20px rgba(0,0,0,.30); scroll-behavior:smooth;
}
html { scroll-behavior:smooth; }
html, body, [class*="css"], p, span, div, label, .stMarkdown { font-family:'Inter',system-ui,sans-serif; color:#111; }
h1,h2,h3,.disp { font-family:'Space Grotesk','Inter',sans-serif; }

/* ---- fixed dark icon rail — one entry per panel ---- */
#oe-rail { position:fixed; left:calc((100vw - min(1440px,100vw))/2 + 1.35rem); top:1.9rem;
  height:calc(100vh - 3.8rem); width:3.9rem; background:#0D0D0D; border-radius:26px;
  display:flex; flex-direction:column; align-items:center; padding:.85rem 0; z-index:1000; gap:.15rem; }
#oe-rail a { text-decoration:none; }
#oe-rail .logo { width:2.7rem; height:2.7rem; border-radius:14px; background:#2E90FA;
  display:flex; align-items:center; justify-content:center; color:#0D0D0D; margin-bottom:.7rem; }
#oe-rail .logo .material-symbols-rounded { font-size:26px; }
#oe-rail .ic { width:2.7rem; height:2.7rem; border-radius:13px; display:flex;
  align-items:center; justify-content:center; color:#8C8C8C; transition:.13s; }
#oe-rail .ic .material-symbols-rounded { font-size:24px; }
#oe-rail .ic:hover { background:#232323; color:#F5F4EF; }
#oe-rail .ic.on { background:#2E90FA; color:#0D0D0D; }
#oe-rail .sp { flex:1; min-height:.4rem; }
#oe-rail .av { width:2.5rem; height:2.5rem; border-radius:50%; margin-top:.55rem;
  background:linear-gradient(135deg,#2E90FA,#7FB4E8); }
.oe-anchor { position:relative; top:-1.4rem; visibility:hidden; }
@media (max-width:1024px){ #oe-rail{ display:none; }
  [data-testid="stMainBlockContainer"]{ padding-left:2.3rem !important; } }

/* ---- display heading ---- */
.oe-brand { width:2.7rem; height:2.7rem; border-radius:14px; background:#2E90FA;
  display:flex; align-items:center; justify-content:center;
  color:#0D0D0D; margin-bottom:.9rem; }
.oe-brand .material-symbols-rounded { font-size:24px; }
.oe-h1 { font-weight:700; font-size:2.9rem; line-height:1.04; letter-spacing:-1.4px;
  color:#0D0D0D; margin:0 0 1.3rem; }
.oe-h1 .hl { background:#2E90FA; border-radius:11px; padding:0 .26rem; box-decoration-break:clone; }

/* ---- pill tab bar ---- */
[data-testid="stRadio"] div[role="radiogroup"] { flex-direction:row !important;
  flex-wrap:wrap; gap:.38rem; align-items:center; }
[data-testid="stRadio"] div[role="radiogroup"] > label { background:#FFF; border:1px solid #E4E3DB;
  border-radius:999px; padding:.4rem .85rem !important; margin:0 !important; cursor:pointer; }
[data-testid="stRadio"] div[role="radiogroup"] > label:hover { border-color:#0D0D0D; }
[data-testid="stRadio"] div[role="radiogroup"] > label p { font-size:.8rem !important;
  font-weight:500; color:#111 !important; letter-spacing:0 !important; text-transform:none !important; }
[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) { background:#0D0D0D; border-color:#0D0D0D; }
[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p { color:#F5F4EF !important; }
[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child { display:none !important; }

/* ---- section header ---- */
.oe-sec { display:flex; align-items:center; gap:.75rem; margin:2.4rem 0 1rem; }
.oe-sec .ic { width:2.5rem; height:2.5rem; border-radius:13px; background:#0D0D0D; color:#F5F4EF;
  display:flex; align-items:center; justify-content:center; flex:none; }
.oe-sec .ic .material-symbols-rounded { font-size:22px; }
.oe-sec .t { font-family:'Space Grotesk',sans-serif; font-size:1.4rem; font-weight:600;
  color:#0D0D0D; letter-spacing:-.4px; }
.oe-sec .h { flex:1; }
.oe-sec .hint { font-size:.76rem; color:#9A9D96; }

/* ---- cards ---- */
.oe-card { background:#FFF; border:1px solid #EEEDE6; border-radius:24px; padding:1.15rem 1.3rem; height:100%; }
.oe-card.lime { background:#2E90FA; border-color:#2E90FA; }
.oe-card.ink  { background:#0D0D0D; border-color:#0D0D0D; }
.oe-card .lab { font-size:.72rem; font-weight:600; letter-spacing:.6px; text-transform:uppercase; color:#9A9D96; }
.oe-card.ink .lab { color:#9A9A9A; }
.oe-card .val { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:2.9rem;
  line-height:1.05; color:#0D0D0D; margin:.35rem 0 .1rem; }
.oe-card.ink .val { color:#F5F4EF; }
.oe-card .unit { font-size:.82rem; color:#9A9D96; }
.oe-card.ink .unit { color:#8A8A8A; }
.oe-badge { display:inline-flex; align-items:center; gap:.3rem; font-size:.72rem; font-weight:600;
  background:#0D0D0D; color:#F5F4EF; border-radius:999px; padding:.16rem .5rem; }
.oe-card.ink .oe-badge { background:#2E90FA; color:#0D0D0D; }
.oe-dom { display:flex; gap:.34rem; margin-top:1rem; }
.oe-dom i { flex:1; height:13px; border-radius:4px; }
.oe-dom i.f { background:#0D0D0D; }
.oe-card.lime .oe-dom i.f { background:#0D0D0D; }
.oe-card.ink .oe-dom i.f { background:#2E90FA; }
.oe-dom i.e { border:1.5px dashed #C4C3BA; }
.oe-card.lime .oe-dom i.e { border-color:rgba(0,0,0,.28); }
.oe-card.ink .oe-dom i.e { border-color:#333; }
.oe-insight { margin-top:1rem; font-size:.82rem; line-height:1.5; color:#C9C9C9; }
.oe-insight b { color:#2E90FA; }
.oe-mini { background:#FFF; border:1px solid #EEEDE6; border-radius:18px; padding:.85rem 1rem; }
.oe-mini .lab { font-size:.66rem; font-weight:600; letter-spacing:.5px; text-transform:uppercase; color:#9A9D96; }
.oe-mini .val { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.4rem; color:#0D0D0D; margin-top:.15rem; }

.stPlotlyChart { background:#FFF; border:1px solid #EEEDE6; border-radius:22px; padding:.5rem .3rem; }
.stRadio label[data-testid], .stSlider label, .stSelectbox label {
  color:#8B8E86 !important; font-size:.7rem !important; letter-spacing:.6px; text-transform:uppercase; font-weight:600; }
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background:#0D0D0D; }
hr { border-color:#E4E3DB; }
.oe-link { background:#FFF; border:1px solid #EEEDE6; border-radius:18px; padding:.9rem 1rem;
  display:flex; gap:.7rem; align-items:flex-start; height:100%; }
.oe-link .ic { width:2rem; height:2rem; border-radius:10px; background:#EDECE4; display:flex;
  align-items:center; justify-content:center; font-size:.9rem; flex:none; }
.oe-link .tt { font-weight:600; font-size:.86rem; color:#0D0D0D; }
.oe-link .ds { font-size:.74rem; color:#9A9D96; margin-top:.15rem; line-height:1.4; }
.oe-link .ar { margin-left:auto; color:#B7B7AE; flex:none; }
[data-testid="stExpander"] { border:1px solid #EEEDE6; border-radius:16px; background:#FFF; }
[data-testid="stExpander"] summary { font-size:.8rem; }
[data-testid="stDataFrame"] { border-radius:12px; }
</style>
""", unsafe_allow_html=True)


# ---- helpers ----------------------------------------------------------- #
def sec(anchor, icon, title, hint=""):
    st.markdown(f'<div id="{anchor}" class="oe-anchor"></div>'
                f'<div class="oe-sec"><div class="ic">'
                f'<span class="material-symbols-rounded">{icon}</span></div>'
                f'<div class="t">{title}</div><div class="h"></div>'
                f'<div class="hint">{hint}</div></div>', unsafe_allow_html=True)


def domino(pct, n=10):
    f = int(round(np.clip(pct, 0, 1) * n))
    return ('<div class="oe-dom">'
            + "".join('<i class="f"></i>' for _ in range(f))
            + "".join('<i class="e"></i>' for _ in range(n - f)) + "</div>")


def kpi_big(kind, lab, val, unit, pct, badge=None):
    b = f'<span class="oe-badge">{badge}</span>' if badge else ""
    return (f'<div class="oe-card {kind}"><div class="lab">{lab}</div>'
            f'<div class="val">{val} {b}</div><div class="unit">{unit}</div>'
            f'{domino(pct)}</div>')


def mini(lab, val):
    return f'<div class="oe-mini"><div class="lab">{lab}</div><div class="val">{val}</div></div>'


def link_card(icon, title, desc):
    return (f'<div class="oe-link"><div class="ic">{icon}</div><div>'
            f'<div class="tt">{title}</div><div class="ds">{desc}</div></div>'
            f'<div class="ar">↗</div></div>')


def style_fig(fig, h=270, legend_top=True):
    fig.update_layout(
        height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk, Inter, sans-serif", size=11, color="#5C5F58"),
        margin=dict(l=8, r=8, t=10, b=8),
        colorway=[INK, AQUA, "#B0A98F", "#8FA3B8"],
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#5C5F58"),
                    orientation="h", y=1.16 if legend_top else -0.22, x=0),
        hoverlabel=dict(bgcolor="#0D0D0D", font_color="#F5F4EF",
                        font_family="Space Grotesk"))
    fig.update_xaxes(gridcolor="#EDECE4", zerolinecolor="#E4E3DB", linecolor="#E4E3DB",
                     tickfont=dict(size=9, color="#9A9D96"))
    fig.update_yaxes(gridcolor="#EDECE4", zerolinecolor="#E4E3DB", linecolor="#E4E3DB",
                     tickfont=dict(size=9, color="#9A9D96"))
    return fig


def heat(arr, x, y, scale, label):
    fig = px.imshow(arr, x=x, y=y, origin="lower", aspect="auto",
                    color_continuous_scale=scale)
    fig.update_layout(coloraxis_colorbar=dict(
        title=dict(text=label, font=dict(size=9, color="#9A9D96")),
        thickness=7, len=0.88, tickfont=dict(size=8, color="#9A9D96"), outlinewidth=0))
    return style_fig(fig, h=248)


# ---- loaders --------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_assets():
    if not ASSETS.exists():
        st.error("dashboard/assets.npz missing — run  `python -m dashboard.prep_assets`")
        st.stop()
    z = np.load(ASSETS, allow_pickle=True)
    return {k: z[k] for k in z.files}


@st.cache_data(show_spinner=False)
def load_csv(name):
    p = METRICS / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_logs():
    out = {}
    for f in METRICS.glob("train_log_*.csv"):
        n = f.stem[len("train_log_"):]
        if not n.startswith("smoke"):
            out[n] = pd.read_csv(f)
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
    g = A["regime_grid"][month - 1].astype(float)
    g[g < 0] = np.nan
    return g, A["regime_grid_lat"], A["regime_grid_lon"]


# ---- fixed dark icon rail (one entry per panel) --------------------- #
def _mi(name):
    return f'<span class="material-symbols-rounded">{name}</span>'


RAIL = [
    ("s1", "satellite_alt", "Satellite input fields"),
    ("s2", "waves", "Subsurface reconstruction"),
    ("s3", "stacked_line_chart", "Skill by depth"),
    ("s4", "scatter_plot", "Predicted vs observed"),
    ("s5", "grid_view", "Regime overlay"),
    ("s6", "compare_arrows", "OceanEmbed vs baseline"),
    ("s7", "timeline", "Training dynamics"),
    ("s8", "science", "Physics consistency"),
]
st.markdown(
    '<div id="oe-rail">'
    f'<a class="logo" href="#oe-top" title="Top">{_mi("blur_on")}</a>'
    + "".join(f'<a class="ic" href="#{a}" title="{t}">{_mi(ic)}</a>' for a, ic, t in RAIL)
    + '<div class="sp"></div>'
    f'<a class="ic" href="#s9" title="Data &amp; scope">{_mi("info")}</a>'
    f'<a class="ic" href="https://github.com/Specter842/oceanembed" target="_blank" '
    f'title="Repository">{_mi("open_in_new")}</a>'
    '<div class="av"></div></div>'
    '<div id="oe-top" class="oe-anchor"></div>',
    unsafe_allow_html=True)


# ---- header ----------------------------------------------------------- #
A = load_assets()
metrics = load_csv("per_regime_rmse.csv")
phys = load_csv("physics_diagnostics.csv")
logs = load_logs()
models = models_avail()
weeks = pd.to_datetime(A["sat_weeks"])
LAT, LON = A["sat_lat"], A["sat_lon"]

st.markdown('<div class="oe-brand"><span class="material-symbols-rounded">'
            'sailing</span></div>'
            '<div class="oe-h1">Reconstructing the Ocean Interior<br>'
            'from the <span class="hl">Surface</span> Alone</div>',
            unsafe_allow_html=True)

cbar = st.columns([2.3, 2, 6])
model = cbar[0].radio("model", models, horizontal=True,
                      index=models.index("oceanembed") if "oceanembed" in models else 0)
holdout = cbar[1].radio("holdout", ["spatial", "temporal"], horizontal=True,
                        help="spatial = Bay of Bengal block · temporal = JJAS 2022")
pred = load_pred(model, holdout)


def pooled(m, ts, reg, col):
    if metrics is None:
        return np.nan
    r = metrics[(metrics.model == m) & (metrics.test_set == ts)
                & (metrics.regime_class == reg) & (metrics.depth_level == -1)]
    return float(r[col].iloc[0]) if len(r) else np.nan


rt = pooled(model, holdout, "barrier_layer_stratified", "rmse_T")
rb = pooled("baseline", holdout, "barrier_layer_stratified", "rmse_T")
rr = pooled(model, holdout, "barrier_layer_stratified", "r_T")
n_ho = len(pred["lat"]) if pred else 0


def hardest_band():
    if metrics is None:
        return None
    d = metrics[(metrics.model == model) & (metrics.test_set == holdout)
                & (metrics.regime_class == "barrier_layer_stratified")
                & (metrics.depth_level > 0)]
    if not len(d):
        return None
    row = d.loc[d.rmse_T.idxmax()]
    return int(row.depth_level), float(row.rmse_T)


hb = hardest_band()

h = st.columns([1, 1, 1])
h[0].markdown(kpi_big("", "matched profiles", "8,340", "of ~10,000 target",
                      min(8340 / 10000, 1)), unsafe_allow_html=True)
h[1].markdown(kpi_big("lime", "barrier-layer RMSE·T",
                      f"{rt:.3f}" if rt == rt else "—", "°C   ·   lower is better",
                      1 - min(rt / 2.0, 1) if rt == rt else 0,
                      badge=f"{rt - rb:+.3f} vs base" if rt == rt and rb == rb else None),
              unsafe_allow_html=True)
_ins = (f'error is not uniform — it peaks at <b>{hb[0]} m</b> '
        f'(±{hb[1]:.2f} °C in the thermocline) and is near-zero above 30 m and '
        f'below 500 m. Surface fields carry least information about thermocline depth.'
        if hb else 'near-perfect profile shape overall.')
h[2].markdown(
    f'<div class="oe-card ink"><div class="lab">profile fit  ·  {holdout} holdout</div>'
    f'<div class="val">{rr:.3f}</div>'
    f'<div class="unit">Pearson r  ·  predicted vs Argo temperature</div>'
    f'<div class="oe-insight">{_ins}</div></div>', unsafe_allow_html=True)

m = st.columns(4)
m[0].markdown(mini("weekly satellite fields", f"{len(weeks)}"), unsafe_allow_html=True)
m[1].markdown(mini("depth range", "0–2000 m"), unsafe_allow_html=True)
m[2].markdown(mini(f"{holdout} holdout · n", f"{n_ho}"), unsafe_allow_html=True)
m[3].markdown(mini("standard depth levels", "18"), unsafe_allow_html=True)


# ---- 1 · satellite input fields ------------------------------------------- #
sec("s1", "satellite_alt", "Satellite input fields", "surface observations — the only model input")
wk = st.select_slider("week", [d.strftime("%Y-%m-%d") for d in weeks],
                      value=weeks[min(len(weeks) - 1, 86)].strftime("%Y-%m-%d"))
wi = [d.strftime("%Y-%m-%d") for d in weeks].index(wk)
sc_sst = [[0, "#F5F4EF"], [.5, "#2E90FA"], [1, "#0D0D0D"]]
sc_ssh = [[0, "#0D0D0D"], [.5, "#F5F4EF"], [1, "#2E90FA"]]
sc_sss = [[0, "#F5F4EF"], [.5, "#B0A98F"], [1, "#0D0D0D"]]
g = st.columns(3)
for col, var, lab, s in zip(g, ("sst", "ssh", "sss"),
                            ("SST °C", "SLA m", "SSS PSU"), (sc_sst, sc_ssh, sc_sss)):
    col.plotly_chart(heat(A[f"sat_{var}"][wi], LON, LAT, s, lab), use_container_width=True)


# ---- 2 · subsurface reconstruction ----------------------------------- #
sec("s2", "waves", "Subsurface reconstruction", "click a float location")
if pred is None:
    st.warning(f"no predictions for {model}/{holdout} — run `python -m src.evaluate`")
    st.stop()
lv = pred["depth_levels"]
regnames = np.array(REGIMES)[pred["regime"].astype(int)]
pts = pd.DataFrame({"lat": pred["lat"], "lon": pred["lon"], "regime": regnames,
                    "i": np.arange(len(pred["lat"]))})
L, R = st.columns([1, 1.25])
with L:
    fm = px.scatter(pts, x="lon", y="lat", color="regime", color_discrete_map=RC,
                    custom_data=["i"])
    fm.update_traces(marker=dict(size=7, line=dict(width=0)))
    ev = st.plotly_chart(style_fig(fm, h=380), use_container_width=True,
                         on_select="rerun", key="pmap")
    pk = ev.get("selection", {}).get("points", []) if ev else []
    idx = int(pk[0]["customdata"][0]) if pk else (176 if len(pts) > 176 else 0)
with R:
    P = {k: pred[k][idx] for k in ("T_pred", "S_pred", "T_std", "S_std",
                                   "T_obs", "S_obs", "mask")}
    mk = P["mask"] > 0
    row = pts.iloc[idx]
    st.markdown(f'<span style="font-family:Space Grotesk;color:#8B8E86;font-size:.8rem">'
                f'#{idx} · {row.lat:.2f}°N {row.lon:.2f}°E · <b style="color:#0D0D0D">'
                f'{row.regime}</b></span>', unsafe_allow_html=True)
    fig = go.Figure()
    for var, pv, sv, ov, cc in [("T", "T_pred", "T_std", "T_obs", INK),
                                ("S", "S_pred", "S_std", "S_obs", "#8FA3B8")]:
        xa = "x" if var == "T" else "x2"
        fig.add_trace(go.Scatter(
            x=np.r_[P[pv][mk] - 2 * P[sv][mk], (P[pv][mk] + 2 * P[sv][mk])[::-1]],
            y=np.r_[lv[mk], lv[mk][::-1]], fill="toself", fillcolor=cc, opacity=.12,
            line=dict(width=0), xaxis=xa, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=P[pv][mk], y=lv[mk], name=f"{var} pred",
                                 line=dict(color=cc, width=2.5), xaxis=xa))
        fig.add_trace(go.Scatter(x=P[ov][mk], y=lv[mk], name=f"{var} Argo",
                                 line=dict(color=cc, width=1, dash="dot"),
                                 mode="lines+markers", marker=dict(size=4), xaxis=xa))
    fig.update_layout(height=380, yaxis=dict(title="depth m", autorange="reversed"),
                      xaxis=dict(title="T °C", domain=[0, 1]),
                      xaxis2=dict(title="S PSU", overlaying="x", side="top"))
    st.plotly_chart(style_fig(fig, h=380), use_container_width=True)
resid = (P["T_pred"] - P["T_obs"])[mk]
rt_i = float(np.sqrt(np.mean((P["T_pred"][mk] - P["T_obs"][mk]) ** 2)))
rs_i = float(np.sqrt(np.mean((P["S_pred"][mk] - P["S_obs"][mk]) ** 2)))
rf = go.Figure(go.Bar(x=lv[mk], y=resid, marker_color=np.where(resid >= 0, INK, AQUA),
                      marker_cornerradius=6))
rf.update_layout(height=150, xaxis_title="depth m", yaxis_title="T resid °C", bargap=.55)
c1, c2 = st.columns([3, 1])
c1.plotly_chart(style_fig(rf, h=150, legend_top=False), use_container_width=True)
c2.markdown(mini("this profile RMSE·T", f"{rt_i:.2f} °C"), unsafe_allow_html=True)
c2.markdown(mini("RMSE·S", f"{rs_i:.2f} PSU"), unsafe_allow_html=True)


# ---- 3 · skill by depth --------------------------------------------------- #
sec("s3", "stacked_line_chart", "Skill by depth", f"{holdout} holdout · barrier-layer regime · per level")
if metrics is not None:
    d = metrics[(metrics.test_set == holdout) & (metrics.depth_level >= 0)
                & (metrics.regime_class == "barrier_layer_stratified")
                & (metrics.model.isin(["baseline", model]))]
    gg = st.columns(2)
    for col, yv, lab in zip(gg, ("rmse_T", "rmse_S"), ("RMSE T °C", "RMSE S PSU")):
        f = go.Figure()
        for mdl in ["baseline", model]:
            s = d[d.model == mdl].sort_values("depth_level")
            f.add_trace(go.Scatter(x=s[yv], y=s.depth_level, name=mdl,
                                   line=dict(color=MC.get(mdl, INK), width=2.5),
                                   mode="lines+markers", marker=dict(size=5)))
        f.update_layout(height=300, xaxis_title=lab,
                        yaxis=dict(title="depth m", autorange="reversed"))
        col.plotly_chart(style_fig(f, h=300), use_container_width=True)


# ---- 4 · predicted vs observed ------------------------------------------ #
sec("s4", "scatter_plot", "Predicted vs observed", f"all {len(pts)} holdout profiles · {model}")
gg = st.columns(2)
for col, pk_, ok_, lab, cc in [(gg[0], "T_pred", "T_obs", "temperature °C", INK),
                               (gg[1], "S_pred", "S_obs", "salinity PSU", "#8FA3B8")]:
    mm = pred["mask"] > 0
    pv, ov = pred[pk_][mm], pred[ok_][mm]
    r = float(np.corrcoef(pv, ov)[0, 1])
    f = go.Figure()
    f.add_trace(go.Histogram2dContour(x=ov, y=pv, colorscale=[[0, "#F5F4EF"], [1, cc]],
                                      showscale=False, ncontours=13, line=dict(width=0)))
    lim = [min(pv.min(), ov.min()), max(pv.max(), ov.max())]
    f.add_trace(go.Scatter(x=lim, y=lim, line=dict(color="#B7B7AE", dash="dash", width=1),
                           showlegend=False, hoverinfo="skip"))
    f.update_layout(height=300, xaxis_title=f"observed {lab}",
                    yaxis_title=f"predicted {lab}",
                    annotations=[dict(x=.05, y=.93, xref="paper", yref="paper",
                                      text=f"r = {r:.3f}", showarrow=False,
                                      font=dict(color="#0D0D0D", size=14,
                                                family="Space Grotesk"))])
    col.plotly_chart(style_fig(f, h=300, legend_top=False), use_container_width=True)


# ---- 5 · regime overlay ------------------------------------------------- #
sec("s5", "grid_view", "Regime overlay",
    "class from independent climatology + discharge — never the satellite channels")
mo = st.slider("month", 1, 12, 8)
L2, R2 = st.columns([1.4, 1])
with L2:
    gm, glat, glon = regime_map(A, mo)
    f = px.imshow(gm, x=glon, y=glat, origin="lower", aspect="auto",
                  range_color=[-0.5, 2.5],
                  color_continuous_scale=[RC[r] for r in REGIMES])
    f.update_layout(coloraxis_colorbar=dict(
        tickvals=[0, 1, 2], ticktext=["mixed", "barrier", "upwell"],
        thickness=7, len=0.8, tickfont=dict(size=8, color="#9A9D96")))
    st.plotly_chart(style_fig(f, h=330), use_container_width=True)
with R2:
    clat, clon = A["clim_lat"], A["clim_lon"]
    my = (clat >= 5) & (clat <= 22)
    mx = (clon >= 85) & (clon <= 99)
    mi = mo - 1
    blt = float(np.nanmean(A["clim_blt_m"][mi][np.ix_(my, mx)]))
    mld = float(np.nanmean(A["clim_mld_m"][mi][np.ix_(my, mx)]))
    strat = float(np.nanmean(A["clim_strat"][mi][np.ix_(my, mx)]))
    f = go.Figure(go.Bar(x=["BLT m", "MLD m", "strat×50"], y=[blt, mld, strat * 50],
                         marker_color=[INK, "#8FA3B8", AQUA], marker_cornerradius=8))
    f.update_layout(height=150, yaxis_title="Bay of Bengal mean", bargap=.5)
    st.plotly_chart(style_fig(f, h=150, legend_top=False), use_container_width=True)
    vc = pd.Series(regnames).value_counts().reindex(REGIMES, fill_value=0)
    f = go.Figure(go.Bar(y=vc.index, x=vc.values, orientation="h",
                         marker_color=[RC[r] for r in vc.index], marker_cornerradius=8))
    f.update_layout(height=150, xaxis_title=f"{holdout} holdout profiles", bargap=.45)
    st.plotly_chart(style_fig(f, h=150, legend_top=False), use_container_width=True)


# ---- 6 · OceanEmbed vs baseline --------------------------------------- #
sec("s6", "compare_arrows", "OceanEmbed vs baseline", "per-regime RMSE, pooled over depth")
if metrics is not None:
    var = st.radio("variable", ["rmse_T", "rmse_S"], horizontal=True,
                   format_func=lambda s: "Temperature" if s.endswith("T") else "Salinity")
    pool = metrics[metrics.depth_level == -1]
    keep = [m_ for m_ in ["baseline", "oceanembed", model] if m_ in pool.model.unique()]
    sub = pool[pool.model.isin(keep) & pool.regime_class.isin(REGIMES)]
    f = px.bar(sub, x="regime_class", y=var, color="model", barmode="group",
               facet_col="test_set", color_discrete_map=MC)
    f.update_traces(marker_cornerradius=14)
    f.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].upper(),
                                             font=dict(size=10, color="#9A9D96")))
    f.update_layout(height=340)
    st.plotly_chart(style_fig(f, h=340), use_container_width=True)
    with st.expander("full table — per test set × regime"):
        st.dataframe(pool[pool.regime_class.isin(REGIMES + ["all"])]
                     .pivot_table(index=["test_set", "regime_class"], columns="model",
                                  values=var).round(3), use_container_width=True)


# ---- 7 · training dynamics ------------------------------------------- #
sec("s7", "timeline", "Training dynamics", "validation RMSE per epoch")
if logs:
    gg = st.columns(2)
    for col, yv, lab in zip(gg, ("rmseT", "rmseS"), ("val RMSE T °C", "val RMSE S PSU")):
        f = go.Figure()
        for name, dfl in logs.items():
            v = dfl[dfl.split == "val"]
            f.add_trace(go.Scatter(x=v.epoch, y=v[yv], name=name,
                                   line=dict(color=MC.get(name, "#B7B7AE"), width=2)))
        f.update_layout(height=260, xaxis_title="epoch", yaxis_title=lab)
        col.plotly_chart(style_fig(f, h=260), use_container_width=True)


# ---- 8 · physics consistency ----------------------------------------- #
sec("s8", "science", "Physics consistency", "TEOS-10 density-inversion fraction · lower is better")
if phys is not None:
    f = go.Figure()
    for ts, cc in [("spatial", INK), ("temporal", AQUA)]:
        s = phys[phys.test_set == ts]
        f.add_trace(go.Bar(name=ts, x=s.model, y=s.inversion_fraction * 100,
                           marker_color=cc, marker_cornerradius=10))
    f.update_layout(height=230, barmode="group", yaxis_title="% inverted level pairs")
    st.plotly_chart(style_fig(f, h=230), use_container_width=True)
    st.caption("≈ 0 for every model — profiles are already density-stable, so the "
               "physics loss term has little to correct at this data scale.")


# ---- 9 · data & scope ----------------------------------------------- #
sec("s9", "info", "Data & scope")
st.markdown(
    '<div class="oe-card" style="border-radius:20px;border-left:3px solid #2E90FA;'
    'background:#FCFCF9">'
    'This demo runs on <b>historical downloaded satellite and Argo data</b>. Real-time '
    'ingestion and the float-deployment / cyclone-flagging features in the roadmap are '
    '<b>not implemented</b> in this build.<br><br>'
    '· Satellite inputs are NOAA-hosted (OISST v2.1, CoastWatch blended SLA, CoastWatch '
    'SMAP SSS), substituted for CMEMS / PODAAC. No ocean-colour channel.<br>'
    '· <b>INCOIS moored-buoy data could not be accessed</b> — the India-specific-data / '
    'data-sovereignty differentiator is future work. River discharge is a literature '
    'climatology, not Indian gauge data.<br>'
    '· Barrier-layer climatology is WOA23-derived, not the published de Boyer Montégut product.<br>'
    '· Backbone shown: ResNet-18 / 20 epochs / CPU — preliminary. See RUN.md.'
    '</div>', unsafe_allow_html=True)
lk = st.columns(3)
lk[0].markdown(link_card("▤", "Phase 0 report", "Full data reality check and every "
                         "endpoint substitution."), unsafe_allow_html=True)
lk[1].markdown(link_card("◔", "Phase 3 findings", "Honest per-regime result analysis "
                         "and likely causes."), unsafe_allow_html=True)
lk[2].markdown(link_card("⧉", "Repository", "github.com/Specter842/oceanembed — code, "
                         "docs, RUN.md."), unsafe_allow_html=True)
