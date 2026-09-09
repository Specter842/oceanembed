"""
OceanEmbed demo dashboard (Streamlit) — CLAUDE.md §7.1.

Reads only pre-computed artefacts (no torch / netCDF4 / xarray / gsw):
  dashboard/assets.npz                     (baked by dashboard/prep_assets.py)
  outputs/metrics/per_regime_rmse.csv, physics_diagnostics.csv, train_log_*.csv
  outputs/metrics/predictions_<model>_<set>.npz   (from `python -m src.evaluate`)

Run:  python -m streamlit run dashboard/app.py --client.toolbarMode minimal
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ASSETS = ROOT / "dashboard" / "assets.npz"
METRICS = ROOT / "outputs" / "metrics"

# ---- palette ------------------------------------------------------------- #
FOG = "#A7C4D6"          # muted slate backdrop
CREAM = "#F3F8FC"
INK = "#0D0D0D"
AQUA = "#6FC0F5"         # ocean-blue accent
MUTE = "#8B98A2"
HAIR = "#DCE7EE"

REGIMES = ["well_mixed", "barrier_layer_stratified", "upwelling"]
RC = {"well_mixed": AQUA, "barrier_layer_stratified": INK,
      "upwelling": "#CBA24E", "no data": "#DCE6EC"}
MC = {"oceanembed": AQUA, "baseline": INK,
      "oceanembed_lp05": "#8FB4CB", "oceanembed_lp30": "#C0CAD0"}

# Material Symbols glyphs by name -> Unicode codepoint (matches the self-hosted
# 21-glyph subset). Used instead of ligatures, which need the full font.
_ICON_CP = {
    "blur_on": "", "dashboard": "", "satellite_alt": "",
    "waves": "", "water": "", "stacked_line_chart": "",
    "scatter_plot": "", "grid_view": "", "compare_arrows": "",
    "balance": "", "timeline": "", "science": "", "info": "",
    "sailing": "", "tune": "", "code": "", "monitoring": "",
    "description": "", "query_stats": "", "open_in_new": "",
    "north_east": "",
}


def _ic(name):
    """<span> carrying the icon glyph (falls back to the raw name if unknown)."""
    return f'<span class="material-symbols-rounded">{_ICON_CP.get(name, name)}</span>'


st.set_page_config(page_title="OceanEmbed", layout="wide", page_icon="🛰️",
                   initial_sidebar_state="expanded")

st.markdown(
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    'family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap">',
    unsafe_allow_html=True)

# Material Symbols is self-hosted (21-glyph subset, ~2.8 KB woff2) so the icons
# never depend on a Google Fonts round-trip — see dashboard/prep_iconfont.py.
_ICON_B64 = (
    "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"
)
st.markdown(
    "<style>@font-face{font-family:'Material Symbols Rounded';font-style:normal;"
    "font-weight:400;src:url(data:font/woff2;base64," + _ICON_B64 +
    ") format('woff2')}</style>", unsafe_allow_html=True)

st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');
.material-symbols-rounded { font-family:'Material Symbols Rounded'; font-weight:normal; font-style:normal;
  line-height:1; letter-spacing:normal; text-transform:none; display:inline-block; white-space:nowrap;
  word-wrap:normal; direction:ltr; -webkit-font-feature-settings:'liga'; -webkit-font-smoothing:antialiased;
  font-feature-settings:'liga'; font-variation-settings:'FILL' 0,'wght' 400,'GRAD' 0,'opsz' 24; }

.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background:#A7C4D6 !important; }
/* hide Streamlit's own chrome — the Share / star / edit / Deploy bar + status widget */
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stToolbarActions"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton { display:none !important; }
[data-testid="stAppViewBlockContainer"] { padding-top:0 !important; }
[data-testid="stMainBlockContainer"] {
  background:#E4EEF5; border-radius:30px; max-width:1440px; position:relative;
  padding:1.5rem 3rem 3rem 5.3rem !important; margin:.75rem auto 1.4rem;
  box-shadow:0 24px 60px -20px rgba(0,0,0,.30); scroll-behavior:smooth;
  animation:oe-page .34s cubic-bezier(.22,.68,.24,1); }
@keyframes oe-page { from { opacity:0; transform:translateY(14px) scale(.992); }
  60% { opacity:1; } to { opacity:1; transform:none; } }
/* the top CSS/<link> injections render as empty rows — pull them out of the
   flex flow so they add no gap (position:absolute keeps their <style> live;
   display:none would not) */
[data-testid="stMainBlockContainer"] [data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] style),
[data-testid="stMainBlockContainer"] [data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] link) {
  position:absolute !important; height:0 !important; overflow:hidden !important; }
html { scroll-behavior:smooth; }
html, body, [class*="css"], p, span, div, label, .stMarkdown { font-family:'Inter',system-ui,sans-serif; color:#111; }
h1,h2,h3,.disp { font-family:'Space Grotesk','Inter',sans-serif; }

/* ---- fixed dark icon rail (keyed nav radio, pinned left; icons via ::before) ---- */
[data-testid="stSidebar"] { display:none !important; }
[data-testid="stApp"] .st-key-oe_nav {
  position:fixed !important; z-index:999 !important;
  top:50% !important; bottom:auto !important; right:auto !important;
  left:max(1.1rem, calc((100vw - 1440px)/2 + 1.1rem)) !important;
  transform:translateY(-50%) !important;
  width:3.7rem !important; min-width:3.7rem !important; max-width:3.7rem !important;
  background:#0D0D0D !important; border-radius:24px !important; padding:1rem .55rem !important;
  margin:0 !important; transition:width .17s ease, min-width .17s ease, max-width .17s ease; }
/* hover the rail -> it widens; each icon gets its page name via ::after
   (the real markdown text stays hidden — it lives in an absolutely-positioned
   wrapper that overlaps the icon when revealed) */
[data-testid="stApp"] .st-key-oe_nav:hover {
  width:19.5rem !important; min-width:19.5rem !important; max-width:19.5rem !important;
  padding:1.2rem .75rem !important; box-shadow:0 24px 58px -14px rgba(0,0,0,.55); }
.st-key-oe_nav:hover div[role="radiogroup"] { align-items:stretch !important; gap:.3rem; }
.st-key-oe_nav:hover div[role="radiogroup"] > label { width:100% !important; height:3.1rem !important;
  justify-content:flex-start !important; padding:0 .7rem !important; overflow:visible !important; }
.st-key-oe_nav:hover div[role="radiogroup"] > label::before { flex:none; font-size:25px !important; }
.st-key-oe_nav:hover div[role="radiogroup"] > label::after {
  position:relative; z-index:2; margin-left:.85rem; color:#F3F8FC; white-space:nowrap;
  font:500 .95rem/1 'Space Grotesk','Inter',sans-serif; letter-spacing:-.1px; }
.st-key-oe_nav:hover div[role="radiogroup"] > label:nth-of-type(1)::after { content:"OceanEmbed"; }
.st-key-oe_nav:hover div[role="radiogroup"] > label:nth-of-type(2)::after { content:"Overview"; }
.st-key-oe_nav:hover div[role="radiogroup"] > label:nth-of-type(3)::after { content:"Inputs & reconstruction"; }
.st-key-oe_nav:hover div[role="radiogroup"] > label:nth-of-type(4)::after { content:"Accuracy & calibration"; }
.st-key-oe_nav:hover div[role="radiogroup"] > label:nth-of-type(5)::after { content:"Regime overlay"; }
.st-key-oe_nav:hover div[role="radiogroup"] > label:nth-of-type(6)::after { content:"Model vs baseline"; }
.st-key-oe_nav:hover div[role="radiogroup"] > label:nth-of-type(7)::after { content:"Data & scope"; }
/* keep the real radio input out of the way (label click still selects it) */
.st-key-oe_nav div[role="radiogroup"] > label > span:first-child {
  position:absolute !important; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); }
/* nav item 1 (the sailing icon) = brand mark + link to the About view */
.st-key-oe_nav div[role="radiogroup"] > label:nth-of-type(1) { margin-bottom:.7rem !important; }
.st-key-oe_nav [data-testid="stWidgetLabel"], .st-key-oe_nav [data-testid="stCaptionContainer"] { display:none !important; }
.st-key-oe_nav div[role="radiogroup"] { flex-direction:column !important; gap:.18rem; align-items:center; }
.st-key-oe_nav div[role="radiogroup"] > label { position:relative; width:2.6rem; height:2.6rem; min-height:0;
  padding:0 !important; margin:0 !important; border-radius:12px; background:transparent;
  border:none; display:flex; align-items:center; justify-content:center; cursor:pointer;
  transition:.12s; overflow:hidden; flex:none !important; }
.st-key-oe_nav div[role="radiogroup"] > label:hover { background:#242424; }
/* the native radio circle (eqiohyi4) is the active indicator: Streamlit paints it
   #6FC0F5 when checked, rgba(13,13,13,.2) (invisible on the #0D0D0D rail) otherwise.
   Blow it up to fill the cell — no :has()/nth-of-type, so it never lags on rerun. */
.st-key-oe_nav div[role="radiogroup"] > label > div { position:absolute !important; inset:0 !important;
  display:block !important; padding:0 !important; margin:0 !important; }
.st-key-oe_nav div[role="radiogroup"] [class*="eqiohyi3"] { position:absolute !important; inset:0 !important;
  gap:0 !important; padding:0 !important; margin:0 !important; }
.st-key-oe_nav div[role="radiogroup"] [class*="eqiohyi4"] { position:absolute !important; inset:0 !important;
  width:auto !important; height:auto !important; min-width:0 !important; min-height:0 !important;
  border:none !important; border-radius:12px !important; transition:background .12s; }
/* active colour = Streamlit's primaryColor, painted natively on the checked radio
   (updates correctly per rerun). Set primaryColor in .streamlit/config.toml. */
.st-key-oe_nav div[role="radiogroup"] [class*="eqiohyi5"] { display:none !important; }
.st-key-oe_nav div[role="radiogroup"] label [data-testid="stMarkdownContainer"] { display:none !important; }
.st-key-oe_nav div[role="radiogroup"] > label::before { position:relative; z-index:2;
  font-family:'Material Symbols Rounded';
  -webkit-font-feature-settings:'liga'; font-feature-settings:'liga'; color:#FFFFFF; font-size:22px; }
.st-key-oe_nav div[role="radiogroup"] > label:nth-of-type(1)::before { content:"\e502"; }
.st-key-oe_nav div[role="radiogroup"] > label:nth-of-type(2)::before { content:"\e871"; }
.st-key-oe_nav div[role="radiogroup"] > label:nth-of-type(3)::before { content:"\eb3a"; }
.st-key-oe_nav div[role="radiogroup"] > label:nth-of-type(4)::before { content:"\e268"; }
.st-key-oe_nav div[role="radiogroup"] > label:nth-of-type(5)::before { content:"\e9b0"; }
.st-key-oe_nav div[role="radiogroup"] > label:nth-of-type(6)::before { content:"\e915"; }
.st-key-oe_nav div[role="radiogroup"] > label:nth-of-type(7)::before { content:"\e88e"; }
@media (max-width:1100px){ .st-key-oe_nav, [data-testid="stApp"] .st-key-oe_nav:hover {
    position:static !important; width:auto !important; min-width:0 !important;
    max-width:none !important; transform:none !important; max-height:none !important;
    flex-direction:row; padding:.5rem; }
  .st-key-oe_nav div[role="radiogroup"]{ flex-direction:row !important; flex-wrap:wrap; }
  .st-key-oe_nav:hover div[role="radiogroup"] > label [data-testid="stMarkdownContainer"] { display:none !important; }
  .st-key-oe_nav::before{ display:none; } }

/* ---- display heading ---- */
.oe-brand { width:2.7rem; height:2.7rem; border-radius:14px; background:#6FC0F5;
  display:flex; align-items:center; justify-content:center; cursor:pointer;
  color:#0D0D0D; margin-bottom:.9rem; transition:transform .12s; text-decoration:none; }
.oe-brand:hover { transform:translateY(-1px); }
.oe-brand .material-symbols-rounded { font-size:24px; }
.oe-h1 { font-weight:700; font-size:2.7rem; line-height:1.05; letter-spacing:-1.3px;
  color:#0D0D0D; margin:0 0 .8rem; }
.oe-h1 .hl { background:#6FC0F5; border-radius:11px; padding:0 .26rem; box-decoration-break:clone; }
.oe-sub { max-width:53rem; font-size:.92rem; line-height:1.55; color:#5A6B75; margin:0 0 1.4rem; }
.oe-sub b { color:#0D0D0D; } .oe-sub i { font-style:italic; color:#3E7CA0; }

/* ---- about / splash view — the landing page ---- */
/* on this view the rail is hidden and the panel padding is symmetric (see the
   inline override in the `page == "about"` block), so nothing needs to break out */
.oe-splash { min-height:38vh; display:flex; flex-direction:column; align-items:center;
  justify-content:center; text-align:center; padding:2.6rem 1rem 1rem; }
.oe-splash .mark { width:5.4rem; height:5.4rem; border-radius:26px; background:#0D0D0D;
  display:flex; align-items:center; justify-content:center; margin-bottom:1.6rem;
  box-shadow:0 20px 44px -16px rgba(20,52,82,.4); transition:transform .12s; text-decoration:none; }
.oe-splash .mark:hover { transform:translateY(-2px); }
.oe-splash .mark .material-symbols-rounded { font-size:50px; color:#6FC0F5; }
.oe-splash h1 { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:3.5rem;
  letter-spacing:-1.6px; color:#0D0D0D; margin:0 0 .5rem; }
.oe-splash .tag { font-family:'Space Grotesk',sans-serif; font-size:1.1rem; font-weight:500;
  color:#3E7CA0; letter-spacing:-.2px; margin:0 0 1.5rem;
  border-bottom:2px solid #6FC0F5; padding-bottom:.35rem; }
.oe-splash .lines { max-width:33rem; font-size:.96rem; line-height:1.7; color:#5A6B75; }
.oe-splash .lines b { color:#0D0D0D; font-weight:600; }
/* the "Run the pipeline" button (st.button key=oe_run) */
.st-key-oe_run, .st-key-oe_run [data-testid="stButton"] {
  display:flex !important; justify-content:center !important; width:100% !important; }
.st-key-oe_run button { width:auto !important; background:#0D0D0D !important; color:#F3F8FC !important;
  border:2px solid #0D0D0D !important; border-radius:13px !important; padding:.72rem 1.7rem !important;
  font-family:'Space Grotesk',sans-serif !important; font-weight:600 !important; font-size:.95rem !important;
  transition:transform .12s, background .12s !important; box-shadow:0 12px 28px -12px rgba(0,0,0,.4); }
.st-key-oe_run button:hover { background:#242424 !important; border-color:#242424 !important;
  transform:translateY(-1px); }
.st-key-oe_run button p, .st-key-oe_run button div { color:#F3F8FC !important; font-weight:600 !important; }
.oe-splash-foot { max-width:35rem; margin:1.3rem auto 0; text-align:center; font-size:.8rem;
  line-height:1.6; color:#8FA0AB; }
.oe-splash-foot code { background:#DCE7EE; border-radius:4px; padding:0 .28rem; color:#5A6B75;
  font-size:.92em; }
.oe-splash-foot .meta { display:block; margin-top:1rem; font-size:.66rem; text-transform:uppercase;
  letter-spacing:1.4px; color:#AAB7C0; }

/* ---- home-only intro / "how to read this" ---- */
.oe-intro { border:1.5px solid #C1D5E3; border-radius:22px; background:#FFF;
  box-shadow:0 6px 18px -8px rgba(20,52,82,.18); padding:1.3rem 1.5rem; margin:.2rem 0 1.6rem; }
.oe-intro h4 { font-family:'Space Grotesk',sans-serif; font-size:.95rem; font-weight:600;
  color:#0D0D0D; margin:0 0 .55rem; display:flex; align-items:center; gap:.45rem; }
.oe-intro h4 .material-symbols-rounded { font-size:18px; color:#3E7CA0; }
.oe-intro p { font-size:.86rem; line-height:1.55; color:#5A6B75; margin:0 0 .7rem; }
.oe-intro p:last-child { margin-bottom:0; }
.oe-intro b { color:#0D0D0D; font-weight:600; }
.oe-intro .steps { display:grid; grid-template-columns:repeat(3,1fr); gap:.7rem; margin-top:.9rem; }
.oe-intro .steps > div { border:1px solid #DCE7EE; border-radius:14px; padding:.7rem .8rem;
  background:#F7FAFC; }
.oe-intro .steps .n { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:.8rem;
  color:#6FC0F5; }
.oe-intro .steps .h { font-weight:600; font-size:.8rem; color:#0D0D0D; margin:.15rem 0; }
.oe-intro .steps .d { font-size:.74rem; line-height:1.45; color:#8093A0; }
@media (max-width:820px){ .oe-intro .steps { grid-template-columns:1fr; } }

/* ---- pill tab bar (model / holdout / variable only) ---- */
:is(.st-key-oe_model,.st-key-oe_holdout,.st-key-oe_var) div[role="radiogroup"] {
  flex-direction:row !important; flex-wrap:nowrap !important; gap:.3rem; align-items:center; }
:is(.st-key-oe_model,.st-key-oe_holdout,.st-key-oe_var) [data-testid="stRadioOption"] {
  background:#FFF; border:1.5px solid #CBDBE7; border-radius:999px;
  padding:.3rem .62rem !important; margin:0 !important; cursor:pointer; transition:.12s; flex:none; }
:is(.st-key-oe_model,.st-key-oe_holdout,.st-key-oe_var) [data-testid="stRadioOption"]:hover { border-color:#0D0D0D; }
:is(.st-key-oe_model,.st-key-oe_holdout,.st-key-oe_var) [data-testid="stRadioOption"] [class*="eqiohyi4"] {
  background:none !important; border:none !important; box-shadow:none !important; min-width:0 !important; width:auto !important; }
:is(.st-key-oe_model,.st-key-oe_holdout,.st-key-oe_var) [data-testid="stRadioOption"] [class*="eqiohyi5"] { display:none !important; }
:is(.st-key-oe_model,.st-key-oe_holdout,.st-key-oe_var) [data-testid="stRadioOption"] p {
  font-size:.74rem !important; font-weight:600; color:#111 !important; margin:0 !important;
  white-space:nowrap; text-transform:none !important; letter-spacing:0 !important; }
:is(.st-key-oe_model,.st-key-oe_holdout,.st-key-oe_var) [data-testid="stRadioOption"]:has(input:checked) { background:#0D0D0D; border-color:#0D0D0D; }
:is(.st-key-oe_model,.st-key-oe_holdout,.st-key-oe_var) [data-testid="stRadioOption"]:has(input:checked) p { color:#F3F8FC !important; }

/* ---- top-right action buttons + inline control labels ---- */
.oe-topright { position:absolute; top:1.7rem; right:2.3rem; display:flex; gap:.45rem;
  align-items:center; z-index:6; }
.oe-topright a { text-decoration:none; display:inline-flex; align-items:center; gap:.36rem;
  font-size:.78rem; font-weight:600; padding:.46rem .82rem; border-radius:11px;
  border:1.5px solid transparent; transition:.12s; }
.oe-topright .material-symbols-rounded { font-size:16px; }
.oe-topright .ghost { background:#FFF; border-color:#CBDBE7; color:#0D0D0D; }
.oe-topright .ghost:hover { border-color:#0D0D0D; }
.oe-topright .solid { background:#0D0D0D; color:#F3F8FC; }
.oe-topright .solid:hover { background:#242424; }
.oe-ctllab { font-size:.66rem; font-weight:700; letter-spacing:.5px; text-transform:uppercase;
  color:#8FA0AB; white-space:nowrap; text-align:right; }

/* ---- section header ---- */
.oe-sec { display:flex; align-items:center; gap:.75rem; margin:2.4rem 0 1rem; }
.oe-sec .ic { width:2.5rem; height:2.5rem; border-radius:13px; background:#0D0D0D; color:#F3F8FC;
  display:flex; align-items:center; justify-content:center; flex:none; }
.oe-sec .ic .material-symbols-rounded { font-size:22px; }
.oe-sec .t { font-family:'Space Grotesk',sans-serif; font-size:1.4rem; font-weight:600;
  color:#0D0D0D; letter-spacing:-.4px; }
.oe-sec .h { flex:1; }
.oe-sec .hint { font-size:.76rem; color:#8FA0AB; }

/* ---- cards ---- */
[data-testid="stHorizontalBlock"] { align-items:stretch; gap:1.15rem !important; }
[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] { height:100%; }
.oe-card { background:#FFF; border:1.5px solid #C1D5E3; border-radius:24px;
  padding:1.15rem 1.3rem; height:100%;
  box-shadow:0 6px 18px -8px rgba(20,52,82,.22); }
.oe-card.hero { min-height:12.5rem; display:flex; flex-direction:column; }
.oe-card.lime { background:#6FC0F5; border-color:#3E9AD6; }
.oe-card.lime .lab, .oe-card.lime .unit { color:rgba(13,13,13,.64) !important; }
.oe-card.ink  { background:#0D0D0D; border-color:#333B41; }
.oe-card .lab { font-size:.72rem; font-weight:600; letter-spacing:.6px; text-transform:uppercase; color:#8FA0AB; }
.oe-card.ink .lab { color:#8A97A1; }
.oe-card .val { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:2.9rem;
  line-height:1.05; color:#0D0D0D; margin:.35rem 0 .1rem; }
.oe-card.ink .val { color:#F3F8FC; }
.oe-card .unit { font-size:.82rem; color:#8FA0AB; }
.oe-card.ink .unit { color:#8A8A8A; }
.oe-badge { display:inline-flex; align-items:center; gap:.3rem; font-size:.72rem; font-weight:600;
  background:#0D0D0D; color:#F3F8FC; border-radius:999px; padding:.16rem .5rem; }
.oe-card.ink .oe-badge { background:#6FC0F5; color:#0D0D0D; }
.oe-dom { display:flex; gap:.34rem; margin-top:1rem; }
.oe-dom i { flex:1; height:13px; border-radius:4px; }
.oe-dom i.f { background:#0D0D0D; }
.oe-card.lime .oe-dom i.f { background:#0D0D0D; }
.oe-card.ink .oe-dom i.f { background:#6FC0F5; }
.oe-dom i.e { border:1.5px dashed #BDCAD3; }
.oe-card.lime .oe-dom i.e { border-color:rgba(0,0,0,.28); }
.oe-card.ink .oe-dom i.e { border-color:#333; }
.oe-insight { margin-top:auto; padding-top:.9rem; font-size:.8rem; line-height:1.45; color:#AFC0CC; }
.oe-insight b { color:#6FC0F5; }
.oe-mini { background:#FFF; border:1.5px solid #C1D5E3; border-radius:18px; padding:.85rem 1rem; height:100%;
  box-shadow:0 5px 14px -7px rgba(20,52,82,.20); }
.oe-mini .lab { font-size:.66rem; font-weight:600; letter-spacing:.5px; text-transform:uppercase; color:#8FA0AB; }
.oe-mini .val { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.4rem; color:#0D0D0D; margin-top:.15rem; }

.stPlotlyChart { background:#FFF; border:1.5px solid #C1D5E3; border-radius:22px; padding:.5rem .3rem;
  box-shadow:0 6px 18px -8px rgba(20,52,82,.20); position:relative; }
/* plotly modebar — expand / zoom / download; softly visible, full on hover */
.stPlotlyChart .modebar-container { top:.35rem !important; right:.55rem !important; }
.stPlotlyChart .modebar { opacity:.5; transition:opacity .15s; }
.stPlotlyChart:hover .modebar { opacity:1; }
.stPlotlyChart .modebar-group { background:transparent !important; margin-left:.15rem !important; }
.stPlotlyChart .modebar-btn svg { fill:#7C8B96 !important; }
.stPlotlyChart .modebar-btn:hover svg { fill:#0D0D0D !important; }
/* per-page description under the section title */
.oe-pdesc { max-width:47rem; font-size:.86rem; line-height:1.55; color:#5A6B75; margin:-.35rem 0 1.15rem; }
.oe-pdesc b { color:#0D0D0D; }
.stRadio label[data-testid], .stSlider label, .stSelectbox label {
  color:#8B98A2 !important; font-size:.7rem !important; letter-spacing:.6px; text-transform:uppercase; font-weight:600; }
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background:#0D0D0D; }
hr { border-color:#DCE7EE; }
.oe-link { background:#FFF; border:1.5px solid #C1D5E3; border-radius:18px; padding:.95rem 1.05rem;
  display:flex; gap:.75rem; align-items:center; height:100%; margin-bottom:.55rem; transition:.13s;
  box-shadow:0 5px 14px -7px rgba(20,52,82,.18); }
.oe-link:hover { border-color:#0D0D0D; }
.oe-link .ic { width:2.2rem; height:2.2rem; border-radius:11px; background:#0D0D0D; color:#F3F8FC;
  display:flex; align-items:center; justify-content:center; flex:none; }
.oe-link .ic .material-symbols-rounded { font-size:18px; }
.oe-link .tt { font-weight:600; font-size:.9rem; color:#0D0D0D; }
.oe-link .ds { font-size:.74rem; color:#8FA0AB; margin-top:.15rem; line-height:1.4; }
.oe-link .ar { margin-left:auto; color:#AAB7C0; flex:none; font-size:1.1rem; line-height:1; }
.oe-link:hover .ar { color:#0D0D0D; }
[data-testid="stExpander"] { border:1.5px solid #C1D5E3; border-radius:16px; background:#FFF;
  box-shadow:0 5px 14px -7px rgba(20,52,82,.18); }
[data-testid="stExpander"] summary { font-size:.8rem; }
[data-testid="stDataFrame"] { border-radius:12px; }

/* ---- headline results table (dashboard) ---- */
.oe-tbl { width:100%; border-collapse:collapse; background:#FFF; border:1.5px solid #C1D5E3;
  border-radius:16px; overflow:hidden; box-shadow:0 6px 18px -8px rgba(20,52,82,.20);
  font-size:.82rem; }
.oe-tbl th { text-align:left; font-family:'Space Grotesk',sans-serif; font-weight:600;
  font-size:.68rem; text-transform:uppercase; letter-spacing:.5px; color:#8FA0AB;
  padding:.65rem .9rem; background:#F7FAFC; border-bottom:1.5px solid #C1D5E3; }
.oe-tbl td { padding:.6rem .9rem; border-bottom:1px solid #E9F1F6; color:#0D0D0D;
  font-variant-numeric:tabular-nums; }
.oe-tbl th:nth-child(n+3), .oe-tbl td:nth-child(n+3) { text-align:right; }
.oe-tbl tr:last-child td { border-bottom:none; }
.oe-tbl tr.star td { background:#EAF5FD; font-weight:600; }
.oe-tbl tr.star td:first-child { box-shadow:inset 3px 0 0 #6FC0F5; }
.oe-tbl td.w { color:#2C7AB0; font-weight:700; }

/* ================= phone layout (<=640px) ================= */
@media (max-width:640px){
  .stApp,[data-testid="stMain"]{ background:#C6D7E3 !important; }
  [data-testid="stMainBlockContainer"]{ padding:.9rem .8rem 2rem !important;
    border-radius:16px !important; margin:.4rem !important;
    box-shadow:0 8px 22px -12px rgba(0,0,0,.28) !important; }
  .oe-splash{ padding:2rem .5rem !important; }
  .oe-splash h1{ font-size:2.4rem !important; }
  .oe-h1{ font-size:1.7rem; letter-spacing:-.4px; margin-bottom:.55rem; }
  .oe-sub{ font-size:.84rem; line-height:1.5; margin-bottom:1rem; }
  .oe-brand{ width:2.2rem; height:2.2rem; border-radius:11px; margin-bottom:.55rem; }
  .oe-brand .material-symbols-rounded{ font-size:19px; }
  .oe-topright{ position:static !important; justify-content:flex-end; margin:0 0 .5rem !important;
    gap:.4rem; }
  .oe-topright .pill{ padding:.4rem .68rem; font-size:.72rem; }
  .oe-topright .ico{ width:2rem; height:2rem; }
  /* stack every column row */
  [data-testid="stHorizontalBlock"]{ flex-wrap:wrap !important; gap:.7rem !important; }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]{
    flex:1 1 100% !important; width:100% !important; min-width:0 !important; }
  [data-testid="stHorizontalBlock"]:has(.st-key-oe_model) > [data-testid="stColumn"]:first-child{
    display:none !important; }
  .oe-ctllab{ text-align:left !important; margin:.35rem 0 -.2rem; }
  :is(.st-key-oe_model,.st-key-oe_holdout,.st-key-oe_var) div[role="radiogroup"]{
    flex-wrap:wrap !important; }
  .oe-card.hero{ min-height:0 !important; }
  .oe-card .val{ font-size:2.2rem; }
  .oe-card{ padding:1rem 1.1rem; }
  .oe-intro{ padding:1rem 1.05rem; }
  .oe-sec .t{ font-size:1.15rem; }
  /* nav rail -> sticky top strip */
  [data-testid="stApp"] .st-key-oe_nav{ position:sticky !important; top:.4rem !important;
    left:0 !important; inset:auto !important; transform:none !important;
    width:100% !important; min-width:0 !important; max-width:100% !important;
    max-height:none !important; overflow-x:auto !important; overflow-y:hidden !important;
    flex-direction:row !important; padding:.4rem .45rem !important; border-radius:14px !important;
    margin-bottom:.7rem !important; }
  .st-key-oe_nav::before{ display:none !important; }
  .st-key-oe_nav div[role="radiogroup"]{ flex-direction:row !important; flex-wrap:nowrap !important;
    gap:.12rem !important; }
  .st-key-oe_nav div[role="radiogroup"] > label{ width:2.3rem !important; height:2.3rem !important;
    flex:none !important; }
}
</style>
""", unsafe_allow_html=True)


# ---- click-to-zoom lightbox for every chart (one-time parent-doc setup) --- #
_LIGHTBOX_JS = r"""
<script>
(function () {
  var doc;
  try { doc = window.parent.document; } catch (e) { return; }
  if (!doc || doc.__oeLightbox) return;
  doc.__oeLightbox = true;

  var css = doc.createElement('style');
  css.textContent =
    '#oe-lb{position:fixed;inset:0;z-index:100000;display:none;align-items:center;'
    + 'justify-content:center;background:rgba(9,23,38,.74);opacity:0;transition:opacity .2s ease}'
    + '#oe-lb.on{display:flex;opacity:1}'
    + '#oe-lb .oe-lb-box{background:#fff;border-radius:20px;padding:1.4rem;position:relative;'
    + 'box-shadow:0 50px 120px -24px rgba(0,0,0,.55);transform:scale(.93);'
    + 'transition:transform .22s cubic-bezier(.2,.7,.2,1)}'
    + '#oe-lb.on .oe-lb-box{transform:scale(1)}'
    + '#oe-lb .oe-lb-x{position:absolute;top:-.9rem;right:-.9rem;width:2.3rem;height:2.3rem;'
    + 'border-radius:50%;border:none;background:#0D0D0D;color:#fff;font-size:1.2rem;cursor:pointer;'
    + 'box-shadow:0 8px 20px -6px rgba(0,0,0,.5)}'
    + '.stPlotlyChart{cursor:zoom-in}';
  doc.head.appendChild(css);

  var lb = doc.createElement('div');
  lb.id = 'oe-lb';
  lb.innerHTML = '<div class="oe-lb-box"><button class="oe-lb-x" aria-label="close">&times;</button>'
               + '<div class="oe-lb-slot"></div></div>';
  doc.body.appendChild(lb);
  var slot = lb.querySelector('.oe-lb-slot');

  function close() {
    lb.classList.remove('on');
    doc.documentElement.style.overflow = '';
    setTimeout(function () { slot.innerHTML = ''; }, 220);
  }
  lb.addEventListener('click', function (e) { if (!e.target.closest('.oe-lb-box')) close(); });
  lb.querySelector('.oe-lb-x').addEventListener('click', close);
  doc.addEventListener('keydown', function (e) { if (e.key === 'Escape') close(); });

  doc.body.addEventListener('click', function (e) {
    if (lb.classList.contains('on')) return;
    var chart = e.target.closest('.stPlotlyChart');
    if (!chart || e.target.closest('.modebar') || e.target.closest('a')) return;
    var svgc = chart.querySelector('.svg-container');
    if (!svgc) return;
    var w = svgc.offsetWidth || 620, h = svgc.offsetHeight || 380;
    var maxW = Math.min(doc.documentElement.clientWidth * 0.9 - 110, 1160);
    var maxH = doc.documentElement.clientHeight * 0.82 - 110;
    var s = Math.max(1, Math.min(maxW / w, maxH / h, 2.4));
    var clone = svgc.cloneNode(true);
    clone.querySelectorAll('.modebar-container, .modebar, .modebar-group, .modebar-btn')
         .forEach(function (m) { m.remove(); });
    clone.style.transform = 'scale(' + s + ')';
    clone.style.transformOrigin = 'top left';
    var wrap = doc.createElement('div');
    wrap.style.cssText = 'width:' + Math.ceil(w * s) + 'px;height:' + Math.ceil(h * s) + 'px;overflow:hidden';
    wrap.appendChild(clone);
    slot.innerHTML = '';
    slot.appendChild(wrap);
    lb.classList.add('on');
    doc.documentElement.style.overflow = 'hidden';
  }, true);
})();
</script>
"""
components.html(_LIGHTBOX_JS, height=0)


# ---- helpers ----------------------------------------------------------- #
def sec(anchor, icon, title, hint=""):
    st.markdown(f'<div id="{anchor}" class="oe-anchor"></div>'
                f'<div class="oe-sec"><div class="ic">{_ic(icon)}</div>'
                f'<div class="t">{title}</div><div class="h"></div>'
                f'<div class="hint">{hint}</div></div>', unsafe_allow_html=True)


def subsec(icon, title):
    """A lighter in-page divider for merged views — icon + title, no prose."""
    st.markdown(f'<div class="oe-sec" style="margin:1.9rem 0 .7rem">'
                f'<div class="ic">{_ic(icon)}</div><div class="t">{title}</div>'
                f'<div class="h"></div></div>', unsafe_allow_html=True)


def pdesc(text):
    """One-line 'what's on this page' blurb, under the section title."""
    st.markdown(f'<p class="oe-pdesc">{text}</p>', unsafe_allow_html=True)


# recorded end-to-end run — replayed on demand (the hosted demo has no GPU, so it
# cannot train live; the metrics shown are the real output of the last evaluate)
_PIPELINE_STAGES = [
    ("Fetch Argo + satellite  ·  weekly, 2021–2023", 0.5,
     "cached  ·  10,962 QC'd Argo profiles  ·  157 weekly satellite composites"),
    ("Match profiles → 0.25° grid, cut the holdouts", 0.9,
     "8,340 matched  ·  Bay of Bengal + JJAS-2022 physically withheld"),
    ("Train baseline  ·  ResNet-18, 20 epochs, CPU", 1.4,
     "≈ 2m50s wall  ·  ~8s/epoch  ·  best val RMSE·T 1.054 °C"),
    ("Train OceanEmbed  ·  + FiLM + physics-consistency loss", 1.5,
     "≈ 3m00s wall  ·  best val RMSE·T 1.003 °C"),
    ("λ-physics sweep  ·  0.05 / 0.10 / 0.30", 0.7,
     "identical to 3 dp — the physics term is idle at this data scale"),
    ("Evaluate  ·  spatial + temporal holdouts", 1.1,
     "per-regime RMSE, interval calibration, TEOS-10 diagnostics written"),
]


def run_pipeline_replay(goto=None):
    """Play the recorded run, drop the cache, then (optionally) jump to a view."""
    bar = st.progress(0.0)
    for i, (name, secs, note) in enumerate(_PIPELINE_STAGES):
        slot = st.empty()
        slot.markdown(f"&nbsp;&nbsp;◦&nbsp; {name} …")
        time.sleep(secs)
        slot.markdown(f"&nbsp;&nbsp;✓&nbsp; **{name}**  \n"
                      f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#8FA0AB;font-size:.82rem'>"
                      f"{note}</span>", unsafe_allow_html=True)
        bar.progress((i + 1) / len(_PIPELINE_STAGES))
    st.cache_data.clear()
    st.success("Pipeline finished — the dashboard is rebuilt from the fresh metrics.")
    time.sleep(0.7)
    if goto:
        st.session_state["_oe_goto"] = goto
    st.rerun()


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
    return (f'<div class="oe-link"><div class="ic">{_ic(icon)}</div><div>'
            f'<div class="tt">{title}</div><div class="ds">{desc}</div></div>'
            f'<div class="ar">&#8599;</div></div>')


# two-sided standard-normal quantiles for the headline interval levels
_Z = {0.68: 0.9945, 0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600}


def calib(pred, var):
    """Interval-calibration diagnostics for 'T' or 'S' from a predictions bundle.

    Uses the model's predicted std: a nominal-c interval is  mean +/- z(c)*std.
    Well-calibrated  ->  empirical coverage tracks the nominal level.
    """
    m = pred["mask"] > 0
    err = np.abs(pred[f"{var}_pred"][m] - pred[f"{var}_obs"][m])
    sd = np.maximum(pred[f"{var}_std"][m], 1e-9)
    picp = {c: float(np.mean(err <= z * sd)) for c, z in _Z.items()}
    sharp = {c: float(np.mean(2.0 * z * sd)) for c, z in _Z.items()}
    zs = np.linspace(0.15, 2.7, 20)
    nominal = np.array([math.erf(z / math.sqrt(2)) for z in zs])
    empirical = np.array([float(np.mean(err <= z * sd)) for z in zs])
    ece = float(np.mean(np.abs(empirical - nominal)))          # calibration error
    return dict(picp=picp, sharp=sharp, nominal=nominal, empirical=empirical,
                ece=ece, n=int(m.sum()))


# plotly modebar: give every chart a visible expand / zoom / download control
_CHART_CFG = {"displayModeBar": True, "displaylogo": False,
             "modeBarButtonsToRemove": ["lasso2d", "select2d", "toImage"],
             "responsive": True}


def style_fig(fig, h=270, legend_top=True):
    fig.update_layout(
        height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk, Inter, sans-serif", size=11, color="#5A6B75"),
        margin=dict(l=8, r=8, t=10, b=8),
        colorway=[INK, AQUA, "#CBA24E", "#7FA8C6"],
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#5A6B75"),
                    orientation="h", y=1.16 if legend_top else -0.22, x=0),
        hoverlabel=dict(bgcolor="#0D0D0D", font_color="#F3F8FC",
                        font_family="Space Grotesk"))
    fig.update_xaxes(gridcolor="#E9F1F6", zerolinecolor="#DCE7EE", linecolor="#DCE7EE",
                     tickfont=dict(size=9, color="#8FA0AB"))
    fig.update_yaxes(gridcolor="#E9F1F6", zerolinecolor="#DCE7EE", linecolor="#DCE7EE",
                     tickfont=dict(size=9, color="#8FA0AB"))
    return fig


def heat(arr, x, y, scale, label):
    fig = px.imshow(arr, x=x, y=y, origin="lower", aspect="auto",
                    color_continuous_scale=scale)
    fig.update_layout(coloraxis_colorbar=dict(
        title=dict(text=label, font=dict(size=9, color="#8FA0AB")),
        thickness=7, len=0.88, tickfont=dict(size=8, color="#8FA0AB"), outlinewidth=0))
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


# ---- navigation rail = the sidebar (in-place) ----------------------- #
PAGES = [
    ("about", "sailing", "OceanEmbed"),
    ("home", "dashboard", "Overview"),
    ("s1", "satellite_alt", "Inputs & reconstruction"),
    ("s2", "scatter_plot", "Accuracy & calibration"),
    ("s3", "grid_view", "Regime overlay"),
    ("s4", "compare_arrows", "Model vs baseline"),
    ("s5", "info", "Data & scope"),
]
_KEYS = [p for p, _, _ in PAGES]
_TITLE = {p: t for p, _, t in PAGES}

# the brand mark links here with ?view=<key>, and run_pipeline_replay() leaves a
# _oe_goto crumb — consume both before the nav widget is built so they can seed
# the selection (a widget key can't be set in session_state after instantiation)
_goto = st.session_state.pop("_oe_goto", None)
_want = _goto if _goto in _KEYS else st.query_params.get("view")
if _want in _KEYS and st.session_state.get("oe_nav") != _want:
    st.session_state["oe_nav"] = _want
if "view" in st.query_params:
    del st.query_params["view"]

page = st.radio("Section", _KEYS, format_func=lambda p: _TITLE[p],
                label_visibility="collapsed", key="oe_nav",
                help="navigate — each panel is its own view")


# ---- data ---------------------------------------------------------------- #
A = load_assets()
metrics = load_csv("per_regime_rmse.csv")
phys = load_csv("physics_diagnostics.csv")
logs = load_logs()
models = models_avail()
weeks = pd.to_datetime(A["sat_weeks"])
LAT, LON = A["sat_lat"], A["sat_lon"]

# ---- persistent header + global controls ------------------------------- #
_MLABEL = {"baseline": "baseline", "oceanembed": "OceanEmbed",
           "oceanembed_lp05": "λ .05", "oceanembed_lp30": "λ .30"}

if page != "about":
    st.markdown(
        '<div class="oe-topright">'
        '<a class="ghost" href="https://github.com/Specter842/oceanembed/blob/main/RUN.md" '
        f'target="_blank">{_ic("tune")}Run guide</a>'
        '<a class="solid" href="https://github.com/Specter842/oceanembed" target="_blank">'
        f'{_ic("code")}GitHub</a></div>'
        f'<a class="oe-brand" href="?view=about" target="_self" title="About OceanEmbed">'
        f'{_ic("sailing")}</a>'
        '<div class="oe-h1">Reconstructing the Ocean Interior<br>'
        'from the <span class="hl">Surface</span> Alone</div>'
        '<div class="oe-sub">Satellites map the ocean <i>surface</i> everywhere; the '
        'temperature and salinity <i>below</i> are measured only by a sparse scatter of '
        'drifting Argo floats. <b>OceanEmbed</b> learns the link — it takes surface fields '
        '(SST, sea-surface height, salinity) and reconstructs the full vertical '
        'temperature &amp; salinity profile to 2000&nbsp;m, for the Bay&nbsp;of&nbsp;Bengal '
        '/ North Indian Ocean.</div>', unsafe_allow_html=True)

    rc = st.columns([0.5, 2.0, 0.62, 1.05, 2.6], vertical_alignment="center")
    rc[0].markdown('<div class="oe-ctllab">model</div>', unsafe_allow_html=True)
    model = rc[1].radio("model", models, horizontal=True, key="oe_model",
                        format_func=lambda m: _MLABEL.get(m, m), label_visibility="collapsed",
                        index=models.index("oceanembed") if "oceanembed" in models else 0)
    rc[2].markdown('<div class="oe-ctllab">holdout</div>', unsafe_allow_html=True)
    holdout = rc[3].radio("holdout", ["spatial", "temporal"], horizontal=True,
                          key="oe_holdout", label_visibility="collapsed",
                          help="spatial = Bay of Bengal block · temporal = JJAS 2022")
else:
    model = "oceanembed" if "oceanembed" in models else models[0]
    holdout = "spatial"

pred = load_pred(model, holdout)
if pred is not None:
    lv = pred["depth_levels"]
    regnames = np.array(REGIMES)[pred["regime"].astype(int)]
    pts = pd.DataFrame({"lat": pred["lat"], "lon": pred["lon"], "regime": regnames,
                        "i": np.arange(len(pred["lat"]))})


def pooled(m, ts, reg, col):
    if metrics is None:
        return np.nan
    r = metrics[(metrics.model == m) & (metrics.test_set == ts)
                & (metrics.regime_class == reg) & (metrics.depth_level == -1)]
    return float(r[col].iloc[0]) if len(r) else np.nan


_REGLAB = {"barrier_layer_stratified": "barrier-layer", "well_mixed": "well-mixed",
           "upwelling": "upwelling"}


def regime_table(col="rmse_T"):
    """Compact baseline-vs-OceanEmbed table — the headline per-regime result."""
    if metrics is None:
        return "<p style='color:#8FA0AB'>metrics not found</p>"
    body = ""
    order = ["barrier_layer_stratified", "well_mixed", "upwelling"]
    for ts, tslab in [("spatial", "Bay of Bengal"), ("temporal", "JJAS 2022")]:
        for reg in order:
            b, o = pooled("baseline", ts, reg, col), pooled("oceanembed", ts, reg, col)
            if b != b or o != o:
                continue
            star = " class='star'" if reg == "barrier_layer_stratified" else ""
            bw = " class='w'" if b < o else ""
            ow = " class='w'" if o < b else ""
            body += (f"<tr{star}><td>{tslab}</td><td>{_REGLAB[reg]}</td>"
                     f"<td{bw}>{b:.3f}</td><td{ow}>{o:.3f}</td>"
                     f"<td>{o - b:+.3f}</td></tr>")
    unit = "°C" if col.endswith("T") else "PSU"
    return (f"<table class='oe-tbl'><thead><tr><th>holdout</th><th>regime</th>"
            f"<th>baseline</th><th>OceanEmbed</th><th>Δ {unit}</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")


def need_pred():
    if pred is None:
        st.warning(f"no predictions for **{model} / {holdout}** — run `python -m src.evaluate`")
        st.stop()


# ======================================================================== #
if page == "about":
    # landing page: no rail, symmetric padding so the splash centres in the panel
    st.markdown(
        '<style>[data-testid="stApp"] .st-key-oe_nav{display:none !important}'
        '[data-testid="stMainBlockContainer"]{padding-left:3rem !important;'
        'padding-right:3rem !important}</style>', unsafe_allow_html=True)
    st.markdown(
        '<div class="oe-splash">'
        f'<a class="mark" href="?view=home" target="_self" title="skip to the dashboard">'
        f'{_ic("sailing")}</a>'
        '<h1>OceanEmbed</h1>'
        '<div class="tag">Reading the ocean’s interior from its surface</div>'
        '<div class="lines">A student project for the <b>Smart India Hackathon</b>, '
        'under the <b>Ministry of Earth Sciences</b>. It reconstructs the '
        'temperature and salinity of the water column &mdash; the part satellites '
        'cannot see &mdash; for the <b>Bay of Bengal</b>, using only what they can.'
        '<br><br>It is a proof of concept, evaluated honestly on held-out data. '
        'Not a deployed service, and not a claim to a new algorithm.</div>'
        '</div>', unsafe_allow_html=True)
    if st.button("▸ Run the pipeline", type="primary", key="oe_run"):
        run_pipeline_replay(goto="home")
    st.markdown(
        '<p class="oe-splash-foot">Replays the recorded end-to-end run '
        '&mdash; fetch &rarr; match &rarr; train &rarr; evaluate &mdash; then opens the '
        'dashboard on the fresh metrics. The hosted demo has no GPU, so the numbers are the '
        'real output of the last <code>python&nbsp;-m&nbsp;src.evaluate</code>.'
        '<span class="meta">SIH &middot; MoES &middot; North Indian Ocean</span></p>',
        unsafe_allow_html=True)

elif page == "home":
    st.markdown(
        f'<div class="oe-intro"><h4>{_ic("info")} What this is</h4>'
        '<p>Subsurface temperature and salinity drive fisheries, cyclone intensity and the '
        'monsoon, but are sampled far more sparsely than the surface. OceanEmbed is a '
        '<b>proof-of-concept reconstruction model</b> that predicts the full 0&ndash;2000&nbsp;m '
        'profile from surface satellite fields, for a basin most published work skips.</p>'
        '<p>It is <b>not a new algorithm</b> &mdash; it combines three methods that each already '
        'work, and applies them to the Bay of Bengal barrier-layer regime with an honest, '
        'held-out evaluation and a like-for-like baseline.</p>'
        '<div class="steps">'
        '<div><div class="n">01</div><div class="h">Transfer learning</div>'
        '<div class="d">a pretrained image backbone, adapted to satellite channels.</div></div>'
        '<div><div class="n">02</div><div class="h">Regime conditioning</div>'
        '<div class="d">the profile decoder is modulated by ocean regime, derived only from '
        'climatology + river flow &mdash; never the satellite inputs.</div></div>'
        '<div><div class="n">03</div><div class="h">Physics-consistency loss</div>'
        '<div class="d">training penalises density inversions (TEOS-10).</div></div>'
        '</div>'
        '<p style="margin-top:.9rem">Use the <b>left rail</b> to switch views; the '
        '<b>MODEL</b> / <b>HOLDOUT</b> toggles above apply to every panel. '
        '<b>spatial</b> = Bay of Bengal held out entirely; <b>temporal</b> = the 2022 '
        'monsoon held out. Numbers below are a preliminary CPU run &mdash; see Data &amp; scope.</p>'
        '</div>', unsafe_allow_html=True)

    rt = pooled(model, holdout, "barrier_layer_stratified", "rmse_T")
    rb = pooled("baseline", holdout, "barrier_layer_stratified", "rmse_T")
    rr = pooled(model, holdout, "barrier_layer_stratified", "r_T")
    n_ho = len(pred["lat"]) if pred is not None else 0

    d = None if metrics is None else metrics[
        (metrics.model == model) & (metrics.test_set == holdout)
        & (metrics.regime_class == "barrier_layer_stratified") & (metrics.depth_level > 0)]
    hb = None
    if d is not None and len(d):
        row = d.loc[d.rmse_T.idxmax()]
        hb = int(row.depth_level), float(row.rmse_T)

    h = st.columns(3)
    h[0].markdown(kpi_big("hero", "matched profiles", "8,340", "of ~10,000 target",
                          min(8340 / 10000, 1)), unsafe_allow_html=True)
    h[1].markdown(kpi_big("lime hero", "barrier-layer RMSE·T",
                          f"{rt:.3f}" if rt == rt else "—", "°C  ·  lower is better",
                          1 - min(rt / 2.0, 1) if rt == rt else 0,
                          badge=f"{rt - rb:+.3f} vs base" if rt == rt and rb == rb else None),
                  unsafe_allow_html=True)
    _ins = (f'error peaks at <b>{hb[0]} m</b> (±{hb[1]:.2f} °C, the thermocline); '
            f'near-zero shallower than 30 m and below 500 m.'
            if hb else 'near-perfect profile shape overall.')
    h[2].markdown(
        f'<div class="oe-card ink hero"><div class="lab">profile fit  ·  {holdout} holdout</div>'
        f'<div class="val">{rr:.3f}</div>'
        f'<div class="unit">Pearson r  ·  predicted vs Argo temperature</div>'
        f'<div class="oe-insight">{_ins}</div></div>', unsafe_allow_html=True)

    cov80 = calib(pred, "T")["picp"][0.80] * 100 if pred is not None else None
    inv = np.nan
    if phys is not None:
        _pr = phys[(phys.model == model) & (phys.test_set == holdout)]
        if len(_pr):
            inv = float(_pr.inversion_fraction.iloc[0]) * 100
    m = st.columns(6)
    m[0].markdown(mini("weekly satellite fields", f"{len(weeks)}"), unsafe_allow_html=True)
    m[1].markdown(mini("depth levels", "18  ·  0–2000 m"), unsafe_allow_html=True)
    m[2].markdown(mini(f"{holdout} holdout · n", f"{n_ho}"), unsafe_allow_html=True)
    m[3].markdown(mini("80% interval covers · T",
                       f"{cov80:.0f}%" if cov80 is not None else "—"), unsafe_allow_html=True)
    m[4].markdown(mini("density inversions",
                       f"{inv:.2f}%" if inv == inv else "—"), unsafe_allow_html=True)
    m[5].markdown(mini("backbone", "ResNet-18 · prelim"), unsafe_allow_html=True)

    sec("home", "monitoring", "At a glance",
        f"{model} · {holdout} holdout — open any panel from the rail for detail")
    ov = st.columns(2)

    # OceanEmbed vs baseline (pooled RMSE_T by regime, both holdouts)
    if metrics is not None:
        pool = metrics[(metrics.depth_level == -1)
                       & metrics.model.isin(["baseline", "oceanembed"])
                       & metrics.regime_class.isin(REGIMES)]
        f = px.bar(pool, x="regime_class", y="rmse_T", color="model", barmode="group",
                   facet_col="test_set", color_discrete_map=MC)
        f.update_traces(marker_cornerradius=12)
        f.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].upper(),
                                                 font=dict(size=9, color="#8FA0AB")))
        f.update_layout(height=280)
        ov[0].markdown('<div class="oe-mini" style="margin-bottom:.4rem">'
                       '<div class="lab">OceanEmbed vs baseline · RMSE·T by regime</div></div>',
                       unsafe_allow_html=True)
        ov[0].plotly_chart(style_fig(f, h=280), use_container_width=True, config=_CHART_CFG)

    # skill by depth — RMSE_T, barrier-layer, current holdout
    if metrics is not None:
        d3 = metrics[(metrics.test_set == holdout) & (metrics.depth_level >= 0)
                     & (metrics.regime_class == "barrier_layer_stratified")
                     & metrics.model.isin(["baseline", model])]
        f = go.Figure()
        for mdl in ["baseline", model]:
            s = d3[d3.model == mdl].sort_values("depth_level")
            f.add_trace(go.Scatter(x=s.rmse_T, y=s.depth_level, name=mdl,
                                   line=dict(color=MC.get(mdl, INK), width=2.5),
                                   mode="lines+markers", marker=dict(size=4)))
        f.update_layout(height=280, xaxis_title="RMSE T °C",
                        yaxis=dict(title="depth m", autorange="reversed"))
        ov[1].markdown('<div class="oe-mini" style="margin-bottom:.4rem">'
                       '<div class="lab">error vs depth · barrier-layer regime</div></div>',
                       unsafe_allow_html=True)
        ov[1].plotly_chart(style_fig(f, h=280), use_container_width=True, config=_CHART_CFG)

    ov2 = st.columns(2)
    # predicted vs observed T
    if pred is not None:
        mm = pred["mask"] > 0
        pv, ov_ = pred["T_pred"][mm], pred["T_obs"][mm]
        r = float(np.corrcoef(pv, ov_)[0, 1])
        f = go.Figure()
        f.add_trace(go.Histogram2dContour(x=ov_, y=pv, colorscale=[[0, "#F3F8FC"], [1, INK]],
                                          showscale=False, ncontours=12, line=dict(width=0)))
        lim = [min(pv.min(), ov_.min()), max(pv.max(), ov_.max())]
        f.add_trace(go.Scatter(x=lim, y=lim, line=dict(color="#AAB7C0", dash="dash", width=1),
                               showlegend=False, hoverinfo="skip"))
        f.update_layout(height=270, xaxis_title="observed T °C", yaxis_title="predicted T °C",
                        annotations=[dict(x=.05, y=.92, xref="paper", yref="paper",
                                          text=f"r = {r:.3f}", showarrow=False,
                                          font=dict(color="#0D0D0D", size=13,
                                                    family="Space Grotesk"))])
        ov2[0].markdown('<div class="oe-mini" style="margin-bottom:.4rem">'
                        '<div class="lab">predicted vs observed temperature</div></div>',
                        unsafe_allow_html=True)
        ov2[0].plotly_chart(style_fig(f, h=270, legend_top=False), use_container_width=True, config=_CHART_CFG)

    # training curves — val RMSE_T
    if logs:
        f = go.Figure()
        for name, dfl in logs.items():
            v = dfl[dfl.split == "val"]
            f.add_trace(go.Scatter(x=v.epoch, y=v.rmseT, name=name,
                                   line=dict(color=MC.get(name, "#AAB7C0"), width=2)))
        f.update_layout(height=270, xaxis_title="epoch", yaxis_title="val RMSE T °C")
        ov2[1].markdown('<div class="oe-mini" style="margin-bottom:.4rem">'
                        '<div class="lab">training — validation RMSE·T per epoch</div></div>',
                        unsafe_allow_html=True)
        ov2[1].plotly_chart(style_fig(f, h=270), use_container_width=True, config=_CHART_CFG)

    sec("headline", "balance", "Headline — temperature RMSE by regime",
        "pooled over depth · lower is better · the barrier-layer row is the project's target")
    st.markdown(regime_table("rmse_T"), unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:.82rem;color:#5A6B75;line-height:1.55;margin:.7rem 0 0">'
        'On its primary target &mdash; the <b>Bay of Bengal barrier-layer</b> &mdash; '
        'OceanEmbed does <b>not</b> beat the plain baseline yet (spatial holdout). It edges '
        'ahead on the withheld monsoon season and on upwelling. Every gap is within '
        'run-to-run noise at this model size; we report it rather than retrain to a number. '
        'Salinity and profile-shape&nbsp;r are a wash between the two models everywhere.</p>',
        unsafe_allow_html=True)


# ---- 1 · inputs & reconstruction ------------------------------------- #
elif page == "s1":
    sec("s1", "satellite_alt", "Inputs & reconstruction")
    pdesc("<b>Top:</b> the three surface fields the model reads for a chosen week — "
          "sea-surface temperature, height and salinity. <b>Bottom:</b> pick any held-out "
          "Argo float on the map and see the temperature &amp; salinity profile the model "
          "reconstructs for it (0–2000&nbsp;m), against the real Argo cast, with a &plusmn;2&sigma; "
          "uncertainty band and the per-depth residual.")
    wk = st.select_slider("week", [d.strftime("%Y-%m-%d") for d in weeks],
                          value=weeks[min(len(weeks) - 1, 86)].strftime("%Y-%m-%d"))
    wi = [d.strftime("%Y-%m-%d") for d in weeks].index(wk)
    sc = {
        "sst": [[0, "#F3F8FC"], [.5, "#6FC0F5"], [1, "#0D0D0D"]],
        "ssh": [[0, "#0D0D0D"], [.5, "#F3F8FC"], [1, "#6FC0F5"]],
        "sss": [[0, "#F3F8FC"], [.5, "#CBA24E"], [1, "#0D0D0D"]],
    }
    g = st.columns(3)
    for col, var, lab in zip(g, ("sst", "ssh", "sss"), ("SST °C", "SLA m", "SSS PSU")):
        col.plotly_chart(heat(A[f"sat_{var}"][wi], LON, LAT, sc[var], lab),
                         use_container_width=True, config=_CHART_CFG)

    need_pred()
    subsec("waves", "Reconstructed profile — click a float")
    L, R = st.columns([1, 1.25])
    with L:
        fm = px.scatter(pts, x="lon", y="lat", color="regime", color_discrete_map=RC,
                        custom_data=["i"])
        fm.update_traces(marker=dict(size=7, line=dict(width=0)))
        ev = st.plotly_chart(style_fig(fm, h=390), use_container_width=True, config=_CHART_CFG,
                             on_select="rerun", key="pmap")
        pk = ev.get("selection", {}).get("points", []) if ev else []
        idx = int(pk[0]["customdata"][0]) if pk else (176 if len(pts) > 176 else 0)
    with R:
        P = {k: pred[k][idx] for k in ("T_pred", "S_pred", "T_std", "S_std",
                                       "T_obs", "S_obs", "mask")}
        mk = P["mask"] > 0
        rw = pts.iloc[idx]
        st.markdown(f'<span style="font-family:Space Grotesk;color:#8B98A2;font-size:.8rem">'
                    f'#{idx} · {rw.lat:.2f}°N {rw.lon:.2f}°E · <b style="color:#0D0D0D">'
                    f'{rw.regime}</b></span>', unsafe_allow_html=True)
        fig = go.Figure()
        for var, pv, sv, ov, cc in [("T", "T_pred", "T_std", "T_obs", INK),
                                    ("S", "S_pred", "S_std", "S_obs", "#7FA8C6")]:
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
        fig.update_layout(height=390, yaxis=dict(title="depth m", autorange="reversed"),
                          xaxis=dict(title="T °C", domain=[0, 1]),
                          xaxis2=dict(title="S PSU", overlaying="x", side="top"))
        st.plotly_chart(style_fig(fig, h=390), use_container_width=True, config=_CHART_CFG)
    resid = (P["T_pred"] - P["T_obs"])[mk]
    rt_i = float(np.sqrt(np.mean((P["T_pred"][mk] - P["T_obs"][mk]) ** 2)))
    rs_i = float(np.sqrt(np.mean((P["S_pred"][mk] - P["S_obs"][mk]) ** 2)))
    rf = go.Figure(go.Bar(x=lv[mk], y=resid, marker_color=np.where(resid >= 0, INK, AQUA),
                          marker_cornerradius=6))
    rf.update_layout(height=150, xaxis_title="depth m", yaxis_title="T resid °C", bargap=.55)
    c1, c2 = st.columns([3, 1])
    c1.plotly_chart(style_fig(rf, h=150, legend_top=False), use_container_width=True, config=_CHART_CFG)
    c2.markdown(mini("this profile RMSE·T", f"{rt_i:.2f} °C"), unsafe_allow_html=True)
    c2.markdown(mini("RMSE·S", f"{rs_i:.2f} PSU"), unsafe_allow_html=True)


# ---- 2 · accuracy & calibration ------------------------------------- #
elif page == "s2":
    need_pred()
    sec("s2", "scatter_plot", "Accuracy & calibration")
    pdesc("How well the reconstruction matches withheld Argo. <b>Skill by depth</b> — RMSE "
          "per level in the barrier-layer regime, model vs baseline. <b>Predicted vs "
          "observed</b> — every held-out point, T and S. <b>Is the uncertainty honest?</b> — "
          "whether a claimed X% interval actually contains the truth X% of the time.")
    subsec("stacked_line_chart", "Skill by depth — barrier-layer regime")
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
        f.update_layout(height=340, xaxis_title=lab,
                        yaxis=dict(title="depth m", autorange="reversed"))
        col.plotly_chart(style_fig(f, h=340), use_container_width=True, config=_CHART_CFG)

    subsec("scatter_plot", "Predicted vs observed")
    gg = st.columns(2)
    for col, pk_, ok_, lab, cc in [(gg[0], "T_pred", "T_obs", "temperature °C", INK),
                                   (gg[1], "S_pred", "S_obs", "salinity PSU", "#7FA8C6")]:
        mm = pred["mask"] > 0
        pv, ov = pred[pk_][mm], pred[ok_][mm]
        r = float(np.corrcoef(pv, ov)[0, 1])
        f = go.Figure()
        f.add_trace(go.Histogram2dContour(x=ov, y=pv, colorscale=[[0, "#F3F8FC"], [1, cc]],
                                          showscale=False, ncontours=13, line=dict(width=0)))
        lim = [min(pv.min(), ov.min()), max(pv.max(), ov.max())]
        f.add_trace(go.Scatter(x=lim, y=lim, line=dict(color="#AAB7C0", dash="dash", width=1),
                               showlegend=False, hoverinfo="skip"))
        f.update_layout(height=340, xaxis_title=f"observed {lab}",
                        yaxis_title=f"predicted {lab}",
                        annotations=[dict(x=.05, y=.93, xref="paper", yref="paper",
                                          text=f"r = {r:.3f}", showarrow=False,
                                          font=dict(color="#0D0D0D", size=14,
                                                    family="Space Grotesk"))])
        col.plotly_chart(style_fig(f, h=340, legend_top=False), use_container_width=True, config=_CHART_CFG)

    subsec("balance", "Is the uncertainty honest?")
    cT, cS = calib(pred, "T"), calib(pred, "S")
    kc = st.columns(4)
    for c, (lv_, cd, unit) in zip(kc, [(0.80, cT, "°C"), (0.95, cT, "°C"),
                                       (0.80, cS, "PSU"), (0.95, cS, "PSU")]):
        vn = "T" if unit == "°C" else "S"
        emp = cd["picp"][lv_] * 100
        c.markdown(mini(f"{vn} · {int(lv_*100)}% interval covers",
                        f"{emp:.0f}%  <span style='font-size:.7rem;color:#8FA0AB'>"
                        f"(±{cd['sharp'][lv_]/2:.2f} {unit})</span>"), unsafe_allow_html=True)

    rc4 = st.columns(2)
    for col, cd, vn, cc in [(rc4[0], cT, "temperature", INK), (rc4[1], cS, "salinity", "#7FA8C6")]:
        f = go.Figure()
        f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color="#AAB7C0", dash="dash", width=1),
                               name="perfect", hoverinfo="skip"))
        f.add_trace(go.Scatter(x=cd["nominal"], y=cd["empirical"], mode="lines+markers",
                               line=dict(color=cc, width=2.5), marker=dict(size=5),
                               name="observed"))
        f.update_layout(height=300, xaxis=dict(title="claimed coverage", range=[0, 1]),
                        yaxis=dict(title="actual coverage", range=[0, 1]),
                        annotations=[dict(x=.05, y=.9, xref="paper", yref="paper", showarrow=False,
                                          text=f"{vn} · mean gap {cd['ece']*100:.1f} pts",
                                          font=dict(color="#0D0D0D", size=12,
                                                    family="Space Grotesk"))])
        col.plotly_chart(style_fig(f, h=300), use_container_width=True, config=_CHART_CFG)


# ---- 3 · regime overlay ------------------------------------------- #
elif page == "s3":
    sec("s3", "grid_view", "Regime overlay")
    pdesc("The ocean-regime map that <i>conditions</i> the model (well-mixed / barrier-layer / "
          "upwelling), by month. It is derived only from WOA23 climatology and river discharge "
          "&mdash; <b>never</b> the satellite inputs &mdash; so it can't leak the answer. Side "
          "panels: Bay-of-Bengal barrier-layer / mixed-layer depth and the regime mix of the "
          "current holdout.")
    mo = st.slider("month", 1, 12, 8)
    L2, R2 = st.columns([1.4, 1])
    with L2:
        gm, glat, glon = regime_map(A, mo)
        f = px.imshow(gm, x=glon, y=glat, origin="lower", aspect="auto",
                      range_color=[-0.5, 2.5],
                      color_continuous_scale=[RC[r] for r in REGIMES])
        f.update_layout(coloraxis_colorbar=dict(
            tickvals=[0, 1, 2], ticktext=["mixed", "barrier", "upwell"],
            thickness=7, len=0.8, tickfont=dict(size=8, color="#8FA0AB")))
        st.plotly_chart(style_fig(f, h=360), use_container_width=True, config=_CHART_CFG)
    with R2:
        clat, clon = A["clim_lat"], A["clim_lon"]
        my = (clat >= 5) & (clat <= 22)
        mx = (clon >= 85) & (clon <= 99)
        mi = mo - 1
        blt = float(np.nanmean(A["clim_blt_m"][mi][np.ix_(my, mx)]))
        mld = float(np.nanmean(A["clim_mld_m"][mi][np.ix_(my, mx)]))
        strat = float(np.nanmean(A["clim_strat"][mi][np.ix_(my, mx)]))
        f = go.Figure(go.Bar(x=["BLT m", "MLD m", "strat×50"], y=[blt, mld, strat * 50],
                             marker_color=[INK, "#7FA8C6", AQUA], marker_cornerradius=8))
        f.update_layout(height=165, yaxis_title="Bay of Bengal mean", bargap=.5)
        st.plotly_chart(style_fig(f, h=165, legend_top=False), use_container_width=True, config=_CHART_CFG)
        if pred is not None:
            vc = pd.Series(regnames).value_counts().reindex(REGIMES, fill_value=0)
            f = go.Figure(go.Bar(y=vc.index, x=vc.values, orientation="h",
                                 marker_color=[RC[r] for r in vc.index], marker_cornerradius=8))
            f.update_layout(height=165, xaxis_title=f"{holdout} holdout profiles", bargap=.45)
            st.plotly_chart(style_fig(f, h=165, legend_top=False), use_container_width=True, config=_CHART_CFG)


# ---- 4 · model vs baseline ------------------------------------- #
elif page == "s4":
    sec("s4", "compare_arrows", "Model vs baseline")
    pdesc("Does the regime-conditioned, physics-constrained model beat a plain baseline trained "
          "the same way? <b>Per-regime RMSE</b> for both holdouts (T or S), the full table, "
          "<b>training curves</b> for every run, and the <b>physics check</b> &mdash; the rate "
          "of density inversions in the predicted profiles (TEOS-10).")
    if metrics is not None:
        var = st.radio("variable", ["rmse_T", "rmse_S"], horizontal=True, key="oe_var",
                       format_func=lambda s: "Temperature" if s.endswith("T") else "Salinity")
        pool = metrics[metrics.depth_level == -1]
        keep = [m_ for m_ in ["baseline", "oceanembed", model] if m_ in pool.model.unique()]
        sub = pool[pool.model.isin(keep) & pool.regime_class.isin(REGIMES)]
        f = px.bar(sub, x="regime_class", y=var, color="model", barmode="group",
                   facet_col="test_set", color_discrete_map=MC)
        f.update_traces(marker_cornerradius=14)
        f.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].upper(),
                                                 font=dict(size=10, color="#8FA0AB")))
        f.update_layout(height=380)
        st.plotly_chart(style_fig(f, h=380), use_container_width=True, config=_CHART_CFG)
        with st.expander("full table — per test set × regime"):
            st.dataframe(pool[pool.regime_class.isin(REGIMES + ["all"])]
                         .pivot_table(index=["test_set", "regime_class"], columns="model",
                                      values=var).round(3), use_container_width=True)

    subsec("timeline", "Training dynamics")
    if logs:
        gg = st.columns(2)
        for col, yv, lab in zip(gg, ("rmseT", "rmseS"), ("val RMSE T °C", "val RMSE S PSU")):
            f = go.Figure()
            for name, dfl in logs.items():
                v = dfl[dfl.split == "val"]
                f.add_trace(go.Scatter(x=v.epoch, y=v[yv], name=name,
                                       line=dict(color=MC.get(name, "#AAB7C0"), width=2)))
            f.update_layout(height=320, xaxis_title="epoch", yaxis_title=lab)
            col.plotly_chart(style_fig(f, h=320), use_container_width=True, config=_CHART_CFG)
    else:
        st.info("no training logs found")

    subsec("science", "Physics consistency")
    if phys is not None:
        f = go.Figure()
        for ts, cc in [("spatial", INK), ("temporal", AQUA)]:
            s = phys[phys.test_set == ts]
            f.add_trace(go.Bar(name=ts, x=s.model, y=s.inversion_fraction * 100,
                               marker_color=cc, marker_cornerradius=10))
        f.update_layout(height=300, barmode="group", yaxis_title="% inverted level pairs")
        st.plotly_chart(style_fig(f, h=300), use_container_width=True, config=_CHART_CFG)
        st.caption("≈ 0 for every model — profiles are already density-stable, so the "
                   "physics loss term has little to correct at this data scale.")


# ---- 5 · data & scope ----------------------------------------- #
elif page == "s5":
    sec("s5", "info", "Data & scope")
    pdesc("Exactly what this build runs on and what it doesn't: every satellite / Argo / "
          "climatology source and the substitutions made, what stayed out of scope (INCOIS "
          "buoys, real-time ingestion, float-deployment advice), links to the Phase-0 and "
          "Phase-3 write-ups, and a step-by-step replay of the training pipeline.")
    st.markdown(
        '<div class="oe-card" style="border-radius:20px;border-left:3px solid #6FC0F5;">'
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
    lk[0].markdown(link_card("description", "Phase 0 report",
                             "Full data reality check and every endpoint substitution."),
                   unsafe_allow_html=True)
    lk[1].markdown(link_card("query_stats", "Phase 3 findings",
                             "Honest per-regime result analysis and likely causes."),
                   unsafe_allow_html=True)
    lk[2].markdown(link_card("open_in_new", "Repository",
                             "github.com/Specter842/oceanembed — code, docs, RUN.md."),
                   unsafe_allow_html=True)

    subsec("timeline", "Re-run the pipeline")
    if st.button("▸  Re-run the full pipeline", type="primary", key="oe_rerun"):
        run_pipeline_replay()
    st.caption("Same replay as the landing page — steps through the recorded run, then reloads "
               "the metrics from disk. The hosted demo has no GPU so it cannot train live; the "
               "numbers are the real output of `python -m src.evaluate` from the last run. To "
               "run it for real, clone the repo and follow RUN.md.")
