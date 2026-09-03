"""Generate architecture / flow diagrams for the OceanEmbed design document."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path(__file__).parent / "assets"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10})

C = dict(src="#E8EEF4", proc="#DCE8DC", store="#FBF0D9", model="#EADCF0",
         out="#F3DEDE", ext="#EDEDED", accent="#33475b")


def box(ax, xy, w, h, text, fc="#EEE", ec="#333", fs=9, bold=False, r=0.02):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0.006,rounding_size={r}",
                 linewidth=1.1, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=3)
    return (x, y, w, h)


def arrow(ax, a, b, style="-|>", ls="-", color="#333", lw=1.2, rad=0.0, text=None):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=13, lw=lw,
                 color=color, linestyle=ls,
                 connectionstyle=f"arc3,rad={rad}", zorder=1))
    if text:
        ax.text((a[0]+b[0])/2, (a[1]+b[1])/2, text, fontsize=7.5, ha="center",
                va="center", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"),
                zorder=4)


def cbot(b): x, y, w, h = b; return (x + w/2, y)
def ctop(b): x, y, w, h = b; return (x + w/2, y + h)
def cl(b):   x, y, w, h = b; return (x, y + h/2)
def cr(b):   x, y, w, h = b; return (x + w, y + h/2)


def fig(w=12, h=7):
    f, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    return f, ax


def save(f, name):
    ax = f.axes[0]
    ys = []
    for p in ax.patches:
        bb = p.get_extents().transformed(ax.transData.inverted())
        ys += [bb.y0, bb.y1]
    for t in ax.texts:
        ys.append(t.get_position()[1])
    if ys:
        ax.set_ylim(min(ys) - 3, max(ys) + 3)
    f.savefig(OUT / name, dpi=150, bbox_inches="tight", pad_inches=0.15,
              facecolor="white")
    plt.close(f); print("wrote", name)


def title(ax, t):
    ax.text(50, 99, t, ha="center", fontsize=12, fontweight="bold", va="top")


# ------------------------------------------------------------------ fig 1
def fig1_context():
    f, ax = fig(12, 6.8)
    title(ax, "Figure 1  -  System Context (High-Level)")
    ax.text(15, 92, "EXTERNAL DATA SOURCES  (public, no auth)", ha="center",
            fontsize=8.5, fontweight="bold", color=C["accent"])
    srcs = [
        "AOML ERDDAP  -  Argo T/S profiles",
        "NCEI THREDDS  -  OISST v2.1 daily SST",
        "NOAA CoastWatch  -  blended SLA + SMAP SSS",
        "NCEI THREDDS  -  WOA23 monthly T/S climatology",
        "Papa et al. 2012  -  G-B discharge curve",
        "PMEL ERDDAP  -  RAMA moorings (probe)",
    ]
    ys = [80, 70, 60, 50, 40, 30]
    sb = [box(ax, (2, y), 27, 7.5, s, C["ext"], fs=7.6) for s, y in zip(srcs, ys)]
    core = box(ax, (38, 33), 27, 46,
               "OceanEmbed\n\noffline batch pipeline\n\nfetch  ->  match  ->  train\n"
               "->  evaluate  ->  serve", C["model"], fs=10, bold=True, r=0.03)
    outs = ["data/processed/*.npz\n(train / test_spatial / test_temporal)",
            "outputs/checkpoints/*.pt\n(OceanEmbed + baseline)",
            "outputs/metrics/per_regime_rmse.csv\n+ figures",
            "Streamlit dashboard\n(5-section demo)"]
    ob = [box(ax, (73, y), 25, 8, s, C["out"], fs=7.6)
          for s, y in zip(outs, [64, 53, 42, 31])]
    for s in sb:
        arrow(ax, cr(s), cl(core), rad=0.05)
    for o in ob:
        arrow(ax, cr(core), cl(o), rad=-0.05)
    ax.text(15, 22, "Unreachable from build host -> substituted:  Ifremer GDAC,\n"
                    "CMEMS, PODAAC, coastwatch.pfeg, SEANOE", ha="center",
            fontsize=7.4, style="italic", color="#822")
    save(f, "fig1_context.png")


# ------------------------------------------------------------------ fig 2
def fig2_modules():
    f, ax = fig(12, 7.2)
    title(ax, "Figure 2  -  Module / Package Architecture")
    da = box(ax, (4, 78), 32, 13,
             "src/data_access/\ncommon - fetch_argo - fetch_satellite - fetch_woa\n"
             "fetch_discharge - fetch_incois - check_sources", C["src"], fs=7.8)
    rg = box(ax, (4, 60), 32, 11,
             "src/regime/\nbuild_climatology  (WOA23 -> MLD/BLT/strat)\n"
             "regime_labels  (compute_regime_label, regime_grid)", C["proc"], fs=7.8)
    dp = box(ax, (44, 74), 26, 9, "src/data_pipeline.py\nmatch + physical holdout split",
             C["proc"], fs=8.3, bold=True)
    ds = box(ax, (44, 60), 26, 8, "src/datasets.py\nNormStats - OceanProfileDataset",
             C["proc"], fs=8.3)
    md = box(ax, (76, 55), 22, 28,
             "src/models/\nbackbone\n(ResNet-50/18 - Prithvi)\nconditioning\n"
             "(FiLMLayer - ConditionEncoder)\nphysics_loss\n(EOS-80 torch - gsw diag)\n"
             "oceanembed_model", C["model"], fs=7.6)
    tr = box(ax, (30, 38), 24, 9, "src/train.py\nConfig - loop - AMP - checkpoints",
             C["proc"], fs=8, bold=True)
    ev = box(ax, (60, 38), 24, 9, "src/evaluate.py\nper-regime RMSE / r", C["proc"],
             fs=8, bold=True)
    vz = box(ax, (60, 24), 24, 7, "src/viz/plots.py", C["proc"], fs=8)
    db = box(ax, (4, 22), 32, 10,
             "dashboard/app.py\nStreamlit - reads artefacts only (no torch)", C["out"], fs=7.8)
    ts = box(ax, (60, 11), 24, 7, "tests/  independence - split", C["ext"], fs=8)

    arrow(ax, cbot(da), ctop(rg))
    arrow(ax, cr(da), (44, 80), rad=0.05)
    arrow(ax, cr(rg), (44, 66), rad=0.05)
    arrow(ax, cbot(dp), ctop(ds))
    arrow(ax, cr(ds), cl(md))
    arrow(ax, cbot(ds), ctop(tr), rad=0.15)
    arrow(ax, cbot(md), ctop(ev), rad=-0.1)
    arrow(ax, cr(tr), cl(ev), text="checkpoints")
    arrow(ax, cbot(ev), ctop(vz))
    arrow(ax, cbot(tr), ctop(db), rad=0.1, text="artefacts")
    arrow(ax, cl(vz), cr(db), rad=0.12, text="figures")
    ax.text(cl(ts)[0] - 1, cl(ts)[1], "checks the\nregime + split\ninvariants", ha="right",
            va="center", fontsize=7, style="italic", color="#555")
    save(f, "fig2_modules.png")


# ------------------------------------------------------------------ fig 3
def fig3_pipeline():
    f, ax = fig(12, 8.4)
    title(ax, "Figure 3  -  Phase 1 Data Pipeline (flowchart)")
    a = box(ax, (3, 82), 24, 9, "fetch_argo\nAOML ERDDAP - QC 1-2 - adjusted\n"
            "2021-2023, NIO box", C["src"], fs=8)
    s = box(ax, (37, 82), 26, 9, "fetch_satellite\nOISST midweek - SLA/SSS week-mean\n"
            "-> 0.25 deg weekly stack", C["src"], fs=8)
    w = box(ax, (72, 82), 25, 9, "fetch_woa\nWOA23 T/S monthly\nNIO subset (OPeNDAP)",
            C["src"], fs=8)
    ap = box(ax, (3, 66), 24, 8, "argo_profiles.parquet\n~5.6 M levels / 11 k profiles",
             C["store"], fs=8)
    sp = box(ax, (37, 66), 26, 8, "satellite_weekly.nc\n157 wk x 3 ch x 120 x 220",
             C["store"], fs=8)
    cl_ = box(ax, (72, 66), 25, 8, "build_climatology\nMLD - ILD - BLT - strat index",
              C["proc"], fs=8)
    rc = box(ax, (72, 53), 25, 7, "regime_climatology.nc\nby month x 1 deg cell",
             C["store"], fs=8)
    rl = box(ax, (72, 40), 25, 8, "regime_labels.compute_regime_label\n"
             "(lat, lon, month, discharge)", C["proc"], fs=7.8)
    dc = box(ax, (72, 28), 25, 7, "fetch_discharge\nmonthly climatology", C["src"], fs=8)
    mp = box(ax, (24, 40), 35, 13,
             "data_pipeline.build()\n\n- profile -> 18 std depths (+ sanity checks)\n"
             "- nearest week (+/-4 d) + nearest cell (+/-0.5 deg)\n"
             "- extract 32x32x3 patch (reflect-pad)\n- attach regime label + strat index",
             C["proc"], fs=7.8, bold=True)
    spl = box(ax, (24, 26), 35, 9, "physical split  (by profile, never random)\n"
              "spatial = Bay of Bengal (lat>=5 and lon>=85)\ntemporal = JJAS 2022",
              C["proc"], fs=7.8)
    tn = box(ax, (9, 8), 19, 8, "train.npz\n6,656", C["out"], fs=8, bold=True)
    xs = box(ax, (31, 8), 19, 8, "test_spatial.npz\n1,044 (BoB)", C["out"], fs=8, bold=True)
    xt = box(ax, (53, 8), 19, 8, "test_temporal.npz\n640 (JJAS'22)", C["out"], fs=8, bold=True)
    nz = box(ax, (76, 8), 18, 8, "norm_stats.json", C["store"], fs=8)
    arrow(ax, cbot(a), ctop(ap)); arrow(ax, cbot(s), ctop(sp))
    arrow(ax, cbot(w), ctop(cl_)); arrow(ax, cbot(cl_), ctop(rc))
    arrow(ax, cbot(rc), ctop(rl)); arrow(ax, ctop(dc), cbot(rl))
    arrow(ax, cbot(ap), (28, 50), rad=0.05)
    arrow(ax, cbot(sp), (44, 53), rad=0.0)
    arrow(ax, cl(rl), cr(mp))
    arrow(ax, cbot(mp), ctop(spl))
    for t in (tn, xs, xt, nz):
        arrow(ax, cbot(spl), ctop(t), rad=0.03)
    save(f, "fig3_pipeline.png")


# ------------------------------------------------------------------ fig 4
def fig4_noncircular():
    f, ax = fig(11, 6.2)
    title(ax, "Figure 4  -  Regime Conditioning: the non-circularity guarantee")
    woa = box(ax, (4, 72), 27, 8, "WOA23 T/S climatology\n(historical, 1970-2022)",
              C["ext"], fs=8)
    bc = box(ax, (4, 58), 27, 9, "build_climatology\nsigma0 threshold MLD - dT ILD\n"
             "BLT = ILD - MLD - strat index", C["proc"], fs=7.8)
    dis = box(ax, (4, 46), 27, 7, "discharge_for_month(m)\nstatic G-B curve", C["ext"], fs=8)
    mon = box(ax, (4, 35), 27, 7, "month (1-12)\nmonsoon-phase geography", C["ext"], fs=8)
    rl = box(ax, (40, 42), 27, 24,
             "compute_regime_label\n(lat, lon, month, discharge)\n\n"
             "-> regime_class :\n   well_mixed /\n   barrier_layer_stratified /\n   upwelling\n"
             "-> stratification_index in [0,1]", C["proc"], fs=8, bold=True)
    film = box(ax, (75, 52), 22, 11, "FiLM conditioning\nin the model\n(gamma, beta modulation)",
               C["model"], fs=8)
    sat = box(ax, (75, 30), 22, 9, "satellite SST / SSH / SSS\n(prediction inputs)",
              C["src"], fs=8)
    pred = box(ax, (75, 16), 22, 8, "T / S prediction", C["out"], fs=8, bold=True)
    for b in (bc, dis, mon):
        arrow(ax, cr(b), cl(rl), rad=0.04)
    arrow(ax, cbot(woa), ctop(bc))
    arrow(ax, cr(rl), cl(film))
    arrow(ax, cbot(film), ctop(pred), rad=-0.35)
    arrow(ax, cbot(sat), ctop(pred))
    ax.add_patch(FancyArrowPatch((cl(sat)[0], cl(sat)[1] + 1),
                 (cr(rl)[0] + 1, cr(rl)[1] - 7), arrowstyle="-|>", mutation_scale=13,
                 lw=1.7, color="#c0392b", linestyle=(0, (4, 3)),
                 connectionstyle="arc3,rad=-0.35", zorder=1))
    ax.text(70, 30, "NEVER\n(enforced by\ntest_regime_independence)", fontsize=7.6,
            color="#c0392b", ha="center", fontweight="bold")
    save(f, "fig4_noncircular.png")


# ------------------------------------------------------------------ fig 5
def fig5_model():
    f, ax = fig(11.5, 6.4)
    title(ax, "Figure 5  -  OceanEmbedModel (low-level architecture)")
    inp = box(ax, (3, 64), 20, 12, "patch\n[B, 3, 32, 32]\nSST - SLA - SSS", C["src"],
              fs=8.5, bold=True)
    cnd = box(ax, (3, 34), 20, 15, "regime one-hot [3]\nstrat index [1]\nmonth sin/cos [2]",
              C["src"], fs=8.5)
    bb = box(ax, (29, 62), 24, 16, "backbone (frozen)\nResNet-50 / 18 / Prithvi-EO\n"
             "+ 1x1 channel adapter\n-> features [B, F]", C["model"], fs=8)
    ce = box(ax, (29, 33), 24, 12, "ConditionEncoder\nMLP 6 -> 32\n-> cond [B, 32]",
             C["model"], fs=8)
    film = box(ax, (59, 54), 15, 12, "FiLM\nx*(1+g)+b\n(identity-init)", C["model"],
               fs=8.5, bold=True)
    dec = box(ax, (59, 34), 15, 12, "decoder MLP\n512-512\nGELU + dropout", C["model"], fs=8.5)
    hT = box(ax, (80, 64), 17, 7, "head_T -> [B,18]", C["out"], fs=8)
    hS = box(ax, (80, 53), 17, 7, "head_S -> [B,18]", C["out"], fs=8)
    hV = box(ax, (80, 42), 17, 7, "head_logvar -> [B,2,18]", C["out"], fs=8)
    arrow(ax, cr(inp), cl(bb))
    arrow(ax, cr(cnd), cl(ce))
    arrow(ax, cr(bb), (59, 60), text="features")
    arrow(ax, cr(ce), (59, 42), text="cond")
    arrow(ax, ctop(ce), cl(film), rad=-0.3)
    arrow(ax, cbot(film), ctop(dec))
    for h in (hT, hS, hV):
        arrow(ax, cr(dec), cl(h), rad=0.06)
    ax.text(60, 24, "baseline model  =  this graph with FiLM removed  and  lambda_physics = 0",
            ha="center", fontsize=8.5, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FBF0D9", ec="#999"))
    ax.text(6, 15, "loss  =  masked GaussianNLL(T)  +  masked GaussianNLL(S)\n"
                   "         +  lambda_physics * EOS-80 density-inversion penalty",
            ha="left", fontsize=8.5, family="monospace")
    save(f, "fig5_model.png")


# ------------------------------------------------------------------ fig 6
def fig6_train():
    f, ax = fig(9.5, 9.4)
    title(ax, "Figure 6  -  Training loop (src/train.py)")
    steps = [
        ("Config  /  baseline_of(cfg)", C["proc"], 1),
        ("device = cuda if available else cpu ;  GradScaler + autocast when cuda", C["ext"], 1),
        ("OceanProfileDataset('train')  ->  90/10 train / val subsets", C["src"], 1),
        ("build_model(cfg)  -  AdamW(trainable params)  -  CosineAnnealingLR", C["model"], 1),
        ("for epoch in 1..N", C["proc"], 1),
        ("forward  ->  out{ T, S, logvar_T, logvar_S }", C["model"], 1),
        ("nll = GaussianNLL(T) + GaussianNLL(S)  (masked)\n"
         "phys = EOS-80 inversion penalty (denormalised)\nloss = nll + lambda * phys", C["out"], 3),
        ("backward  -  clip_grad_norm(5)  -  optim.step  -  sched.step", C["proc"], 1),
        ("val pass  ->  RMSE_T / RMSE_S  ;  append CSV row", C["store"], 1),
        ("val loss < best ?   ->  save outputs/checkpoints/<name>.pt", C["out"], 1),
        ("NaN / inf loss ?   ->  abort", C["ext"], 1),
    ]
    y = 90; W = 74; X = 10; prev = None; loop_from = None; loop_to = None
    for i, (txt, col, lines) in enumerate(steps):
        h = 5.2 + 2.6 * (lines - 1)
        b = box(ax, (X, y - h), W, h, txt, col, fs=8, bold=(i in (4, 6)))
        if prev:
            arrow(ax, cbot(prev), ctop(b))
        if i == 4:
            loop_to = b
        if i == 9:
            loop_from = b
        prev = b
        y -= h + 2.4
    ax.add_patch(FancyArrowPatch(cr(loop_from), cr(loop_to), arrowstyle="-|>",
                 mutation_scale=14, lw=1.3, color="#666",
                 connectionstyle="arc,angleA=0,angleB=0,armA=55,armB=55,rad=6", zorder=1))
    ax.text(X + W + 13, (loop_from[1] + loop_to[1]) / 2 + loop_to[3] / 2,
            "next epoch", fontsize=8, ha="center", color="#666", rotation=-90)
    save(f, "fig6_train.png")


# ------------------------------------------------------------------ fig 7
def fig7_eval():
    f, ax = fig(12, 6.2)
    title(ax, "Figure 7  -  Phase 3 evaluation & evidence")
    ck = box(ax, (3, 58), 20, 11, "checkpoints\noceanembed / baseline\n(+ lp05 / lp30)",
             C["out"], fs=8)
    ho = box(ax, (3, 40), 20, 11, "test_spatial.npz\ntest_temporal.npz", C["store"], fs=8)
    pr = box(ax, (30, 46), 22, 16, "evaluate.predict()\nbatched forward\ndenorm T/S + sigma\n"
             "(logvar or MC-dropout)", C["proc"], fs=8, bold=True)
    mt = box(ax, (58, 55), 22, 11, "metrics_for()\nRMSE_T / RMSE_S / r\nper regime x depth",
             C["proc"], fs=8)
    pg = box(ax, (58, 40), 22, 11, "physics_diagnostics_gsw\nTEOS-10 inversion fraction",
             C["proc"], fs=8)
    c1 = box(ax, (84, 57), 14, 7, "per_regime_rmse.csv", C["out"], fs=7.4, bold=True)
    c2 = box(ax, (84, 47), 14, 7, "physics_diagnostics.csv", C["out"], fs=7.4)
    c3 = box(ax, (84, 37), 14, 7, "predictions_*.npz", C["store"], fs=7.4)
    vz = box(ax, (34, 16), 33, 9, "viz/plots.py\nrmse_by_regime - profiles_barrier_layer\n"
             "- uncertainty_map_bob", C["proc"], fs=7.8)
    fg = box(ax, (74, 17), 22, 8, "outputs/figures/*.png", C["out"], fs=8, bold=True)
    arrow(ax, cr(ck), cl(pr), rad=0.05); arrow(ax, cr(ho), cl(pr), rad=-0.05)
    arrow(ax, cr(pr), cl(mt), rad=0.05); arrow(ax, cr(pr), cl(pg), rad=-0.05)
    arrow(ax, cr(mt), cl(c1)); arrow(ax, cr(pg), cl(c2))
    arrow(ax, cbot(pr), cl(c3), rad=-0.3)
    arrow(ax, cbot(c1), ctop(vz), rad=0.35)
    arrow(ax, cr(vz), cl(fg))
    save(f, "fig7_eval.png")


# ------------------------------------------------------------------ fig 8
def fig8_dashboard():
    f, ax = fig(12, 6.0)
    title(ax, "Figure 8  -  Dashboard data flow (dashboard/app.py)")
    src = [box(ax, (3, y), 31, 7, s, C["store"], fs=7.9) for s, y in [
        ("satellite_weekly.nc", 78), ("predictions_<model>_<set>.npz", 66),
        ("regime_climatology.nc  +  regime_grid()", 54),
        ("per_regime_rmse.csv  -  physics_diagnostics.csv", 42)]]
    secs = [box(ax, (44, y), 53, 7, s, c, fs=7.8) for s, y, c in [
        ("1 - Satellite input fields  -  week slider -> 3 heatmaps", 80, C["out"]),
        ("2 - Profile viewer  -  click point -> pred vs Argo T/S + 2 sigma band", 68, C["out"]),
        ("3 - Regime overlay  -  month slider -> class map", 56, C["out"]),
        ("4 - OceanEmbed vs baseline  -  RMSE bars, T/S toggle, per holdout", 44, C["out"]),
        ("5 - Data & Scope footer  -  disclaimer + substitution notes", 32, C["ext"])]]
    for i in range(4):
        arrow(ax, cr(src[i]), cl(secs[i]), rad=0.05)
    ax.text(50, 22, "Streamlit  -  @st.cache_data loaders  -  no torch import  -  "
            "runs anywhere the artefacts exist", ha="center", fontsize=8.3, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="#EEE", ec="#999"))
    save(f, "fig8_dashboard.png")


# ------------------------------------------------------------------ fig 9
def fig9_phases():
    f, ax = fig(12, 5.0)
    title(ax, "Figure 9  -  End-to-end phase workflow with verification gates")
    ph = [
        ("Phase 0\nData reality check\n-> phase0_report.md", C["ext"]),
        ("Phase 1\nPipeline + holdouts\n-> *.npz", C["src"]),
        ("Phase 2\nModel + physics loss\n+ baseline -> *.pt", C["model"]),
        ("Phase 3\nPer-regime evaluation\n-> per_regime_rmse.csv", C["proc"]),
        ("Phase 4\nDashboard + README", C["out"]),
    ]
    x = 3; prev = None
    for txt, col in ph:
        b = box(ax, (x, 52), 17, 22, txt, col, fs=8.2, bold=True)
        if prev:
            arrow(ax, cr(prev), cl(b))
            ax.text((prev[0] + prev[2] + b[0]) / 2, 47, "gate", fontsize=7,
                    ha="center", color="#a33", style="italic")
        prev = b; x += 19.4
    ax.text(50, 28,
            "Gates (checklist, must pass before the next phase):\n"
            "0->1  report reviewed (INCOIS / discharge flagged)\n"
            "1->2  independence test passes + split-disjoint assertion\n"
            "2->3  both models train, no NaN, checkpoints saved\n"
            "3->4  csv is script-generated + at least one regime plot",
            ha="center", fontsize=8, family="monospace")
    save(f, "fig9_phases.png")


for fn in (fig1_context, fig2_modules, fig3_pipeline, fig4_noncircular, fig5_model,
           fig6_train, fig7_eval, fig8_dashboard, fig9_phases):
    fn()
print("done")
