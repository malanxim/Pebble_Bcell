"""single_figs.py —— 每张图独立单面板(不拼接),文字不重叠。

每函数 = 一张单面板图;输出 final_figures/panels/{name}.{pdf,png}(矢量+300dpi)。
"""
from __future__ import annotations
import gc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Patch
from matplotlib.lines import Line2D

from . import config as C
from . import paper_design as D
from .tf_analysis import _nets, _lognorm_pb
D.apply_style()

T = C.TAB_DIR
BC = C.H5AD
PAN = D.OUT / "panels"; PAN.mkdir(parents=True, exist_ok=True)


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(PAN / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig); print(f"  panels/{name}")


def _curve(d, col, ax, color, z=False):
    dd = d[["age", col]].dropna().rename(columns={col: "s"}).copy()
    s = dd["s"].astype(float)
    if z:
        s = (s - s.mean()) / (s.std() + 1e-9)
    dd["s"] = s
    m = smf.ols("s ~ bs(age, df=4)", data=dd).fit()
    grid = pd.DataFrame({"age": np.linspace(20, 95, 120)})
    pred = m.get_prediction(grid)
    ax.scatter(dd["age"], dd["s"], s=9, alpha=0.22, color=D.MUTED, rasterized=True)
    ax.fill_between(grid["age"], pred.conf_int()[:, 0], pred.conf_int()[:, 1], color=color, alpha=0.16)
    ax.plot(grid["age"], pred.predicted_mean, color=color, lw=2.2)
    D.age_rug(ax, dd["age"].values)


# ============ atlas / cohort ============
def design():
    fig, ax = plt.subplots(figsize=(9, 4.2)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4)
    def b(x, y, w, h, t, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc=fc, ec=D.BASE, lw=1.4))
        ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=8.5, fontweight="bold")
    b(0.2, 1.5, 2.0, 1.0, "981 donors\n1.25M PBMC\n129,579 B cells", "#eef2f8")
    b(3.0, 2.3, 2.2, 0.9, "Azimuth B labels\n4 subtypes", D.SURF)
    b(3.0, 0.7, 2.2, 0.9, "donor×subtype\npseudobulk\n(raw counts)", "#fdf3ec")
    b(6.0, 2.3, 2.0, 0.9, "Phase 1\nDEG · GSEA", D.SURF)
    b(6.0, 0.7, 2.0, 0.9, "Phase 2\nTF regulon", D.SURF)
    b(8.6, 1.4, 1.2, 1.2, "age model\nspline+linear", "#f1f6f2")
    for (x1, y1, x2, y2) in [(2.2, 2.0, 3.0, 2.75), (2.2, 2.0, 3.0, 1.15),
                             (5.2, 2.75, 6.0, 2.75), (5.2, 1.15, 6.0, 1.15),
                             (8.0, 2.75, 8.6, 2.2), (8.0, 1.15, 8.6, 1.8)]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=D.INK2, lw=1.0))
    ax.set_title("Study design: donor-level framework", fontsize=11, fontweight="bold", pad=10)
    _save(fig, "01_design")


def age_dist():
    don = pd.read_csv(T / "donor_summary.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hist(don["age"], bins=np.arange(18, 99, 4), color=D.SUBTYPE_COLOR["B naive"], alpha=0.85, edgecolor="white")
    ax.set_xlabel("age (years)"); ax.set_ylabel("# donors")
    ax.set_title(f"Cohort age distribution (n={len(don)}, median {int(don['age'].median())})", fontsize=10.5)
    D.style_ax(ax); _save(fig, "02_age_distribution")


def coverage():
    cov = pd.read_csv(T / "subtype_donor_coverage.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(6, 4.2)); x = np.arange(cov.shape[1]); w = 0.26
    for i, thr in enumerate([10, 20, 30]):
        ax.bar(x + (i-1)*w, cov.loc[thr].values, w, label=f"≥{thr} cells",
               color=[D.MUTED, D.SUBTYPE_COLOR["B memory"], D.SUBTYPE_COLOR["B intermediate"]][i], alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels([D.SUBTYPE_SHORT[s] for s in cov.columns], rotation=12)
    ax.set_ylabel("# donors"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Donors per subtype (DEG coverage)", fontsize=10.5)
    D.style_ax(ax)
    fig.text(0.5, -0.02, "Note: plasmablast has only 14 donors at >=20 cells -> exploratory only",
             ha="center", fontsize=7.5, color=D.INK2, style="italic")
    _save(fig, "03_coverage")


def umap():
    b = ad.read_h5ad(BC, backed="r")
    umap = np.asarray(b.obsm["X_umap"])
    lab = b.obs[C.COL_CELLTYPE].astype(str).values
    b.file.close()
    fig, ax = plt.subplots(figsize=(6, 5.5))
    o = np.random.default_rng(3).permutation(len(umap))[:40000]
    for s in ["B naive", "B memory", "B intermediate", "Plasmablast"]:
        m = lab[o] == s
        ax.scatter(umap[o][m, 0], umap[o][m, 1], s=3, alpha=0.35, color=D.SUBTYPE_COLOR[s], label=s, rasterized=True)
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2"); ax.legend(fontsize=8, markerscale=3, frameon=False)
    ax.set_title("B-cell subtypes (Azimuth annotation)", fontsize=10.5); D.style_ax(ax, ygrid=False)
    _save(fig, "04_umap")


# ============ composition ============
def comp_forest():
    rob = pd.read_csv(T / "composition_robustness.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.2)); y = np.arange(len(rob))
    ax.errorbar(rob["beta_age"], y - 0.16, xerr=1.96*rob["SE_binomial"], fmt="o", ms=6, color=D.MUTED, capsize=3, label="binomial (naive)")
    ax.errorbar(rob["beta_age"], y + 0.16, xerr=1.96*rob["SE_quasi"], fmt="o", ms=6, color=D.SUBTYPE_COLOR["B memory"], capsize=3, label="quasi-binomial (corrected)")
    ax.axvline(0, color=D.MUTED, ls="--", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([f"{r.subtype} (φ={r.dispersion_pearson:.0f})" for r in rob.itertuples()], fontsize=8)
    ax.set_xlabel("log-odds / 10 yr"); ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("Overdispersion widens CIs (naive p too optimistic)", fontsize=10)
    D.style_ax(ax); _save(fig, "05_comp_overdispersion_forest")


def comp_absolute():
    rob = pd.read_csv(T / "composition_robustness.csv").sort_values("delta_40to80_pp")
    fig, ax = plt.subplots(figsize=(6.5, 4.2)); y = np.arange(len(rob))
    ax.scatter(rob["delta_40to80_pp"], y, s=90, c=[D.SUBTYPE_COLOR[s] for s in rob["subtype"]], zorder=3, edgecolors="white")
    ax.axvline(0, color=D.MUTED, lw=0.8)
    ax.set_xlim(rob["delta_40to80_pp"].min() - 0.5, rob["delta_40to80_pp"].max() + 1.5)
    for i, v in enumerate(rob["delta_40to80_pp"]):
        ax.text(v + 0.12, i, f"{v:+.1f} pp", va="center", ha="left", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(rob["subtype"]); ax.set_xlabel("Δ predicted fraction, age 40→80 (pp)")
    ax.set_title("Absolute composition change is small", fontsize=10.5); D.style_ax(ax)
    _save(fig, "06_comp_absolute_change")


def comp_merge():
    mm = pd.read_csv(T / "check3_merge_composition.csv")
    mm["lab"] = mm["group"].replace({"B naive": "naive", "memory+intermediate": "mem+inter\n(merged)", "Plasmablast": "plasma"})
    fig, ax = plt.subplots(figsize=(6.5, 4.2)); y = np.arange(len(mm))
    cols = {"naive": D.SUBTYPE_COLOR["B naive"], "mem+inter\n(merged)": D.SUBTYPE_COLOR["B intermediate"], "plasma": D.SUBTYPE_COLOR["Plasmablast"]}
    ax.barh(y, mm["beta_age10yr"], color=[cols[l] for l in mm["lab"]], alpha=0.9)
    ax.axvline(0, color=D.MUTED, lw=0.8)
    ax.set_xlim(-0.035, 0.06)
    for i, r in enumerate(mm.itertuples()):
        lab = "NS" if r.p_quasi >= 0.05 else f"p={r.p_quasi:.1e}"
        ax.text(max(r.beta_age10yr, 0) + 0.0015, i, lab, va="center", ha="left", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(mm["lab"]); ax.set_xlabel("log-odds / 10 yr (quasi-binomial)")
    ax.set_title("Merging memory+intermediate → effect disappears", fontsize=10); D.style_ax(ax)
    _save(fig, "07_comp_merge_sensitivity")


def comp_fraction_age():
    frac = pd.read_csv(T / "per_donor_subtype_fractions.csv")
    fmap = {"B naive": "frac_Bnaive", "B memory": "frac_Bmemory", "B intermediate": "frac_Bintermediate", "Plasmablast": "frac_Plasmablast"}
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for s, col in fmap.items():
        d = frac[["age", col]].dropna().sort_values("age")
        ax.scatter(d["age"], d[col], s=6, alpha=0.2, color=D.SUBTYPE_COLOR[s], rasterized=True)
        ax.plot(d["age"], d[col].rolling(80, min_periods=15, center=True).mean(), color=D.SUBTYPE_COLOR[s], lw=1.8, label=s)
    ax.set_xlabel("age"); ax.set_ylabel("fraction of B cells"); ax.set_ylim(-0.02, 1.0)
    ax.legend(fontsize=7.5, frameon=False); ax.set_title("Subtype fraction vs age (donor-level)", fontsize=10.5)
    D.style_ax(ax); _save(fig, "08_comp_fraction_vs_age")


# ============ DEG ============
def volcano(subtype, highlight):
    d = pd.read_csv(T / "deg" / f"{subtype.replace(' ', '')}_deg_continuous.csv").dropna(subset=["log2FC_per10yr", "padj"])
    d = d[~d["is_IG"]]; d["nlogp"] = -np.log10(d["padj"].clip(lower=1e-300))
    sig = (d["padj"] < 0.05) & (d["log2FC_per10yr"].abs() > 0.05)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(d.loc[~sig, "log2FC_per10yr"], d.loc[~sig, "nlogp"], s=5, alpha=0.2, color=D.MUTED, rasterized=True)
    up = sig & (d["log2FC_per10yr"] > 0); dn = sig & (d["log2FC_per10yr"] < 0)
    ax.scatter(d.loc[dn, "log2FC_per10yr"], d.loc[dn, "nlogp"], s=8, alpha=0.6, color=D.SUBTYPE_COLOR["B naive"])
    ax.scatter(d.loc[up, "log2FC_per10yr"], d.loc[up, "nlogp"], s=8, alpha=0.6, color=D.SUBTYPE_COLOR["B memory"])
    for _, r in d[d["symbol"].isin(highlight) & sig].iterrows():
        ax.annotate(r["symbol"], (r["log2FC_per10yr"], r["nlogp"]), fontsize=6.8, xytext=(3, 3), textcoords="offset points")
    ax.axhline(-np.log10(0.05), color=D.MUTED, ls=":", lw=0.7); ax.axvline(0, color=D.MUTED, lw=0.6)
    ax.set_xlim(-0.35, 0.35); ax.set_xlabel("log2FC per 10 yr"); ax.set_ylabel("-log10 FDR")
    ax.set_title(f"{subtype} age DEG (n={int(d['n_donors'].iloc[0])} donors, {(d['padj']<0.05).sum()} sig)", fontsize=10)
    D.style_ax(ax); _save(fig, f"09_volcano_{D.SUBTYPE_SHORT[subtype]}")


def keygene_trend():
    a = ad.read_h5ad(C.PB_DIR / "pb_Bmemory.h5ad")
    X = np.asarray(a.X.todense()) if sp.issparse(a.X) else np.asarray(a.X)
    lib = X.sum(1); lcpm = np.log1p(X / np.where(lib > 0, lib, 1)[:, None] * 1e4)
    sym = a.var["symbol"].astype(str).values; smap = {g: i for i, g in enumerate(sym)}
    df = pd.DataFrame({"age": a.obs["age"].values})
    for g in ["CD69", "NR4A2", "BTG1"]:
        if g in smap: df[g] = lcpm[:, smap[g]]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for g, col in [("CD69", D.SUBTYPE_COLOR["B memory"]), ("NR4A2", D.SUBTYPE_COLOR["B naive"]), ("BTG1", D.SUBTYPE_COLOR["B intermediate"])]:
        if g in df.columns:
            _curve(df, g, ax, col, z=True)
    ax.set_xlabel("age"); ax.set_ylabel("expression (z-scored log1p CPM)")
    from matplotlib.lines import Line2D as L2
    ax.legend(handles=[L2([0],[0],color=D.SUBTYPE_COLOR["B memory"],lw=2,label="CD69"),
               L2([0],[0],color=D.SUBTYPE_COLOR["B naive"],lw=2,label="NR4A2"),
               L2([0],[0],color=D.SUBTYPE_COLOR["B intermediate"],lw=2,label="BTG1")], fontsize=8, frameon=False)
    ax.set_title("Memory B: activation/anti-proliferation genes vs age", fontsize=10)
    D.style_ax(ax); del a; gc.collect(); _save(fig, "10_keygene_age_memory")


def deg_heatmap():
    subs = ["B naive", "B memory", "B intermediate"]; frames = {}
    for s in subs:
        dd = pd.read_csv(T / "deg" / f"{s.replace(' ','')}_deg_continuous.csv")
        dd = dd[~dd["is_IG"]].set_index("symbol")["log2FC_per10yr"]; frames[s] = dd
    allg = pd.concat(frames, axis=1)
    sel = allg.abs().sum(axis=1).sort_values(ascending=False).head(20).index
    M = allg.loc[sel, subs]
    fig, ax = plt.subplots(figsize=(4.5, 6))
    im = ax.imshow(M.values, cmap=D.DIVERGE_CMAP, vmin=-0.12, vmax=0.12, aspect="auto")
    ax.set_yticks(range(len(sel))); ax.set_yticklabels(sel, fontsize=7.5)
    ax.set_xticks(range(3)); ax.set_xticklabels([D.SUBTYPE_SHORT[s] for s in subs])
    fig.colorbar(im, ax=ax, fraction=0.04, label="log2FC/10yr")
    ax.set_title("Top age DEG across subtypes", fontsize=10); ax.grid(False)
    _save(fig, "11_deg_heatmap")


# ============ pathway ============
def hallmark_heatmap():
    hf = pd.read_csv(T / "gsea" / "hallmark_focus_pathways.csv")
    keep = ["TNF-alpha Signaling via NF-kB", "Interferon Gamma Response", "Interferon Alpha Response",
            "Inflammatory Response", "IL-6/JAK/STAT3 Signaling", "Apoptosis", "p53 Pathway",
            "Myc Targets V1", "E2F Targets", "G2-M Checkpoint"]
    hf = hf[hf["Term"].isin(keep)]
    piv = hf.pivot_table(index="Term", columns="subtype", values="NES", aggfunc="first").reindex(keep)
    fdr = hf.pivot_table(index="Term", columns="subtype", values="FDR q-val", aggfunc="first").reindex(keep)
    fig, ax = plt.subplots(figsize=(5, 6))
    im = ax.imshow(piv.values, cmap=D.DIVERGE_CMAP, vmin=-2.6, vmax=2.6, aspect="auto")
    ax.set_yticks(range(len(keep))); ax.set_yticklabels([t.replace(" Signaling via NF-kB", "/NF-κB").replace(" Response", "") for t in keep], fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels([D.SUBTYPE_SHORT[s] for s in piv.columns])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            f = fdr.values[i, j]
            ax.text(j, i, "**" if f < 0.05 else ("*" if f < 0.1 else ""), ha="center", va="center", fontsize=9, color="white" if abs(piv.values[i, j]) > 1.6 else D.INK)
    fig.colorbar(im, ax=ax, fraction=0.04, label="NES"); ax.set_title("Hallmark GSEA (*FDR<0.1 **<0.05)", fontsize=10); ax.grid(False)
    _save(fig, "12_hallmark_nes_heatmap")


def pathway_module(tag, term, color, title):
    df = pd.read_csv(T / f"check2_scores_{tag}.csv")
    if term not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    _curve(df, term, ax, color)
    ax.set_xlabel("age"); ax.set_ylabel("module score (log1p CPM)")
    ax.set_title(title, fontsize=10); D.style_ax(ax); _save(fig, f"13_pathway_{tag}_{term[:8].replace(' ','')}")


# ============ TF ============
def tf_heatmap():
    fdr = pd.read_csv(T / "tf_age_full_fdr.csv")
    tfocus = ["FOXO3", "PAX5", "STAT1", "NFKB", "MYC", "NR4A2", "ATF4", "E2F1"]
    subs = C.MAIN_SUBTYPES
    rows = []
    for s in subs:
        for t in tfocus:
            r = fdr[(fdr.subtype == s) & (fdr.tf == t)]
            if len(r): rows.append({"subtype": s, "tf": t, "lin": r["linear_per10yr"].values[0], "fdr": r["p_spline_fdr_full"].values[0]})
    rdf = pd.DataFrame(rows)
    mat = rdf.pivot_table(index="tf", columns="subtype", values="lin").reindex(tfocus)
    matf = rdf.pivot_table(index="tf", columns="subtype", values="fdr").reindex(tfocus)
    vm = np.nanmax(np.abs(mat.values))
    CONV = {("B naive", "STAT1"), ("B memory", "NFKB"), ("B intermediate", "NFKB"), ("B naive", "NFKB")}
    fig, ax = plt.subplots(figsize=(4.8, 5.5))
    im = ax.imshow(mat.values, cmap=D.DIVERGE_CMAP, vmin=-vm, vmax=vm, aspect="auto")
    ax.set_yticks(range(len(tfocus))); ax.set_yticklabels(tfocus, fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels([D.SUBTYPE_SHORT[s] for s in mat.columns])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            s_ = mat.columns[j]; t_ = mat.index[i]; fd = matf.values[i, j]
            mk = "*" if fd < 0.05 else ("•" if (s_, t_) in CONV else "")
            ax.text(j, i, mk, ha="center", va="center", fontsize=10, color="white" if abs(mat.values[i, j]) > vm*0.55 else D.INK)
    fig.colorbar(im, ax=ax, fraction=0.04, label="TF act /10yr")
    ax.set_title("TF age effect  (*strict FDR  •convergent)", fontsize=9.5); ax.grid(False)
    _save(fig, "14_tf_effect_heatmap")


def tf_curve(subtype, tf):
    ct, _ = _nets()
    import decoupler as dc
    mat, obs = _lognorm_pb(subtype)
    acts = dc.run_ulm(mat, ct, min_n=5, use_raw=False)[0]
    if tf not in acts.columns:
        del mat, acts; return
    fdr = pd.read_csv(T / "tf_age_full_fdr.csv")
    fd = fdr[(fdr.subtype == subtype) & (fdr.tf == tf)]["p_spline_fdr_full"]
    tier = "strict" if (len(fd) and fd.values[0] < 0.05) else "convergent"
    d = acts[[tf]].copy(); d.columns = ["s"]; d = d.join(obs[["age"]]).dropna()
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    _curve(d, "s", ax, D.SUBTYPE_COLOR[subtype])
    ax.set_xlabel("age"); ax.set_ylabel(f"{tf} activity (ULM)")
    ax.set_title(f"{D.SUBTYPE_SHORT[subtype]} | {tf}  [{tier}, FDR={fd.values[0]:.3g}]", fontsize=9.8)
    D.style_ax(ax); del mat, acts; gc.collect(); _save(fig, f"15_tf_curve_{tf}_{D.SUBTYPE_SHORT[subtype]}")


def evidence_matrix():
    sc = pd.read_csv(T / "tf_age_results_scored.csv")
    _tfs = ["FOXO3", "PAX5", "STAT1", "NFKB", "MYC", "NR4A2", "ATF4", "E2F1"]
    kr = sc[sc["tf"].isin(_tfs)].copy()
    kr = kr[kr["subtype"].isin(list(C.MAIN_SUBTYPES))]
    kr = kr[kr[["doro_sign", "mlm_sign", "tf_pathway_corr"]].notna().sum(axis=1) >= 2]
    kr["lab"] = kr["subtype"].replace(D.SUBTYPE_SHORT) + " | " + kr["tf"]
    cols = ["main_sign", "doro_sign", "mlm_sign", "tf_pathway_corr"]
    M = kr.set_index("lab")[cols].copy()
    M["tf_pathway_corr"] = np.sign(M["tf_pathway_corr"]); vals = np.where(pd.isna(M.values), np.nan, M.values.astype(float))
    masked = np.ma.masked_invalid(vals)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    cmap = plt.matplotlib.colors.ListedColormap([D.SUBTYPE_COLOR["B naive"], "#e8e8e6", D.SUBTYPE_COLOR["B memory"]]); cmap.set_bad("#f2f2f0")
    ax.imshow(masked, aspect="auto", cmap=cmap, vmin=-1, vmax=1)
    ax.set_yticks(range(len(M))); ax.set_yticklabels(M.index, fontsize=7.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(["linear", "DoRothEA", "MLM", "Hallmark"], fontsize=8, rotation=15)
    ax.set_title("Evidence convergence (orange↑ blue↓ grey=n/a)", fontsize=9.5); ax.grid(False)
    _save(fig, "16_tf_evidence_matrix")


def tf_network():
    from .mechanism_fig import _targets, NODES, TIER_FILL, TIER_EDGE
    ct, _ = _nets()
    fig, ax = plt.subplots(figsize=(9, 5.5)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    XF, XT = 0.18, 0.45; n_tf = len(NODES)
    tf_y = {NODES[i][0]: 0.85 - i*(0.72/(n_tf-1)) for i in range(n_tf)}
    for tf, st, te in NODES:
        subs = _targets(ct, tf, st); m = len(subs)
        if m == 0: continue
        yc = tf_y[tf]; span = min(0.14, 0.032*m); ys = np.linspace(yc-span/2, yc+span/2, m) if m > 1 else [yc]
        for (sym, r), yy in zip(subs.iterrows(), ys):
            w = float(r["w"])
            # 边:灰色;实线=激活,虚线=抑制(不再用颜色混年龄方向)
            ax.annotate("", xy=(XT-0.01, yy), xytext=(XF+0.09, yc),
                        arrowprops=dict(arrowstyle="-|>" if w > 0 else "-[", color="#aaaaaa",
                                        lw=1.0+min(abs(w),3)*0.2, alpha=0.5,
                                        linestyle="-" if w > 0 else "--",
                                        connectionstyle="arc3,rad=0.04", shrinkA=2, shrinkB=2))
            # 点:颜色=年龄方向(橙=随龄↑,蓝=随龄↓);大小=显著性(-log10 padj)
            c = D.SUBTYPE_COLOR["B memory"] if float(r["log2FC_per10yr"]) > 0 else D.SUBTYPE_COLOR["B naive"]
            s = 70 + min(-np.log10(max(float(r["padj"]),1e-10)),6)*15
            ax.scatter(XT, yy, s=s, c=c, alpha=0.92, edgecolors="white", linewidth=0.4, zorder=4)
            ax.text(XT+0.012, yy, sym, fontsize=8, va="center", zorder=5)
        ax.add_patch(FancyBboxPatch((XF-0.05, yc-0.026), 0.14, 0.052,
                     boxstyle="round,pad=0.003,rounding_size=0.01",
                     fc=TIER_FILL[te], ec=TIER_EDGE[te], lw=1.4, zorder=3))
        ax.text(XF+0.02, yc, tf, fontsize=10, fontweight="bold", ha="center", va="center", zorder=4)
        ax.text(XF+0.02, yc-0.055, D.SUBTYPE_SHORT[st], fontsize=6.5, ha="center", color=D.INK2, style="italic")
    leg = [Patch(facecolor=TIER_FILL["strict"], edgecolor="black", label="TF strict"),
           Patch(facecolor=TIER_FILL["convergent"], edgecolor="black", label="TF convergent"),
           Line2D([0],[0],marker="o",color="none",markerfacecolor=D.SUBTYPE_COLOR["B memory"],markersize=7,label="target: up with age"),
           Line2D([0],[0],marker="o",color="none",markerfacecolor=D.SUBTYPE_COLOR["B naive"],markersize=7,label="target: down with age"),
           Line2D([0],[0],color="#aaaaaa",lw=1.2,linestyle="-",label="edge: activation"),
           Line2D([0],[0],color="#aaaaaa",lw=1.2,linestyle="--",label="edge: inhibition"),
           Line2D([0],[0],marker="o",color="none",markerfacecolor="grey",markersize=5,label="dot size = -log10(FDR)")]
    ax.legend(handles=leg, loc="lower right", fontsize=7, frameon=True, facecolor="white", edgecolor=D.GRID)
    ax.set_title("TF - target regulatory network", fontsize=11, fontweight="bold", pad=8)
    _save(fig, "17_tf_target_network")


def state_model():
    fig, ax = plt.subplots(figsize=(8.5, 4.6)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    def box(x, y, w, h, ec=D.BASE, lw=1.4, fc=D.SURF):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.025", fc=fc, ec=ec, lw=lw))
    box(0.03, 0.43, 0.16, 0.14); ax.text(0.11, 0.50, "age", fontsize=13, fontweight="bold", ha="center", va="center")
    box(0.28, 0.16, 0.34, 0.68, ec=D.BASE)
    ax.text(0.45, 0.79, "regulatory programs", fontsize=9, fontweight="bold", ha="center", color=D.INK2)
    for i, (txt, tier) in enumerate([("FOXO3 ↑ (all subtypes)", "strict"), ("PAX5 ↓ (intermediate)", "strict"),
                                     ("STAT1 / IFN ↑ (naive)", "convergent"), ("NF-κB ↑ (memory)", "convergent")]):
        yy = 0.66 - i*0.12
        ax.add_patch(Rectangle((0.31, yy-0.018), 0.035, 0.036, fc=TIER_FILL_map()[tier], ec=TIER_EDGE_map()[tier], lw=1.0))
        ax.text(0.355, yy, txt, fontsize=8.4, va="center")
    box(0.69, 0.20, 0.28, 0.60, ec=D.BASE)
    ax.text(0.83, 0.75, "candidate B-cell state", fontsize=8.8, fontweight="bold", ha="center", color=D.INK2)
    for i, t in enumerate(["chronic low-level", "inflammation", "", "↓ growth / cycle", "readiness", "", "altered identity", "maintenance", "", "stress / quiescence"]):
        ax.text(0.83, 0.66 - i*0.042, t, fontsize=7.6, ha="center", color=D.INK if t else "none")
    for (x1, x2) in [(0.19, 0.28), (0.62, 0.69)]:
        ax.annotate("", xy=(x2, 0.50), xytext=(x1, 0.50), arrowprops=dict(arrowstyle="-|>", color=D.INK2, lw=1.4, linestyle=(0, (3, 2))))
    ax.text(0.5, 0.04, "solid = strict FDR<0.05 · light = convergent · dashed = associated with, not causal", ha="center", fontsize=7, color=D.MUTED, style="italic")
    ax.set_title("Candidate state model", fontsize=11, fontweight="bold")
    _save(fig, "18_state_model")


def TIER_FILL_map(): return {"strict": "#1baf7a", "convergent": "#9ed8c0", "exploratory": "#d8d8d6"}
def TIER_EDGE_map(): return {"strict": "#0f6e4f", "convergent": "#5b9b86", "exploratory": "#999"}


def run_all():
    design(); age_dist(); coverage(); umap()
    comp_forest(); comp_absolute(); comp_merge(); comp_fraction_age()
    volcano("B naive", ["TSC22D3", "YBX3", "CD69", "NEAT1", "MARCKSL1", "VPREB3"])
    volcano("B memory", ["CD69", "NR4A2", "BTG1", "LGALS1", "LTB", "VPREB3"])
    volcano("B intermediate", ["LGALS1", "LILRA4", "JCHAIN", "MARCKSL1", "LTB"])
    keygene_trend(); deg_heatmap(); hallmark_heatmap()
    pathway_module("Bnaive", "Interferon Gamma Response", D.SUBTYPE_COLOR["B naive"], "naive | IFN-γ module vs age")
    pathway_module("Bmemory", "TNF-alpha Signaling via NF-kB", D.SUBTYPE_COLOR["B memory"], "memory | TNF/NF-κB module vs age")
    tf_heatmap()
    tf_curve("B naive", "STAT1"); tf_curve("B intermediate", "PAX5"); tf_curve("B memory", "NFKB")
    for s in C.MAIN_SUBTYPES: tf_curve(s, "FOXO3")
    evidence_matrix(); tf_network(); state_model()
    print("[single] all single-panel figures done.")


if __name__ == "__main__":
    run_all()
