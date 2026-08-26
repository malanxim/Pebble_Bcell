"""honest_figs.py —— 诚实表达"小但协调"的年龄效应。

1) forest_40to80:核心 TF 的 40→80 岁变化(Δ ± 95%CI + 占 donor SD 比例 + FDR + tier)。
2) tf_curve_delta:TF 活性"相对 40 岁基线"的变化曲线(诚实展示小累积效应)。
3) direction_heatmap:按通路分组的基因年龄效应热图(每基因小、整组同向)。
"""
from __future__ import annotations
import gc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import config as C
from . import paper_design as D
from .tf_analysis import _nets, _lognorm_pb
D.apply_style()

T = C.TAB_DIR
PAN = D.OUT / "panels"
CORE = [("FOXO3", "B naive"), ("FOXO3", "B memory"), ("FOXO3", "B intermediate"),
        ("PAX5", "B intermediate"), ("STAT1", "B naive"), ("NFKB", "B memory")]
TIER_OF = {("FOXO3", "B naive"): "strict", ("FOXO3", "B memory"): "strict", ("FOXO3", "B intermediate"): "strict",
           ("PAX5", "B intermediate"): "strict", ("STAT1", "B naive"): "convergent", ("NFKB", "B memory"): "convergent"}
TIER_COLOR = {"strict": "#1baf7a", "convergent": "#9ed8c0", "exploratory": "#cccccc"}


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(PAN / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig); print(f"  panels/{name}")


# ============ 1) 40→80 森林图 ============
def forest_40to80():
    ct, _ = _nets()
    import decoupler as dc
    fdr = pd.read_csv(T / "tf_age_full_fdr.csv")
    rows = []
    for tf, sub in CORE:
        mat, obs = _lognorm_pb(sub)
        acts = dc.run_ulm(mat, ct, min_n=5, use_raw=False)[0]
        if tf not in acts.columns:
            del mat, acts; continue
        d = pd.DataFrame({"s": acts[tf].values}, index=obs.index).join(obs[["age", "sex", "pool"]]).dropna()
        d["age_scaled"] = (d["age"] - d["age"].mean()) / 10.0
        m = smf.ols("s ~ age_scaled + sex + pool", data=d).fit()
        coef = float(m.params["age_scaled"]); se = float(m.bse["age_scaled"])
        sd = float(d["s"].std())
        delta = 4 * coef; ci = 4 * 1.96 * se; frac = delta / sd if sd > 0 else np.nan
        fd = fdr[(fdr.subtype == sub) & (fdr.tf == tf)]["p_spline_fdr_full"]
        fd = float(fd.values[0]) if len(fd) else np.nan
        rows.append({"label": f"{D.SUBTYPE_SHORT[sub]} | {tf}", "tf": tf, "subtype": sub,
                     "delta": delta, "lo": delta - ci, "hi": delta + ci, "frac_sd": frac, "fdr": fd,
                     "tier": TIER_OF[(tf, sub)]})
        del mat, acts; gc.collect()
    res = pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)   # 自上而下:naive->...
    res.to_csv(D.SRC / "forest_40to80_tf.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    fig.subplots_adjust(right=0.62)
    y = np.arange(len(res))
    for i, r in enumerate(res.itertuples()):
        ax.errorbar(r.delta, i, xerr=[[r.delta - r.lo], [r.hi - r.delta]], fmt="none",
                    ecolor=D.MUTED, capsize=3, alpha=0.8, zorder=1)
        ax.scatter(r.delta, i, s=140, c=TIER_COLOR[r.tier], edgecolors="black",
                   linewidth=1.3, zorder=3, marker="s" if r.tier == "strict" else "o")
        ax.annotate(f"Δ={r.delta:+.3f}  ({r.frac_sd:+.2f} sd)\nFDR={r.fdr:.3g}",
                    xy=(1.0, i), xycoords=("axes fraction", "data"),
                    xytext=(8, 0), textcoords="offset points", va="center", ha="left",
                    fontsize=7.6, color=D.INK)
    ax.axvline(0, color=D.MUTED, ls="--", lw=0.8)
    pad = (res["hi"].max() - res["lo"].min()) * 0.10
    ax.set_xlim(res["lo"].min() - pad, res["hi"].max() + pad)
    ax.set_yticks(y); ax.set_yticklabels(res["label"], fontsize=9)
    ax.set_xlabel("predicted TF-activity change, age 40→80 (ULM units)")
    ax.set_title("Core TF effects: small but tiered", fontsize=10.5)
    leg = [Patch(facecolor=TIER_COLOR["strict"], edgecolor="black", label="strict FDR<0.05"),
           Patch(facecolor=TIER_COLOR["convergent"], edgecolor="black", label="convergent (FDR NS)")]
    ax.legend(handles=leg, loc="upper left", fontsize=8, frameon=False, bbox_to_anchor=(0.0, 1.0))
    D.style_ax(ax); _save(fig, "19_forest_40to80_tf")


# ============ 2) TF 曲线 = 相对 40 岁基线变化 ============
def tf_curve_delta(subtype, tf):
    ct, _ = _nets()
    import decoupler as dc
    mat, obs = _lognorm_pb(subtype)
    acts = dc.run_ulm(mat, ct, min_n=5, use_raw=False)[0]
    if tf not in acts.columns:
        del mat, acts; return
    d = pd.DataFrame({"s": acts[tf].values}, index=obs.index).join(obs[["age"]]).dropna()
    ms = smf.ols("s ~ bs(age, df=4)", data=d).fit()
    grid = pd.DataFrame({"age": np.linspace(40, 95, 90)})
    pred = ms.get_prediction(grid).predicted_mean
    base = float(ms.get_prediction(pd.DataFrame({"age": [40]})).predicted_mean[0])
    delta = pred - base
    ci = ms.get_prediction(grid).conf_int(alpha=0.05)
    dlo = ci[:, 0] - base; dhi = ci[:, 1] - base
    fdr = pd.read_csv(T / "tf_age_full_fdr.csv")
    fd = fdr[(fdr.subtype == subtype) & (fdr.tf == tf)]["p_spline_fdr_full"]
    fd = float(fd.values[0]) if len(fd) else np.nan
    tier = TIER_OF.get((tf, subtype), "exploratory")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.axhline(0, color=D.MUTED, lw=0.7)
    ax.fill_between(grid["age"], dlo, dhi, color=D.SUBTYPE_COLOR[subtype], alpha=0.16)
    ax.plot(grid["age"], delta, color=D.SUBTYPE_COLOR[subtype], lw=2.3)
    for ag in [60, 80]:
        v = float(ms.get_prediction(pd.DataFrame({"age": [ag]})).predicted_mean[0]) - base
        ax.scatter([ag], [v], s=35, color=D.SUBTYPE_COLOR[subtype], zorder=4)
        ax.annotate(f"{v:+.2f}", (ag, v), fontsize=7.5, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("age"); ax.set_ylabel(f"Δ {tf} activity vs age 40")
    ax.set_title(f"{D.SUBTYPE_SHORT[subtype]} | {tf}  [{tier}]", fontsize=10.5)
    ax.text(0.03, 0.95, f"FDR = {fd:.3g}", transform=ax.transAxes, va="top", fontsize=8.5, color=D.INK2,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=D.GRID, alpha=0.8))
    D.style_ax(ax); del mat, acts; gc.collect()
    _save(fig, f"20_delta_{tf}_{D.SUBTYPE_SHORT[subtype]}")


# ============ 3) 方向收敛热图(按通路分组的基因年龄效应) ============
def direction_heatmap():
    hf = pd.read_csv(T / "gsea" / "hallmark_focus_pathways.csv")
    groups = {"NF-κB": "TNF-alpha Signaling via NF-kB", "IFN-γ": "Interferon Gamma Response",
              "IFN-α": "Interferon Alpha Response", "MYC": "Myc Targets V1", "E2F": "E2F Targets"}
    subs = ["B naive", "B memory", "B intermediate"]
    rows, row_meta = [], []
    seen = set()
    for gname, term in groups.items():
        sub_rows = hf[hf["Term"] == term]
        if len(sub_rows) == 0:
            continue
        lead = []
        for lg in sub_rows["Lead_genes"].dropna():
            lead += str(lg).replace(";", ",").split(",")
        lead = [g.strip() for g in lead if g.strip()]
        # 取该通路 leading-edge 基因在各亚型的 log2FC/10yr
        for s in subs:
            d = pd.read_csv(T / "deg" / f"{s.replace(' ', '')}_deg_continuous.csv")
            d = d[~d["is_IG"]].set_index("symbol")
            common = [g for g in lead if g in d.index and g not in seen]
            for g in common[:8]:   # 每通路每亚型最多 8
                rows.append({"gene": g, "pathway": gname, "subtype": s, "lfc": float(d.loc[g, "log2FC_per10yr"])})
                seen.add(g)
    df = pd.DataFrame(rows)
    if df.empty:
        return
    # 透视:行=gene(按通路排序),列=subtype;用各亚型各自的 lfc(同一基因在 3 亚型行)
    # 结构:每基因一行,3 列(naive/memory/intermediate 的 lfc)
    pv = df.pivot_table(index=["pathway", "gene"], columns="subtype", values="lfc")
    pv = pv.reindex(columns=subs)
    order = ["NF-κB", "IFN-γ", "IFN-α", "MYC", "E2F"]
    pv = pv.reindex([p for p in order if p in pv.index.get_level_values(0)], level=0)
    fig, ax = plt.subplots(figsize=(7.5, 11))
    fig.subplots_adjust(left=0.38, right=0.82)
    im = ax.imshow(pv.values, cmap=D.DIVERGE_CMAP, vmin=-0.12, vmax=0.12, aspect="auto")
    ax.set_yticks(range(len(pv.index)))
    ax.set_yticklabels([f"{g}" for (_, g) in pv.index], fontsize=7.5)
    ax.set_xticks(range(3)); ax.set_xticklabels([D.SUBTYPE_SHORT[s] for s in subs], fontsize=9)
    # 通路分隔线 + 通路标签(竖排在左外侧)
    paths_list = pv.index.get_level_values(0).tolist()
    block_ranges = {}
    for i, p in enumerate(paths_list):
        if p not in block_ranges:
            block_ranges[p] = [i, i]
        else:
            block_ranges[p][1] = i
    for p, (s, e) in block_ranges.items():
        mid = (s + e) / 2
        ax.axhline(s - 0.5, color=D.INK, lw=1.2)
        ax.annotate(p, xy=(0, mid), xycoords=("axes fraction", "data"),
                    xytext=(-160, 0), textcoords="offset points",
                    fontsize=10, fontweight="bold", color=D.INK, ha="center", va="center", rotation=90)
    ax.axhline(len(paths_list) - 0.5, color=D.INK, lw=1.2)
    # colorbar 右移
    cax = fig.add_axes([0.88, 0.25, 0.022, 0.5])
    fig.colorbar(im, cax=cax, label="log2FC / 10 yr")
    ax.set_title("Leading-edge gene age effects by pathway\n(each gene small, but pathway-coordinated)", fontsize=10)
    ax.grid(False)
    _save(fig, "21_direction_convergence_heatmap")


def run():
    forest_40to80()
    for tf, sub in [("FOXO3", "B memory"), ("PAX5", "B intermediate"), ("STAT1", "B naive"), ("NFKB", "B memory")]:
        tf_curve_delta(sub, tf)
    direction_heatmap()
    print("[honest] done.")


if __name__ == "__main__":
    run()
