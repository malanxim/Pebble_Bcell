"""plots.py —— DEG 与富集相关图(组成图在 composition.py 内)。

火山/MA/热图/重点基因年龄趋势/跨亚型一致性/GO dotplot/GSEA NES 热图,figstyle。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C
from . import figstyle
figstyle.set_style()

UP, DOWN, NS = figstyle.ORANGE, figstyle.BLUE, "#c8c8c4"


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(C.FIG_DIR / f"deg_{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] -> figures/deg_{name}.{{pdf,png}}")


def _sig_mask(deg):
    return (deg["padj"] < C.DEG_PADJ) & (deg["log2FC_per10yr"].abs() > C.DEG_LFC)


def volcano(deg, subtype):
    d = deg.dropna(subset=["log2FC_per10yr", "padj"]).copy()
    d["nlogp"] = -np.log10(d["padj"].clip(lower=1e-300))
    sig = _sig_mask(d)
    up = sig & (d["log2FC_per10yr"] > 0); dn = sig & (d["log2FC_per10yr"] < 0)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(d.loc[~sig, "log2FC_per10yr"], d.loc[~sig, "nlogp"], s=4, alpha=0.25, color=NS, rasterized=True)
    ax.scatter(d.loc[dn, "log2FC_per10yr"], d.loc[dn, "nlogp"], s=6, alpha=0.5, color=DOWN, label=f"down ({dn.sum()})", rasterized=True)
    ax.scatter(d.loc[up, "log2FC_per10yr"], d.loc[up, "nlogp"], s=6, alpha=0.5, color=UP, label=f"up ({up.sum()})", rasterized=True)
    # 标注:重点基因 + top 显著
    lab = d[d["symbol"].isin(C.HIGHLIGHT_GENES) & sig].copy()
    top = d.sort_values("padj").head(12)
    lab = pd.concat([lab, top]).drop_duplicates("symbol")
    for _, r in lab.iterrows():
        if isinstance(r["symbol"], str):
            ax.scatter(r["log2FC_per10yr"], r["nlogp"], s=12, facecolor="none", edgecolor="k", lw=0.5, zorder=5)
            ax.annotate(r["symbol"], (r["log2FC_per10yr"], r["nlogp"]), fontsize=6.5,
                        xytext=(3, 3), textcoords="offset points")
    ax.axhline(-np.log10(C.DEG_PADJ), color=figstyle.MUTED, ls=":", lw=0.8)
    ax.axvline(0, color=figstyle.MUTED, lw=0.6)
    ax.set_xlabel("log2FC per 10 yr (age)"); ax.set_ylabel("-log10(padj)")
    ax.set_title(f"{subtype}: age-related DEG (n_donors={int(d['n_donors'].iloc[0])})")
    ax.legend(fontsize=8); figstyle.thin_despine(ax); _save(fig, f"{subtype.replace(' ','')}_volcano")


def ma_plot(deg, subtype):
    d = deg.dropna(subset=["log2FC_per10yr", "baseMean"]).copy()
    d["lbase"] = np.log10(d["baseMean"].clip(lower=1e-3))
    sig = _sig_mask(d)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(d.loc[~sig, "lbase"], d.loc[~sig, "log2FC_per10yr"], s=4, alpha=0.25, color=NS, rasterized=True)
    ax.scatter(d.loc[sig, "lbase"], d.loc[sig, "log2FC_per10yr"], s=6, alpha=0.5,
               c=np.where(d.loc[sig, "log2FC_per10yr"] > 0, UP, DOWN), rasterized=True)
    ax.axhline(0, color=figstyle.MUTED, lw=0.6)
    ax.set_xlabel("log10 baseMean"); ax.set_ylabel("log2FC per 10 yr")
    ax.set_title(f"{subtype}: MA plot"); figstyle.thin_despine(ax); _save(fig, f"{subtype.replace(' ','')}_ma")


def deg_heatmap(deg, pb, subtype, n=15):
    sig = deg[_sig_mask(deg)].copy()
    if len(sig) == 0:
        print(f"[plots] {subtype} 无显著 DEG,跳过热图"); return
    top = pd.concat([sig.sort_values("log2FC_per10yr").head(n),
                     sig.sort_values("log2FC_per10yr", ascending=False).head(n)])
    genes = top["ensembl"].tolist()
    avail = [g for g in genes if g in pb.var_names]
    if not avail:
        return
    X = pb[:, avail].X
    X = np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X)
    lib = X.sum(axis=1); lcpm = np.log1p(X / np.where(lib > 0, lib, 1)[:, None] * 1e6)
    z = (lcpm - lcpm.mean(axis=0)) / np.where(lcpm.std(axis=0) > 0, lcpm.std(axis=0), 1)
    order = np.argsort(pb.obs["age"].values)
    z = z[order, :]; sym = pb.var.loc[avail, "symbol"].tolist()
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(avail) + 1))
    ax.imshow(z.T, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_yticks(range(len(sym))); ax.set_yticklabels(sym, fontsize=7)
    ax.set_xticks([]); ax.set_xlabel(f"donors sorted by age (n={pb.n_obs})")
    ax.set_title(f"{subtype}: top ±{n} age-DEG"); ax.grid(False); _save(fig, f"{subtype.replace(' ','')}_heatmap")


def keygene_age_trend(deg, pb, subtype):
    want = [g for g in C.HIGHLIGHT_GENES if g in set(pb.var["symbol"])]
    want = want[:12]
    if not want:
        return
    sym2e = pb.var.set_index("symbol")["ensembl"].to_dict()
    age = pb.obs["age"].values
    ncol = min(4, len(want)); nrow = int(np.ceil(len(want) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2.6 * nrow), sharex=True)
    axes = np.array(axes).reshape(-1)
    for k, g in enumerate(want):
        ax = axes[k]; e = sym2e.get(g)
        if e is None or e not in pb.var_names:
            ax.axis("off"); continue
        x = pb[:, e].X; x = np.asarray(x.todense()).ravel() if sp.issparse(x) else np.asarray(x).ravel()
        lib = pb.X.sum(axis=1); lib = np.asarray(lib.todense()).ravel() if sp.issparse(lib) else np.asarray(lib).ravel()
        cpm = x / np.where(lib > 0, lib, 1) * 1e6
        lv = np.log1p(cpm)
        ax.scatter(age, lv, s=5, alpha=0.35, color=figstyle.BLUE)
        srt = np.argsort(age)
        ax.plot(age[srt], pd.Series(lv[srt]).rolling(60, min_periods=10, center=True).mean().values,
                color=figstyle.ORANGE, lw=1.2)
        ax.set_title(g, fontsize=9); figstyle.thin_despine(ax)
        ax.set_ylabel("log1p(CPM)" if k % ncol == 0 else "")
        if k >= len(want) - ncol: ax.set_xlabel("age")
    for k in range(len(want), len(axes)):
        axes[k].axis("off")
    fig.suptitle(f"{subtype}: key gene pseudobulk expr vs age", fontsize=11); fig.tight_layout()
    _save(fig, f"{subtype.replace(' ','')}_keygene_age")


def cross_subtype(deg_dict):
    subs = [s for s in C.MAIN_SUBTYPES if s in deg_dict]
    if len(subs) < 2:
        return
    # LFC 相关(成对)
    base = {s: deg_dict[s].set_index("ensembl")["log2FC_per10yr"] for s in subs}
    pairs = [(subs[i], subs[j]) for i in range(len(subs)) for j in range(i + 1, len(subs))]
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.5 * len(pairs), 3.5))
    axes = np.array(axes).reshape(-1)
    for k, (a, b) in enumerate(pairs):
        ax = axes[k]; df = base[a].to_frame("a").join(base[b].rename("b"), how="inner")
        ax.scatter(df["a"], df["b"], s=3, alpha=0.2, color=figstyle.AQUA, rasterized=True)
        r = df.corr().iloc[0, 1]
        ax.set_xlabel(f"{a}\nlog2FC/10yr"); ax.set_ylabel(f"{b}\nlog2FC/10yr")
        ax.set_title(f"ρ={r:.2f}"); ax.axhline(0, color=figstyle.MUTED, lw=0.5); ax.axvline(0, color=figstyle.MUTED, lw=0.5)
        figstyle.thin_despine(ax)
    fig.suptitle("Age-effect (log2FC/10yr) correlation across subtypes"); fig.tight_layout(); _save(fig, "cross_subtype_lfc_corr")

    # 共同/特异 显著 DEG 计数
    sigs = {s: set(deg_dict[s].loc[_sig_mask(deg_dict[s]), "symbol"].dropna()) for s in subs}
    from itertools import combinations
    rows = []
    for s in subs:
        others = set().union(*[sigs[o] for o in subs if o != s])
        rows.append({"category": f"{s} only", "n": len(sigs[s] - others)})
    common = set.intersection(*[sigs[s] for s in subs])
    rows.append({"category": "shared (all)", "n": len(common)})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(df["category"], df["n"], color=figstyle.BLUE); ax.set_xlabel("# sig DEG (padj<.05, |LFC|>0.1)")
    ax.set_title("Shared vs subtype-specific age DEG"); figstyle.thin_despine(ax); _save(fig, "deg_shared_specific")


def consistency(deg_dir_files):
    """连续 vs young/old vs Q4/Q1 方向一致性(top 基因)。deg_dir_files: {subtype: (cont_df, yo_df, q_df)}"""
    rows = []
    for s, (c, yo, q) in deg_dir_files.items():
        if yo is None or q is None:
            continue
        m = c.set_index("ensembl"); yo2 = yo.set_index("ensembl"); q2 = q.set_index("ensembl")
        common = m.index.intersection(yo2.index).intersection(q2.index)
        a, b, cc = m.loc[common, "log2FC_per10yr"], yo2.loc[common].get("log2FC_old_vs_young"), q2.loc[common].get("log2FC_Q4_vs_Q1")
        if b is None or cc is None:
            continue
        rows.append({"subtype": s, "n": len(common),
                     "cont~youngold_rho": round(np.corrcoef(a, b)[0, 1], 3),
                     "cont~Q4Q1_rho": round(np.corrcoef(a, cc)[0, 1], 3)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.to_csv(C.DEG_DIR / "age_scheme_consistency.csv", index=False)
    print(f"[plots] 年龄方案一致性 -> deg/age_scheme_consistency.csv\n{df.to_string(index=False)}")
    return df


def ora_dotplot(ora_df, subtype):
    if ora_df is None or len(ora_df) == 0 or "Term" not in ora_df.columns:
        return
    col_p = "P-value" if "P-value" in ora_df.columns else ora_df.columns[2]
    fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    for k, d in enumerate(["up", "down"]):
        ax = axes[k]
        sub = ora_df[ora_df["direction"] == d].copy()
        bp = sub["library"].str.contains("Biological_Process")
        sub = sub[bp] if bp.any() else sub
        sub = sub.sort_values(col_p).head(8).iloc[::-1]
        if len(sub) == 0:
            ax.set_title(f"{d}: no terms"); continue
        terms = sub["Term"].astype(str).str.split("__").str[-1].values
        ax.barh(range(len(sub)), -np.log10(sub[col_p].clip(lower=1e-20)),
                color=UP if d == "up" else DOWN)
        ax.set_yticks(range(len(sub))); ax.set_yticklabels([t[:50] for t in terms], fontsize=7)
        ax.set_xlabel("-log10(P)"); ax.set_title(f"{d} (top terms)"); figstyle.thin_despine(ax)
    fig.suptitle(f"{subtype}: pathway over-representation", fontsize=11); fig.tight_layout()
    _save(fig, f"{subtype.replace(' ','')}_ora_dotplot")


def gsea_nes_heatmap(gsea_dict):
    subs = [s for s in C.MAIN_SUBTYPES if s in gsea_dict and len(gsea_dict[s])]
    if not subs:
        return
    # 取 Hallmark 条目,跨亚型 NES
    frames = []
    for s in subs:
        g = gsea_dict[s]
        col = "NES" if "NES" in g.columns else None
        if col is None or "Term" not in g.columns:
            continue
        gg = g[g["library"].str.contains("Hallmark")].copy()
        gg["subtype"] = s; frames.append(gg[["Term", col, "subtype"]])
    if not frames:
        return
    allg = pd.concat(frames)
    mat = allg.pivot_table(index="Term", columns="subtype", values=col, aggfunc="first")
    mat = mat.loc[mat.abs().sum(axis=1).sort_values(ascending=False).index[:25]]
    fig, ax = plt.subplots(figsize=(0.4 * mat.shape[1] + 3, 0.3 * mat.shape[0] + 1))
    ax.imshow(mat.values, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels([t.split("__")[-1][:45] for t in mat.index], fontsize=7)
    ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns, rotation=30, ha="right", fontsize=8)
    ax.set_title("GSEA NES (Hallmark, age)"); ax.grid(False); _save(fig, "gsea_nes_heatmap")
