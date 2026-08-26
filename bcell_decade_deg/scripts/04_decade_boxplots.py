"""04_decade_boxplots.py — per-decade expression "oscillation" for sig DEGs.

For the union of significant young-vs-old DEGs (primary model), show log2-CPM
across the 20s..90s decades: top-gene boxplots, a gene-trajectory spaghetti
(oscillation), and a z-scored mean heatmap. Done at all-B level (the user's
focus) and as a per-subtype trajectory panel.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import expr as E
from plotting import set_style, save, decade_color
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

log = C.get_logger("04_decade", C.LOGS_DIR / f"04_decade_{C.timestamp()}.log")


def per_decade_tables(Y: np.ndarray, meta: pd.DataFrame, gene_names, decades_present):
    """Y: donors x genes log2CPM. Return long table of per-decade summary stats."""
    didx = E.decade_index(meta["age"].values)
    rows = []
    for di, dlab in enumerate(decades_present):
        m = didx == di
        if m.sum() < C.DECADE_MIN_DONORS:
            continue
        for gi, g in enumerate(gene_names):
            v = Y[m, gi]
            rows.append({"decade": dlab, "n": int(m.sum()), "gene": g,
                         "mean": float(v.mean()), "median": float(np.median(v)),
                         "sd": float(v.std(ddof=1)), "iqr": float(np.subtract(*np.percentile(v, [75, 25])))})
    return pd.DataFrame(rows)


def topgene_boxplots(unit, Y, meta, gene_syms, deg, decades_present):
    """4x3 grid of per-gene boxplots across decades for the top |stat| genes."""
    order = deg.reindex(gene_syms)["abs_stat"].sort_values(ascending=False).index[:12]
    fig, axs = plt.subplots(4, 3, figsize=(11.5, 12.0), sharex=True)
    didx = E.decade_index(meta["age"].values)
    pos = np.arange(len(decades_present))
    cols = [decade_color(d) for d in decades_present]
    for k, g in enumerate(order):
        ax = axs.flat[k]
        gi = list(gene_syms).index(g)
        data = [Y[(didx == di), gi] for di in range(len(decades_present))]
        bp = ax.boxplot(data, positions=pos, widths=0.7, patch_artist=True,
                        showfliers=False, medianprops=dict(color="white", lw=1.3))
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c); patch.set_edgecolor(C.BASELINE); patch.set_alpha(0.9)
        ax.set_xticks(pos); ax.set_xticklabels(decades_present, fontsize=8)
        tag = "↑old" if deg.loc[g, "lfc"] > 0 else "↓old"
        ax.set_title(f"{g}  {tag} (lfc {deg.loc[g,'lfc']:+.1f})", fontsize=9.5)
        ax.set_ylabel("log2-CPM", fontsize=8)
        ax.tick_params(labelsize=8)
    for k in range(len(order), 12):
        axs.flat[k].axis("off")
    fig.suptitle(f"{unit}: significant DEG expression across decades (young→old)",
                 y=0.995, fontweight="semibold", fontsize=13)
    save(fig, C.DEC_F / f"{E.SLUG[unit]}_deg_decade_boxplots")


def trajectory_spaghetti(unit, Y, meta, gene_syms, decades_present):
    """Each gene's per-decade mean (z-scored across decades); bold = group median."""
    didx = E.decade_index(meta["age"].values)
    M = np.full((len(gene_syms), len(decades_present)), np.nan)
    for di in range(len(decades_present)):
        m = didx == di
        if m.sum() < C.DECADE_MIN_DONORS:
            continue
        M[:, di] = Y[m, :].mean(axis=0)
    Z = (M - M.mean(axis=1, keepdims=True)) / (M.std(axis=1, ddof=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(decades_present))
    for gi in range(len(gene_syms)):
        ax.plot(x, Z[gi], color=C.COLOR_YOUNG, alpha=0.18, lw=0.9)
    grp = np.nanmedian(Z, axis=0)
    ax.plot(x, grp, color=C.COLOR_OLD, lw=2.6, marker="o", label="median across genes")
    ax.axhline(0, color=C.MUTED, lw=0.7, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(decades_present)
    ax.set_xlabel("Decade (young → old)"); ax.set_ylabel("per-decade mean (z-scored)")
    ax.set_title(f"{unit}: sig-DEG expression trajectory (n={len(gene_syms)} genes)\n"
                 f"thin = each gene; bold = median")
    ax.legend(loc="upper left")
    save(fig, C.DEC_F / f"{E.SLUG[unit]}_deg_trajectory_spaghetti")


def trajectory_heatmap(unit, Y, meta, gene_syms, deg, decades_present):
    didx = E.decade_index(meta["age"].values)
    M = np.full((len(gene_syms), len(decades_present)), np.nan)
    for di in range(len(decades_present)):
        m = didx == di
        if m.sum() < C.DECADE_MIN_DONORS:
            continue
        M[:, di] = Y[m, :].mean(axis=0)
    Z = (M - M.mean(axis=1, keepdims=True)) / (M.std(axis=1, ddof=1, keepdims=True) + 1e-9)
    # sort genes by linear trend (corr with decade index) so monotonic genes cluster
    trend = np.array([np.corrcoef(np.arange(M.shape[1]), Z[i])[0, 1] for i in range(len(gene_syms))])
    order = np.argsort(-trend)
    Zs = Z[order]; labs = np.array(gene_syms)[order]

    fig, ax = plt.subplots(figsize=(5.4, max(4.5, 0.28 * len(labs) + 1)))
    im = ax.imshow(Zs, aspect="auto", cmap="RdBu_r", vmin=-1.5, vmax=1.5, interpolation="nearest")
    ax.set_xticks(range(len(decades_present))); ax.set_xticklabels(decades_present)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=7.5)
    ax.set_xlabel("Decade (young → old)")
    ax.set_title(f"{unit}: sig-DEG mean expression (z-scored)\nsorted by age trend")
    fig.colorbar(im, ax=ax, label="z(mean log2-CPM)", shrink=0.5)
    save(fig, C.DEC_F / f"{E.SLUG[unit]}_deg_trajectory_heatmap")


def run_unit(unit):
    X, meta, genes = E.load_counts(unit)
    deg_all = E.sig_deg_ensg(models=("primary",))
    deg_u = deg_all[deg_all["unit"] == unit].copy()
    if unit == "allB":   # all-B draws on the union across subtypes (global view)
        deg_u = deg_all.drop_duplicates("ensg").copy()
    if len(deg_u) == 0:
        log.info(f"{unit}: 0 sig DEGs -> skip"); return
    # one row per unique symbol/ensg (prefer max |stat|)
    deg_u["abs_stat"] = deg_u["stat"].abs()
    deg_u = deg_u.sort_values("abs_stat", ascending=False).drop_duplicates("symbol")
    ens = deg_u["ensg"].astype(str).tolist()
    gidx = np.array([np.where(genes == e)[0][0] for e in ens])
    Y = E.log2cpm_subset(X, gidx)
    syms = deg_u.set_index("symbol")  # index = symbol
    deg_sym = pd.DataFrame({"lfc": deg_u.set_index("symbol")["log2FoldChange"],
                            "abs_stat": deg_u.set_index("symbol")["abs_stat"]})
    decades_present = [d for d in E.DECADE_ORDER
                       if ((E.decade_index(meta["age"].values) == E.DECADE_ORDER.index(d)).sum() >= C.DECADE_MIN_DONORS)]

    log.info(f"{unit}: {len(deg_sym)} sig DEGs, plotting across {len(decades_present)} decades")
    per_decade_tables(Y, meta, deg_sym.index.tolist(), E.DECADE_ORDER).to_csv(
        C.DEC_R / f"{E.SLUG[unit]}_deg_decade_summary.csv", index=False)
    topgene_boxplots(unit, Y, meta, deg_sym.index.tolist(), deg_sym, decades_present)
    trajectory_spaghetti(unit, Y, meta, deg_sym.index.tolist(), decades_present)
    trajectory_heatmap(unit, Y, meta, deg_sym.index.tolist(), deg_sym, decades_present)


def main():
    C.ensure_dirs(); set_style()
    run_unit("allB")              # main: all B cells, union of sig DEGs
    for u in C.B_SUBTYPES:        # per-subtype trajectory heatmaps where sig exists
        run_unit(u)
    log.info("DONE 04_decade")


if __name__ == "__main__":
    main()
