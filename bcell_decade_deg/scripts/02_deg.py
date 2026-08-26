"""02_deg.py — young(20-30) vs old(80-90) DEG, all-B + each B subtype.

PyDESeq2 0.4.12 on per-donor pseudobulk counts.
  Primary:     ~ group            (old vs young)            -- the requested contrast
  Sensitivity: ~ pool + group     (mixed-pool donors only)  -- within-pool, batch-robust

Positive log2FC = higher in OLD. GSEA rank = Wald `stat` (saved per unit for 03_gsea).
Genes are ENSG; we map to symbol (from raw h5ad feature_name) for tables/labels/GSEA.
Inflation lambda reported per fit (median stat^2 / 0.4549).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from plotting import set_style, save
import matplotlib.pyplot as plt

log = C.get_logger("02_deg", C.LOGS_DIR / f"02_deg_{C.timestamp()}.log")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
try:
    from adjustText import adjust_text as _adjust_text
except Exception:
    _adjust_text = None

MINCELL = {
    "allB": C.ALLB_MIN_CELLS,
    "B naive": C.MAIN_SUBTYPE_MIN_CELLS,
    "B intermediate": C.MAIN_SUBTYPE_MIN_CELLS,
    "B memory": C.MAIN_SUBTYPE_MIN_CELLS,
    "Plasmablast": C.PLASMABLAST_MIN_CELLS,
}
SLUG = {"allB": "allB", **C.SUBTYPE_SLUG}
TITLE = {"allB": "all B cells", "B naive": "B naive", "B intermediate": "B intermediate",
         "B memory": "B memory", "Plasmablast": "Plasmablast"}
SYM: dict[str, str] = {}   # ensg -> symbol, populated in main()
MODEL_STATUS: list[dict] = []


def load_unit(name: str):
    if name == "allB":
        X = sp.load_npz(C.QC_R / "allb_counts.npz").tocsr()
        meta = pd.read_csv(C.QC_R / "allb_donor_meta.csv")
    else:
        slug = C.SUBTYPE_SLUG[name]
        X = sp.load_npz(C.QC_R / "persub" / f"{slug}_counts.npz").tocsr()
        meta = pd.read_csv(C.QC_R / "persub" / f"{slug}_meta.csv")
    genes = pd.read_csv(C.QC_R / "genes.csv")["gene"].astype(str).values
    assert X.shape[1] == len(genes)
    meta = meta.copy(); meta["donor_id"] = meta["donor_id"].astype(str)
    return X, meta, genes


def filter_genes(X: sp.csr_matrix, min_count: int, min_samples: int) -> np.ndarray:
    return np.asarray((X >= min_count).sum(axis=0) >= min_samples).ravel()


def run_deseq2(counts: pd.DataFrame, meta: pd.DataFrame, design: list[str],
               min_samples: int, tag: str):
    """Fit PyDESeq2; return results_df with added 'symbol' col. None on failure."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    keep = filter_genes(sp.csr_matrix(counts.values), C.MIN_COUNT, min_samples)
    cnt = counts.loc[:, keep]
    log.info(f"  [{tag}] design={design}, matrix={cnt.shape}")
    try:
        dds = DeseqDataSet(counts=cnt, metadata=meta, design_factors=design,
                           ref_level=["group", "young"], quiet=True)
        dds.deseq2()
        ds = DeseqStats(dds, contrast=["group", "old", "young"], alpha=0.05, quiet=True)
        ds.summary()
        res = ds.results_df.copy()
    except Exception as e:
        log.warning(f"  [{tag}] PyDESeq2 failed: {type(e).__name__}: {e}")
        raise
    res["symbol"] = [SYM.get(g, g) for g in res.index]
    stat = res["stat"].replace([np.inf, -np.inf], np.nan)
    res.attrs["lambda"] = float(np.nanmedian(stat ** 2) / 0.4549) if stat.notna().any() else np.nan
    sig = (res["padj"] < C.PADJ_THR) & (res["log2FoldChange"].abs() > C.LFC_THR)
    log.info(f"  [{tag}] genes={len(res)}, sig(↑old/↓old)="
             f"{int((sig&(res.log2FoldChange>0)).sum())}/{int((sig&(res.log2FoldChange<0)).sum())}, "
             f"lambda={res.attrs['lambda']:.2f}")
    return res


def deg_unit(name: str):
    log.info(f"==== {name} ====")
    X, meta, genes = load_unit(name)
    mc = MINCELL[name]
    mask = (meta["n_cells"] >= mc) & meta["group"].isin(["young", "old"])
    yo = meta.loc[mask].copy()
    Xyo = X[mask.values]
    yo["group"] = pd.Categorical(yo["group"], categories=["young", "old"])
    yo["sex"], yo["pool_number"] = yo["sex"].astype(str), yo["pool_number"].astype(int)
    log.info(f"{name}: {len(yo)} donors (young {int((yo.group=='young').sum())}, "
             f"old {int((yo.group=='old').sum())}), mincell>={mc}")
    MODEL_STATUS.append({"unit": name, "model": "primary", "status": "run",
                         "min_cells": mc, "n_donors": len(yo),
                         "n_young": int((yo.group == "young").sum()),
                         "n_old": int((yo.group == "old").sum()),
                         "n_mixed_pools": np.nan})

    cnt = pd.DataFrame(np.asarray(Xyo.todense()).astype(np.int64),
                       index=yo["donor_id"].values, columns=genes)
    md = yo.set_index("donor_id")[["group", "sex", "pool_number"]]

    res_pri = run_deseq2(cnt, md, ["group"], C.MIN_SAMPLES, "primary")

    res_pool = None
    pool_grp = yo.groupby("pool_number")["group"].agg(lambda s: {"young", "old"} <= set(s))
    mixed = pool_grp[pool_grp].index
    n_mixed = int(yo["pool_number"].isin(mixed).sum())
    if n_mixed >= C.MIN_MIXED_POOL_DONORS:
        km = yo["pool_number"].isin(mixed).values
        MODEL_STATUS.append({"unit": name, "model": "within_pool", "status": "run",
                             "min_cells": mc, "n_donors": n_mixed,
                             "n_young": int(((yo.group == "young") & yo.pool_number.isin(mixed)).sum()),
                             "n_old": int(((yo.group == "old") & yo.pool_number.isin(mixed)).sum()),
                             "n_mixed_pools": len(mixed)})
        res_pool = run_deseq2(cnt.loc[km], md.loc[km], ["pool_number", "group"],
                              max(5, C.MIN_SAMPLES // 2), "within-pool")
    else:
        MODEL_STATUS.append({"unit": name, "model": "within_pool", "status": "skipped",
                             "min_cells": mc, "n_donors": n_mixed,
                             "n_young": int(((yo.group == "young") & yo.pool_number.isin(mixed)).sum()),
                             "n_old": int(((yo.group == "old") & yo.pool_number.isin(mixed)).sum()),
                             "n_mixed_pools": len(mixed)})
        log.info(f"  mixed-pool donors={n_mixed} <{C.MIN_MIXED_POOL_DONORS} "
                 "-> skip within-pool")

    slug = SLUG[name]
    for tag, res in [("primary", res_pri), ("within_pool", res_pool)]:
        if res is None:
            continue
        res.index.name = "ensg"
        res.to_csv(C.DEG_R / f"{slug}_deg_{tag}.csv")
    if res_pri is not None:   # rank file for GSEA (primary)
        rk = res_pri.dropna(subset=["stat"]).reset_index().rename(columns={"index": "ensg"})
        rk[["ensg", "symbol", "log2FoldChange", "stat", "pvalue", "padj"]].to_csv(
            C.DEG_R / f"{slug}_rank.csv", index=False)
    volcano(name, res_pri)
    return res_pri, res_pool


def volcano(name: str, res: pd.DataFrame | None):
    if res is None:
        return
    res = res.dropna(subset=["log2FoldChange", "padj"]).copy()
    res["nlp"] = -np.log10(res["padj"].clip(lower=1e-300))
    sig = (res["padj"] < C.PADJ_THR) & (res["log2FoldChange"].abs() > C.LFC_THR)
    up, dn = sig & (res["log2FoldChange"] > 0), sig & (res["log2FoldChange"] < 0)

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ax.scatter(res.loc[~sig, "log2FoldChange"], res.loc[~sig, "nlp"],
               s=6, color=C.MUTED, alpha=0.35, edgecolors="none", label="ns")
    ax.scatter(res.loc[dn, "log2FoldChange"], res.loc[dn, "nlp"],
               s=14, color=C.COLOR_YOUNG, alpha=0.85, edgecolors="none", label="higher in young")
    ax.scatter(res.loc[up, "log2FoldChange"], res.loc[up, "nlp"],
               s=14, color=C.COLOR_OLD, alpha=0.85, edgecolors="none", label="higher in old")
    top = pd.concat([res[up].nlargest(8, "nlp"), res[dn].nlargest(8, "nlp")])
    if len(top):
        texts = [ax.text(r["log2FoldChange"], r["nlp"], str(r["symbol"]),
                         fontsize=7.5, color=C.INK, ha="left", va="center")
                 for _, r in top.iterrows()]
        if _adjust_text is not None:
            _adjust_text(texts, ax=ax, only_move={"text": "xy", "static": "xy"},
                         expand=(1.3, 1.6), force_text=(0.6, 0.9), force_points=(0.4, 0.5),
                         arrowprops=dict(arrowstyle="-", color=C.MUTED, lw=0.5, shrinkA=2, shrinkB=5))
        else:   # fallback: stagger by index
            for i, t in enumerate(texts):
                t.set_position((t.get_position()[0] + 0.05 * (i % 3),
                                t.get_position()[1] + 0.4 * (i - 3)))
    ax.axhline(-np.log10(C.PADJ_THR), color=C.BASELINE, lw=0.7, ls="--")
    ax.axvline(C.LFC_THR, color=C.BASELINE, lw=0.7, ls="--")
    ax.axvline(-C.LFC_THR, color=C.BASELINE, lw=0.7, ls="--")
    ax.set_xlabel("log2 fold change  (old / young)"); ax.set_ylabel("-log10 padj")
    ax.set_title(f"{TITLE[name]}: young(20–30) vs old(80–90)  λ={res.attrs.get('lambda', float('nan')):.2f}")
    ax.legend(loc="upper right", markerscale=1.2)
    save(fig, C.DEG_F / f"{SLUG[name]}_volcano")


def main() -> None:
    global SYM
    C.ensure_dirs(); set_style()
    gm = pd.read_csv(C.QC_R / "gene_map_ensg_symbol.csv")
    SYM = dict(zip(gm["ensg"].astype(str), gm["symbol"].astype(str)))
    log.info(f"loaded {len(SYM)} ensg->symbol mappings")

    summary = []
    for u in ["allB"] + C.B_SUBTYPES:
        rp, rw = deg_unit(u)
        for tag, res in [("primary", rp), ("within_pool", rw)]:
            if res is None:
                continue
            sig = (res["padj"] < C.PADJ_THR) & (res["log2FoldChange"].abs() > C.LFC_THR)
            summary.append({"unit": u, "model": tag, "n_tested": len(res),
                            "min_cells": MINCELL[u],
                            "analysis_tier": "exploratory" if u == "Plasmablast" else "main",
                            "n_sig": int(sig.sum()),
                            "n_up_old": int((sig & (res["log2FoldChange"] > 0)).sum()),
                            "n_dn_old": int((sig & (res["log2FoldChange"] < 0)).sum()),
                            "lambda": round(res.attrs.get("lambda", np.nan), 3)})
    pd.DataFrame(summary).to_csv(C.DEG_R / "deg_summary.csv", index=False)
    pd.DataFrame(MODEL_STATUS).to_csv(C.DEG_R / "model_status.csv", index=False)
    if not summary:
        raise RuntimeError("No DEG model completed; refusing to continue with empty results")
    log.info(f"DEG summary:\n{pd.DataFrame(summary).to_string(index=False)}")
    log.info("DONE 02_deg")


if __name__ == "__main__":
    main()
