"""expr.py — shared helpers: load pseudobulk, log2-CPM, gene sets, decade order."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import config as C

DECADE_ORDER = [d[2] for d in C.DECADES]               # ['20s',...,'90s']
SLUG = {"allB": "allB", **C.SUBTYPE_SLUG}


def load_counts(unit: str):
    """Return (X csr [n_donor x 27066], meta_df, gene_names)."""
    if unit == "allB":
        X = sp.load_npz(C.QC_R / "allb_counts.npz").tocsr()
        meta = pd.read_csv(C.QC_R / "allb_donor_meta.csv")
    else:
        X = sp.load_npz(C.QC_R / "persub" / f"{C.SUBTYPE_SLUG[unit]}_counts.npz").tocsr()
        meta = pd.read_csv(C.QC_R / "persub" / f"{C.SUBTYPE_SLUG[unit]}_meta.csv")
    genes = pd.read_csv(C.QC_R / "genes.csv")["gene"].astype(str).values
    meta = meta.copy(); meta["donor_id"] = meta["donor_id"].astype(str)
    return X, meta, genes


def log2cpm_subset(X: sp.csr_matrix, gene_idx: np.ndarray) -> np.ndarray:
    """Library-size-normalized log2(CPM+1) for a column subset -> dense (n_donor x |gene_idx|)."""
    Xs = np.asarray(X[:, gene_idx].todense()).astype(np.float64)
    lib = np.asarray(X.sum(axis=1)).ravel()
    lib[lib == 0] = 1.0
    cpm = Xs / lib[:, None] * 1e6
    return np.log2(cpm + 1.0)


def well_expressed_genes(X: sp.csr_matrix, genes: np.ndarray,
                         min_mean_log2cpm: float = C.EXPR_MIN_MEAN_LOG2CPM) -> np.ndarray:
    """Boolean mask over genes: donors' mean log2-CPM >= threshold (for variance analysis)."""
    lib = np.asarray(X.sum(axis=1)).ravel(); lib[lib == 0] = 1.0
    sums = np.asarray(X.sum(axis=0)).ravel()
    means = sums / X.shape[0] / lib.mean() * 1e6
    return np.log2(means + 1.0) >= min_mean_log2cpm


def sig_deg_ensg(models=("primary",), padj=C.PADJ_THR, lfc=C.LFC_THR) -> pd.DataFrame:
    """Union of significant DEGs across units/models. Returns df[unit,ensg,symbol,lfc,stat,padj]."""
    rows = []
    for u in ["allB"] + C.B_SUBTYPES:
        for m in models:
            p = C.DEG_R / f"{SLUG[u]}_deg_{m}.csv"
            if not p.exists():
                continue
            d = pd.read_csv(p)
            s = d[(d["padj"] < padj) & (d["log2FoldChange"].abs() > lfc)].copy()
            s["unit"] = u; s["model"] = m
            rows.append(s[["unit", "model", "ensg", "symbol",
                           "log2FoldChange", "stat", "padj"]])
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["unit", "model", "ensg", "symbol", "log2FoldChange", "stat", "padj"])
    return out


def decade_index(age: np.ndarray) -> np.ndarray:
    """Map age -> decade index (0..7), NaN outside [20,100)."""
    idx = np.full(len(age), np.nan)
    for i, (lo, hi, _) in enumerate(C.DECADES):
        idx[(age >= lo) & (age < hi)] = i
    return idx
