"""03_gsea.py — Hallmark GSEA on the young-vs-old Wald-stat ranking.

For each unit (allB + subtypes) and model (primary, within_pool), build a
symbol-level rank (dedup ENSG->symbol by max |stat|) and run gseapy.prerank
against MSigDB_Hallmark_2020. Output per-unit NES tables + a combined NES
heatmap (primary) and a within-pool heatmap.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from plotting import set_style, save
import matplotlib.pyplot as plt

log = C.get_logger("03_gsea", C.LOGS_DIR / f"03_gsea_{C.timestamp()}.log")
warnings.filterwarnings("ignore")

SLUG = {"allB": "allB", **C.SUBTYPE_SLUG}
ORDER = ["allB"] + C.B_SUBTYPES
PERM = 1000


def symbol_rank(deg: pd.DataFrame) -> pd.Series:
    """ENSG-level stat -> symbol-level, dedup by max |stat|, sorted desc."""
    d = deg.dropna(subset=["stat"]).copy()
    d["abs"] = d["stat"].abs()
    d = (d.sort_values("abs", ascending=False)
           .drop_duplicates("symbol")
           .set_index("symbol")["stat"]
           .sort_values(ascending=False))
    return d[~d.index.duplicated(keep="first")]


def run_unit(slug: str, model: str) -> pd.DataFrame | None:
    path = C.DEG_R / f"{slug}_deg_{model}.csv"
    if not path.exists():
        return None
    deg = pd.read_csv(path)
    rnk = symbol_rank(deg)
    if len(rnk) < 20:
        log.warning(f"{slug}/{model}: rank too short ({len(rnk)})")
        return None
    log.info(f"[{slug}/{model}] prerank on {len(rnk)} ranked symbols")
    import gseapy as gp
    gene_sets = str(C.HALLMARK_GMT) if C.HALLMARK_GMT is not None else "MSigDB_Hallmark_2020"
    if C.HALLMARK_GMT is not None and not C.HALLMARK_GMT.exists():
        raise FileNotFoundError(f"BCELL_HALLMARK_GMT does not exist: {C.HALLMARK_GMT}")
    try:
        pre = gp.prerank(rnk=rnk, gene_sets=gene_sets,
                         outdir=None, seed=C.RANDOM_SEED, permutation_num=PERM,
                         no_plot=True, verbose=False, min_size=5, max_size=500,
                         threads=4)
    except Exception as e:
        log.warning(f"[{slug}/{model}] gseapy failed: {e}")
        return None
    df = pre.res2d.copy()
    df["term"] = df["Term"].str.replace("MSigDB_Hallmark_2020__", "", regex=False)
    df["unit"] = slug
    df["model"] = model
    keep = ["term", "ES", "NES", "NOM p-val", "FWER p-val", "FDR q-val", "Tag %"]
    out = df[keep].rename(columns={"NOM p-val": "pval", "FDR q-val": "fdr",
                                   "FWER p-val": "fwer"})
    out.to_csv(C.GSEA_R / f"{slug}_{model}_gsea.csv", index=False)
    return out


def heatmap(model: str, tables: dict[str, pd.DataFrame]):
    """Units (cols) x hallmark (rows) NES heatmap; annotate FDR<0.05."""
    cols = {SLUG[u]: tables[u] for u in ORDER if u in tables and tables[u] is not None}
    if not cols:
        log.info(f"no tables for {model}, skip heatmap"); return
    nes = pd.DataFrame({k: t.set_index("term")["NES"] for k, t in cols.items()})
    fdr = pd.DataFrame({k: t.set_index("term")["fdr"] for k, t in cols.items()})
    nes = nes.loc[nes.abs().max(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(5.4, max(6.0, 0.30 * len(nes) + 1)))
    v = max(2.0, float(np.nanmax(nes.abs())))
    im = ax.imshow(nes.values, cmap="RdBu_r", aspect="auto",
                   vmin=-v, vmax=v, interpolation="nearest")
    ax.set_xticks(range(nes.shape[1])); ax.set_xticklabels(nes.columns, rotation=30, ha="right")
    ax.set_yticks(range(nes.shape[0])); ax.set_yticklabels(nes.index, fontsize=8)
    for i in range(nes.shape[0]):
        for j in range(nes.shape[1]):
            f = fdr.iloc[i, j]
            if pd.notna(f) and f < 0.05:
                ax.text(j, i, "*", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    ax.set_title(f"Hallmark GSEA (NES): young vs old — {model}\n* = FDR<0.05 (blue=higher young, red=higher old)")
    fig.colorbar(im, ax=ax, label="NES", shrink=0.5)
    save(fig, C.GSEA_F / f"hallmark_nes_{model}")


def main() -> None:
    C.ensure_dirs(); set_style()
    tables_pri, tables_pool = {}, {}
    for u in ORDER:
        slug = SLUG[u]
        rp = run_unit(slug, "primary")
        if rp is not None:
            tables_pri[u] = rp
        rw = run_unit(slug, "within_pool")
        if rw is not None:
            tables_pool[u] = rw

    heatmap("primary", tables_pri)
    heatmap("within_pool", tables_pool)

    # save combined wide tables
    for model, tables in [("primary", tables_pri), ("within_pool", tables_pool)]:
        rows = []
        for u, t in tables.items():
            tt = t.copy(); tt["unit"] = u; rows.append(tt)
        if rows:
            pd.concat(rows).to_csv(C.GSEA_R / f"hallmark_combined_{model}.csv", index=False)
    log.info("DONE 03_gsea")


if __name__ == "__main__":
    main()
