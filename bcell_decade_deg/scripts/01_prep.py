"""01_prep.py — derive full-transcriptome pseudobulks + cohort QC.

Source: aligned bcell_aging pseudobulk (pb_all_groups.npy; donor x subtype rows
on the full gene set). We derive, with NO cell-level reload:
  - all-B per donor:        sum the donor's subtype rows
  - per subtype per donor:  filter rows by subtype label

Output (results/qc/):
  genes.csv, allb_counts.npz, allb_donor_meta.csv,
  persub/<subtype>_counts.npz, persub/<subtype>_meta.csv
QC figures under figures/qc/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from plotting import set_style, save, decade_color
import matplotlib.pyplot as plt

log = C.get_logger("01_prep", C.LOGS_DIR / f"01_prep_{C.timestamp()}.log")


def build_indicator(rows2group: np.ndarray, n_groups: int, n_rows: int) -> sp.csr_matrix:
    """Sparse 0/1 indicator G[n_groups, n_rows]; (G @ X) sums X rows by group."""
    data = np.ones(len(rows2group), dtype=np.float32)
    return sp.csr_matrix((data, (rows2group, np.arange(len(rows2group)))),
                         shape=(n_groups, n_rows))


def main() -> None:
    C.ensure_dirs(); set_style()
    log.info(f"load aligned pseudobulk from {C.PB_COUNTS_NPY}")
    X_dense = np.load(C.PB_COUNTS_NPY, mmap_mode="r")
    X = sp.csr_matrix(X_dense)
    del X_dense
    meta = pd.read_csv(C.PB_META_CSV)
    genes = pd.read_csv(C.PB_GENES_CSV)
    assert X.shape == (len(meta), len(genes)), (X.shape, len(meta), len(genes))
    required = {"donor_id", "subtype", "n_cells", "age", "sex", "pool"}
    missing = required - set(meta.columns)
    assert not missing, f"pseudobulk metadata missing columns: {sorted(missing)}"
    assert not meta.duplicated(["donor_id", "subtype"]).any(), "duplicate donor-subtype rows"
    for col in ["age", "sex", "pool"]:
        assert meta.groupby("donor_id")[col].nunique().max() == 1, f"{col} varies within donor"

    row_sums = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    if "library_size" in meta.columns:
        np.testing.assert_allclose(meta["library_size"].astype(float), row_sums,
                                   rtol=0, atol=0,
                                   err_msg="count rows and metadata rows are misaligned")
    meta = meta.rename(columns={"subtype": "predicted.celltype.l2", "pool": "pool_number"})
    meta["library_size"] = row_sums
    assert np.all(row_sums >= meta["n_cells"].astype(float).values), "implausible row library sizes"

    gene_names = genes["ensembl"].astype(str).values
    gene_symbols = genes["symbol"].astype(str).values
    log.info(f"pb_counts {X.shape}, donors={meta['donor_id'].nunique()}, genes={len(genes)}")
    log.info("row-alignment checks passed")

    # ---- donor-level (donor_id is consistent within; age/sex/pool identical across rows) ----
    donors = meta["donor_id"].astype(str).values
    uniq, inv = np.unique(donors, return_inverse=True)   # inv maps row -> donor idx
    log.info(f"unique donors: {len(uniq)}")

    # all-B per donor = sum subtype rows within donor
    G = build_indicator(inv, len(uniq), X.shape[0])
    allb = (G @ X).tocsr()                                # 981 x 27066
    log.info(f"all-B pseudobulk per donor: {allb.shape}, nnz={allb.nnz}")

    # donor metadata (first within group for constant cols; sum for additive cols)
    agg = (meta.assign(_i=inv)
              .sort_values("_i")
              .groupby("donor_id", sort=False)
              .agg(age=("age", "first"), sex=("sex", "first"), pool_number=("pool_number", "first"),
                   n_cells=("n_cells", "sum"), library_size=("library_size", "sum"))
              .reset_index())
    assert (agg["donor_id"].astype(str).values == uniq).all(), "donor order mismatch"
    dmeta = pd.DataFrame({
        "donor_id": uniq,
        "age": agg["age"].values.astype(float),
        "sex": agg["sex"].values,
        "pool_number": agg["pool_number"].values,
        "n_cells": agg["n_cells"].values.astype(int),
        "library_size": agg["library_size"].values.astype(float),
    })
    dmeta["decade"] = dmeta["age"].map(C.decade_label)
    dmeta["group"] = dmeta["age"].map(C.age_group)
    log.info(f"all-B donors by group: {dmeta['group'].value_counts().to_dict()}")
    log.info(f"all-B donors by decade:\n{dmeta['decade'].value_counts().sort_index().to_string()}")

    # ---- save all-B ----
    sp.save_npz(C.QC_R / "allb_counts.npz", allb)
    dmeta.to_csv(C.QC_R / "allb_donor_meta.csv", index=False)
    pd.DataFrame({"gene": gene_names}).to_csv(C.QC_R / "genes.csv", index=False)
    pd.DataFrame({"ensg": gene_names, "symbol": gene_symbols}).to_csv(
        C.QC_R / "gene_map_ensg_symbol.csv", index=False)
    (C.QC_R / "persub").mkdir(exist_ok=True)

    # ---- per-subtype (filter rows by subtype; full gene set) ----
    sub_summary = []
    for sub in C.B_SUBTYPES:
        m = meta["predicted.celltype.l2"].astype(str).values == sub
        if m.sum() == 0:
            log.warning(f"no rows for {sub}"); continue
        Xs = X[m]
        ms = meta.loc[m].copy()
        ms = ms[["donor_id", "age", "sex", "pool_number", "n_cells", "library_size"]].copy()
        ms["decade"] = ms["age"].map(C.decade_label)
        ms["group"] = ms["age"].map(C.age_group)
        ms["donor_id"] = ms["donor_id"].astype(str)
        slug = C.SUBTYPE_SLUG[sub]
        sp.save_npz(C.QC_R / "persub" / f"{slug}_counts.npz", Xs.tocsr())
        ms.to_csv(C.QC_R / "persub" / f"{slug}_meta.csv", index=False)
        sub_summary.append({"subtype": sub, "n_donors": len(ms),
                            "young": int((ms["group"] == "young").sum()),
                            "old": int((ms["group"] == "old").sum())})
        log.info(f"{sub}: {len(ms)} donors (young {int((ms['group']=='young').sum())}, "
                 f"old {int((ms['group']=='old').sum())})")
    pd.DataFrame(sub_summary).to_csv(C.QC_R / "subtype_group_counts.csv", index=False)

    qc_figures(dmeta, meta, sub_summary)
    log.info("DONE 01_prep")


def qc_figures(dmeta: pd.DataFrame, pbmeta: pd.DataFrame, sub_summary: list) -> None:
    a = dmeta["age"].values

    # 1. age histogram with young/old bands
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.hist(a, bins=np.arange(15, 101, 5), color="#9ec5f4", edgecolor="white", linewidth=0.6)
    ax.axvspan(*C.YOUNG, color=C.COLOR_YOUNG, alpha=0.16, zorder=0)
    ax.axvspan(*C.OLD, color=C.COLOR_OLD, alpha=0.16, zorder=0)
    ax.text(np.mean(C.YOUNG), ax.get_ylim()[1] * 0.95, "young\n20–30",
            ha="center", va="top", color=C.COLOR_YOUNG, fontsize=9, fontweight="medium")
    ax.text(np.mean(C.OLD), ax.get_ylim()[1] * 0.95, "old\n80–90",
            ha="center", va="top", color=C.COLOR_OLD, fontsize=9, fontweight="medium")
    ax.set_xlabel("Donor age (years)"); ax.set_ylabel("Donors")
    ax.set_title(f"Cohort age distribution (n={len(dmeta)} donors, 981 B-cell donors)")
    save(fig, C.QC_F / "age_histogram")

    # 2. donors per decade (all-B + per subtype)
    decades = [d[2] for d in C.DECADES]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    x = np.arange(len(decades))
    counts_all = [int((dmeta["decade"] == d).sum()) for d in decades]
    ax.bar(x, counts_all, color=[decade_color(d) for d in decades],
           edgecolor="white", linewidth=0.6, label="all-B")
    for xi, c in zip(x, counts_all):
        ax.text(xi, c + 3, str(c), ha="center", va="bottom", fontsize=8, color=C.INK_SEC)
    ax.set_xticks(x); ax.set_xticklabels(decades)
    ax.set_xlabel("Decade"); ax.set_ylabel("Donors")
    ax.set_title("Donor counts per decade (all-B pseudobulk)")
    save(fig, C.QC_F / "donors_per_decade")

    # 3. per-subtype young/old donor counts
    ss = pd.DataFrame(sub_summary)
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    xpos = np.arange(len(ss)); w = 0.36
    ax.bar(xpos - w/2, ss["young"], w, color=C.COLOR_YOUNG, label=f"young {C.YOUNG}")
    ax.bar(xpos + w/2, ss["old"], w, color=C.COLOR_OLD, label=f"old {C.OLD}")
    for xi, yv, ov in zip(xpos, ss["young"], ss["old"]):
        ax.text(xi - w/2, yv + 1, str(int(yv)), ha="center", fontsize=8, color=C.INK_SEC)
        ax.text(xi + w/2, ov + 1, str(int(ov)), ha="center", fontsize=8, color=C.INK_SEC)
    ax.set_xticks(xpos); ax.set_xticklabels(ss["subtype"], rotation=15, ha="right")
    ax.set_ylabel("Donors"); ax.set_title("Donors per subtype in young vs old groups")
    ax.legend()
    save(fig, C.QC_F / "subtype_young_old_counts")

    # 4. n_cells & library_size vs age (variance-confound check)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, col, lab in [(axs[0], "n_cells", "B cells / donor"),
                         (axs[1], "library_size", "library size / donor")]:
        ax.scatter(dmeta["age"], dmeta[col], s=8, color="#5598e7", alpha=0.55, edgecolors="none")
        r = np.corrcoef(dmeta["age"], dmeta[col])[0, 1]
        ax.set_xlabel("Donor age"); ax.set_ylabel(lab)
        ax.set_title(f"{lab}: corr(age)={r:+.3f}")
    fig.suptitle("Pseudobulk depth vs age (confound check for variance)", y=1.02, fontweight="semibold")
    save(fig, C.QC_F / "depth_vs_age")


if __name__ == "__main__":
    main()
