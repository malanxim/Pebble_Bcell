"""data_check.py —— Part 1&3:数据结构检查 + donor 汇总表 + QC 图。

只读原始 obs(backed,不加载 X),产 donor_summary / 计数 / 比例 表与 QC 图。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C
from . import figstyle
figstyle.set_style()


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(C.FIG_DIR / f"qc_{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[data_check] -> figures/qc_{name}.{{pdf,png}}")


def load_obs() -> pd.DataFrame:
    """backed 读取原始 obs(不加载 X),返回细胞级 obs + B 掩码列。"""
    print(f"[data_check] backed 读 {C.H5AD.name} obs ...")
    a = ad.read_h5ad(C.H5AD, backed="r")
    cols = [C.COL_DONOR, C.COL_AGE, C.COL_SEX, C.COL_POOL, C.COL_CELLTYPE]
    obs = a.obs[cols].copy()
    a.file.close()
    obs[C.COL_POOL] = obs[C.COL_POOL].astype(str)
    obs["is_B"] = obs[C.COL_CELLTYPE].isin(C.SUBTYPES)
    return obs


def donor_tables(obs: pd.DataFrame) -> pd.DataFrame:
    """细胞级 obs -> donor 级汇总(总细胞/B细胞/各亚型计数/比例)。"""
    ct = pd.crosstab(obs[C.COL_DONOR], obs[C.COL_CELLTYPE])
    meta = (obs.groupby(C.COL_DONOR, observed=True)
            .agg(age=(C.COL_AGE, "first"), sex=(C.COL_SEX, "first"),
                 pool=(C.COL_POOL, "first"), total_cells=(C.COL_DONOR, "size"))
            .reset_index().rename(columns={"donor_id": C.COL_DONOR}))
    for s in C.SUBTYPES:
        meta[f"n_{_short(s)}"] = meta[C.COL_DONOR].map(ct[s]).fillna(0).astype(int) if s in ct else 0
    meta["n_B"] = meta[[f"n_{_short(s)}" for s in C.SUBTYPES]].sum(axis=1)
    sub = meta[meta["n_B"] > 0].copy()
    for s in C.SUBTYPES:
        c = f"n_{_short(s)}"
        with np.errstate(invalid="ignore", divide="ignore"):
            sub[f"frac_{_short(s)}"] = np.where(sub["n_B"] > 0, sub[c] / sub["n_B"], np.nan)
    # 存表
    meta.to_csv(C.TAB_DIR / "donor_summary.csv", index=False)
    sub.to_csv(C.TAB_DIR / "per_donor_subtype_counts.csv", index=False)
    frac = sub[[C.COL_DONOR, "age"] + [f"frac_{_short(s)}" for s in C.SUBTYPES]]
    frac.to_csv(C.TAB_DIR / "per_donor_subtype_fractions.csv", index=False)
    print(f"[data_check] donors={len(meta)}  (有B细胞 {len(sub)})  "
          f"age {int(meta['age'].min())}-{int(meta['age'].max())} mean {meta['age'].mean():.1f}")
    return sub


def checks(obs: pd.DataFrame, donors: pd.DataFrame):
    """打印数据结构与混杂检查。"""
    print("\n=== [check] donor→age 唯一性 ===")
    nu = obs.groupby(C.COL_DONOR, observed=True)[C.COL_AGE].nunique()
    print(f"  年龄不一致的 donor: {int((nu > 1).sum())}")

    print("\n=== [check] 协变量变异(常量者不入模型) ===")
    for col in [C.COL_SEX, C.COL_POOL, "disease", "self_reported_ethnicity_ontology_term_id"]:
        if col in obs.columns:
            print(f"  {col}: {obs[col].nunique()} 个值")
        elif col == C.COL_SEX:
            print(f"  {C.COL_SEX}: {obs[C.COL_SEX].nunique()} 个值")

    print("\n=== [check] 各亚型 donor 覆盖(不同细胞门槛) ===")
    cov = {}
    for thr in C.MIN_CELLS_SENS:
        row = {}
        for s in C.SUBTYPES:
            c = f"n_{_short(s)}"
            row[s] = int((donors[c] >= thr).sum())
        cov[thr] = row
    print(pd.DataFrame(cov).T.to_string())

    print("\n=== [check] 年龄组 vs sex/pool 组成(混杂检查) ===")
    donors[AGE_GROUP_COL_TMP] = pd.cut(donors["age"], [17, 39, 64, 98],
                                       labels=["young(<40)", "middle(40-64)", "old(>=65)"])
    sex_ct = pd.crosstab(donors[AGE_GROUP_COL_TMP], donors["sex"], normalize="index") * 100
    print("  各年龄组 sex 构成(%):"); print(sex_ct.round(1).to_string())
    pool_div = donors.groupby(AGE_GROUP_COL_TMP, observed=True)["pool"].nunique()
    print(f"  各年龄组涉及的 pool 数: {pool_div.to_dict()}")
    return pd.DataFrame(cov).T


AGE_GROUP_COL_TMP = "age_group_tmp"


def _short(s: str) -> str:
    return s.replace(" ", "").replace("naive", "naive")  # B naive->Bnaive


def plot_qc(obs: pd.DataFrame, donors: pd.DataFrame):
    d = donors.copy()
    # 1 年龄直方图
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(d["age"], bins=np.arange(18, 99, 5), color=figstyle.BLUE, edgecolor="white")
    ax.set_xlabel("age (years)"); ax.set_ylabel("# donors")
    ax.set_title(f"Donor age distribution (n={len(d)})")
    figstyle.thin_despine(ax); _save(fig, "age_hist")

    # 2 每岁 donor 柱
    vc = d["age"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(vc.index, vc.values, color=figstyle.BLUE)
    ax.set_xlabel("age"); ax.set_ylabel("# donors"); ax.set_title("Donors per age (year)")
    figstyle.thin_despine(ax); _save(fig, "donors_per_age")

    # 3 10 岁组柱
    g = pd.cut(d["age"], C.AGE_BIN_EDGES, labels=C.AGE_BIN_LABELS)
    vc2 = g.value_counts().reindex(C.AGE_BIN_LABELS)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(vc2.index.astype(str), vc2.values, color=figstyle.ORANGE)
    ax.set_xlabel("age group"); ax.set_ylabel("# donors")
    ax.set_title("Donors per 10-year group"); figstyle.thin_despine(ax)
    for i, v in enumerate(vc2.values):
        ax.text(i, v + 2, str(int(v)), ha="center", fontsize=8)
    _save(fig, "donors_per_10yr")

    # 4 每 donor B 细胞数分布
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(d["n_B"], bins=50, color=figstyle.AQUA, edgecolor="white")
    ax.set_xlabel("# B cells / donor"); ax.set_ylabel("# donors")
    ax.set_title(f"B cells per donor (n={len(d)})")
    ax.axvline(C.MIN_CELLS, color=figstyle.ORANGE, ls="--", label=f"min={C.MIN_CELLS}")
    ax.legend(); figstyle.thin_despine(ax); _save(fig, "bcells_per_donor")

    # 5 各亚型 donor 细胞数分布(box)
    fig, ax = plt.subplots(figsize=(7, 4))
    data, labels = [], []
    for s in C.SUBTYPES:
        c = f"n_{_short(s)}"
        vals = d[d[c] >= 1][c].values
        data.append(np.log10(vals + 1)); labels.append(f"{s}\n(n={len(vals)})")
    ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
    ax.set_ylabel("log10(cells/donor + 1)"); ax.set_title("Per-donor cell counts by subtype (donors with ≥1)")
    figstyle.thin_despine(ax); _save(fig, "subtype_cells_per_donor")

    # 6 年龄 vs B 细胞数 散点
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(d["age"], d["n_B"], s=8, alpha=0.35, color=figstyle.BLUE)
    ax.set_xlabel("age"); ax.set_ylabel("# B cells / donor")
    ax.set_title("Age vs B-cell count per donor"); figstyle.thin_despine(ax)
    _save(fig, "age_vs_bcount")

    # 7 年龄组 × sex 堆叠
    d["_ag"] = pd.cut(d["age"], [17, 39, 64, 98], labels=["<40", "40-64", "≥65"])
    ct = pd.crosstab(d["_ag"], d["sex"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(5, 4))
    bottom = np.zeros(len(ct))
    for i, sx in enumerate(ct.columns):
        ax.bar(ct.index.astype(str), ct[sx], bottom=bottom,
               color=[figstyle.BLUE, figstyle.ORANGE][i % 2], label=sx)
        bottom += ct[sx].values
    ax.set_ylabel("% donors"); ax.set_title("Sex composition by age group")
    ax.legend(); figstyle.thin_despine(ax); _save(fig, "agegroup_sex")


def run():
    obs = load_obs()
    donors = donor_tables(obs)
    cov = checks(obs, donors)
    plot_qc(obs, donors)
    cov.to_csv(C.TAB_DIR / "subtype_donor_coverage.csv")
    print("\n[data_check] 完成。")
    return obs, donors
