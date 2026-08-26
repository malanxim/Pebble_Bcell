"""deg.py —— Part 5/7:亚型内年龄相关 DEG(PyDESeq2,donor 级 pseudobulk)。

主分析:连续年龄,design ~ sex + pool + age_scaled(末位),测 age 系数 = 每 10 岁 log2FC。
辅助:<40 vs ≥65 二分组;Q1 vs Q4 敏感性。
统计单位 = donor(每 donor 每亚型 1 个 pseudobulk 样本),非细胞。
"""
from __future__ import annotations
import gc
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import config as C
from .pseudobulk import build_counts


# ----------------------------------------------------------------- 数据准备
def _load_pb_inputs():
    """载入 pseudobulk 全量(pb_all + meta + gene_map),供任意门槛/亚型重建。"""
    pb_all = np.load(C.PB_DIR / "pb_all_groups.npy", mmap_mode="r")
    grp_meta = pd.read_csv(C.PB_DIR / "pb_all_groups_meta.csv")
    gene_map = pd.read_csv(C.PB_DIR / "pb_gene_map.csv")
    import anndata as ad
    var = pd.DataFrame({"ensembl": gene_map["ensembl"].astype(str),
                        "symbol": gene_map["symbol"].astype(str)})
    var.index = var["ensembl"].values
    return pb_all, grp_meta, var


def get_pseudobulk(subtype, threshold=C.MIN_CELLS):
    pb_all, grp_meta, var = _load_pb_inputs()
    a = build_counts(np.asarray(pb_all), grp_meta, var, subtype, threshold)
    return a


def _mat_for_deseq(a):
    """AnnData -> (counts_df, meta_df, symbol_series) 供 PyDESeq2。"""
    counts = pd.DataFrame(np.asarray(a.X.todense()) if sp.issparse(a.X) else np.asarray(a.X),
                          index=a.obs_names, columns=a.var_names).astype(int)
    meta = a.obs[["sex", "pool", "age"]].copy()
    meta["sex"] = meta["sex"].astype(str)
    meta["pool"] = meta["pool"].astype(str)
    sym = a.var["symbol"].astype(str)
    return counts, meta, sym


def _enrich_result(df, sym, n_donors, counts_df):
    """补 symbol/n_donors/pct_donors/is_IG 列。"""
    out = df.copy()
    out["ensembl"] = out.index
    out["symbol"] = out["ensembl"].map(sym.to_dict())
    out["n_donors"] = n_donors
    nz = (counts_df > 0).mean(axis=0)              # 每基因表达样本占比
    out["pct_donors"] = out["ensembl"].map(nz.to_dict())
    out["is_IG"] = out["symbol"].fillna("").str.startswith(C.IG_PREFIXES)
    out = out.rename(columns={"log2FoldChange": "log2FC_per10yr"})
    return out


# ----------------------------------------------------------------- 主分析:连续年龄
def fit_continuous(a):
    """PyDESeq2 连续年龄;返回 results_df(含 symbol/log2FC_per10yr/padj/...)。"""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    counts, meta, sym = _mat_for_deseq(a)
    n_donors = len(meta)
    mean_age = float(meta["age"].mean())
    meta["age_scaled"] = (meta["age"] - mean_age) / C.AGE_SCALE   # 系数 = 每 10 岁
    # pool 过稀(donor 少或 pool 数 > donor/3)时去掉 pool,避免设计矩阵过饱和
    n_pool = meta["pool"].nunique()
    factors = ["sex", "pool", "age_scaled"] if (n_donors >= 50 and n_pool <= n_donors / 3) \
        else (["sex", "age_scaled"] if meta["sex"].nunique() > 1 else ["age_scaled"])
    print(f"[deg] 连续年龄: {n_donors} donor, mean_age={mean_age:.1f}, "
          f"genes={counts.shape[1]}, design ~ {'+'.join(factors)}")
    dds = DeseqDataSet(counts=counts, metadata=meta,
                       design_factors=factors,
                       continuous_factors=["age_scaled"], refit_cooks=False, quiet=True)
    dds.deseq2()
    res = DeseqStats(dds, quiet=True)          # 测末位系数 age-scaled
    res.summary()
    try:
        res.lfc_shrink(coeff="age-scaled")
    except Exception as e:
        print(f"[deg] lfc_shrink 跳过: {str(e)[:80]}")
    df = res.results_df.copy()
    out = _enrich_result(df, sym, n_donors, counts)
    out.attrs["mean_age"] = mean_age
    return out


# ----------------------------------------------------------------- 辅助:young vs old
def fit_young_old(a):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    counts, meta, sym = _mat_for_deseq(a)
    ag = pd.cut(meta["age"], [0, C.YOUNG_MAX - 1, C.OLD_MIN - 1, 200],
                labels=["young", "middle", "old"])
    meta["agegrp"] = ag.astype(str)
    keep = meta["agegrp"].isin(["young", "old"])
    counts, meta = counts[keep], meta[keep]
    if len(meta) < 20 or (meta["agegrp"] == "young").sum() < 5 \
            or (meta["agegrp"] == "old").sum() < 5:
        print("[deg] young/old 样本不足,跳过"); return None
    print(f"[deg] <40 vs ≥65: young={int((meta.agegrp=='young').sum())} "
          f"old={int((meta.agegrp=='old').sum())}")
    dds = DeseqDataSet(counts=counts, metadata=meta,
                       design_factors=["sex", "pool", "agegrp"],
                       ref_level=["agegrp", "young"], refit_cooks=False, quiet=True)
    dds.deseq2()
    res = DeseqStats(dds, contrast=["agegrp", "old", "young"], quiet=True)
    res.summary()
    try:
        res.lfc_shrink(coeff="agegrp_old_vs_young")
    except Exception:
        pass
    out = _enrich_result(res.results_df.copy(), sym, len(meta), counts)
    out = out.rename(columns={"log2FC_per10yr": "log2FC_old_vs_young"})
    return out


# ----------------------------------------------------------------- 敏感性:Q1 vs Q4
def fit_quantile(a):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    counts, meta, sym = _mat_for_deseq(a)
    q1, q4 = a.obs["age"].quantile(C.QUANTILE_COMPARE).values
    meta["ageq"] = pd.cut(meta["age"], [0, q1, q4, 200], labels=["Q1", "mid", "Q4"]).astype(str)
    keep = meta["ageq"].isin(["Q1", "Q4"])
    counts, meta = counts[keep], meta[keep]
    if (meta["ageq"] == "Q1").sum() < 5 or (meta["ageq"] == "Q4").sum() < 5:
        print(f"[deg] Q1/Q4 不足(Q1<{q1:.0f}, Q4>{q4:.0f}),跳过"); return None
    print(f"[deg] Q1 vs Q4: Q1<{q1:.0f} n={int((meta.ageq=='Q1').sum())}, "
          f"Q4>{q4:.0f} n={int((meta.ageq=='Q4').sum())}")
    dds = DeseqDataSet(counts=counts, metadata=meta,
                       design_factors=["sex", "pool", "ageq"],
                       ref_level=["ageq", "Q1"], refit_cooks=False, quiet=True)
    dds.deseq2()
    res = DeseqStats(dds, contrast=["ageq", "Q4", "Q1"], quiet=True)
    res.summary()
    try:
        res.lfc_shrink(coeff="ageq_Q4_vs_Q1")
    except Exception:
        pass
    out = _enrich_result(res.results_df.copy(), sym, len(meta), counts)
    out = out.rename(columns={"log2FC_per10yr": "log2FC_Q4_vs_Q1"})
    return out


# ----------------------------------------------------------------- 入口
def run_deg(subtype, threshold=C.MIN_CELLS):
    tag = subtype.replace(" ", "")
    cont_path = C.DEG_DIR / f"{tag}_deg_continuous.csv"
    yo_path = C.DEG_DIR / f"{tag}_deg_young_old.csv"
    q_path = C.DEG_DIR / f"{tag}_deg_Q4vsQ1.csv"
    print(f"\n==== DEG: {subtype} (≥{threshold} cells) ====")

    need_fit = not cont_path.exists() or not yo_path.exists() or not q_path.exists()
    a = get_pseudobulk(subtype, threshold) if need_fit else None

    # 主分析:连续年龄
    if cont_path.exists():
        cont = pd.read_csv(cont_path)
        print(f"[deg] 连续年龄(已存): {int((cont['padj']<C.DEG_PADJ).sum())} sig")
    else:
        cont = fit_continuous(a)
        cont.to_csv(cont_path, index=False)
        print(f"[deg] 连续年龄 DEG: {int((cont['padj']<C.DEG_PADJ).sum())} sig -> {cont_path.name}")

    # 辅助:<40 vs ≥65
    if yo_path.exists():
        yo = pd.read_csv(yo_path)
    else:
        yo = fit_young_old(a)
        if yo is not None:
            yo.to_csv(yo_path, index=False)
            print(f"[deg] young/old -> {yo_path.name}")

    # 敏感性:Q1 vs Q4
    if q_path.exists():
        q = pd.read_csv(q_path)
    else:
        q = fit_quantile(a)
        if q is not None:
            q.to_csv(q_path, index=False)
            print(f"[deg] Q1/Q4 -> {q_path.name}")

    # 敏感性:门槛一致性(≥10/≥30 vs ≥20)
    conc_path = C.DEG_DIR / f"{tag}_sensitivity_concordance.csv"
    if conc_path.exists():
        conc = pd.read_csv(conc_path)
    else:
        conc = _sensitivity_concordance(subtype, cont)
        conc.to_csv(conc_path, index=False)
        print(f"[deg] 门槛一致性 -> {conc_path.name}")

    if a is not None:
        del a; gc.collect()
    return {"continuous": cont, "young_old": yo, "Q4vsQ1": q, "concordance": conc}


def _sensitivity_concordance(subtype, cont_main, ref_thr=C.MIN_CELLS):
    """比较 ≥10/≥30 连续年龄 与主分析(≥20)的方向一致性(top基因)。"""
    rows = []
    main = cont_main.set_index("ensembl")["log2FC_per10yr"]
    for thr in C.MIN_CELLS_SENS:
        if thr == ref_thr:
            continue
        try:
            a = get_pseudobulk(subtype, thr)
            d = fit_continuous(a)
            del a; gc.collect()
        except Exception as e:
            print(f"[deg] 敏感性 ≥{thr} 失败: {str(e)[:80]}"); continue
        s = d.set_index("ensembl")["log2FC_per10yr"]
        common = main.index.intersection(s.index)
        m, o = main.loc[common], s.loc[common]
        rho = float(np.corrcoef(m, o)[0, 1])
        sign_agree = float((np.sign(m) == np.sign(o)).mean())
        rows.append({"threshold": thr, "n_donors": d["n_donors"].iloc[0],
                     "pearson_rho": round(rho, 3), "sign_agreement": round(sign_agree, 3)})
    return pd.DataFrame(rows)
