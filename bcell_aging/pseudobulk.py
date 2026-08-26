"""pseudobulk.py —— Part 4:donor×亚型 pseudobulk 构建(内存安全)。

从原始 h5ad 的 .X(原始整数 counts)按 (donor, B 亚型) 求和,塌缩成 donor×基因。
内存策略:载入 B 细胞稀疏矩阵(~1GB)→ 一次性稀疏指示矩阵 G → 按基因块 G.T@Xb 累加(峰值~300MB)。
统计单位 = donor:每个 donor 在每个亚型只形成一个 pseudobulk 样本。
"""
from __future__ import annotations
import gc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

from . import config as C
from . import figstyle
figstyle.set_style()


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(C.FIG_DIR / f"pb_{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[pseudobulk] -> figures/pb_{name}.{{pdf,png}}")


def _load_b_cells():
    """载入 B 细胞子集(原始 counts)到内存 + obs + gene 符号映射。"""
    print("[pseudobulk] backed 读原始 h5ad,切 B 细胞 ...")
    a = ad.read_h5ad(C.H5AD, backed="r")
    mask = a.obs[C.COL_CELLTYPE].isin(C.SUBTYPES)
    idx = np.where(mask.values)[0]
    b = a[idx, :].to_memory()
    a.file.close()
    obs = b.obs[[C.COL_DONOR, C.COL_AGE, C.COL_SEX, C.COL_POOL, C.COL_CELLTYPE]].copy()
    obs[C.COL_POOL] = obs[C.COL_POOL].astype(str)
    var = b.var.copy()
    var["ensembl"] = var.index.astype(str)
    var["symbol"] = var["feature_name"].astype(str)
    # 保留 ensembl 为基因 ID(唯一),symbol 另存一列
    print(f"[pseudobulk] B 细胞 {b.n_obs:,} × 基因 {b.n_vars};X.max={float(b.X.max()):.0f} (整数计数)")
    assert float(b.X.max()) > 30, "X 非原始计数?"
    return b, obs, var


def _sum_by_group(b, obs):
    """稀疏指示矩阵 G (n_cells × n_groups);按基因块累加 pb (n_groups × n_genes)。"""
    keys = list(zip(obs[C.COL_DONOR].astype(str), obs[C.COL_CELLTYPE].astype(str)))
    codes, uniq = pd.factorize(keys)            # codes: 每个 cell 的 group id(int); uniq: 去重 key
    n_cells, n_groups, n_genes = b.n_obs, len(uniq), b.n_vars
    G = sp.csr_matrix((np.ones(n_cells, dtype=np.float32),
                       (np.arange(n_cells), codes)),
                      shape=(n_cells, n_groups)).tocsc()
    pb = np.zeros((n_groups, n_genes), dtype=np.float32)
    print(f"[pseudobulk] 流式求和: {n_groups} 个 (donor×亚型) 组, 基因块 {C.PB_CHUNK} ...")
    for j0 in range(0, n_genes, C.PB_CHUNK):
        j1 = min(j0 + C.PB_CHUNK, n_genes)
        Xb = b.X[:, j0:j1]                       # n_cells × block (稀疏)
        pb[:, j0:j1] = np.asarray((G.T @ Xb).todense())
    # 校验:pb 总和 == B 细胞原始 counts 总和
    tot = float(pb.sum()); chk = float(b.X.sum())
    print(f"[pseudobulk] 校验 pb.sum={tot:.0f} vs X.sum={chk:.0f} (应相等)")
    assert abs(tot - chk) / chk < 1e-4, "pseudobulk 求和与原始 counts 不符!"
    # 组元信息
    ncells = np.asarray(G.sum(axis=0)).ravel()
    meta_rows = []
    for code, (donor, subtype) in enumerate(uniq):
        sub = obs.iloc[np.where(codes == code)[0]]
        meta_rows.append({C.COL_DONOR: donor, "subtype": subtype,
                          "n_cells": int(ncells[code]),
                          "age": int(sub[C.COL_AGE].iloc[0]),
                          "sex": str(sub[C.COL_SEX].iloc[0]),
                          "pool": str(sub[C.COL_POOL].iloc[0])})
    grp_meta = pd.DataFrame(meta_rows)
    # Persist a row-level checksum-like quantity.  Downstream consumers can
    # verify that metadata rows still match count-matrix rows after transfer.
    grp_meta["library_size"] = pb.sum(axis=1).astype(np.float64)
    return pb, grp_meta


def _filter_genes(counts, var):
    """保留 总counts≥MIN 且 CPM≥1 的样本占比≥prop 的基因(不限 HVG)。"""
    lib = counts.sum(axis=1)
    lib_safe = np.where(lib > 0, lib, 1)[:, None]
    cpm = counts / lib_safe * 1e6
    tot = counts.sum(axis=0)
    keep = (tot >= C.GENE_MIN_COUNT) & ((cpm >= C.GENE_MIN_CPM).mean(axis=0) >= C.GENE_MIN_SAMP_PROP)
    keep = np.asarray(keep).ravel()
    print(f"[pseudobulk] 基因过滤: {int(keep.sum())}/{counts.shape[1]} 通过")
    return counts[:, keep], var[keep].copy(), keep


def build_counts(pb_all, grp_meta, var, subtype, threshold):
    """从全量 pb_all 取某亚型、≥threshold 细胞的 donor,过滤基因,返回 AnnData。"""
    sel = grp_meta.index[(grp_meta["subtype"] == subtype) & (grp_meta["n_cells"] >= threshold)].values
    obs = grp_meta.loc[sel].reset_index(drop=True)
    counts = pb_all[sel, :]
    counts_f, var_f, _ = _filter_genes(counts, var)
    a = ad.AnnData(X=sp.csr_matrix(counts_f.astype(np.float32)),
                   obs=obs, var=var_f.reset_index(drop=True))
    a.obs_names = (obs[C.COL_DONOR].astype(str) + "__" + subtype.replace(" ", "")).values
    a.var_names = a.var["ensembl"].astype(str).values
    return a


def _per_subtype(pb_all, grp_meta, var, threshold):
    """按亚型筛 donor(≥threshold 细胞),过滤基因,存 AnnData;返回 {subtype: AnnData}。"""
    out = {}
    for s in C.SUBTYPES:
        a = build_counts(pb_all, grp_meta, var, s, threshold)
        if a.n_obs < 10:
            print(f"[pseudobulk] {s}: 仅 {a.n_obs} donor ≥{threshold} 细胞 → 探索性")
        tag = s.replace(" ", "")
        a.write_h5ad(C.PB_DIR / f"pb_{tag}.h5ad")
        a.obs.to_csv(C.PB_DIR / f"pb_{tag}_meta.csv", index=False)
        out[s] = a
        print(f"[pseudobulk] {s}: {a.n_obs} donor × {a.n_vars} 基因 (≥{threshold}细胞) -> pb_{tag}.h5ad")
    return out


def _diagnostics(pbs: dict):
    """library size / PCA / 相关性 诊断(用 naive 作代表 + 每亚型 libsize)。"""
    # 每亚型 library size 分布
    fig, ax = plt.subplots(figsize=(6, 4))
    for s, a in pbs.items():
        ls = np.asarray(a.X.sum(axis=1)).ravel()
        ax.hist(np.log10(ls + 1), bins=40, alpha=0.5, label=f"{s} (n={a.n_obs})")
    ax.set_xlabel("log10(library size + 1)"); ax.set_ylabel("# samples")
    ax.set_title("Pseudobulk library size by subtype"); ax.legend(fontsize=8)
    figstyle.thin_despine(ax); _save(fig, "libsize")

    # PCA(用最大的 naive 亚型),按 age/sex/pool 着色
    a = pbs.get("B naive")
    if a is not None:
        lib = np.asarray(a.X.sum(axis=1)).ravel()
        Xlog = np.log1p(a.X / np.where(lib > 0, lib, 1)[:, None] * 1e4)
        svd = TruncatedSVD(n_components=min(10, a.n_obs - 1), random_state=C.RANDOM_SEED)
        pc = svd.fit_transform(sp.csr_matrix(Xlog) if not sp.issparse(Xlog) else Xlog)
        for col, cmap, title in [("age", "viridis", "age"), ("sex", None, "sex"), ("pool", None, "pool")]:
            fig, ax = plt.subplots(figsize=(5.5, 5))
            v = a.obs[col].values
            if col == "age":
                sc = ax.scatter(pc[:, 0], pc[:, 1], c=v.astype(float), s=12, alpha=0.6, cmap="viridis")
                fig.colorbar(sc, ax=ax, label="age")
            else:
                cats = pd.Series(v).astype(str)
                for cval in cats.unique():
                    mm = cats.values == cval
                    ax.scatter(pc[mm, 0], pc[mm, 1], s=8, alpha=0.4, label=cval)
                ax.legend(fontsize=7, loc="best")
            ax.set_xlabel(f"PC1 ({svd.explained_variance_ratio_[0]*100:.1f}%)")
            ax.set_ylabel(f"PC2 ({svd.explained_variance_ratio_[1]*100:.1f}%)")
            ax.set_title(f"naive pseudobulk PCA by {title}")
            figstyle.thin_despine(ax); _save(fig, f"pca_naive_{col}")
        # age vs libsize
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(a.obs["age"], lib, s=8, alpha=0.4, color=figstyle.BLUE)
        ax.set_xlabel("age"); ax.set_ylabel("library size")
        ax.set_title("naive pseudobulk: age vs library size")
        r = np.corrcoef(a.obs["age"], ls)[0, 1]
        ax.text(0.05, 0.95, f"r={r:.2f}", transform=ax.transAxes, va="top")
        figstyle.thin_despine(ax); _save(fig, "naive_age_vs_libsize")


def run(threshold=C.MIN_CELLS):
    pb_path = C.PB_DIR / "pb_all_groups.npy"
    if pb_path.exists():
        print("[pseudobulk] 检测到已存 pb_all_groups.npy,跳过重建(复用)。")
        grp_meta = pd.read_csv(C.PB_DIR / "pb_all_groups_meta.csv")
        gene_map = pd.read_csv(C.PB_DIR / "pb_gene_map.csv")
        var = pd.DataFrame({"ensembl": gene_map["ensembl"].astype(str),
                            "symbol": gene_map["symbol"].astype(str)})
        var.index = var["ensembl"].values
        pbs = {s: ad.read_h5ad(C.PB_DIR / f"pb_{s.replace(' ', '')}.h5ad") for s in C.SUBTYPES
               if (C.PB_DIR / f"pb_{s.replace(' ', '')}.h5ad").exists()}
        return pbs
    b, obs, var = _load_b_cells()
    pb_all, grp_meta = _sum_by_group(b, obs)
    del b; gc.collect()
    # 存全量 group 计数矩阵(donor×亚型,供敏感性复用)
    np.save(C.PB_DIR / "pb_all_groups.npy", pb_all)
    grp_meta.to_csv(C.PB_DIR / "pb_all_groups_meta.csv", index=False)
    pd.DataFrame({"ensembl": var["ensembl"].values, "symbol": var["symbol"].values}) \
        .to_csv(C.PB_DIR / "pb_gene_map.csv", index=False)
    # 门槛 donor 覆盖
    cov = {}
    for thr in C.MIN_CELLS_SENS:
        cov[thr] = {s: int(((grp_meta["subtype"] == s) & (grp_meta["n_cells"] >= thr)).sum())
                    for s in C.SUBTYPES}
    print("[pseudobulk] donor 覆盖:"); print(pd.DataFrame(cov).T.to_string())

    pbs = _per_subtype(pb_all, grp_meta, var, threshold)
    try:
        _diagnostics(pbs)
    except Exception as e:
        print(f"[pseudobulk] 诊断绘图跳过: {str(e)[:80]}")
    del pb_all; gc.collect()
    print("[pseudobulk] 完成。")
    return pbs


if __name__ == "__main__":
    run()
