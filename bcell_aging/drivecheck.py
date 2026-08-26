"""drivecheck.py —— 检查2:年龄相关通路是否由少数 donor 或单个 pool 驱动。

每 donor×亚型 用 Hallmark 基因集算模块评分(mean log1p(CPM));donor 级 OLS score~age+sex+pool。
检查:去 top 1%/5% 高分 donor 后效应;leave-one-pool-out 符号一致率;ribosome 与技术变量相关。
区分"广泛轻度升高 / 少数高分 donor / 某 pool 特异"三种情形。
"""
from __future__ import annotations
import gc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C
from . import figstyle
figstyle.set_style()

PATHWAYS = ["Interferon Alpha Response", "Interferon Gamma Response",
            "TNF-alpha Signaling via NF-kB", "Unfolded Protein Response",
            "p53 Pathway", "Apoptosis", "Myc Targets V1", "Myc Targets V2",
            "E2F Targets"]


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(C.FIG_DIR / f"check2_{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _gene_sets():
    import gseapy as gp
    lib = gp.get_library(C.HALLMARK_LIB, organism="human")
    var = pd.read_csv(C.PB_DIR / "pb_gene_map.csv")["symbol"].astype(str)
    vset = set(var)
    out = {}
    for k, genes in lib.items():
        if k in PATHWAYS:
            out[k] = [g for g in genes if g in vset]
    out["Ribosome (RPL/RPS)"] = [g for g in var if g.startswith(("RPL", "RPS"))]
    return out, vset


def _scores_for_subtype(subtype, gsets):
    """读 pseudobulk -> donor×pathway 模块评分(mean log1p CPM) + 技术变量。"""
    a = ad.read_h5ad(C.PB_DIR / f"pb_{subtype.replace(' ', '')}.h5ad")
    X = a.X
    if sp.issparse(X):
        X = np.asarray(X.todense())
    lib = X.sum(axis=1); lib_safe = np.where(lib > 0, lib, 1)[:, None]
    cpm = X / lib_safe * 1e6
    lcpm = np.log1p(cpm)
    sym = a.var["symbol"].astype(str).values
    symidx = {g: i for i, g in enumerate(sym)}
    rows = {}
    for name, genes in gsets.items():
        idx = [symidx[g] for g in genes if g in symidx]
        rows[name] = lcpm[:, idx].mean(axis=1) if idx else np.full(a.n_obs, np.nan)
    sc = pd.DataFrame(rows, index=a.obs_names)
    sc["age"] = a.obs["age"].values; sc["sex"] = a.obs["sex"].astype(str).values
    sc["pool"] = a.obs["pool"].astype(str).values; sc["n_cells"] = a.obs["n_cells"].values
    sc["donor"] = a.obs["donor_id"].astype(str).values
    sc["lib_size"] = lib
    # 技术变量:IG 分数、MT 分数、核糖体分数(counts 占比)
    is_ig = np.array([s.startswith(("IGH", "IGL", "IGK", "IGJ")) for s in sym])
    is_mt = np.array([s.startswith("MT-") for s in sym])
    is_ribo = np.array([s.startswith(("RPL", "RPS")) for s in sym])
    sc["ig_frac"] = (X[:, is_ig].sum(axis=1) / np.where(lib > 0, lib, 1)) if is_ig.any() else 0
    sc["mt_frac"] = (X[:, is_mt].sum(axis=1) / np.where(lib > 0, lib, 1)) if is_mt.any() else 0
    sc["ribo_frac"] = (X[:, is_ribo].sum(axis=1) / np.where(lib > 0, lib, 1)) if is_ribo.any() else 0
    del a; gc.collect()
    return sc


def _fit(df, drop_top=None, score_col=None):
    d = df.copy()
    if drop_top:
        thr = d[score_col].quantile(drop_top)
        d = d[d[score_col] <= thr]
    if len(d) < 30 or d["age"].nunique() < 3:
        return np.nan, np.nan, len(d)
    try:
        # sex/pool 为字符串列 -> patsy 自动作分类(勿用 C(),与本模块别名 C 冲突)
        m = smf.ols(f"{score_col} ~ age + sex + pool", data=d).fit()
        return float(m.params["age"]), float(m.pvalues["age"]), len(d)
    except Exception:
        return np.nan, np.nan, len(d)


def _leave_one_pool(df, score_col):
    main = _fit(df, score_col=score_col)[0]
    signs = []
    coefs = []
    worst = (None, 0)
    for p in df["pool"].unique():
        d = df[df["pool"] != p]
        c, _, n = _fit(d, score_col=score_col)
        if not np.isnan(c):
            signs.append(np.sign(c) == np.sign(main)); coefs.append(c)
            if abs(c - main) > abs(worst[1]):
                worst = (p, c - main)
    agree = float(np.mean(signs)) if signs else np.nan
    return main, agree, (min(coefs), max(coefs)) if coefs else (np.nan, np.nan), worst[0]


def run():
    gsets, _ = _gene_sets()
    print(f"[check2] 基因集大小: {[(k, len(v)) for k, v in gsets.items()]}")
    stab, pool_tab, ribo_rows = [], [], []
    for s in C.MAIN_SUBTYPES:
        sc = _scores_for_subtype(s, gsets)
        sc.to_csv(C.TAB_DIR / f"check2_scores_{s.replace(' ','')}.csv", index=False)
        for pw in list(gsets.keys()):
            d = sc[["age", "sex", "pool", pw]].rename(columns={pw: "score"}).dropna()
            b0, p0, n0 = _fit(d, score_col="score")
            b1, _, _ = _fit(d, drop_top=0.99, score_col="score")
            b5, _, _ = _fit(d, drop_top=0.95, score_col="score")
            main_agree = not (np.isnan(b0) or np.isnan(b1) or np.isnan(b5)) and \
                (np.sign(b0) == np.sign(b1) == np.sign(b5))
            lp_main, lp_agree, (lomin, lomax), worst_pool = _leave_one_pool(d, "score")
            stab.append({"subtype": s, "pathway": pw, "n": n0,
                         "age_coef": round(b0, 4), "p": p0,
                         "coef_dropTop1%": round(b1, 4), "coef_dropTop5%": round(b5, 4),
                         "sign_stable_droptop": bool(main_agree)})
            pool_tab.append({"subtype": s, "pathway": pw,
                             "main_coef": round(lp_main, 4),
                             "lop_sign_agreement": round(lp_agree, 3),
                             "lop_min": round(lomin, 4), "lop_max": round(lomax, 4),
                             "most_changing_pool": worst_pool})
        # ribosome vs 技术 + 调整 lib_size/n_cells 后年龄效应是否残留
        if "Ribosome (RPL/RPS)" in sc.columns:
            r = sc["Ribosome (RPL/RPS)"]
            for cov in ["lib_size", "n_cells", "mt_frac", "ig_frac", "ribo_frac", "age"]:
                ribo_rows.append({"subtype": s, "covariate": cov,
                                  "pearson_r": round(float(r.corr(sc[cov])), 3)})
            rd = sc[["age", "sex", "pool", "lib_size", "n_cells"]].copy()
            rd["ribo"] = r.values
            try:
                madj = smf.ols("ribo ~ age + sex + pool + lib_size + n_cells", data=rd).fit()
                ribo_rows.append({"subtype": s, "covariate": "age_coef_adj(lib+nCells)",
                                  "pearson_r": round(float(madj.params["age"]), 4)})
                ribo_rows.append({"subtype": s, "covariate": "age_p_adj(lib+nCells)",
                                  "pearson_r": float(madj.pvalues["age"])})
            except Exception as e:
                print(f"[check2] ribosome 调整拟合失败 {s}: {str(e)[:60]}")
    stab = pd.DataFrame(stab); stab.to_csv(C.TAB_DIR / "check2_donor_stability.csv", index=False)
    pool_tab = pd.DataFrame(pool_tab); pool_tab.to_csv(C.TAB_DIR / "check2_leave_one_pool.csv", index=False)
    ribo = pd.DataFrame(ribo_rows); ribo.to_csv(C.TAB_DIR / "check2_ribosome_technical.csv", index=False)
    print("\n[check2] donor 稳定性(score~age,去 top donor):")
    print(stab.to_string(index=False))
    print("\n[check2] leave-one-pool-out 符号一致率:")
    print(pool_tab[["subtype", "pathway", "main_coef", "lop_sign_agreement",
                    "lop_min", "lop_max", "most_changing_pool"]].to_string(index=False))
    print("\n[check2] ribosome 评分与技术变量相关:")
    print(ribo.to_string(index=False))
    _forest(stab)
    _pool_plot(pool_tab)
    _ribo_plot()
    return stab, pool_tab, ribo


def _forest(stab):
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(stab) + 1.5))
    y = np.arange(len(stab))
    colors = [figstyle.BLUE if a else figstyle.ORANGE for a in stab["sign_stable_droptop"]]
    ax.barh(y, stab["age_coef"], color=colors)
    ax.set_yticks(y); ax.set_yticklabels([f"{r.subtype}|{r.pathway}" for r in stab.itertuples()], fontsize=6.5)
    ax.axvline(0, color=figstyle.MUTED, lw=1)
    ax.set_xlabel("module-score slope vs age (OLS, per donor)")
    ax.set_title("Pathway module score ~ age (blue=stable to donor drop)")
    figstyle.thin_despine(ax); _save(fig, "pathway_age_effects")


def _pool_plot(pool_tab):
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(pool_tab) + 1.5))
    y = np.arange(len(pool_tab))
    colors = [figstyle.BLUE if a >= 0.9 else (figstyle.YELLOW if a >= 0.75 else figstyle.ORANGE)
              for a in pool_tab["lop_sign_agreement"]]
    ax.barh(y, pool_tab["lop_sign_agreement"], color=colors)
    ax.set_yticks(y); ax.set_yticklabels([f"{r.subtype}|{r.pathway}" for r in pool_tab.itertuples()], fontsize=6.5)
    ax.axvline(0.9, color=figstyle.MUTED, ls="--", lw=0.8)
    ax.set_xlabel("leave-one-pool-out sign agreement")
    ax.set_title("Pool-robustness (>=0.9 robust, 0.75-0.9 moderate)")
    ax.set_xlim(0, 1); figstyle.thin_despine(ax); _save(fig, "leavepool_signagreement")


def _ribo_plot():
    ribo = pd.read_csv(C.TAB_DIR / "check2_ribosome_technical.csv")
    p = ribo.pivot_table(index="subtype", columns="covariate", values="pearson_r")
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(p) + 2))
    im = ax.imshow(p.values, aspect="auto", cmap="RdBu_r", vmin=-0.8, vmax=0.8)
    ax.set_yticks(range(len(p.index))); ax.set_yticklabels(p.index, fontsize=8)
    ax.set_xticks(range(len(p.columns))); ax.set_xticklabels(p.columns, rotation=30, ha="right", fontsize=8)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            ax.text(j, i, f"{p.values[i,j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Ribosome module score vs technical covariates (Pearson r)")
    fig.colorbar(im, ax=ax); ax.grid(False); _save(fig, "ribosome_technical")
