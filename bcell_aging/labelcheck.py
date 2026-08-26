"""labelcheck.py —— 检查3:组成标签敏感性 + 连续状态评分 + 年龄非线性。

A) 合并 memory+intermediate 重做组成(quasi-binomial):memory 类总群是否仍随龄降。
B) 注明:本分析组成/DEG 均已用 Azimuth 原标签(predicted.celltype.l2),非 de novo 重聚类边界。
C) 连续模块评分(naive/memory/ABC/plasma)每 donor×亚型 ~ age:状态轴漂移。
D) 非线性:对组成比例用 B 样条(age) vs 线性,看 60+/80+ 是否有转折。
"""
from __future__ import annotations
import gc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import patsy
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C
from . import figstyle
figstyle.set_style()

SIGS = {"naive": C.MARKER_GENES["naive"][:6], "memory": C.MARKER_GENES["memory"],
        "abc": C.MARKER_GENES["abc"], "plasma": ["MZB1", "JCHAIN", "XBP1", "PRDM1", "IRF4"]}


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(C.FIG_DIR / f"check3_{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


# ---------- A: 合并 memory+intermediate 组成 ----------
def merge_composition():
    dc = pd.read_csv(C.TAB_DIR / "per_donor_subtype_counts.csv")
    dc["n_meminter"] = dc["n_Bmemory"] + dc["n_Bintermediate"]
    rows = []
    for label, col in [("B naive", "n_Bnaive"), ("memory+intermediate", "n_meminter"),
                       ("Plasmablast", "n_Plasmablast")]:
        d = dc[["age", "sex", "pool", "n_B", col]].rename(columns={col: "k"}).copy()
        d = d[d["n_B"] >= C.MIN_CELLS]
        d["k"] = d["k"].astype(int); d["frac"] = (d["k"] / d["n_B"]).clip(1e-4, 1 - 1e-4)
        mean_age = float(d["age"].mean())
        d["age_scaled"] = (d["age"] - mean_age) / C.AGE_SCALE
        y, X = patsy.dmatrices("frac ~ age_scaled + sex + pool", data=d, return_type="dataframe")
        m = sm.GLM(y, X, family=sm.families.Binomial(), var_weights=d["n_B"].values).fit()
        phi = m.pearson_chi2 / m.df_resid
        mq = sm.GLM(y, X, family=sm.families.Binomial(), var_weights=d["n_B"].values).fit(scale=phi)
        X40, X80 = X.copy(), X.copy()
        X40["age_scaled"] = (40 - mean_age) / C.AGE_SCALE
        X80["age_scaled"] = (80 - mean_age) / C.AGE_SCALE
        mu40, mu80 = float(m.predict(X40).mean()), float(m.predict(X80).mean())
        rows.append({"group": label, "n": len(d), "beta_age10yr": round(float(mq.params["age_scaled"]), 4),
                     "p_quasi": float(mq.pvalues["age_scaled"]),
                     "delta_40to80_pp": round((mu80 - mu40) * 100, 2)})
    res = pd.DataFrame(rows)
    res.to_csv(C.TAB_DIR / "check3_merge_composition.csv", index=False)
    print("[check3-A] 合并 memory+intermediate 组成(quasi-binomial):")
    print(res.round(4).to_string(index=False))
    return res


# ---------- C: 连续模块评分 ----------
def module_scores():
    varmap = pd.read_csv(C.PB_DIR / "pb_gene_map.csv")
    out = []
    for s in C.MAIN_SUBTYPES:
        a = ad.read_h5ad(C.PB_DIR / f"pb_{s.replace(' ', '')}.h5ad")
        X = a.X; X = np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X)
        lib = X.sum(axis=1); lcpm = np.log1p(X / np.where(lib > 0, lib, 1)[:, None] * 1e6)
        sym = a.var["symbol"].astype(str).values; smap = {g: i for i, g in enumerate(sym)}
        df = pd.DataFrame({"age": a.obs["age"].values, "sex": a.obs["sex"].astype(str).values,
                           "pool": a.obs["pool"].astype(str).values})
        for signame, genes in SIGS.items():
            idx = [smap[g] for g in genes if g in smap]
            if not idx:
                continue
            df[signame] = lcpm[:, idx].mean(axis=1)
            try:
                m = smf_ols(f"{signame} ~ age + sex + pool", df).fit()
                out.append({"subtype": s, "signature": signame, "n": len(df),
                            "slope_age": round(float(m.params["age"]), 5), "p": float(m.pvalues["age"])})
            except Exception:
                pass
        del a; gc.collect()
    res = pd.DataFrame(out)
    res.to_csv(C.TAB_DIR / "check3_module_scores.csv", index=False)
    print("\n[check3-C] 连续模块评分 ~ age(donor 级):")
    print(res.round(4).to_string(index=False))
    return res


def smf_ols(formula, df):
    import statsmodels.formula.api as smf
    return smf.ols(formula, data=df)


# ---------- D: 非线性(B 样条 vs 线性) ----------
def nonlinearity():
    dc = pd.read_csv(C.TAB_DIR / "per_donor_subtype_counts.csv")
    dc = dc[dc["n_B"] >= C.MIN_CELLS].copy()
    rows = []
    curves = {}
    for label, col in [("B naive", "n_Bnaive"), ("B memory", "n_Bmemory"),
                       ("B intermediate", "n_Bintermediate")]:
        d = dc[["age", "sex", "pool", "n_B", col]].rename(columns={col: "k"}).copy()
        d["frac"] = (d["k"] / d["n_B"]).clip(1e-4, 1 - 1e-4)
        try:
            lin = smf_glm("frac ~ age + sex + pool", d, d["n_B"].values).fit()
            spl = smf_glm("frac ~ bs(age, df=4) + sex + pool", d, d["n_B"].values).fit()
            bs_cols = [c for c in spl.params.index if c.startswith("bs(")]
            rows.append({"group": label, "linear_p": float(lin.pvalues["age"]),
                         "spline_dev_vs_linear": round(lin.aic - spl.aic, 1),
                         "spline_terms": len(bs_cols)})
        except Exception as e:
            print(f"[check3-D] {label} 非线性失败: {str(e)[:60]}")
    res = pd.DataFrame(rows)
    res.to_csv(C.TAB_DIR / "check3_nonlinearity.csv", index=False)
    print("\n[check3-D] 非线性(spline_dev_vs_linear>0 = 样条更优):")
    print(res.round(4).to_string(index=False) if len(res) else "(无)")
    _nl_plot(dc)
    return res


def smf_glm(formula, d, vw):
    import statsmodels.formula.api as smf
    return smf.glm(formula, data=d, family=sm.families.Binomial(), var_weights=vw)


def _nl_plot(dc):
    """各亚型比例 随年龄 的滚动均值(直观非线性)。"""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharex=True)
    pal = [figstyle.BLUE, figstyle.ORANGE, figstyle.YELLOW]
    for k, (label, col) in enumerate([("B naive", "n_Bnaive"), ("B memory", "n_Bmemory"),
                                      ("B intermediate", "n_Bintermediate")]):
        ax = axes[k]
        d = dc[["age", "n_B", col]].dropna().copy()
        d["frac"] = d[col] / d["n_B"]
        ax.scatter(d["age"], d["frac"], s=5, alpha=0.2, color=pal[k])
        srt = d.sort_values("age")
        ax.plot(srt["age"], srt["frac"].rolling(80, min_periods=15, center=True).mean(),
                color=pal[k], lw=1.8)
        ax.set_title(label, fontsize=9); figstyle.thin_despine(ax); ax.set_xlabel("age")
        if k == 0: ax.set_ylabel("donor fraction")
    fig.suptitle("Composition fraction vs age (rolling mean — check for nonlinearity)")
    fig.tight_layout(); _save(fig, "nonlinearity_rolling")


def run():
    A = merge_composition()
    C_res = module_scores()
    D = nonlinearity()
    print("\n[check3] 完成。")
    return A, C_res, D
