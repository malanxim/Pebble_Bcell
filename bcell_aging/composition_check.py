"""composition_check.py —— 组成模型稳健性复核。

针对用户指出的问题:donor 级 binomial GLM 把每 donor 细胞总数当成大量二项试验,
可能严重低估 donor 间额外生物学变异 -> P 值过乐观。
复核:
  1) quasi-binomial(Pearson 离散 φ)校正年龄斜率 SE/p;
  2) β-binomial 精度 φ(MLE, 两步:均值用 binomial GLM 拟合 μ, 再估精度);
  3) 绝对比例变化:在 40 vs 80 岁 做边际标准化(marginal standardization),报 百分点差;
  4) 组成约束说明:4 亚型比例和为 1,memory/plasmablast 降会机械地让 naive/intermediate 升。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy.stats import betabinom
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C
from . import figstyle
figstyle.set_style()

SHORT = {"B naive": "Bnaive", "B memory": "Bmemory",
         "B intermediate": "Bintermediate", "Plasmablast": "Plasmablast"}


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(C.FIG_DIR / f"comp_{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _bb_precision(k, n, mu):
    """两步 β-binomial:均值 μ 已由 GLM 给定,MLE 估精度 φ(a=μφ,b=(1-μ)φ)。"""
    mu = np.clip(mu, 1e-3, 1 - 1e-3); k = k.astype(float); n = n.astype(float)

    def negll(logphi):
        phi = np.exp(logphi)
        return -np.sum(betabinom.logpmf(k, n, mu * phi, (1 - mu) * phi))
    res = minimize_scalar(negll, bounds=(np.log(0.05), np.log(5000)), method="bounded")
    return float(np.exp(res.x))


def check_one(donors, subtype):
    col = f"n_{SHORT[subtype]}"
    d = donors[["age", "sex", "pool", "n_B", col]].rename(columns={col: "k"}).copy()
    d = d[d["n_B"] >= C.MIN_CELLS].copy()
    d["k"] = d["k"].astype(int); d["n_B"] = d["n_B"].astype(int)
    mean_age = float(d["age"].mean())
    d["age_scaled"] = (d["age"] - mean_age) / C.AGE_SCALE
    d["frac"] = (d["k"] / d["n_B"]).clip(1e-4, 1 - 1e-4)
    y, X = patsy.dmatrices("frac ~ age_scaled + sex + pool", data=d, return_type="dataframe")
    vw = d["n_B"].values.astype(float)
    fam = sm.families.Binomial()

    m_bin = sm.GLM(y, X, family=fam, var_weights=vw).fit()         # 普通 binomial(过乐观)
    phi_pearson = float(m_bin.pearson_chi2 / m_bin.df_resid)        # 离散
    m_quasi = sm.GLM(y, X, family=fam, var_weights=vw).fit(scale=phi_pearson)  # quasi-binomial
    bb_phi = _bb_precision(d["k"].values, d["n_B"].values, m_bin.fittedvalues.values)

    # 边际标准化:40 vs 80 岁(保持各 donor 的 sex/pool)
    X40, X80 = X.copy(), X.copy()
    X40["age_scaled"] = (40 - mean_age) / C.AGE_SCALE
    X80["age_scaled"] = (80 - mean_age) / C.AGE_SCALE
    mu40, mu80 = float(m_bin.predict(X40).mean()), float(m_bin.predict(X80).mean())

    return {
        "subtype": subtype, "n_donors": len(d),
        "beta_age": round(float(m_bin.params["age_scaled"]), 4),
        "SE_binomial": round(float(m_bin.bse["age_scaled"]), 4),
        "p_binomial": float(m_bin.pvalues["age_scaled"]),
        "dispersion_pearson": round(phi_pearson, 2),
        "SE_quasi": round(float(m_quasi.bse["age_scaled"]), 4),
        "p_quasi": float(m_quasi.pvalues["age_scaled"]),
        "bb_precision_phi": round(bb_phi, 2),
        "pred_frac_age40": round(mu40, 4), "pred_frac_age80": round(mu80, 4),
        "delta_40to80_pp": round((mu80 - mu40) * 100, 2),
    }


def run(donors=None):
    if donors is None:
        donors = pd.read_csv(C.TAB_DIR / "per_donor_subtype_counts.csv")
    rows = [check_one(donors, s) for s in C.SUBTYPES]
    res = pd.DataFrame(rows)
    res.to_csv(C.TAB_DIR / "composition_robustness.csv", index=False)
    print("[composition_check] -> tables/composition_robustness.csv")
    show = res[["subtype", "n_donors", "beta_age", "p_binomial", "dispersion_pearson",
                "p_quasi", "bb_precision_phi", "pred_frac_age40", "pred_frac_age80",
                "delta_40to80_pp"]]
    print(show.round(4).to_string(index=False))
    _forest(res)
    _abschange_bar(res)
    return res


def _forest(res):
    """quasi-binomial 校正后的年龄效应森林图(对比 binomial 的 CI)。"""
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(res) + 1.5))
    y = np.arange(len(res))
    # binomial CI(过窄)与 quasi CI(校正)对比
    for i, r in enumerate(res.itertuples()):
        b = r.beta_age
        ax.errorbar(b, i - 0.12, xerr=1.96 * r.SE_binomial, fmt="o", color=figstyle.MUTED,
                    capsize=3, label="binomial (naive)" if i == 0 else None)
        ax.errorbar(b, i + 0.12, xerr=1.96 * r.SE_quasi, fmt="o", color=figstyle.BLUE,
                    capsize=3, label="quasi-binomial (overdisp-corrected)" if i == 0 else None)
    ax.axvline(0, color=figstyle.MUTED, ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels([f"{r.subtype}\n(n={r.n_donors})" for r in res.itertuples()], fontsize=9)
    ax.set_xlabel("fraction change per 10 yr (log-odds)")
    ax.set_title("B subtype fraction vs age: overdispersion-corrected (quasi-binomial)")
    ax.legend(fontsize=8); figstyle.thin_despine(ax); _save(fig, "forest_robust")


def _abschange_bar(res):
    """绝对比例变化(40→80 岁,百分点)—— 体现"统计显著但效应小"。"""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = [figstyle.BLUE, figstyle.ORANGE, figstyle.YELLOW, figstyle.MAGENTA]
    rr = res.sort_values("delta_40to80_pp")
    ax.barh(rr["subtype"], rr["delta_40to80_pp"], color=colors)
    ax.axvline(0, color=figstyle.MUTED, lw=1)
    for i, v in enumerate(rr["delta_40to80_pp"]):
        ax.text(v + (0.1 if v >= 0 else -0.1), i, f"{v:+.1f} pp", va="center",
                ha="left" if v >= 0 else "right", fontsize=8)
    ax.set_xlabel("predicted fraction change, age 40 → 80 (percentage points)")
    ax.set_title("Absolute composition change (marginal standardization)")
    figstyle.thin_despine(ax); _save(fig, "abschange_40to80")
