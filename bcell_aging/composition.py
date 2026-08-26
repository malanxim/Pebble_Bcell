"""composition.py —— Part 5:donor 级 B 亚型比例 vs 连续年龄(组成分析,与 DEG 分开)。

统计:每亚型 success=该亚型细胞数, trials=该 donor 总 B 细胞数;
binomial GLM(带 var_weights=n_B)frac ~ age_scaled + sex + C(pool) -> 每 10 岁斜率+CI+p。
注:这是"组成变化",与亚型内表达变化(DEG)是两类不同效应,必须分开报告。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
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
    print(f"[composition] -> figures/comp_{name}.{{pdf,png}}")


def fit_one(frac_df, subtype):
    """binomial GLM frac ~ age_scaled + sex + C(pool),返回 slope/CI/p/per10yr。"""
    col = f"frac_{SHORT[subtype]}"
    d = frac_df[[col, "age", "sex", "pool", "n_B"]].rename(
        columns={col: "frac"}).dropna()
    d = d[d["n_B"] >= C.MIN_CELLS].copy()
    mean_age = float(d["age"].mean())
    d["age_scaled"] = (d["age"] - mean_age) / C.AGE_SCALE
    d["frac"] = d["frac"].clip(1e-4, 1 - 1e-4)
    try:
        # pool 为字符串列,patsy 自动作分类变量(勿用 C(),会与本模块别名 C 冲突)
        m = smf.glm("frac ~ age_scaled + sex + pool", data=d,
                    family=sm.families.Binomial(), var_weights=d["n_B"].values).fit()
        c = m.params["age_scaled"]; se = m.bse["age_scaled"]; p = m.pvalues["age_scaled"]
    except Exception as e:
        print(f"[composition] {subtype} GLM 失败({str(e)[:60]}),改连续年龄简单相关")
        r = d["age"].corr(d["frac"])
        return dict(subtype=subtype, n=len(d), slope_per10yr=np.nan, ci_lo=np.nan,
                    ci_hi=np.nan, p=np.nan, pearson=round(r, 3))
    return dict(subtype=subtype, n=len(d),
                slope_per10yr=round(c, 4),
                ci_lo=round(c - 1.96 * se, 4), ci_hi=round(c + 1.96 * se, 4),
                p=float(p), pearson=round(d["age"].corr(d["frac"]), 3))


def run(donors):
    """donors: data_check 产出的 donor 级表(含 frac_* / n_B)。"""
    frac = donors.copy()
    frac["n_B"] = frac["n_B"].astype(int)
    rows = []
    for s in C.SUBTYPES:
        r = fit_one(frac, s)
        rows.append(r)
        print(f"  {s:15s}: n={r['n']}  slope/10yr={r['slope_per10yr']}  "
              f"p={r['p']:.3g}" if not np.isnan(r['p']) else
              f"  {s:15s}: n={r['n']}  (GLM 失败)")
    res = pd.DataFrame(rows)
    res.to_csv(C.TAB_DIR / "composition_age_effects.csv", index=False)
    print(f"[composition] -> tables/composition_age_effects.csv")
    _plot_forest(res)
    _plot_scatter(frac)
    _plot_stacked(frac)
    _plot_heatmap(frac)
    return res


def _plot_forest(res):
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(res) + 1.5))
    y = np.arange(len(res))
    ax.errorbar(res["slope_per10yr"], y,
                xerr=[res["slope_per10yr"] - res["ci_lo"], res["ci_hi"] - res["slope_per10yr"]],
                fmt="o", color=figstyle.BLUE, capsize=3)
    ax.axvline(0, color=figstyle.MUTED, ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels([f"{r['subtype']}\n(n={r['n']})" for _, r in res.iterrows()], fontsize=9)
    ax.set_xlabel("fraction change per 10 yr (log-odds, binomial GLM)")
    ax.set_title("B subtype fraction vs age (donor-level)")
    for yi, r in enumerate(res.itertuples()):
        if not np.isnan(r.p):
            ax.text(r.ci_hi + 0.02, yi, f"p={r.p:.2g}", va="center", fontsize=7, color=figstyle.INK_SEC)
    figstyle.thin_despine(ax); _save(fig, "forest")


def _plot_scatter(frac):
    cats = [s for s in C.SUBTYPES]
    fig, axes = plt.subplots(1, len(cats), figsize=(3.4 * len(cats), 3.2), sharex=True)
    pal = [figstyle.BLUE, figstyle.ORANGE, figstyle.YELLOW, figstyle.MAGENTA]
    for k, s in enumerate(cats):
        ax = axes[k]; col = f"frac_{SHORT[s]}"
        d = frac[[col, "age"]].dropna()
        ax.scatter(d["age"], d[col], s=6, alpha=0.3, color=pal[k])
        srt = d.sort_values("age")
        ax.plot(srt["age"], srt[col].rolling(60, min_periods=10, center=True).mean(),
                color=pal[k], lw=1.5)
        ax.set_title(s, fontsize=9); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("age"); figstyle.thin_despine(ax)
        if k == 0: ax.set_ylabel("donor fraction")
    fig.suptitle("B subtype fraction vs age (donor-level)", fontsize=11); fig.tight_layout()
    _save(fig, "scatter_age")


def _plot_stacked(frac):
    d = frac.copy()
    d["ag"] = pd.cut(d["age"], C.AGE_BIN_EDGES, labels=C.AGE_BIN_LABELS)
    cols = [f"frac_{SHORT[s]}" for s in C.SUBTYPES]
    means = d.groupby("ag", observed=True)[cols].mean() * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    bottom = np.zeros(len(means)); pal = [figstyle.BLUE, figstyle.ORANGE, figstyle.YELLOW, figstyle.MAGENTA]
    for i, s in enumerate(C.SUBTYPES):
        c = f"frac_{SHORT[s]}"
        ax.bar(means.index.astype(str), means[c], bottom=bottom, color=pal[i], label=s)
        bottom += means[c].values
    ax.set_ylabel("% composition"); ax.set_title("Mean B subtype composition by age group")
    ax.legend(fontsize=7, loc="lower right"); figstyle.thin_despine(ax); _save(fig, "stacked_agegroup")


def _plot_heatmap(frac):
    d = frac.sort_values("age")
    cols = [f"frac_{SHORT[s]}" for s in C.SUBTYPES]
    mat = d[cols].values.T * 100
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0)
    ax.set_yticks(range(len(C.SUBTYPES))); ax.set_yticklabels(C.SUBTYPES, fontsize=8)
    ax.set_xticks([]); ax.set_xlabel(f"donors (sorted by age), n={len(d)}")
    ax.set_title("B subtype fraction per donor (sorted by age)")
    ax.grid(False); _save(fig, "donor_heatmap")
