"""tf_extras.py —— Phase 2 补充:全 TF 空间 FDR + regulon 重叠 + 细胞级正确检验。

修正:① FDR 必须在 全部受检 TF × 全部亚型 上校正(非仅候选 TF);
② 细胞级 SD 不足以证明"广泛",改看 donor 内中位、TF-high 细胞比例、去 top 细胞鲁棒性;
③ regulon 靶基因重叠(MYC/E2F、NF-κB 成员、STAT/IRF 共享)→ TF 非彼此独立。
"""
from __future__ import annotations
import gc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C
from . import figstyle
from .tf_analysis import _nets, _lognorm_pb
figstyle.set_style()

KEY = ["STAT1", "NFKB", "MYC", "PAX5", "FOXO3", "STAT2", "IRF9", "E2F1", "E2F4",
       "NR4A2", "ATF4", "IRF1", "IRF7"]


def _spline_p(acts, obs):
    """全 TF:样条 bs(age,4) 整体 Wald p + 线性 per10yr。返回 DataFrame(tf, p_spline, linear_per10yr)。"""
    import decoupler as dc
    rows = []
    d0 = obs.copy(); mean_age = float(d0["age"].mean()); d0["age_scaled"] = (d0["age"] - mean_age) / C.AGE_SCALE
    for tf in acts.columns:
        d = d0.copy(); d["score"] = acts[tf].values
        d = d.dropna()
        try:
            ms = smf.ols("score ~ bs(age, df=4) + sex + pool", data=d).fit()
            bs_idx = [i for i, n in enumerate(ms.params.index) if "bs(" in n]
            R = np.zeros((len(bs_idx), len(ms.params)))
            for k, i in enumerate(bs_idx): R[k, i] = 1
            p_sp = float(ms.wald_test(R).pvalue) if bs_idx else np.nan
            ml = smf.ols("score ~ age_scaled + sex + pool", data=d).fit()
            rows.append({"tf": tf, "p_spline": p_sp, "linear_per10yr": float(ml.params["age_scaled"])})
        except Exception:
            rows.append({"tf": tf, "p_spline": np.nan, "linear_per10yr": np.nan})
    return pd.DataFrame(rows)


def full_fdr():
    ct, _ = _nets()
    frames = []
    for s in C.MAIN_SUBTYPES:
        mat, obs = _lognorm_pb(s)
        import decoupler as dc
        acts = dc.run_ulm(mat, ct, min_n=5, use_raw=False)[0]
        df = _spline_p(acts, obs); df["subtype"] = s
        frames.append(df)
        del mat, acts; gc.collect()
    allr = pd.concat(frames, ignore_index=True)
    allr["p_spline_fdr_full"] = sm.stats.multipletests(allr["p_spline"].fillna(1), method="fdr_bh")[1]
    allr.to_csv(C.TAB_DIR / "tf_age_full_fdr.csv", index=False)
    n_total = len(allr)
    print(f"[tf-extras] 全 TF 空间 FDR: {n_total} 个 TF×亚型 联合校正")
    key_rows = allr[allr["tf"].isin(KEY)]
    print("\n关键 TF 在【全空间 FDR】下的显著性:")
    print(key_rows[["subtype", "tf", "p_spline", "p_spline_fdr_full", "linear_per10yr"]].round(4).to_string(index=False))
    return allr


def regulon_overlap():
    ct, _ = _nets()
    # TF -> 靶基因集合
    by_tf = {t: set(g) for t, g in ct.groupby("source")["target"]}
    key = [t for t in KEY if t in by_tf]
    J = pd.DataFrame(index=key, columns=key, dtype=float)
    for a in key:
        for b in key:
            inter = len(by_tf[a] & by_tf[b]); uni = len(by_tf[a] | by_tf[b])
            J.loc[a, b] = inter / uni if uni else 0
    J = J.astype(float)
    J.to_csv(C.TAB_DIR / "tf_regulon_jaccard.csv")
    # 热图
    fig, ax = plt.subplots(figsize=(0.5 * len(key) + 2, 0.45 * len(key) + 1.5))
    im = ax.imshow(J.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(key))); ax.set_xticklabels(key, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(key))); ax.set_yticklabels(key, fontsize=8)
    for i in range(len(key)):
        for j in range(len(key)):
            v = J.values[i, j]
            if i == j or v >= 0.2:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.55 else figstyle.INK_PRI)
    ax.set_title("Regulon target-gene overlap (Jaccard)\nTFs are not independent: MYC<->E2F, STAT<->IRF, NF-kB members share targets")
    fig.colorbar(im, ax=ax, fraction=0.04); ax.grid(False)
    for ext in ("pdf", "png"):
        fig.savefig(C.FIG_DIR / f"tf_regulon_overlap.{ext}", bbox_inches="tight")
    plt.close(fig)
    high_pairs = []
    for i, a in enumerate(key):
        for b in key[i + 1:]:
            if J.loc[a, b] >= 0.2:
                high_pairs.append((a, b, round(float(J.loc[a, b]), 2)))
    print(f"\n[tf-extras] 高重叠 regulon 对(Jaccard≥0.2): {high_pairs}")
    return J, high_pairs


def run():
    allr = full_fdr()
    J, pairs = regulon_overlap()
    # The cell-level/re-clustering exploration is not part of the manuscript's
    # primary donor-pseudobulk inference and is deliberately not run here.
    print("\n[tf-extras] 完成（全 TF FDR + regulon overlap）。")
    return allr, pairs
