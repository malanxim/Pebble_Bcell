"""tf_analysis.py —— Phase 2:donor pseudobulk TF 活性(decoupler)。

路线:donor×亚型 pseudobulk(主)→ CollecTRI 主网络 + DoRothEA 敏感性 → ULM 主 + MLM 敏感性。
年龄模型:**样条 bs(age,df=4) 为主**(非线性,Phase 1 已证显著曲率),线性每10岁为方向摘要。
donor/pool/网络/方法 多交叉验证;TF 成立按多标准(非仅 FDR)。避免同靶基因循环证明。
注:CollecTRI 把 NF-κB 家族合并为单个 "NFKB"(家族级活性,符合预期);无 AP-1(FOS/JUN)。
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
figstyle.set_style()

FOCUS = (["MYC", "MAX", "E2F1", "E2F2", "E2F3", "E2F4", "FOXM1",       # 增殖/生长
          "NFKB", "NR4A1", "NR4A2", "NR4A3", "NFATC1", "NFATC2",        # NF-κB/激活(memory)
          "STAT1", "STAT2", "IRF9", "IRF7", "IRF1",                     # IFN(naive)
          "TP53", "FOXO1", "FOXO3", "BACH2", "PAX5", "EBF1", "SPIB",    # 应激/身份(Tier2)
          "XBP1", "ATF4", "IRF4", "PRDM1"])                             # 浆细胞(探索)
# TF ↔ Hallmark 一致性核对
TF_PATHWAY = {"NFKB": "TNF-alpha Signaling via NF-kB", "STAT1": "Interferon Gamma Response",
              "STAT2": "Interferon Alpha Response", "IRF9": "Interferon Alpha Response",
              "IRF7": "Interferon Alpha Response", "MYC": "Myc Targets V1",
              "E2F1": "E2F Targets", "E2F4": "E2F Targets", "FOXM1": "G2-M Checkpoint"}


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(C.FIG_DIR / f"tf_{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _nets():
    import decoupler as dc
    ct = dc.get_collectri(organism="human")
    ct["weight"] = pd.to_numeric(ct["weight"], errors="coerce")
    ct = ct.dropna(subset=["weight"]).groupby(["source", "target"], as_index=False)["weight"].mean()
    ct["weight"] = ct["weight"].to_numpy(dtype=float)          # 消除 pandas nullable dtype
    doro = dc.get_dorothea(organism="human")
    doro = doro[doro["confidence"].isin(["A", "B"])]
    doro["weight"] = pd.to_numeric(doro["weight"], errors="coerce")
    doro = doro.dropna(subset=["weight"]).groupby(["source", "target"], as_index=False)["weight"].mean()
    doro["weight"] = doro["weight"].to_numpy(dtype=float)
    print(f"[tf] CollecTRI {ct['source'].nunique()} TFs; DoRothEA(A+B) {doro['source'].nunique()} TFs")
    return ct, doro


def _lognorm_pb(subtype):
    a = ad.read_h5ad(C.PB_DIR / f"pb_{subtype.replace(' ', '')}.h5ad")
    X = np.asarray(a.X.todense()) if sp.issparse(a.X) else np.asarray(a.X)
    lib = X.sum(1)
    lcpm = np.log1p(X / np.where(lib > 0, lib, 1)[:, None] * 1e4)
    mat = pd.DataFrame(lcpm, index=a.obs_names, columns=a.var["symbol"].astype(str))
    mat = mat.T.groupby(mat.columns).mean().T              # 同名基因合并
    obs = a.obs[["age", "sex", "pool"]].copy()
    obs["sex"] = obs["sex"].astype(str); obs["pool"] = obs["pool"].astype(str)
    del a; gc.collect()
    return mat, obs


def _activity(mat, net, method):
    import decoupler as dc
    res = dc.run_ulm(mat, net, min_n=5, use_raw=False) if method == "ulm" \
        else dc.run_mlm(mat, net, min_n=5, use_raw=False)
    acts = res[0] if isinstance(res, tuple) else res
    return acts


def _age_models(acts, obs, tf):
    """样条(主)+ 线性(方向)。返回 dict。"""
    d = acts[[tf]].copy(); d.columns = ["score"]
    d = d.join(obs[["age", "sex", "pool"]]).dropna()
    if len(d) < 50:
        return None
    mean_age = float(d["age"].mean())
    d["age_scaled"] = (d["age"] - mean_age) / C.AGE_SCALE
    try:
        ms = smf.ols("score ~ bs(age, df=4) + sex + pool", data=d).fit()
        bs_idx = [i for i, n in enumerate(ms.params.index) if "bs(" in n]
        if bs_idx:
            import numpy as _np
            R = _np.zeros((len(bs_idx), len(ms.params)))
            for k, i in enumerate(bs_idx): R[k, i] = 1
            p_spline = float(ms.wald_test(R).pvalue)
        else:
            p_spline = _np.nan
        ml = smf.ols("score ~ age_scaled + sex + pool", data=d).fit()
        lin = float(ml.params["age_scaled"]); lin_p = float(ml.pvalues["age_scaled"])
    except Exception as e:
        return {"error": str(e)[:60]}
    # 预测曲线(边际:sex/pool 取众数)
    grid = pd.DataFrame({"age": np.linspace(20, 95, 60)})
    grid["sex"] = d["sex"].mode()[0]; grid["pool"] = d["pool"].mode()[0]
    try:
        pred = ms.predict(grid).values
    except Exception:
        pred = _np.full(60, _np.nan)
    return {"n": len(d), "p_spline": p_spline, "linear_per10yr": lin, "linear_p": lin_p,
            "pred_age": list(grid["age"]), "pred_score": list(pred)}


def _robustness(acts, obs, tf):
    d = acts[[tf]].copy(); d.columns = ["score"]
    d = d.join(obs[["age", "sex", "pool"]]).dropna()
    mean_age = float(d["age"].mean()); d["age_scaled"] = (d["age"] - mean_age) / C.AGE_SCALE
    base = _ols_sign(d, "age_scaled")
    # drop top 1%/5% TF-high donor
    s1 = _ols_sign(d[d["score"] <= d["score"].quantile(0.99)], "age_scaled")
    s5 = _ols_sign(d[d["score"] <= d["score"].quantile(0.95)], "age_scaled")
    # leave-one-pool
    pools = [p for p in d["pool"].unique() if (d["pool"] != p).sum() >= 50]
    signs = [_ols_sign(d[d["pool"] != p], "age_scaled") for p in pools]
    signs = [s for s in signs if s is not None]
    lop = float(np.mean([np.sign(s) == np.sign(base) for s in signs])) if signs and base else _np.nan
    return {"drop1_sign": s1, "drop5_sign": s5, "lop_sign_agree": lop, "main_sign": base}


def _ols_sign(d, col):
    try:
        m = smf.ols(f"score ~ {col} + sex + pool", data=d).fit()
        return float(m.params[col])
    except Exception:
        return None


def run():
    ct, doro = _nets()
    focus = [t for t in FOCUS if t in set(ct["source"]) | set(doro["source"])]
    print(f"[tf] 进入分析的 focus TF({len(focus)}): {focus}")
    all_rows, curves, cross = [], {}, {}
    for s in C.MAIN_SUBTYPES:
        mat, obs = _lognorm_pb(s)
        acts_ct_ulm = _activity(mat, ct, "ulm")
        acts_ct_mlm = _activity(mat, ct, "mlm")
        acts_doro_ulm = _activity(mat, doro, "ulm")
        # 存活性矩阵
        acts_ct_ulm.to_csv(C.TAB_DIR / f"tf_activity_{s.replace(' ','')}_collectri_ulm.csv")
        # Hallmark 模块评分(与 TF 活性相关,验证一致性)
        pw_scores = _hallmark_scores(s, mat)
        for tf in focus:
            if tf not in acts_ct_ulm.columns:
                continue
            am = _age_models(acts_ct_ulm, obs, tf)
            rb = _robustness(acts_ct_ulm, obs, tf)
            if not am or "error" in am:
                continue
            # 网络/方法一致性(方向)
            doro_dir = _sign_change(acts_doro_ulm, obs, tf) if tf in acts_doro_ulm.columns else None
            mlm_dir = _sign_change(acts_ct_mlm, obs, tf) if tf in acts_ct_mlm.columns else None
            # TF-Hallmark 相关
            pw_corr = None
            if tf in TF_PATHWAY and TF_PATHWAY[tf] in pw_scores.columns:
                pw_corr = round(float(acts_ct_ulm[tf].corr(pw_scores[TF_PATHWAY[tf]])), 3)
            row = {"subtype": s, "tf": tf, **{k: am[k] for k in ["n", "p_spline", "linear_per10yr", "linear_p"]},
                   "main_sign": rb["main_sign"], "drop1_sign": rb["drop1_sign"], "drop5_sign": rb["drop5_sign"],
                   "lop_sign_agree": rb["lop_sign_agree"], "doro_sign": doro_dir, "mlm_sign": mlm_dir,
                   "tf_pathway_corr": pw_corr}
            all_rows.append(row)
            if tf in ["MYC", "E2F1", "NFKB", "STAT1", "NR4A2", "FOXM1"]:
                curves[(s, tf)] = (am["pred_age"], am["pred_score"])
        del mat, acts_ct_ulm, acts_ct_mlm, acts_doro_ulm; gc.collect()
    res = pd.DataFrame(all_rows)
    res["p_spline_fdr"] = sm.stats.multipletests(res["p_spline"].fillna(1), method="fdr_bh")[1]
    res.to_csv(C.TAB_DIR / "tf_age_results.csv", index=False)
    print("\n[tf] 焦点 TF 年龄效应(样条 p + 线性方向 + 鲁棒性):")
    show = res[res["tf"].isin(["MYC", "E2F1", "E2F4", "FOXM1", "NFKB", "NR4A2", "STAT1", "STAT2", "IRF9", "FOXO1", "PAX5"])]
    print(show[["subtype", "tf", "p_spline_fdr", "linear_per10yr", "lop_sign_agree", "doro_sign", "mlm_sign", "tf_pathway_corr"]].round(3).to_string(index=False))
    _heatmap(res)
    _curves(curves)
    _confidence(res)
    return res


def _sign_change(acts, obs, tf):
    d = acts[[tf]].rename(columns={tf: "score"}).join(obs[["age", "sex", "pool"]]).dropna().copy()
    d["age_scaled"] = (d["age"] - d["age"].mean()) / C.AGE_SCALE
    s = _ols_sign(d, "age_scaled")
    return int(np.sign(s)) if s is not None else None


def _hallmark_scores(subtype, mat):
    """复用 Hallmark 基因集给每 donor 算模块评分(与 drivecheck 同口径)。"""
    import gseapy as gp
    lib = gp.get_library(C.HALLMARK_LIB, organism="human")
    out = {}
    for term, genes in lib.items():
        g = [x for x in genes if x in mat.columns]
        if g:
            out[term] = mat[g].mean(axis=1)
    return pd.DataFrame(out, index=mat.index)


def _heatmap(res):
    piv = res.pivot_table(index="tf", columns="subtype", values="linear_per10yr")
    fig, ax = plt.subplots(figsize=(0.5 * piv.shape[1] + 2.5, 0.3 * piv.shape[0] + 1))
    ax.imshow(piv.values, aspect="auto", cmap="RdBu_r",
              vmin=-np.nanmax(np.abs(piv.values)), vmax=np.nanmax(np.abs(piv.values)))
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=7)
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=20, ha="right", fontsize=8)
    ax.set_title("TF activity: linear age effect (per 10 yr, CollecTRI ULM)")
    ax.grid(False); _save(fig, "age_effect_heatmap")


def _curves(curves):
    if not curves:
        return
    keys = list(curves.keys()); ncol = 3; nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 2.8 * nrow), sharex=True)
    axes = np.array(axes).reshape(-1)
    for k, (s, tf) in enumerate(keys):
        ax = axes[k]; x, y = curves[(s, tf)]
        ax.plot(x, y, color=figstyle.BLUE, lw=1.8)
        ax.axhline(0, color=figstyle.MUTED, lw=0.5)
        ax.set_title(f"{s} | {tf}", fontsize=9); figstyle.thin_despine(ax)
        if k >= len(keys) - ncol: ax.set_xlabel("age")
        if k % ncol == 0: ax.set_ylabel("TF activity (ULM)")
    for k in range(len(keys), len(axes)): axes[k].axis("off")
    fig.suptitle("TF activity vs age (spline, donor pseudobulk)", fontsize=11); fig.tight_layout()
    _save(fig, "nonlinear_curves")


def _confidence(res):
    """多标准置信度打分(非仅 FDR)。"""
    def tier(r):
        score = 0
        if not np.isnan(r["p_spline_fdr"]) and r["p_spline_fdr"] < 0.05: score += 1
        if r["lop_sign_agree"] is not None and not np.isnan(r["lop_sign_agree"]) and r["lop_sign_agree"] >= 0.9: score += 1
        if r["drop5_sign"] is not None and np.sign(r["drop5_sign"]) == np.sign(r["main_sign"]): score += 1
        if r["doro_sign"] is not None and r["doro_sign"] == np.sign(r["main_sign"]): score += 1
        if r["mlm_sign"] is not None and r["mlm_sign"] == np.sign(r["main_sign"]): score += 1
        if r["tf_pathway_corr"] is not None and abs(r["tf_pathway_corr"]) >= 0.3 and np.sign(r["tf_pathway_corr"]) == np.sign(r["main_sign"]): score += 1
        return ("高可信" if score >= 4 else ("中可信" if score >= 2 else "探索性")), score
    res[["confidence", "evidence_score"]] = res.apply(tier, axis=1, result_type="expand")
    res.to_csv(C.TAB_DIR / "tf_age_results_scored.csv", index=False)
    print("\n[tf] 置信度分级:")
    hc = res[res["confidence"] == "高可信"][["subtype", "tf", "linear_per10yr", "p_spline_fdr", "evidence_score"]]
    print("高可信:"); print(hc.round(3).to_string(index=False) if len(hc) else "  (无)")
    mc = res[res["confidence"] == "中可信"][["subtype", "tf", "linear_per10yr", "evidence_score"]]
    print("中可信:"); print(mc.round(3).to_string(index=False) if len(mc) else "  (无)")
    return res
