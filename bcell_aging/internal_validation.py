"""internal_validation.py —— 内部 split-half 复现 + memory–intermediate 连续轴 + age×sex。

不追求外部验证;只用 981 donor 做严格内部复现,证明小效应真实、协调、可重复。
1) 分层 split-half(donor 按 age×sex 分层分两半);
2) 核心 TF(FOXO3×3, PAX5, STAT1, NFKB)+ MYC/E2F 模块评分:discovery→replication 方向+效应;
3) memory–intermediate 连续状态轴(discovery 定权重→replication 验证 vs age);
4) age×sex 交互(仅核心程序)。
"""
from __future__ import annotations
import gc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from . import config as C
from . import paper_design as D
from .tf_analysis import _nets, _lognorm_pb
D.apply_style()

T = C.TAB_DIR
PAN = D.OUT / "panels"
CORE_TF = [("FOXO3", "B naive"), ("FOXO3", "B memory"), ("FOXO3", "B intermediate"),
           ("PAX5", "B intermediate"), ("STAT1", "B naive"), ("NFKB", "B memory")]


def _split_donors():
    """按 age 三分位 × sex 分层 split-half。"""
    d = pd.read_csv(T / "donor_summary.csv")[["donor_id", "age", "sex"]].drop_duplicates()
    d["age_ter"] = pd.qcut(d["age"], 3, labels=["y", "m", "o"])
    d["strat"] = d["age_ter"].astype(str) + "_" + d["sex"].astype(str)
    disc_idx, rep_idx = train_test_split(d.index, test_size=0.5, random_state=2026, stratify=d["strat"])
    disc = set(d.loc[disc_idx, "donor_id"]); rep = set(d.loc[rep_idx, "donor_id"])
    print(f"[val] split: discovery {len(disc)}, replication {len(rep)}")
    return disc, rep


def _fit_linear(acts_df, obs_df, tf):
    """OLS tf~age_scaled+sex+pool;返回 coef(per10yr), SE, delta_40to80, p。"""
    d = pd.DataFrame({"s": acts_df[tf].values}, index=obs_df.index).join(obs_df[["age", "sex", "pool"]]).dropna()
    if len(d) < 30:
        return dict(n=len(d), coef=np.nan, se=np.nan, delta=np.nan, p=np.nan, sd=np.nan)
    d["age_scaled"] = (d["age"] - d["age"].mean()) / 10.0
    m = smf.ols("s ~ age_scaled + sex + pool", data=d).fit()
    c = float(m.params["age_scaled"]); se = float(m.bse["age_scaled"])
    return dict(n=len(d), coef=c, se=se, delta=4*c, p=float(m.pvalues["age_scaled"]), sd=float(d["s"].std()))


def tf_replication(disc_ids, rep_ids):
    """每亚型:在 discovery 和 replication donor 上分别算 ULM,拟合,比较。"""
    ct, _ = _nets()
    import decoupler as dc
    rows = []
    for sub in C.MAIN_SUBTYPES:
        mat, obs = _lognorm_pb(sub)
        obs["did"] = obs.index.str.split("__").str[0]
        # discovery
        m_d = mat[obs["did"].isin(disc_ids)]
        o_d = obs[obs["did"].isin(disc_ids)]
        acts_d = dc.run_ulm(m_d, ct, min_n=5, use_raw=False)[0]
        # replication
        m_r = mat[obs["did"].isin(rep_ids)]
        o_r = obs[obs["did"].isin(rep_ids)]
        acts_r = dc.run_ulm(m_r, ct, min_n=5, use_raw=False)[0]
        for tf, _ in [(t, s) for t, s in CORE_TF if s == sub]:
            if tf not in acts_d.columns or tf not in acts_r.columns:
                continue
            fd = _fit_linear(acts_d, o_d, tf)
            fr = _fit_linear(acts_r, o_r, tf)
            sign_agree = (np.sign(fd["coef"]) == np.sign(fr["coef"])) if not (np.isnan(fd["coef"]) or np.isnan(fr["coef"])) else False
            rows.append({"tf": tf, "subtype": sub,
                         "disc_delta": round(fd["delta"], 4), "disc_p": fd["p"],
                         "rep_delta": round(fr["delta"], 4), "rep_p": fr["p"],
                         "disc_sd_frac": round(fd["delta"] / fd["sd"], 3) if fd["sd"] > 0 else np.nan,
                         "rep_sd_frac": round(fr["delta"] / fr["sd"], 3) if fr["sd"] > 0 else np.nan,
                         "sign_agree": bool(sign_agree),
                         "disc_n": fd["n"], "rep_n": fr["n"]})
        del mat, obs, m_d, o_d, m_r, o_r, acts_d, acts_r; gc.collect()
    res = pd.DataFrame(rows)
    res.to_csv(T / "tf_internal_replication.csv", index=False)
    print("[val] TF split-half replication:")
    print(res[["tf", "subtype", "disc_delta", "rep_delta", "sign_agree", "disc_sd_frac", "rep_sd_frac"]].to_string(index=False))
    return res


def _mem_inter_axis(disc_ids, rep_ids):
    """memory–intermediate 连续状态轴:discovery 定权重→replication 验证 vs age。"""
    am = ad.read_h5ad(C.PB_DIR / "pb_Bmemory.h5ad")
    ai = ad.read_h5ad(C.PB_DIR / "pb_Bintermediate.h5ad")
    Xm = np.asarray(am.X.todense()) if sp.issparse(am.X) else np.asarray(am.X)
    Xi = np.asarray(ai.X.todense()) if sp.issparse(ai.X) else np.asarray(ai.X)
    lm = Xm.sum(1); lm_s = np.where(lm > 0, lm, 1)[:, None]; lcpm_m = np.log1p(Xm / lm_s * 1e4)
    li = Xi.sum(1); li_s = np.where(li > 0, li, 1)[:, None]; lcpm_i = np.log1p(Xi / li_s * 1e4)
    sym_m = am.var["symbol"].astype(str).values
    sym_i = ai.var["symbol"].astype(str).values
    common = sorted(set(sym_m) & set(sym_i))
    mi = {g: i for i, g in enumerate(sym_m)}; ii = {g: i for i, g in enumerate(sym_i)}
    idx_m = [mi[g] for g in common]; idx_i = [ii[g] for g in common]
    lcpm_m = lcpm_m[:, idx_m]; lcpm_i = lcpm_i[:, idx_i]; sym = np.array(common)
    dm_ids = am.obs["donor_id"].astype(str)
    disc_mask_m = dm_ids.isin(disc_ids).values
    disc_mask_i = ai.obs["donor_id"].astype(str).isin(disc_ids).values
    w = lcpm_m[disc_mask_m].mean(0) - lcpm_i[disc_mask_i].mean(0)
    top = np.argsort(np.abs(w))[-200:]
    w_top = w[top]; sym_top = sym[top]
    smap = {g: j for j, g in enumerate(sym)}
    # score = 每个 donor 的 memory 和 intermediate pseudobulk 在轴上的位置
    def score(lcpm):
        idx = [smap.get(g) for g in sym_top if g in smap]
        w = np.array([w_top[list(sym_top).index(g)] for g in sym_top if g in smap])
        sub = lcpm[:, idx]
        return (sub * w[None, :]).mean(axis=1) / (np.abs(w).sum() + 1e-9)
    sm = score(lcpm_m); si = score(lcpm_i)
    # 每 donor 的 memory 和 intermediate 轴位置(均值=该 donor 的记忆-中间状态)
    dm = am.obs[["donor_id", "age", "sex", "pool"]].copy(); dm["ax"] = sm
    di = ai.obs[["donor_id", "age", "sex", "pool"]].copy(); di["ax"] = si
    combined = pd.concat([dm, di])
    # replication donors
    rep = combined[combined["donor_id"].astype(str).isin(rep_ids)].dropna(subset=["age"])
    disc = combined[combined["donor_id"].astype(str).isin(disc_ids)].dropna(subset=["age"])
    # axis ~ age
    def fit(d):
        d = d.copy(); d["age_scaled"] = (d["age"] - d["age"].mean()) / 10.0
        if len(d) < 30: return np.nan, np.nan, len(d)
        m = smf.ols("ax ~ age_scaled + sex + pool", data=d).fit()
        return float(m.params["age_scaled"]), float(m.pvalues["age_scaled"]), len(d)
    dc_coef, dc_p, dc_n = fit(disc)
    rp_coef, rp_p, rp_n = fit(rep)
    print(f"\n[val] memory–intermediate axis: disc coef/10yr={dc_coef:.5f} (p={dc_p:.3g}, n={dc_n})"
          f"  | rep coef/10yr={rp_coef:.5f} (p={rp_p:.3g}, n={rp_n})"
          f"  | sign_agree={np.sign(dc_coef)==np.sign(rp_coef)}")
    combined.to_csv(T / "mem_inter_axis_scores.csv", index=False)
    del am, ai; gc.collect()
    return dict(disc_coef=dc_coef, disc_p=dc_p, rep_coef=rp_coef, rep_p=rp_p,
                sign_agree=bool(np.sign(dc_coef) == np.sign(rp_coef)))


def age_sex():
    """核心程序 age×sex 交互。"""
    ct, _ = _nets()
    import decoupler as dc
    rows = []
    for sub in C.MAIN_SUBTYPES:
        mat, obs = _lognorm_pb(sub)
        acts = dc.run_ulm(mat, ct, min_n=5, use_raw=False)[0]
        for tf, _ in [(t, s) for t, s in CORE_TF if s == sub]:
            if tf not in acts.columns: continue
            d = pd.DataFrame({"s": acts[tf].values}, index=obs.index).join(obs[["age", "sex", "pool"]]).dropna()
            d["age_scaled"] = (d["age"] - d["age"].mean()) / 10.0
            try:
                m = smf.ols("s ~ age_scaled * sex + pool", data=d).fit()
                inter = [c for c in m.params.index if ":" in c and "age" in c.lower() and "sex" in c.lower()]
                if inter:
                    ic = inter[0]
                    rows.append({"tf": tf, "subtype": sub, "interaction_p": float(m.pvalues[ic]),
                                 "interaction_coef": round(float(m.params[ic]), 5)})
            except Exception:
                pass
        del mat, obs, acts; gc.collect()
    res = pd.DataFrame(rows)
    res.to_csv(T / "age_sex_interaction.csv", index=False)
    print("\n[val] age×sex interaction (core TFs):")
    print(res.to_string(index=False))
    return res


def run():
    disc, rep = _split_donors()
    tf_res = tf_replication(disc, rep)
    axis_res = _mem_inter_axis(disc, rep)
    sex_res = age_sex()
    # 汇总
    print("\n=== 内部验证汇总 ===")
    print(f"TF 方向一致率: {tf_res['sign_agree'].mean()*100:.0f}%")
    print(f"memory–intermediate 轴: 方向一致 {axis_res['sign_agree']}")
    print(f"age×sex: {(sex_res['interaction_p']<0.05).sum()}/{len(sex_res)} 显著交互")
    return tf_res, axis_res, sex_res


if __name__ == "__main__":
    run()
