"""05_variance_age.py — does expression variance rise with age?

At all-B level (981 donors), for two gene sets:
  - sig_DEG : union of significant young-vs-old DEGs (the user's "significant genes")
  - global  : all well-expressed genes (mean log2-CPM >= 2.0) — genome-wide read
Compute per-decade (20s..90s) expression variance across donors, and test whether
dispersion increases with age. Done both RAW and POOL-RESIDUALIZED (batch-robust),
plus a depth-restricted (n_cells>=50) robustness check to rule out the sampling-noise
confound. CV = sd/mean separates mean-driven from true dispersion change.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import expr as E
from plotting import set_style, save, decade_color
import matplotlib.pyplot as plt

log = C.get_logger("05_variance", C.LOGS_DIR / f"05_variance_{C.timestamp()}.log")
N_BOOT = 1000


def per_gene_per_decade(Y: np.ndarray, didx: np.ndarray, decades_present):
    """Y donors x genes -> arrays var[g,d], mean[g,d], n[d]; list of valid decades."""
    nd = len(decades_present)
    var = np.full((Y.shape[1], nd), np.nan); mean = np.full_like(var, np.nan)
    n_d = np.zeros(nd, dtype=int)
    for di in range(nd):
        m = didx == di
        n_d[di] = int(m.sum())
        if m.sum() < C.DECADE_MIN_DONORS:
            continue
        var[:, di] = Y[m].var(axis=0, ddof=1)
        mean[:, di] = Y[m].mean(axis=0)
    cv = var ** 0.5 / np.maximum(mean, 1e-9)
    return var, mean, cv, n_d


def gene_trend(var: np.ndarray) -> np.ndarray:
    """Per-gene Spearman(var, decade) over decades with data; NaN if <4 decades."""
    x = np.arange(var.shape[1])
    out = np.full(var.shape[0], np.nan)
    for g in range(var.shape[0]):
        v = var[g]
        ok = ~np.isnan(v)
        if ok.sum() >= 4:
            out[g] = stats.spearmanr(x[ok], v[ok])[0]
    return out


def agg_curve(var: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean variance per decade + bootstrap SEM across genes."""
    mu = np.nanmean(var, axis=0)
    rng = np.random.default_rng(C.RANDOM_SEED)
    gidx = np.arange(var.shape[0])
    boot = np.zeros((N_BOOT, var.shape[1]))
    for b in range(N_BOOT):
        boot[b] = np.nanmean(var[rng.choice(gidx, gidx.size, replace=True)], axis=0)
    se = np.nanstd(boot, axis=0)
    return mu, se


def resid_pool(Y: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """Project out per-pool mean (within-pool residuals) for batch-robust dispersion."""
    P = pd.get_dummies(meta["pool_number"].astype(int), prefix="p").values.astype(np.float64)
    P = np.column_stack([np.ones(len(P)), P])      # intercept + pool dummies
    beta, *_ = np.linalg.lstsq(P, Y, rcond=None)
    return Y - P @ beta


def analyze(name, Y_raw, Y_res, didx, decades_present, axes_curves, axes_cv, axes_trend):
    var_r, mean_r, cv_r, n_d = per_gene_per_decade(Y_raw, didx, decades_present)
    var_e, mean_e, cv_e, _ = per_gene_per_decade(Y_res, didx, decades_present)
    mu_r, se_r = agg_curve(var_r); mu_e, se_e = agg_curve(var_e)
    cvmu_r, cvse_r = agg_curve(cv_r); cvmu_e, cvse_e = agg_curve(cv_e)
    tr_r = gene_trend(var_r); tr_e = gene_trend(var_e)
    x = np.arange(len(decades_present))

    def sp(a):
        ok = ~np.isnan(a)
        return stats.spearmanr(np.arange(a.shape[0])[ok], a[ok]) if ok.sum() >= 4 else (np.nan, np.nan)

    r_r, p_r = sp(mu_r); r_e, p_e = sp(mu_e)
    wil_r = stats.wilcoxon(tr_r[~np.isnan(tr_r)]) if np.isfinite(tr_r).sum() > 10 else (np.nan, np.nan)
    wil_e = stats.wilcoxon(tr_e[~np.isnan(tr_e)]) if np.isfinite(tr_e).sum() > 10 else (np.nan, np.nan)
    pos_r = np.nanmean(tr_r > 0); pos_e = np.nanmean(tr_e > 0)
    log.info(f"[{name}] raw    var-vs-age: Spearman r={r_r:+.3f} p={p_r:.2g} | "
             f"per-gene trend: {pos_r*100:.0f}% positive (Wilcoxon p={wil_r[1]:.2g})")
    log.info(f"[{name}] resid  var-vs-age: Spearman r={r_e:+.3f} p={p_e:.2g} | "
             f"per-gene trend: {pos_e*100:.0f}% positive (Wilcoxon p={wil_e[1]:.2g})")

    # --- variance curve ---
    ax = axes_curves
    ax.errorbar(x, mu_r, yerr=se_r, color=C.COLOR_OLD, lw=2, marker="o",
                capsize=3, label=f"raw (ρ={r_r:+.2f}, p={p_r:.1g})")
    ax.errorbar(x, mu_e, yerr=se_e, color=C.COLOR_YOUNG, lw=2, marker="s", ls="--",
                capsize=3, label=f"pool-residualized (ρ={r_e:+.2f}, p={p_e:.1g})")
    ax.set_xticks(x); ax.set_xticklabels(decades_present, fontsize=8)
    ax.set_title(f"{name} (n={Y_raw.shape[1]} genes)"); ax.set_ylabel("mean expression variance")
    ax.legend(fontsize=8)

    # --- CV curve ---
    ax = axes_cv
    ax.errorbar(x, cvmu_r, yerr=cvse_r, color=C.COLOR_OLD, lw=2, marker="o", capsize=3, label="raw")
    ax.errorbar(x, cvmu_e, yerr=cvse_e, color=C.COLOR_YOUNG, lw=2, marker="s", ls="--", capsize=3, label="pool-residualized")
    ax.set_xticks(x); ax.set_xticklabels(decades_present, fontsize=8)
    ax.set_title(f"{name}"); ax.set_ylabel("mean CV (sd/mean)")
    ax.legend(fontsize=8)

    # --- per-gene trend distribution ---
    ax = axes_trend
    bins = np.linspace(-1, 1, 33)
    ax.hist(tr_r[~np.isnan(tr_r)], bins=bins, color=C.COLOR_OLD, alpha=0.6, label=f"raw ({pos_r*100:.0f}%↑)")
    ax.hist(tr_e[~np.isnan(tr_e)], bins=bins, color=C.COLOR_YOUNG, alpha=0.5, label=f"resid ({pos_e*100:.0f}%↑)")
    ax.axvline(0, color=C.MUTED, lw=0.8); ax.set_xlim(-1, 1)
    ax.set_xlabel("per-gene Spearman(var, age-decade)"); ax.set_ylabel("genes")
    ax.set_title(f"{name}"); ax.legend(fontsize=8)

    # table
    rows = []
    for di, dlab in enumerate(decades_present):
        rows.append({"set": name, "decade": dlab, "n_donors": int(n_d[di]),
                     "mean_var_raw": mu_r[di], "mean_var_resid": mu_e[di],
                     "mean_cv_raw": cvmu_r[di], "mean_cv_resid": cvmu_e[di]})
    return pd.DataFrame(rows), dict(name=name, n_genes=Y_raw.shape[1],
                                    rho_raw=r_r, p_raw=p_r, rho_resid=r_e, p_resid=p_e,
                                    pct_pos_raw=pos_r, pct_pos_resid=pos_e,
                                    wilcoxon_p_raw=wil_r[1], wilcoxon_p_resid=wil_e[1])


def main():
    C.ensure_dirs(); set_style()
    X, meta, genes = E.load_counts("allB")
    didx = E.decade_index(meta["age"].values)
    decades_present = [E.DECADE_ORDER[i] for i in range(len(E.DECADE_ORDER))
                       if int((didx == i).sum()) >= C.DECADE_MIN_DONORS]
    keep = np.array([E.DECADE_ORDER.index(d) for d in decades_present])
    didx_p = np.full_like(didx, -1)
    for new, old in enumerate(keep):
        didx_p[didx == old] = new

    # gene sets
    sig = E.sig_deg_ensg(models=("primary",)).drop_duplicates("ensg")
    sig_idx = np.array([np.where(genes == e)[0][0] for e in sig["ensg"]]) if len(sig) else np.array([], int)
    global_mask = E.well_expressed_genes(X, genes)
    global_idx = np.where(global_mask)[0]
    needed = np.unique(np.concatenate([sig_idx, global_idx]))
    log.info(f"genes: sig={len(sig_idx)}, global(well-expressed)={len(global_idx)}, "
             f"union={len(needed)}, decades={decades_present}")

    Y_raw = E.log2cpm_subset(X, needed)
    Y_res = resid_pool(Y_raw, meta)
    needed_genes = genes[needed]
    gpos = {ens: i for i, ens in enumerate(needed_genes)}
    Y_sig_raw = Y_raw[:, [gpos[e] for e in sig["ensg"]]] if len(sig) else np.empty((len(meta), 0))
    Y_sig_res = Y_res[:, [gpos[e] for e in sig["ensg"]]] if len(sig) else np.empty((len(meta), 0))
    gmask = np.isin(needed, global_idx)
    Y_g_raw, Y_g_res = Y_raw[:, gmask], Y_res[:, gmask]

    fig_c, axs_c = plt.subplots(1, 2, figsize=(12, 4.4), sharex=True)
    fig_cv, axs_cv = plt.subplots(1, 2, figsize=(12, 4.4), sharex=True)
    fig_t, axs_t = plt.subplots(1, 2, figsize=(12, 4.2))
    rows, stats_rows = [], []
    for ax_c, ax_cv, ax_t, (nm, yr, ye) in zip(
            axs_c, axs_cv, axs_t,
            [("sig-DEG", Y_sig_raw, Y_sig_res), ("global", Y_g_raw, Y_g_res)]):
        if yr.shape[1] == 0:
            ax_c.set_title(f"{nm}: no genes"); continue
        tbl, st = analyze(nm, yr, ye, didx_p, decades_present, ax_c, ax_cv, ax_t)
        rows.append(tbl); stats_rows.append(st)

    fig_c.suptitle("Expression variance across decades — does dispersion rise with age?\n"
                   "(bold=raw; dashed=pool-residualized; error bars=bootstrap SEM across genes)",
                   y=1.03, fontweight="semibold")
    save(fig_c, C.VAR_F / "variance_vs_decade")
    fig_cv.suptitle("Coefficient of variation (CV = sd/mean) across decades", y=1.03, fontweight="semibold")
    save(fig_cv, C.VAR_F / "cv_vs_decade")
    fig_t.suptitle("Per-gene variance-age trend (Spearman ρ): is the mass shifted positive?", y=1.03, fontweight="semibold")
    save(fig_t, C.VAR_F / "per_gene_trend_dist")

    # depth-restricted robustness (global, raw)
    depth_robustness(meta, X, needed, global_idx, decades_present, didx_p)

    if rows:
        pd.concat(rows).to_csv(C.VAR_R / "variance_by_decade.csv", index=False)
    pd.DataFrame(stats_rows).to_csv(C.VAR_R / "variance_age_test.csv", index=False)
    log.info(f"tests:\n{pd.DataFrame(stats_rows).to_string(index=False)}")
    log.info("DONE 05_variance")


def depth_robustness(meta, X, needed, global_idx, decades_present, didx_p):
    """Restrict to depth-matched donors (n_cells>=50) -> rule out sampling-noise confound."""
    m = meta["n_cells"].values >= 50
    if m.sum() < 200:
        log.info("depth-restricted set too small, skip"); return
    Xd = X[m]
    Yd = E.log2cpm_subset(Xd, needed)
    gmask = np.isin(needed, global_idx)
    var, _, _, n_d = per_gene_per_decade(Yd[:, gmask], didx_p[m], decades_present)
    mu = np.nanmean(var, axis=0)
    x = np.arange(len(decades_present))
    ok = ~np.isnan(mu)
    r = stats.spearmanr(np.arange(len(mu))[ok], mu[ok])[0] if ok.sum() >= 4 else np.nan
    fig, (ax, axb) = plt.subplots(2, 1, figsize=(6.8, 5.2), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(x, mu, color=C.COLOR_OLD, lw=2.2, marker="o")
    ax.set_ylabel("mean expression variance")
    ax.set_title(f"Depth-restricted robustness (n_cells≥50, n={int(m.sum())} donors)\n"
                 f"global gene-set variance vs age: ρ={r:+.3f}")
    axb.bar(x, n_d, color=[decade_color(d) for d in decades_present])
    axb.set_xticks(x); axb.set_xticklabels(decades_present, fontsize=8)
    axb.set_ylabel("donors retained"); axb.set_xlabel("decade")
    save(fig, C.VAR_F / "depth_restricted_variance")
    log.info(f"depth-restricted (n_cells>=50, n={int(m.sum())}): global var-vs-age ρ={r:+.3f}")


if __name__ == "__main__":
    main()
