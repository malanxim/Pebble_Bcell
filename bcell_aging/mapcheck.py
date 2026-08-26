"""mapcheck.py —— 检查1:基因 ID 映射 + GSEA 去重敏感性。

Ensembl→symbol 映射成功率/重复/IG-HLA-ribo-MT;GSEA 输入重复 symbol;
两套【预定】去重规则下 Hallmark NES 是否一致(核心通路方向不变即通过)。
规则A(现行):同 symbol 保留 padj 最小(最强 Wald);规则B:保留 baseMean 最高。
"""
from __future__ import annotations
import socket
import numpy as np
import pandas as pd
import gseapy as gp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

socket.setdefaulttimeout(90)
from . import config as C
from . import figstyle
figstyle.set_style()

CORE = ["TNF-alpha Signaling via NF-kB", "Interferon Alpha Response", "Interferon Gamma Response",
        "p53 Pathway", "Apoptosis", "Myc Targets V1", "Myc Targets V2", "E2F Targets",
        "Unfolded Protein Response", "IL-6/JAK/STAT3 Signaling"]


def _rank(cont, rule):
    d = cont.dropna(subset=["stat", "symbol", "baseMean"]).copy()
    d["symbol"] = d["symbol"].astype(str)
    if rule == "A":                       # 最强 Wald(padj 最小)
        d = d.sort_values("padj")
    else:                                 # 最高均值(baseMean 最大)
        d = d.sort_values("baseMean", ascending=False)
    d = d.drop_duplicates("symbol")
    r = pd.Series(d["stat"].values, index=d["symbol"].values)
    return r[~r.index.duplicated(keep="first")].sort_values(ascending=False)


def mapping_stats():
    gm = pd.read_csv(C.PB_DIR / "pb_gene_map.csv")
    gm["symbol"] = gm["symbol"].astype(str)
    n = len(gm)
    empty = int((gm["symbol"].isin(["", "nan", "NA"]) | gm["symbol"].isna()).sum())
    dup_sym = gm["symbol"].value_counts()
    dup_sym = dup_sym[dup_sym > 1]
    # 版本号?
    has_ver = gm["ensembl"].astype(str).str.contains(r"\.\d+$").mean()
    # 特征类映射核查
    def has(prefix_or_set):
        if callable(prefix_or_set):
            return int(gm["symbol"].apply(prefix_or_set).sum())
        return int(gm["symbol"].astype(str).str.startswith(prefix_or_set).sum())
    ribo = lambda s: isinstance(s, str) and (s.startswith("RPL") or s.startswith("RPS"))
    feat = {
        "总基因": n,
        "空/无 symbol": empty,
        "Ensembl 带版本号(.x)占比%": round(float(has_ver * 100), 2),
        "重复 symbol 数": int(dup_sym.shape[0]),
        "受重复影响的 Ensembl 条目": int(gm["symbol"].isin(dup_sym.index).sum()),
        "IG 基因(IGH/IGL/IGK/IGJ)": has(("IGH", "IGL", "IGK", "IGJ")),
        "HLA 基因": has("HLA"),
        "核糖体(RPL/RPS)": has(ribo),
        "线粒体(MT-)": has("MT-"),
    }
    rep = pd.DataFrame({"项目": list(feat.keys()), "值": list(feat.values())})
    rep.to_csv(C.TAB_DIR / "check1_mapping_stats.csv", index=False)
    print("[check1] 映射统计:\n" + rep.to_string(index=False))
    return dup_sym


def dedup_sensitivity(subtypes=None):
    subtypes = subtypes or C.MAIN_SUBTYPES
    ruleA = pd.read_csv(C.GSEA_DIR / "hallmark_all.csv")     # 现行(规则A)已存
    ruleA = ruleA[ruleA["subtype"].isin(subtypes)]
    rows = []
    for s in subtypes:
        cont = pd.read_csv(C.DEG_DIR / f"{s.replace(' ','')}_deg_continuous.csv")
        rnkB = _rank(cont, "B")
        res = gp.prerank(rnk=rnkB, gene_sets=C.HALLMARK_LIB, outdir=None,
                         min_size=C.GSEA_MIN_SIZE, max_size=C.GSEA_MAX_SIZE,
                         permutation_num=1000, seed=C.GSEA_SEED, verbose=False, no_plot=True)
        dB = res.res2d.copy(); dB["subtype"] = s; dB["rule"] = "B"
        rows.append(dB)
    ruleB = pd.concat(rows, ignore_index=True)
    ruleA2 = ruleA.copy(); ruleA2["rule"] = "A"
    both = pd.concat([ruleA2[["Term", "subtype", "NES", "FDR q-val", "rule"]],
                      ruleB[["Term", "subtype", "NES", "FDR q-val", "rule"]]])
    piv = both.pivot_table(index=["subtype", "Term"], columns="rule", values="NES")
    rho = float(piv.corr().iloc[0, 1])
    print(f"[check1] 规则A vs 规则B Hallmark NES 相关 ρ={rho:.3f}")
    piv.to_csv(C.GSEA_DIR / "check1_dedup_nes.csv")
    # 核心通路方向一致性
    print("[check1] 核心通路 (规则A_NES / 规则B_NES / 方向一致?):")
    cre = both[both["Term"].isin(CORE)].pivot_table(index=["subtype", "Term"], columns="rule", values="NES")
    cre["方向一致"] = np.sign(cre["A"]) == np.sign(cre["B"])
    print(cre.round(2).to_string())
    cre.to_csv(C.GSEA_DIR / "check1_core_pathway_dedup.csv")
    _scatter(piv, rho)
    return rho, cre


def _scatter(piv, rho):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(piv["A"], piv["B"], s=10, alpha=0.6, color=figstyle.AQUA)
    lim = max(abs(piv[["A", "B"]]).max())
    ax.plot([-lim, lim], [-lim, lim], color=figstyle.MUTED, lw=0.8)
    ax.axhline(0, color=figstyle.MUTED, lw=0.5); ax.axvline(0, color=figstyle.MUTED, lw=0.5)
    ax.set_xlabel("NES (rule A: keep strongest Wald)"); ax.set_ylabel("NES (rule B: keep highest baseMean)")
    ax.set_title(f"Hallmark NES: dedup-rule sensitivity (ρ={rho:.2f})")
    figstyle.thin_despine(ax)
    for ext in ("pdf", "png"):
        fig.savefig(C.FIG_DIR / f"check1_dedup_nes_scatter.{ext}", bbox_inches="tight")
    plt.close(fig)


def run():
    dup = mapping_stats()
    dup.head(20).to_csv(C.TAB_DIR / "check1_duplicate_symbols.csv", index=False)
    rho, cre = dedup_sensitivity()
    agree = cre["方向一致"].mean()
    print(f"\n[check1] 核心通路方向一致率 = {agree*100:.0f}%")
    verdict = "通过(核心通路在两套去重规则下方向稳定)" if agree >= 0.85 and rho >= 0.8 \
        else "注意:去重规则影响较大,需在报告中说明"
    print(f"[check1] 判定: {verdict}")
    return verdict
