"""enrich.py —— Part 6/7:GO/通路 ORA + GSEA(gseapy)。

ORA:对 up/down DEG 分开(padj<0.05 且 |log2FC_per10yr|>阈值),背景=该亚型受检基因;
GSEA:用全基因按 Wald z 排序(保留方向+证据)。
免疫球蛋白(IG)基因:保留完整结果,富集展示另给"去IG"视图。
"""
from __future__ import annotations
import warnings, socket
import numpy as np
import pandas as pd
import gseapy as gp

from . import config as C

# 限制网络调用最长 90s,避免 Enrichr/GSEA 卡死拖垮整条流程
socket.setdefaulttimeout(90)


def _gene_map():
    return pd.read_csv(C.PB_DIR / "pb_gene_map.csv").set_index("ensembl")["symbol"].astype(str)


def _to_symbols(ensembl_iter, gmap):
    syms = [gmap.get(e) for e in ensembl_iter]
    syms = [s for s in syms if isinstance(s, str) and s and s != "nan"]
    return sorted(set(syms))


def _is_ig(sym):
    return isinstance(sym, str) and sym.startswith(C.IG_PREFIXES)


def ora(deg, background_sym, direction, libraries, drop_IG):
    """direction='up'|'down';按方向拆分显著基因(FDR<阈值)。年龄每10岁效应通常很小,
    故不设 |LFC| 量级门槛,只按方向(否则多数显著基因被滤掉)。"""
    if direction == "up":
        sel = (deg["padj"] < C.DEG_PADJ) & (deg["log2FC_per10yr"] > 0)
    else:
        sel = (deg["padj"] < C.DEG_PADJ) & (deg["log2FC_per10yr"] < 0)
    genes = deg.loc[sel, "symbol"].tolist()
    if drop_IG:
        genes = [g for g in genes if not _is_ig(g)]
    genes = [g for g in genes if isinstance(g, str) and g]
    if len(genes) < 5:
        print(f"[enrich] {direction}({'noIG' if drop_IG else 'all'}) 基因过少({len(genes)}),跳过")
        return pd.DataFrame()
    out = []
    for lib in libraries:
        try:
            r = gp.enrich(gene_list=genes, gene_sets=lib, background=background_sym,
                          outdir=None, verbose=False)
            df = r.results.copy() if hasattr(r, "results") else r.res2d.copy()
            df["library"] = lib; df["direction"] = direction
            df["set"] = "noIG" if drop_IG else "all"
            out.append(df)
        except Exception as e:
            print(f"[enrich] ORA {lib}/{direction} 失败: {str(e)[:70]}")
    if not out:
        return pd.DataFrame()
    res = pd.concat(out, ignore_index=True)
    # 标准化列名(gseapy 1.x: Term/Overlap/Pvalue/Adjusted P-value/Genes)
    return res


def gsea(deg, libraries, drop_IG):
    """全基因按 stat(Wald z)排序做 GSEA;返回 res2d。"""
    d = deg.dropna(subset=["stat", "symbol"]).copy()
    if drop_IG:
        d = d[~d["symbol"].apply(_is_ig)]
    d = d.sort_values("padj").drop_duplicates("symbol")     # 同符号取最强
    rank = pd.Series(d["stat"].values, index=d["symbol"].values).sort_values(ascending=False)
    rank = rank[~rank.index.duplicated(keep="first")]
    if len(rank) < 1000:
        print(f"[enrich] GSEA 排序基因过少({len(rank)}),跳过"); return pd.DataFrame()
    out = []
    for lib in libraries:
        try:
            r = gp.prerank(rnk=rank, gene_sets=lib, outdir=None, verbose=False,
                           seed=C.GSEA_SEED, min_size=C.GSEA_MIN_SIZE, max_size=C.GSEA_MAX_SIZE,
                           permutation_num=500, no_plot=True)
            df = r.res2d.copy() if hasattr(r, "res2d") else pd.DataFrame()
            if len(df):
                df["library"] = lib; df["set"] = "noIG" if drop_IG else "all"
                out.append(df)
        except Exception as e:
            print(f"[enrich] GSEA(prerank) {lib} 失败: {str(e)[:70]}")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def run_enrich(subtype, deg_df):
    """deg_df = 连续年龄 DEG 表(含 symbol/stat/padj/log2FC_per10yr)。"""
    tag = subtype.replace(" ", "")
    gmap = _gene_map()
    # 背景 = 该亚型受检基因(进入 DEG 的全部)的符号
    background_sym = _to_symbols(deg_df["ensembl"].tolist(), gmap)
    pd.Series(background_sym).to_csv(C.ENR_DIR / f"{tag}_background_genes.csv", index=False, header=False)

    all_ora, all_gsea = [], []
    for drop_IG in (False, True):
        for direction in ("up", "down"):
            o = ora(deg_df, background_sym, direction, C.ORA_LIBRARIES, drop_IG)
            if len(o):
                all_ora.append(o)
        g = gsea(deg_df, C.GSEA_LIBRARIES, drop_IG)
        if len(g):
            all_gsea.append(g)

    if all_ora:
        ora_df = pd.concat(all_ora, ignore_index=True)
        ora_df.to_csv(C.ENR_DIR / f"{tag}_ora.csv", index=False)
        print(f"[enrich] ORA {tag}: {len(ora_df)} 行 -> {tag}_ora.csv")
    if all_gsea:
        gsea_df = pd.concat(all_gsea, ignore_index=True)
        gsea_df.to_csv(C.GSEA_DIR / f"{tag}_gsea.csv", index=False)
        print(f"[enrich] GSEA {tag}: {len(gsea_df)} 行 -> {tag}_gsea.csv")
    return {"ora": pd.concat(all_ora) if all_ora else pd.DataFrame(),
            "gsea": pd.concat(all_gsea) if all_gsea else pd.DataFrame()}
