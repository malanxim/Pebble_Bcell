"""hallmark.py —— 补齐 MSigDB Hallmark GSEA(gp.prerank, Enrichr 库名 MSigDB_Hallmark_2020)。

对每个 B 亚型用全基因按 Wald z 排序做 prerank GSEA,提取 50 个 Hallmark 通路的
NES/FDR/leading-edge。重点核对用户指定的:IFN-α/γ、TNF-NFκB、IL6-JAK-STAT3、
PI3K-AKT-mTOR、mTORC1、MYC、OXPHOS、糖酵解、脂肪酸、UPR、蛋白分泌、ROS、凋亡、p53、
E2F、G2M、有丝分裂纺锤体 等。
产出:NES 热图(亚型×通路)、leading-edge 表、代表性富集曲线。
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

# 用户重点关注通路 -> Hallmark 条目名(Enrichr MSigDB_Hallmark_2020 的人类可读标签)
FOCUS = [
    "Interferon Alpha Response", "Interferon Gamma Response",
    "TNF-alpha Signaling via NF-kB", "Inflammatory Response",
    "IL-6/JAK/STAT3 Signaling", "IL-2/STAT5 Signaling", "Complement",
    "PI3K/AKT/mTOR Signaling", "mTORC1 Signaling",
    "Myc Targets V1", "Myc Targets V2",
    "Oxidative Phosphorylation", "Glycolysis", "Fatty Acid Metabolism",
    "Unfolded Protein Response", "Protein Secretion",
    "Reactive Oxygen Species Pathway", "Apoptosis", "p53 Pathway",
    "E2F Targets", "G2-M Checkpoint", "Mitotic Spindle", "DNA Repair",
]


def _norm(s):
    return " ".join(str(s).split()).lower()   # 折叠空白、小写,便于匹配(PI3K/AKT/mTOR 有双空格)


def _rank_from_deg(cont):
    d = cont.dropna(subset=["stat", "symbol"]).copy()
    d = d.sort_values("padj").drop_duplicates("symbol")          # 同符号取最强
    r = pd.Series(d["stat"].values, index=d["symbol"].astype(str).values)
    return r[~r.index.duplicated(keep="first")].sort_values(ascending=False)


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(C.FIG_DIR / f"hallmark_{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def run_hallmark(subtypes=None, curves_for=("B naive", "B memory")):
    subtypes = subtypes or C.MAIN_SUBTYPES
    rows = []
    for s in subtypes:
        tag = s.replace(" ", "")
        cont = pd.read_csv(C.DEG_DIR / f"{tag}_deg_continuous.csv")
        rnk = _rank_from_deg(cont)
        if len(rnk) < 1000:
            print(f"[hallmark] {s} 基因过少,跳过"); continue
        # 富集曲线:outdir 让 gseapy 落盘每个 term 的 PNG(目录需先建)
        outdir = None
        if s in curves_for:
            outdir = C.FIG_DIR / f"hallmark_curves_{tag}"
            outdir.mkdir(parents=True, exist_ok=True)
        try:
            res = gp.prerank(rnk=rnk, gene_sets=C.HALLMARK_LIB, outdir=outdir,
                             min_size=C.GSEA_MIN_SIZE, max_size=C.GSEA_MAX_SIZE,
                             permutation_num=1000, seed=C.GSEA_SEED, verbose=False,
                             no_plot=(outdir is None))
            df = res.res2d.copy()
        except Exception as e:
            print(f"[hallmark] {s} prerank 失败: {str(e)[:90]}"); continue
        df["subtype"] = s
        rows.append(df)
        print(f"[hallmark] {s}: {len(df)} Hallmark terms (sig FDR<0.25: "
              f"{int((df['FDR q-val']<0.25).sum())})")
    if not rows:
        return pd.DataFrame()
    allr = pd.concat(rows, ignore_index=True)
    allr.to_csv(C.GSEA_DIR / "hallmark_all.csv", index=False)

    # 每亚型 csv + leading-edge
    for s in subtypes:
        sub = allr[allr["subtype"] == s]
        if len(sub):
            sub.to_csv(C.GSEA_DIR / f"hallmark_{s.replace(' ','')}.csv", index=False)

    _nes_heatmap(allr)
    _focus_table(allr)
    return allr


def _nes_heatmap(allr):
    if "NES" not in allr.columns:
        return
    # 主图:全部 50 Hallmark × 亚型,按某亚型 |NES| 排序
    mat = allr.pivot_table(index="Term", columns="subtype", values="NES", aggfunc="first")
    order = mat.abs().sum(axis=1).sort_values(ascending=False).index
    mat = mat.loc[order]
    fig, ax = plt.subplots(figsize=(0.45 * mat.shape[1] + 2.5, 0.26 * mat.shape[0] + 1))
    ax.imshow(mat.values, aspect="auto", cmap="RdBu_r", vmin=-2.6, vmax=2.6)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels([t.replace("HALLMARK_", "").replace("_", " ").title() for t in mat.index], fontsize=6.5)
    ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns, rotation=20, ha="right", fontsize=8)
    # 标 * 显著
    fdr = allr.pivot_table(index="Term", columns="subtype", values="FDR q-val", aggfunc="first").loc[order]
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if fdr.iloc[i, j] < 0.1:
                ax.text(j, i, "*" if fdr.iloc[i, j] >= 0.05 else "**", ha="center", va="center",
                        fontsize=5, color="k")
    ax.set_title("Hallmark GSEA NES vs age (per B subtype)  (* FDR<0.1, ** FDR<0.05)")
    ax.grid(False); _save(fig, "nes_heatmap")


def _focus_table(allr):
    """用户重点通路 × 亚型 的 NES/FDR/leading-edge 表。"""
    focus_norm = {_norm(t) for t in FOCUS}
    f = allr[allr["Term"].apply(_norm).isin(focus_norm)].copy()
    if len(f) == 0:
        print("[hallmark] 无重点通路命中"); return
    lead_col = "Lead_genes" if "Lead_genes" in f.columns else ("GENES" if "GENES" in f.columns else None)
    cols = ["Term", "subtype", "NES", "FDR q-val"] + ([lead_col] if lead_col else [])
    out = f[cols].copy()
    out["Term"] = out["Term"].replace({t: t.replace("HALLMARK_", "") for t in FOCUS})
    out.to_csv(C.GSEA_DIR / "hallmark_focus_pathways.csv", index=False)
    print("[hallmark] 重点通路 -> tables/gsea/hallmark_focus_pathways.csv")
    pivot = f.pivot_table(index="Term", columns="subtype", values="NES", aggfunc="first")
    pivot = pivot.reindex([t for t in FOCUS if t in pivot.index])
    print(pivot.round(2).to_string())
