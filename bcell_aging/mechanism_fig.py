"""mechanism_fig.py —— 整合 Figure 6(干净版)。

左:TF→靶基因 双分型网络(边短、图例外置到图底)。
右:极简 3 盒机制模型(age | 程序(带 tier 色块) | 候选状态),仅 2 条箭头,无交叉。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Patch
from matplotlib.lines import Line2D

from . import config as C
from . import paper_design as D
from .tf_analysis import _nets
D.apply_style()

NODES = [("FOXO3", "B memory", "strict"), ("NFKB", "B memory", "convergent"),
         ("STAT1", "B naive", "convergent"), ("PAX5", "B intermediate", "strict")]
TIER_FILL = {"strict": "#1baf7a", "convergent": "#9ed8c0", "exploratory": "#d8d8d6"}
TIER_EDGE = {"strict": "#0f6e4f", "convergent": "#5b9b86", "exploratory": "#999"}
PADJ, LFC, N_PER_TF = 0.10, 0.02, 5


def _targets(ct, tf, subtype):
    tgts = ct[ct["source"] == tf].set_index("target")["weight"]
    deg = pd.read_csv(C.DEG_DIR / f"{subtype.replace(' ', '')}_deg_continuous.csv")
    deg = deg[~deg["is_IG"]].dropna(subset=["log2FC_per10yr", "padj"])
    common = tgts.index.intersection(deg["symbol"])
    if len(common) == 0:
        return pd.DataFrame()
    sub = deg.set_index("symbol").loc[common].copy(); sub["w"] = tgts.loc[common]
    sub = sub[(sub["padj"] < PADJ) & (sub["log2FC_per10yr"].abs() > LFC)]
    return sub.reindex(sub["log2FC_per10yr"].abs().sort_values(ascending=False).index).head(N_PER_TF)


def run():
    ct, _ = _nets()
    fig = plt.figure(figsize=(14, 7.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.06)
    axN = fig.add_subplot(gs[0]); axM = fig.add_subplot(gs[1])

    # ============ 左:双分型网络(边短) ============
    axN.set_xlim(0, 1); axN.set_ylim(0.04, 1.0); axN.axis("off")
    XF, XT = 0.30, 0.62                     # 列距收窄 -> 边短
    n_tf = len(NODES)
    tf_y = {NODES[i][0]: 0.90 - i * (0.80 / (n_tf - 1)) for i in range(n_tf)}
    tf_data = {tf: (st, te, _targets(ct, tf, st)) for tf, st, te in NODES}
    tgt_pos = {}
    for tf, subtype, tier in NODES:
        subs = tf_data[tf][2]; m = len(subs)
        if m == 0:
            continue
        yc = tf_y[tf]; span = min(0.15, 0.022 * m)
        ys = np.linspace(yc - span / 2, yc + span / 2, m) if m > 1 else [yc]
        for (sym, r), yy in zip(subs.iterrows(), ys):
            tgt_pos[(tf, sym)] = (XT, yy, float(r["w"]), float(r["log2FC_per10yr"]), float(r["padj"]))
    # 边(短曲线)
    for (tf, sym), (xt, yt, w, lfc, fdr) in tgt_pos.items():
        col = D.SUBTYPE_COLOR["B memory"] if w > 0 else D.SUBTYPE_COLOR["B naive"]
        axN.annotate("", xy=(xt - 0.016, yt), xytext=(XF + 0.045, tf_y[tf]),
                     arrowprops=dict(arrowstyle="-|>" if w > 0 else "-[", color=col,
                                     lw=1.0 + min(abs(w), 3) * 0.3, alpha=0.5,
                                     connectionstyle="arc3,rad=0.06", shrinkA=2, shrinkB=2))
    # 靶基因
    for (tf, sym), (xt, yt, w, lfc, fdr) in tgt_pos.items():
        c = D.SUBTYPE_COLOR["B memory"] if lfc > 0 else D.SUBTYPE_COLOR["B naive"]
        s = 60 + min(-np.log10(max(fdr, 1e-10)), 6) * 20
        axN.scatter(xt, yt, s=s, c=c, alpha=0.92, edgecolors="white", linewidth=0.5, zorder=4)
        axN.text(xt + 0.02, yt, sym, fontsize=7, va="center", color=D.INK, zorder=5)
    # TF 方块
    for tf, subtype, tier in NODES:
        y = tf_y[tf]
        axN.add_patch(FancyBboxPatch((XF - 0.07, y - 0.028), 0.115, 0.056,
                     boxstyle="round,pad=0.004,rounding_size=0.018",
                     fc=TIER_FILL[tier], ec=TIER_EDGE[tier], lw=1.6, zorder=3))
        axN.text(XF - 0.0125, y, tf, fontsize=10.5, fontweight="bold", ha="center", va="center", zorder=4)
        axN.text(XF - 0.0125, y - 0.045, f"{D.SUBTYPE_SHORT[subtype]} · {tier}", fontsize=6.4,
                 ha="center", color=D.INK2, style="italic")
    axN.text(0.5, 0.985, "TF → age-DEG regulon targets", ha="center", fontsize=10.5, fontweight="bold")
    D.panel_label(axN, "A")

    # ============ 右:极简 3 盒模型 ============
    axM.set_xlim(0, 1); axM.set_ylim(0, 1); axM.axis("off")
    def box(x, y, w, h, ec=D.BASE, lw=1.4, fc=D.SURF):
        axM.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.025",
                                     fc=fc, ec=ec, lw=lw))
    # age
    box(0.04, 0.44, 0.16, 0.12, ec=D.BASE)
    axM.text(0.12, 0.50, "age", fontsize=12, fontweight="bold", ha="center", va="center")
    # programs(单高盒,内含 4 行 tier 色块 + 文本)
    box(0.30, 0.18, 0.34, 0.64, ec=D.BASE)
    axM.text(0.47, 0.78, "regulatory programs", fontsize=9, fontweight="bold", ha="center", color=D.INK2)
    progs = [("FOXO3 ↑  (all subtypes)", "strict"),
             ("PAX5 ↓  (intermediate)", "strict"),
             ("STAT1 / IFN ↑  (naive)", "convergent"),
             ("NF-κB ↑  (memory)", "convergent")]
    for i, (txt, tier) in enumerate(progs):
        yy = 0.66 - i * 0.11
        axM.add_patch(Rectangle((0.33, yy - 0.018), 0.035, 0.036,
                                fc=TIER_FILL[tier], ec=TIER_EDGE[tier], lw=1.0, zorder=4))
        axM.text(0.375, yy, txt, fontsize=8.2, va="center", color=D.INK, zorder=4)
    # state
    box(0.70, 0.22, 0.27, 0.56, ec=D.BASE)
    axM.text(0.835, 0.74, "candidate B-cell state", fontsize=8.6, fontweight="bold", ha="center", color=D.INK2)
    state_lines = ["chronic low-level", "inflammation", "", "↓ growth /", "cycle readiness", "",
                   "altered identity", "maintenance", "", "stress /", "quiescence"]
    for i, t in enumerate(state_lines):
        axM.text(0.835, 0.66 - i * 0.038, t, fontsize=7.4, ha="center",
                 color=D.INK if t else "none")
    # 2 条短箭头
    axM.annotate("", xy=(0.30, 0.50), xytext=(0.20, 0.50),
                 arrowprops=dict(arrowstyle="-|>", color=D.INK2, lw=1.3, linestyle=(0, (3, 2))))
    axM.annotate("", xy=(0.70, 0.50), xytext=(0.64, 0.50),
                 arrowprops=dict(arrowstyle="-|>", color=D.INK2, lw=1.3, linestyle=(0, (3, 2))))
    axM.text(0.5, 0.985, "candidate state model", ha="center", fontsize=10.5, fontweight="bold")
    D.panel_label(axM, "B")
    axM.text(0.5, 0.06, "solid = strict FDR<0.05  ·  light = convergent  ·  dashed arrow = associated with, not causal",
             ha="center", fontsize=6.8, color=D.MUTED, style="italic")

    # 图例(figure 底部,水平,不挡网络)
    leg = [Patch(facecolor=TIER_FILL["strict"], edgecolor=TIER_EDGE["strict"], label="TF / program: strict FDR<0.05"),
           Patch(facecolor=TIER_FILL["convergent"], edgecolor=TIER_EDGE["convergent"], label="convergent (FDR NS)"),
           Line2D([0], [0], marker="o", color="none", markerfacecolor=D.SUBTYPE_COLOR["B memory"], markersize=8, label="target ↑ age"),
           Line2D([0], [0], marker="o", color="none", markerfacecolor=D.SUBTYPE_COLOR["B naive"], markersize=8, label="target ↓ age"),
           Line2D([0], [0], color=D.SUBTYPE_COLOR["B memory"], lw=1.5, label="activation"),
           Line2D([0], [0], color=D.SUBTYPE_COLOR["B naive"], lw=1.5, linestyle="--", label="inhibition")]
    fig.legend(handles=leg, loc="lower center", ncol=6, fontsize=7.6, frameon=True,
               facecolor="white", edgecolor=D.GRID, bbox_to_anchor=(0.5, -0.005))

    fig.suptitle("Figure 6 | Regulatory network & candidate model of age-associated B-cell remodeling",
                 fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    D.save(fig, "Figure6_mechanism")
    rows = [{"TF": tf, "subtype": st, "tier": te, "target": s, "weight": float(r["w"]),
             "log2FC_per10yr": float(r["log2FC_per10yr"]), "padj": float(r["padj"])}
            for tf, st, te in NODES for s, r in tf_data[tf][2].iterrows()]
    pd.DataFrame(rows).to_csv(D.SRC / "Figure6__network_edges.csv", index=False)
    print(f"[mechanism] target edges={len(rows)}")


if __name__ == "__main__":
    run()
