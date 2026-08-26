"""paper_design.py —— 论文级统一设计系统(全项目固定)。

配色:NIVE冷蓝、MEMORY暖橙、INTERMEDIATE绿、PLASMA紫;年龄 viridis;
正负效应/NES 用 0 中心发散色盘(暖+冷-,浅灰中点)。
证据等级编码:严格FDR=实心深色粗边;收敛支持=半透明/阴影;探索=灰空心。
字体 Arial;输出 PDF(矢量,可编辑文字)+ 300dpi PNG。
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams, colors as mcolors
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "outputs" / "bcell_aging" / "final_figures"
MAIN = OUT / "main"; SUPP = OUT / "supplementary"; SRC = OUT / "source_data"
for d in (MAIN, SUPP, SRC):
    d.mkdir(parents=True, exist_ok=True)

# ---- 固定亚型配色(全项目一致,不换色) ----
SUBTYPE_COLOR = {"B naive": "#2a78d6", "B memory": "#eb6834",
                 "B intermediate": "#1baf7a", "Plasmablast": "#7b5cff"}
SUBTYPE_SHORT = {"B naive": "naive", "B memory": "memory",
                 "B intermediate": "intermediate", "Plasmablast": "plasmablast"}

# ---- 发散色盘(0中心,暖正冷负,浅灰中点)CVD 友好 ----
_POS, _MID, _NEG = "#c0392b", "#f4f3ef", "#2a78d6"
DIVERGE_CMAP = mcolors.LinearSegmentedColormap.from_list("div", [_NEG, _MID, _POS], N=256)
AGE_CMAP = "viridis"

INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#9a9893"; GRID = "#e6e5df"; BASE = "#c3c2b7"; SURF = "#ffffff"

# ---- 证据等级视觉编码 ----
# strict=FDR<0.05; convergent=网络/方法/Hallmark 收敛但 FDR NS; exploratory=其余/矛盾
TIER = {
    "strict":     dict(facealpha=1.0, edge="black", lw=1.6, marker="o", hatch=None,  label="strict FDR<0.05"),
    "convergent": dict(facealpha=0.55, edge="black", lw=1.2, marker="o", hatch="////", label="convergent (FDR NS)"),
    "exploratory":dict(facealpha=0.25, edge=MUTED,   lw=0.9, marker="o", hatch=None,  label="exploratory"),
}

# 收敛支持白名单(跨网络/方法/Hallmark 一致但全空间 FDR NS)
CONVERGENT = {("B naive", "STAT1"), ("B memory", "NFKB"), ("B intermediate", "NFKB"), ("B naive", "NFKB")}
# 矛盾/降级白名单(即便 FDR 显著也不作核心)
DOWNGRADE = {("B naive", "E2F1")}   # regulon↔Hallmark 不符


def tier_of(subtype, tf, fdr_full):
    if (subtype, tf) in DOWNGRADE:
        return "exploratory"
    if fdr_full < 0.05:
        return "strict"
    if (subtype, tf) in CONVERGENT:
        return "convergent"
    return "exploratory"


def face_color(base_hex, tier):
    """按 tier 调透明度。"""
    a = TIER[tier]["facealpha"]
    rgb = mcolors.to_rgb(base_hex)
    return (*rgb, a)


# ---- 字体/排版 ----
def apply_style():
    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42,   # 保留可编辑文字(TrueType),不转轮廓
        "svg.fonttype": "none",
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.edgecolor": BASE, "axes.linewidth": 0.9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.facecolor": SURF, "figure.facecolor": SURF, "savefig.facecolor": SURF,
        "figure.dpi": 130, "savefig.dpi": 300,
        "axes.grid": False,
    })


def save(fig, name, dir_=MAIN):
    """PDF(矢量,可编辑文字)+ 300dpi PNG。"""
    for ext in ("pdf", "png"):
        fig.savefig(dir_ / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[paper] -> {dir_.name}/{name}.{{pdf,png}}")


def panel_label(ax, letter, x=0.012, y=0.97):
    """面板字母放轴内左上角(带浅底),避免与标题/坐标轴重叠。"""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="top", ha="left", zorder=10,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75))


def style_ax(ax, ygrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if ygrid:
        ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7); ax.set_axisbelow(True)


def age_rug(ax, ages, color=BASE, frac=0.02):
    ymin, ymax = ax.get_ylim(); span = ymax - ymin
    rng = np.random.default_rng(1)
    y = ymin + frac * span + rng.uniform(-0.003, 0.003, len(ages)) * span
    ax.scatter(ages, y, s=2.5, color=color, alpha=0.4, edgecolors="none", zorder=1)
    ax.set_ylim(ymin - 0.03 * span, ymax)
