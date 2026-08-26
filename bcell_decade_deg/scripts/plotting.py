"""Shared matplotlib style + helpers (validated dataviz palette).

Static PNG + PDF outputs for an analysis pipeline. Light surface, recessive
chrome, thin marks, color-by-job, ≥2 series always direct-labeled or legended.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from config import (COLOR_OLD, COLOR_YOUNG, BASELINE, DECADE_RAMP, GRIDLINE,
                    INK, INK_SEC, MUTED, SUBTYPE_COLORS, SURFACE)


def set_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_SEC,
        "axes.titlecolor": INK,
        "axes.titleweight": "semibold",
        "axes.titlesize": 12,
        "axes.titlepad": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.7,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "text.color": INK,
        "font.family": "sans-serif",
        # Arial Unicode MS renders Latin + Greek/math (ρ λ) + arrows (→ ↑ ↓)
        # + dashes + CJK, so no glyph falls back to tofu (Helvetica Neue lacks them).
        "font.sans-serif": ["Arial Unicode MS", "Arial", "Helvetica Neue", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.dpi": 110,
        "savefig.dpi": 200,
    })


def decade_color(label: str):
    """Map a decade label (e.g. '30s') to a sequential ramp color."""
    order = ["20s", "30s", "40s", "50s", "60s", "70s", "80s", "90s"]
    return DECADE_RAMP[order.index(label)]


def save(fig, path_noext: Path) -> None:
    """Save a figure as both PDF (vector) and PNG (preview)."""
    path_noext = Path(path_noext)
    fig.savefig(path_noext.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_noext.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def legend_inline_labels(ax, series: dict, x_last: dict, dy: float = 0.0):
    """Place small colored text labels next to the last point of each series."""
    for name, color in series.items():
        if name in x_last:
            ax.text(x_last[name][0] + 0.12, x_last[name][1] + dy, name,
                    color=color, fontsize=8.5, va="center", fontweight="medium")
