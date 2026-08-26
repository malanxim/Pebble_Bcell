"""Shared Matplotlib style for the B-cell aging analysis figures."""

import matplotlib as mpl

# Surfaces and neutral ink.
SURFACE = "#fcfcfb"
INK_PRI = "#0b0b0b"
INK_SEC = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"

# Categorical palette.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"

# Sequential blue ramp and diverging endpoints.
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
            "#256abf", "#1c5cab", "#104281"]
DIVERGE_NEG = "#2a78d6"
DIVERGE_MID = "#f0efec"
DIVERGE_POS = "#e34948"


def set_style() -> None:
    """Apply the shared publication plotting style."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASE,
        "axes.labelcolor": INK_SEC,
        "axes.titlecolor": INK_PRI,
        "axes.titleweight": "bold",
        "axes.titlesize": 12.5,
        "xtick.color": INK_SEC,
        "ytick.color": INK_SEC,
        "text.color": INK_PRI,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.size": 10,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.dpi": 130,
        "savefig.dpi": 160,
    })


def thin_despine(ax, keep_left: bool = True, keep_bottom: bool = True) -> None:
    """Show only the requested left and bottom axes spines."""
    for side, visible in [("top", False), ("right", False),
                          ("left", keep_left), ("bottom", keep_bottom)]:
        ax.spines[side].set_visible(visible)
