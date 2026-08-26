"""Shared configuration for the B-cell young-vs-old DEG + decade-variance analysis.

Reuses the aligned full-transcriptome pseudobulk artefacts produced by
`bcell_aging/pseudobulk.py` (NO reload of the raw h5ad).

Analysis design (per user request, 2026-08-02):
  - Group comparison:  young = age in [20, 30]  vs  old = age in [80, 90]
  - DEG (PyDESeq2) + GSEA (gseapy prerank, Hallmark) for ALL-B + each subtype
  - For significant DEGs: per-decade (10-yr bin) expression boxplots ("oscillation")
  - Per-decade expression-variance trajectory; test whether variance rises with age

Caveat from the archived cohort audit: corr(age, pool) = -0.27, so
the young-vs-old contrast is partly between-pool. We report a ~group primary
(what was requested) AND a pool-adjusted sensitivity so the batch-confounded vs
batch-robust genes are clearly separated.
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SCRIPTS_DIR = Path(__file__).resolve().parent
PKG_DIR = SCRIPTS_DIR.parent                       # .../bcell_decade_deg
WORK_DIR = PKG_DIR.parent

# Source produced by bcell_aging/pseudobulk.py.  Environment overrides make
# the release portable without hard-coding a local checkout path.
PB_DIR = Path(os.environ.get(
    "BCELL_PB_DIR", WORK_DIR / "outputs" / "bcell_aging" / "tables" / "pseudobulk"
)).expanduser().resolve()
PB_COUNTS_NPY = PB_DIR / "pb_all_groups.npy"
PB_META_CSV = PB_DIR / "pb_all_groups_meta.csv"
PB_GENES_CSV = PB_DIR / "pb_gene_map.csv"
HALLMARK_GMT = os.environ.get("BCELL_HALLMARK_GMT")
HALLMARK_GMT = Path(HALLMARK_GMT).expanduser().resolve() if HALLMARK_GMT else None

# This package's output tree.
OUTPUT_DIR = Path(os.environ.get(
    "BCELL_EXTREME_OUTPUT_DIR", PKG_DIR / "run_outputs"
)).expanduser().resolve()
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOGS_DIR = OUTPUT_DIR / "logs"
REPORT_DIR = OUTPUT_DIR / "report"

QC_R = RESULTS_DIR / "qc";        QC_F = FIGURES_DIR / "qc"
DEG_R = RESULTS_DIR / "deg";      DEG_F = FIGURES_DIR / "deg"
GSEA_R = RESULTS_DIR / "gsea";    GSEA_F = FIGURES_DIR / "gsea"
DEC_R = RESULTS_DIR / "decade";   DEC_F = FIGURES_DIR / "decade"
VAR_R = RESULTS_DIR / "variance"; VAR_F = FIGURES_DIR / "variance"

# --------------------------------------------------------------------------- #
# Analysis parameters
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42

B_SUBTYPES = ["B naive", "B intermediate", "B memory", "Plasmablast"]
ALLB_KEY = "allB"

# Group definitions (inclusive both ends, per "20-30岁" / "80-90岁").
YOUNG = (20, 30)
OLD = (80, 90)

# Decade trajectory bins (half-open [lo, hi)); age 19 falls outside (5 donors).
DECADES = [
    (20, 30, "20s"), (30, 40, "30s"), (40, 50, "40s"), (50, 60, "50s"),
    (60, 70, "60s"), (70, 80, "70s"), (80, 90, "80s"), (90, 100, "90s"),
]
DECADE_MIN_DONORS = 10   # need >= this donors for a stable per-decade variance

# Significant-DEG selection for the decade / variance analyses.
PADJ_THR = 0.05
LFC_THR = 0.5            # |log2FC|

# PyDESeq2 minimum-count gene filtering (applied before fitting).
MIN_COUNT = 10
MIN_SAMPLES = 10

# Donor inclusion.  The three main subtypes use the same >=20-cell threshold
# as the primary manuscript analysis.  Plasmablast is explicitly exploratory.
ALLB_MIN_CELLS = int(os.environ.get("BCELL_ALLB_MIN_CELLS", "20"))
MAIN_SUBTYPE_MIN_CELLS = int(os.environ.get("BCELL_MAIN_SUBTYPE_MIN_CELLS", "20"))
PLASMABLAST_MIN_CELLS = int(os.environ.get("BCELL_PLASMABLAST_MIN_CELLS", "3"))
MIN_MIXED_POOL_DONORS = int(os.environ.get("BCELL_MIN_MIXED_POOL_DONORS", "40"))

# Variance: exclude genes whose mean log2-CPM is too low to interpret dispersion.
EXPR_MIN_MEAN_LOG2CPM = 2.0

# --------------------------------------------------------------------------- #
# Validated palette (dataviz skill; see plotting.py for roles)
# --------------------------------------------------------------------------- #
COLOR_YOUNG = "#2a78d6"      # blue
COLOR_OLD = "#e34948"        # red
SUBTYPE_COLORS = {                       # categorical, direct-labeled (relief)
    "B naive": "#2a78d6",
    "B intermediate": "#eb6834",
    "B memory": "#1baf7a",
    "Plasmablast": "#e87ba4",
}
# Sequential blue ramp (light->dark) for decade magnitude (young->old).
DECADE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef",
    "#6da7ec", "#5598e7", "#3987e5", "#256abf",
]
INK = "#0b0b0b"
INK_SEC = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

SUBTYPE_SLUG = {s: s.replace(" ", "_") for s in B_SUBTYPES}


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    if log_file is not None:
        log_file = Path(log_file); log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8"); fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dirs() -> None:
    for d in [RESULTS_DIR, FIGURES_DIR, LOGS_DIR, REPORT_DIR,
              QC_R, DEG_R, GSEA_R, DEC_R, VAR_R,
              QC_F, DEG_F, GSEA_F, DEC_F, VAR_F]:
        d.mkdir(parents=True, exist_ok=True)


def age_group(age: float) -> str:
    """young / old / mid label."""
    if YOUNG[0] <= age <= YOUNG[1]:
        return "young"
    if OLD[0] <= age <= OLD[1]:
        return "old"
    return "mid"


def decade_label(age: float):
    """Return decade label or None if outside [20,100)."""
    for lo, hi, lab in DECADES:
        if lo <= age < hi:
            return lab
    return None
