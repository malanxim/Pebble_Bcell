# OneK1K B-cell aging — curated analysis code

This directory contains code only: no figures, pseudobulk matrices, raw data,
logs, or result tables are included.

## Layout

- `bcell_aging/`: primary donor-level continuous-age analysis (QC,
  donor-by-subtype pseudobulk, composition, DEG, Hallmark GSEA, regulon
  inference, full-TF FDR, split-half validation, and manuscript panels).
- `bcell_decade_deg/`: corrected secondary young-versus-old and decade/variance
  sensitivity analysis.
- `bcell_aging/figstyle.py`: shared plotting style for the primary analysis.

Legacy `analysis_bcell_aging`, reclustering, Milo exploration, poster code, and
report generators were intentionally excluded. The primary analysis uses the
original Azimuth `predicted.celltype.l2` labels consistently; the UMAP plotting
code was updated to show those same labels.

## Critical correction in the extreme-age branch

The old branch read a count matrix whose rows no longer matched its sorted
metadata. The corrected branch reads `pb_all_groups.npy`,
`pb_all_groups_meta.csv`, and `pb_gene_map.csv` produced together by
`bcell_aging/pseudobulk.py`. It checks matrix dimensions, donor/subtype
uniqueness, within-donor metadata consistency, and exact row library sizes when
the checksum column is present. The primary pseudobulk writer now stores that
row-level `library_size` checksum.

## Environment

For the corrected extreme-age branch:

```bash
uv sync
```

For the complete primary/TF pipeline:

```bash
uv sync --extra full
```

The main pinned versions carried over from the archived analysis are
PyDESeq2 0.4.12 and gseapy 1.3.1.

## Primary analysis

Set the raw AnnData and an output directory outside this code-only checkout:

```bash
export BCELL_H5AD=/path/to/eQTLAutoimmune.h5ad
export BCELL_OUTPUT_DIR=/path/to/primary_outputs
uv run --extra full python -m bcell_aging.run
uv run --extra full python -m bcell_aging.composition_check
uv run --extra full python -m bcell_aging.labelcheck
uv run --extra full python -m bcell_aging.mapcheck
uv run --extra full python -m bcell_aging.hallmark
uv run --extra full python -m bcell_aging.drivecheck
uv run --extra full python -m bcell_aging.tf_analysis
uv run --extra full python -m bcell_aging.tf_extras
uv run --extra full python -m bcell_aging.internal_validation
uv run --extra full python -m bcell_aging.single_figs
uv run --extra full python -m bcell_aging.honest_figs
```

## Corrected extreme-age sensitivity analysis

Point `BCELL_PB_DIR` to the primary pipeline's `tables/pseudobulk` directory.
Supply the exact Hallmark GMT used for the analysis to avoid a changing online
gene-set dependency:

```bash
export BCELL_PB_DIR=/path/to/primary_outputs/tables/pseudobulk
export BCELL_HALLMARK_GMT=/path/to/hallmark.gmt
export BCELL_EXTREME_OUTPUT_DIR=/path/to/extreme_age_outputs
uv run python bcell_decade_deg/run_all.py
```

The five stages are: aligned pseudobulk preparation/QC, PyDESeq2, Hallmark
preranked GSEA, decade trajectories, and decade-level variance sensitivity.
The all-B and three main-subtype models require at least 20 cells per donor;
the relaxed plasmablast model (at least 3 cells) is explicitly exploratory.
Thresholds are configurable. For the archived relaxed-threshold sensitivity,
set `BCELL_MAIN_SUBTYPE_MIN_CELLS=10`. A within-pool model is skipped when
fewer than 40 retained donors come from pools containing both age groups; this
limit can be changed explicitly with `BCELL_MIN_MIXED_POOL_DONORS`.
