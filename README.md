# OneK1K B-cell aging analysis

This repository contains the core analysis and figure-generation code for
`OneK1K_B_cell_aging_reorganized`, a study of age-associated changes across
B-cell subtypes in the OneK1K cohort. The main analyses include data quality
control, donor-by-subtype pseudobulk aggregation, B-cell composition analysis,
age-associated differential expression, Hallmark pathway enrichment,
transcription-factor analysis, internal validation, complementary age-group
analyses, and manuscript figure generation. Data, generated figures, and result
tables are not included.

## Code overview

- `bcell_aging/`: the main donor-level workflow, including data quality checks,
  donor-by-subtype pseudobulk aggregation, B-cell composition analysis,
  differential expression, Hallmark enrichment, transcription-factor analysis,
  internal validation, and figure generation.
- `bcell_decade_deg/`: complementary young-versus-old, age-decade trajectory,
  and variance analyses.
- `bcell_aging/figstyle.py`: shared plotting settings.

## Installation

Python 3.10 is recommended. Dependencies are defined in `pyproject.toml` and can
be installed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra full
```

## Running the main workflow

Set the input AnnData file and an output directory, then run:

```bash
export BCELL_H5AD=/path/to/eQTLAutoimmune.h5ad
export BCELL_OUTPUT_DIR=/path/to/output
uv run --extra full python -m bcell_aging.run
```

Individual analysis and plotting modules in `bcell_aging/` can also be run
separately.

For the complementary age-group analyses, point the workflow to the pseudobulk
tables produced by the main analysis:

```bash
export BCELL_PB_DIR=/path/to/output/tables/pseudobulk
export BCELL_HALLMARK_GMT=/path/to/hallmark.gmt
export BCELL_EXTREME_OUTPUT_DIR=/path/to/age_group_output
uv run python bcell_decade_deg/run_all.py
```
