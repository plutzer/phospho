# phospho

Phosphoproteomics analysis tools for label-free DIA-NN output: normalization,
differential abundance testing, phosphosite occupancy relative to protein
abundance, and PTM-SEA / PTMsigDB export.

## Install

```bash
conda create -n phospho python=3.12 --file requirements.txt -c conda-forge
conda activate phospho
```

## Run

```bash
python -m phospho.cli --input_dir <dir> --output_dir <dir>
```

`--input_dir` must contain `Conditions.csv` and `Comparisons.csv`. By default the
two matrices are read from `report.pg_matrix.tsv` and `report.phosphosites_90.tsv`
in `--input_dir` (one combined DIA-NN search).

When whole-cell and phospho runs are searched **separately**, point each matrix
at its own file with `--whole_matrix` / `--phospho_matrix` (absolute, or relative
to `--input_dir`):

```bash
python -m phospho.cli --input_dir <dir> --output_dir <dir> \
  --whole_matrix WC_report.pg_matrix.tsv \
  --phospho_matrix report.phosphosites_90.tsv
```

Each matrix need only carry its own assay type's runs. A whole-cell run and a
phospho run for the same `Condition`+`Replicate` share a `short_name`, which is
how occupancy pairs sites to protein abundance across the two searches. In
`Comparisons.csv`, `Condition1`/`Condition2` may be a `;`-separated list of
conditions to pool several conditions on one side of a comparison.

PTM-SEA export:

```bash
python phospho_to_ptmsigdb.py --input_file <phospho_*.csv> --output_dir <dir>
```

## Layout

- `phospho/` — the package: `converters`, `datamodel` (`QuantData`),
  `processing`, `exports`, `plotting`, `cli`.
- `tests/` — pytest suite (`pytest`, or `pytest --runslow`).
- `notebooks/` — exploratory analyses (not part of the package).

## Test

```bash
pytest
```
