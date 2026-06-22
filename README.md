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

`--input_dir` must contain `report.pg_matrix.tsv`, `report.phosphosites_90.tsv`,
`Conditions.csv`, and `Comparisons.csv`. PTM-SEA export:

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
