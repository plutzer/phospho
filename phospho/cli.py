"""Command-line entry point: DIA-NN matrices -> normalized tables, differential
statistics, relative occupancy, and QC plots.

The staged outputs are built from the modular package: converters build
:class:`QuantData`, processing normalizes and tests, plotting renders QC figures,
and exports are produced separately.

The whole-proteome and phosphosite matrices may come from one combined DIA-NN
search or from two separate searches; each is located independently, by the
``--whole_matrix`` / ``--phospho_matrix`` flag or, when omitted, by its default
name in ``--input_dir``. Each matrix need only carry its own assay type's runs.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .converters import load_comparisons, load_conditions, quantdata_from_diann
from .datamodel import QuantData
from .processing import (
    differential_stats,
    log10_normalize,
    relative_occupancy_quant,
    total_int_normalize,
)
from . import plotting

#: Default matrix filenames, used when a matrix is not given an explicit path.
PG_MATRIX = "report.pg_matrix.tsv"
PHOSPHO_MATRIX = "report.phosphosites_90.tsv"


def _resolve_matrix(path, input_dir, default_name):
    """Locate a matrix: an explicit path (absolute, or relative to `input_dir`),
    or `default_name` in `input_dir` when `path` is None."""
    if path is None:
        path = default_name
    return path if os.path.isabs(path) else os.path.join(input_dir, path)


def normalize(qd):
    """log10 then total-intensity normalize a container's quant matrix."""
    logged = log10_normalize(qd.quant, qd.quant_cols)
    scaled = total_int_normalize(logged)
    return qd.with_quant(scaled)


def _full_frame(qd):
    """Feature metadata and quant matrix recombined into one wide DataFrame."""
    return qd.feature_meta.join(qd.quant)


def create_dirs(output_dir, comparisons):
    for experiment in comparisons["Experiment"]:
        os.makedirs(os.path.join(output_dir, experiment), exist_ok=True)


def preprocess(qd, label, output_dir):
    """Normalize and emit QC plots (raw/normalized boxplots, PCA). Returns the
    normalized container."""
    raw_logged = log10_normalize(qd.quant, qd.quant_cols)
    fig = plotting.intensity_boxplot(raw_logged, qd.quant_cols)
    fig.savefig(os.path.join(output_dir, f"intensity_boxplot_{label}.png"))
    plt.close(fig)

    normalized = normalize(qd)

    fig = plotting.intensity_boxplot(normalized.quant, normalized.quant_cols)
    fig.savefig(os.path.join(output_dir, f"intensity_boxplot_{label}_normalized.png"))
    plt.close(fig)

    fig = plotting.pca_plot(normalized.quant, normalized.quant_cols)
    fig.savefig(os.path.join(output_dir, f"pca_plot_{label}.png"))
    plt.close(fig)

    return normalized


def _check_pairing(row):
    """A paired test needs a one-to-one sample correspondence, which pooling two
    or more conditions on a side does not define. Fail fast rather than mis-pair."""
    if not row["Paired"]:
        return
    for side in ("Condition1", "Condition2"):
        if ";" in str(row[side]):
            raise ValueError(
                f"comparison {row['Experiment']!r} is Paired but pools multiple "
                f"conditions in {side} ({row[side]!r}); pairing is undefined"
            )


def run_differential(qd, comparisons, output_dir, prefix):
    """Per comparison: subset to the two conditions, test, write CSV + volcano.

    Returns a dict mapping experiment name to the per-comparison result frame so
    occupancy can reuse it without re-reading CSVs.
    """
    results = {}
    full = _full_frame(qd)
    for _, row in comparisons.iterrows():
        _check_pairing(row)
        experiment = row["Experiment"]
        con1_cols = qd.comparison_samples(row["Condition1"])
        con2_cols = qd.comparison_samples(row["Condition2"])
        comparison_cols = con1_cols + con2_cols

        frame = full[list(qd.feature_meta.columns) + comparison_cols].copy()
        frame = differential_stats(frame, con1_cols, con2_cols, row["Paired"])

        frame.to_csv(os.path.join(output_dir, experiment, f"{prefix}{experiment}.csv"), index=False)
        fig = plotting.volcano_plot(frame, f"{experiment}_{qd.assay_type}")
        fig.savefig(
            os.path.join(output_dir, experiment, f"volcano_plot_{experiment}_{qd.assay_type}.png")
        )
        plt.close(fig)
        results[experiment] = frame
    return results


def run_occupancy(protein_qd, phospho_qd, protein_results, phospho_results, comparisons, output_dir):
    """Relative occupancy per comparison from the in-memory differential frames."""
    for _, row in comparisons.iterrows():
        experiment = row["Experiment"]
        con1_cols = phospho_qd.comparison_samples(row["Condition1"])
        con2_cols = phospho_qd.comparison_samples(row["Condition2"])
        quant_cols = con1_cols + con2_cols

        occupancy = relative_occupancy_quant(
            protein_results[experiment],
            phospho_results[experiment],
            quant_cols,
            con1_cols,
            con2_cols,
            row["Paired"],
        )
        occupancy.to_csv(
            os.path.join(output_dir, experiment, f"relative_occupancy_{experiment}.csv"),
            index=False,
        )
        fig = plotting.volcano_plot(occupancy, f"{experiment}_relative_occupancy")
        fig.savefig(
            os.path.join(output_dir, experiment, f"volcano_plot_{experiment}_relative_occupancy.png")
        )
        plt.close(fig)


def run_pipeline(input_dir, output_dir, whole_matrix=None, phospho_matrix=None):
    conditions = load_conditions(os.path.join(input_dir, "Conditions.csv"))
    comparisons = load_comparisons(os.path.join(input_dir, "Comparisons.csv"))

    create_dirs(output_dir, comparisons)

    whole_path = _resolve_matrix(whole_matrix, input_dir, PG_MATRIX)
    phospho_path = _resolve_matrix(phospho_matrix, input_dir, PHOSPHO_MATRIX)
    protein_qd = quantdata_from_diann(whole_path, conditions, "Whole")
    phospho_qd = quantdata_from_diann(phospho_path, conditions, "Phospho")

    protein_qd = preprocess(protein_qd, "proteinlevel", output_dir)
    phospho_qd = preprocess(phospho_qd, "phospholevel", output_dir)

    _full_frame(protein_qd).to_csv(os.path.join(output_dir, "processed_proteins.csv"), index=False)
    _full_frame(phospho_qd).to_csv(os.path.join(output_dir, "processed_phospho.csv"), index=False)

    protein_results = run_differential(protein_qd, comparisons, output_dir, "protein_")
    phospho_results = run_differential(phospho_qd, comparisons, output_dir, "phospho_")

    run_occupancy(protein_qd, phospho_qd, protein_results, phospho_results, comparisons, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Process DIANN data.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files.")
    parser.add_argument(
        "--whole_matrix",
        type=str,
        default=None,
        help=(
            "Whole-proteome pg_matrix path, absolute or relative to --input_dir. "
            f"Defaults to {PG_MATRIX} in --input_dir. Use to point at a separately "
            "searched whole-cell matrix."
        ),
    )
    parser.add_argument(
        "--phospho_matrix",
        type=str,
        default=None,
        help=(
            "Phosphosite matrix path, absolute or relative to --input_dir. "
            f"Defaults to {PHOSPHO_MATRIX} in --input_dir. Use to point at a "
            "separately searched phospho matrix."
        ),
    )
    args = parser.parse_args()
    run_pipeline(args.input_dir, args.output_dir, args.whole_matrix, args.phospho_matrix)
    print("DONE")


if __name__ == "__main__":
    main()
