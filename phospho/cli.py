"""Command-line entry point: DIA-NN matrices -> normalized tables, differential
statistics, relative occupancy, and QC plots.

Reproduces the staged outputs of the original ``DIANN_data_processing.py`` using
the modular package: converters build :class:`QuantData`, processing normalizes
and tests, plotting renders QC figures, and exports are produced separately.
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

PG_MATRIX = "report.pg_matrix.tsv"
PHOSPHO_MATRIX = "report.phosphosites_90.tsv"


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


def run_differential(qd, comparisons, output_dir, prefix):
    """Per comparison: subset to the two conditions, test, write CSV + volcano.

    Returns a dict mapping experiment name to the per-comparison result frame so
    occupancy can reuse it without re-reading CSVs.
    """
    results = {}
    full = _full_frame(qd)
    for _, row in comparisons.iterrows():
        experiment = row["Experiment"]
        con1_cols = qd.condition_samples(row["Condition1"])
        con2_cols = qd.condition_samples(row["Condition2"])
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
        con1_cols = phospho_qd.condition_samples(row["Condition1"])
        con2_cols = phospho_qd.condition_samples(row["Condition2"])
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


def run_pipeline(input_dir, output_dir):
    conditions = load_conditions(os.path.join(input_dir, "Conditions.csv"))
    comparisons = load_comparisons(os.path.join(input_dir, "Comparisons.csv"))

    create_dirs(output_dir, comparisons)

    protein_qd = quantdata_from_diann(os.path.join(input_dir, PG_MATRIX), conditions, "Whole")
    phospho_qd = quantdata_from_diann(os.path.join(input_dir, PHOSPHO_MATRIX), conditions, "Phospho")

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
    args = parser.parse_args()
    run_pipeline(args.input_dir, args.output_dir)
    print("DONE")


if __name__ == "__main__":
    main()
