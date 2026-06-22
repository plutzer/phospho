"""Guard: exact, replicate-aware condition matching must select the same sample
columns as the legacy substring scan (``condition in column``) on real designs.

If this ever fails, the substring approach and the exact approach disagree for
some comparison (e.g. a condition name that is a substring of another's column),
which is a latent bug to investigate rather than silently accept.
"""

import os

import pandas as pd
import pytest

from phospho import load_comparisons, load_conditions, quantdata_from_diann

PG_MATRIX = "report.pg_matrix.tsv"
PHOSPHO_MATRIX = "report.phosphosites_90.tsv"


def _legacy_substring_select(conditions_type, condition1, condition2):
    comparison_cols = conditions_type[
        (conditions_type["Condition"] == condition1)
        | (conditions_type["Condition"] == condition2)
    ]["short_name"].values
    con1 = [c for c in comparison_cols if condition1 in c]
    con2 = [c for c in comparison_cols if condition2 in c]
    return con1, con2


def _check_directory(input_dir):
    conditions = load_conditions(os.path.join(input_dir, "Conditions.csv"))
    comparisons = load_comparisons(os.path.join(input_dir, "Comparisons.csv"))
    protein = quantdata_from_diann(os.path.join(input_dir, PG_MATRIX), conditions, "Whole")
    phospho = quantdata_from_diann(os.path.join(input_dir, PHOSPHO_MATRIX), conditions, "Phospho")

    for _, row in comparisons.iterrows():
        for qd, assay in ((protein, "Whole"), (phospho, "Phospho")):
            conditions_type = conditions[conditions["Type"] == assay]
            legacy1, legacy2 = _legacy_substring_select(
                conditions_type, row["Condition1"], row["Condition2"]
            )
            exact1 = qd.condition_samples(row["Condition1"])
            exact2 = qd.condition_samples(row["Condition2"])
            assert exact1 == legacy1, (row["Experiment"], assay, "Condition1")
            assert exact2 == legacy2, (row["Experiment"], assay, "Condition2")


@pytest.mark.golden
def test_exact_matches_legacy_on_subset(subset_input):
    _check_directory(subset_input)


@pytest.mark.slow
def test_exact_matches_legacy_on_full_example(example_data_dir):
    _check_directory(example_data_dir)
