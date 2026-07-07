"""Whole-cell and phospho matrices from separate DIA-NN searches.

The two matrices carry disjoint run columns and live under caller-chosen
filenames; occupancy still pairs sites to protein abundance through the shared
``short_name``. Also covers pooled (``;``-separated) comparison sides.
"""

import os

import pandas as pd
import pytest

from phospho import run_pipeline
from phospho.cli import PG_MATRIX, PHOSPHO_MATRIX, _resolve_matrix

pytestmark = pytest.mark.unit

# short_names shared by both assay types; occupancy pairs on these.
SHORT_NAMES = ["A_early_1", "A_early_2", "A_late_3", "A_late_4",
               "B_early_1", "B_early_2", "B_late_3", "B_late_4"]
CONDITIONS = ["A_early", "A_early", "A_late", "A_late",
              "B_early", "B_early", "B_late", "B_late"]
REPLICATES = [1, 2, 3, 4, 1, 2, 3, 4]


def _conditions_frame():
    rows = []
    for assay, tag in (("Whole", "wc"), ("Phospho", "ph")):
        for short, cond, rep in zip(SHORT_NAMES, CONDITIONS, REPLICATES):
            rows.append({"Run": f"{tag}_{short}", "Type": assay,
                         "Condition": cond, "Replicate": rep})
    return pd.DataFrame(rows)


def _whole_matrix():
    # Distinct run columns (wc_*) from the phospho matrix; three protein groups.
    data = {"Protein.Group": ["P1", "P2", "P3"]}
    for i, short in enumerate(SHORT_NAMES):
        data[f"wc_{short}"] = [1000.0 + i, 2000.0 + i, 3000.0 + i]
    return pd.DataFrame(data)


def _phospho_matrix():
    data = {"Protein": ["P1", "P2"], "Residue": ["S", "T"], "Site": [10, 20]}
    for i, short in enumerate(SHORT_NAMES):
        data[f"ph_{short}"] = [5000.0 + i, 6000.0 + i]
    return pd.DataFrame(data)


def _comparisons_frame():
    return pd.DataFrame([
        {"Condition1": "B_early", "Condition2": "A_early",
         "Experiment": "B_vs_A_early", "Paired": False},
        {"Condition1": "B_early;B_late", "Condition2": "A_early;A_late",
         "Experiment": "B_vs_A_combined", "Paired": False},
        {"Condition1": "A_early;B_early", "Condition2": "A_late;B_late",
         "Experiment": "early_vs_late", "Paired": False},
    ])


@pytest.fixture
def separate_input(tmp_path):
    _conditions_frame().to_csv(tmp_path / "Conditions.csv", index=False)
    _comparisons_frame().to_csv(tmp_path / "Comparisons.csv", index=False)
    _whole_matrix().to_csv(tmp_path / "WC_report.pg_matrix.tsv", sep="\t", index=False)
    _phospho_matrix().to_csv(tmp_path / "phospho_sites.tsv", sep="\t", index=False)
    return tmp_path


def test_resolve_matrix_default_name_in_input_dir():
    assert _resolve_matrix(None, os.path.join("in", "dir"), PG_MATRIX) == \
        os.path.join("in", "dir", PG_MATRIX)


def test_resolve_matrix_relative_joins_input_dir():
    assert _resolve_matrix("WC.tsv", "in", PHOSPHO_MATRIX) == os.path.join("in", "WC.tsv")


def test_resolve_matrix_absolute_used_verbatim():
    absolute = os.path.abspath(os.path.join("elsewhere", "m.tsv"))
    assert _resolve_matrix(absolute, "in", PG_MATRIX) == absolute


def test_pipeline_runs_on_separate_searches(separate_input, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    run_pipeline(str(separate_input), str(out),
                 whole_matrix="WC_report.pg_matrix.tsv",
                 phospho_matrix="phospho_sites.tsv")

    for exp in ("B_vs_A_early", "B_vs_A_combined", "early_vs_late"):
        assert os.path.isdir(os.path.join(out, exp))
        assert os.path.exists(os.path.join(out, exp, f"protein_{exp}.csv"))
        assert os.path.exists(os.path.join(out, exp, f"phospho_{exp}.csv"))
        assert os.path.exists(os.path.join(out, exp, f"relative_occupancy_{exp}.csv"))


def test_pooled_comparison_gathers_all_member_samples(separate_input, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    run_pipeline(str(separate_input), str(out),
                 whole_matrix="WC_report.pg_matrix.tsv",
                 phospho_matrix="phospho_sites.tsv")

    frame = pd.read_csv(os.path.join(out, "B_vs_A_combined", "protein_B_vs_A_combined.csv"))
    sample_cols = set(frame.columns) & set(SHORT_NAMES)
    assert sample_cols == set(SHORT_NAMES)  # both timepoints of both genotypes pooled


def test_occupancy_pairs_across_shared_short_names(separate_input, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    run_pipeline(str(separate_input), str(out),
                 whole_matrix="WC_report.pg_matrix.tsv",
                 phospho_matrix="phospho_sites.tsv")

    occ = pd.read_csv(os.path.join(out, "B_vs_A_early", "relative_occupancy_B_vs_A_early.csv"))
    # P1 and P2 are present in the whole-cell protein groups, so both sites survive.
    assert len(occ) == 2


def test_paired_pooled_comparison_raises(separate_input, tmp_path_factory):
    comparisons = _comparisons_frame()
    comparisons.loc[len(comparisons)] = {
        "Condition1": "B_early;B_late", "Condition2": "A_early;A_late",
        "Experiment": "bad_paired", "Paired": True,
    }
    comparisons.to_csv(separate_input / "Comparisons.csv", index=False)

    out = tmp_path_factory.mktemp("out")
    with pytest.raises(ValueError, match="Paired but pools multiple conditions"):
        run_pipeline(str(separate_input), str(out),
                     whole_matrix="WC_report.pg_matrix.tsv",
                     phospho_matrix="phospho_sites.tsv")
