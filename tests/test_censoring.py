"""Censoring bad runs via an optional ``Censor`` column in ``Conditions.csv``.

A censored run is excluded from the analysis but its matrix column must still be
recognized as a run (never leaking into ``feature_meta``). Censoring one assay
type's run leaves the matching ``short_name`` in the other type untouched.
"""

import os

import numpy as np
import pandas as pd
import pytest

from phospho import load_conditions, quantdata_from_diann, run_pipeline

pytestmark = pytest.mark.unit


def _write_conditions(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


BASE_ROWS = [
    {"Run": "wc_g1", "Type": "Whole", "Condition": "GFP", "Replicate": 1},
    {"Run": "wc_g2", "Type": "Whole", "Condition": "GFP", "Replicate": 2},
    {"Run": "ph_g1", "Type": "Phospho", "Condition": "GFP", "Replicate": 1},
    {"Run": "ph_g2", "Type": "Phospho", "Condition": "GFP", "Replicate": 2},
]


def test_absent_censor_column_defaults_false(tmp_path):
    path = tmp_path / "Conditions.csv"
    _write_conditions(path, BASE_ROWS)
    conditions = load_conditions(str(path))
    assert "Censor" in conditions.columns
    assert not conditions["Censor"].any()


def test_censor_column_coerced_to_bool(tmp_path):
    path = tmp_path / "Conditions.csv"
    rows = [dict(r, Censor=c) for r, c in zip(BASE_ROWS, ["false", "0", "TRUE", "no"])]
    _write_conditions(path, rows)
    conditions = load_conditions(str(path))
    assert conditions.set_index("Run")["Censor"].to_dict() == {
        "wc_g1": False, "wc_g2": False, "ph_g1": True, "ph_g2": False,
    }


def test_bad_censor_value_raises(tmp_path):
    path = tmp_path / "Conditions.csv"
    rows = [dict(r) for r in BASE_ROWS]
    rows[0]["Censor"] = "maybe"
    _write_conditions(path, rows)
    with pytest.raises(ValueError, match="boolean"):
        load_conditions(str(path))


def _phospho_matrix(tmp_path):
    """Phospho matrix carrying both phospho run columns."""
    matrix = pd.DataFrame({
        "Protein": ["P1", "P2"],
        "Residue": ["S", "T"],
        "Site": [10, 20],
        "ph_g1": [100.0, 200.0],
        "ph_g2": [110.0, 210.0],
    })
    path = tmp_path / "phospho.tsv"
    matrix.to_csv(path, sep="\t", index=False)
    return str(path)


def test_censored_run_excluded_and_not_leaked_into_features(tmp_path):
    cond_path = tmp_path / "Conditions.csv"
    rows = [dict(r) for r in BASE_ROWS]
    rows[2]["Censor"] = True  # ph_g1
    _write_conditions(cond_path, rows)
    conditions = load_conditions(str(cond_path))

    qd = quantdata_from_diann(_phospho_matrix(tmp_path), conditions, "Phospho")

    assert "GFP_1" not in qd.quant.columns  # short_name of the censored phospho run
    assert list(qd.quant.columns) == ["GFP_2"]
    assert "GFP_1" not in qd.sample_meta.index
    # The censored run's matrix column must not become feature metadata.
    assert "ph_g1" not in qd.feature_meta.columns
    assert "ph_g2" not in qd.feature_meta.columns


def test_censoring_phospho_run_leaves_whole_run(tmp_path):
    cond_path = tmp_path / "Conditions.csv"
    rows = [dict(r) for r in BASE_ROWS]
    rows[2]["Censor"] = True  # ph_g1 only
    _write_conditions(cond_path, rows)
    conditions = load_conditions(str(cond_path))

    wc = pd.DataFrame({
        "Protein.Group": ["P1", "P2"],
        "wc_g1": [1.0, 2.0],
        "wc_g2": [3.0, 4.0],
    })
    wc_path = tmp_path / "wc.tsv"
    wc.to_csv(wc_path, sep="\t", index=False)

    whole_qd = quantdata_from_diann(str(wc_path), conditions, "Whole")
    assert "GFP_1" in whole_qd.quant.columns  # whole-cell rep 1 survives


# --- integration: censoring within the full pipeline ---------------------------

SHORT_NAMES = ["GFP_1", "GFP_2", "GFP_3", "BRSK2_1", "BRSK2_2", "BRSK2_3"]
CONDITIONS = ["GFP", "GFP", "GFP", "BRSK2", "BRSK2", "BRSK2"]
REPLICATES = [1, 2, 3, 1, 2, 3]


@pytest.fixture
def censored_input(tmp_path):
    rows = []
    for assay, tag in (("Whole", "wc"), ("Phospho", "ph")):
        for short, cond, rep in zip(SHORT_NAMES, CONDITIONS, REPLICATES):
            censor = assay == "Phospho" and short == "GFP_1"
            rows.append({"Run": f"{tag}_{short}", "Type": assay, "Condition": cond,
                         "Replicate": rep, "Censor": censor})
    pd.DataFrame(rows).to_csv(tmp_path / "Conditions.csv", index=False)

    pd.DataFrame([{"Condition1": "BRSK2", "Condition2": "GFP",
                   "Experiment": "BRSK2vsGFP", "Paired": False}]).to_csv(
        tmp_path / "Comparisons.csv", index=False)

    wc = {"Protein.Group": ["P1", "P2", "P3"]}
    for i, short in enumerate(SHORT_NAMES):
        wc[f"wc_{short}"] = [1000.0 + i, 2000.0 + i, 3000.0 + i]
    pd.DataFrame(wc).to_csv(tmp_path / "wc.tsv", sep="\t", index=False)

    ph = {"Protein": ["P1", "P2"], "Residue": ["S", "T"], "Site": [10, 20]}
    for i, short in enumerate(SHORT_NAMES):
        ph[f"ph_{short}"] = [5000.0 + i, 6000.0 + i]
    pd.DataFrame(ph).to_csv(tmp_path / "ph.tsv", sep="\t", index=False)
    return tmp_path


def test_pipeline_censors_phospho_run_only(censored_input, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    run_pipeline(str(censored_input), str(out),
                 whole_matrix="wc.tsv", phospho_matrix="ph.tsv")

    processed_ph = pd.read_csv(os.path.join(out, "processed_phospho.csv"))
    processed_pr = pd.read_csv(os.path.join(out, "processed_proteins.csv"))
    assert "GFP_1" not in processed_ph.columns
    assert "GFP_1" in processed_pr.columns  # whole-cell rep 1 retained

    phospho_frame = pd.read_csv(os.path.join(out, "BRSK2vsGFP", "phospho_BRSK2vsGFP.csv"))
    protein_frame = pd.read_csv(os.path.join(out, "BRSK2vsGFP", "protein_BRSK2vsGFP.csv"))
    ph_gfp = set(phospho_frame.columns) & {"GFP_1", "GFP_2", "GFP_3"}
    pr_gfp = set(protein_frame.columns) & {"GFP_1", "GFP_2", "GFP_3"}
    assert ph_gfp == {"GFP_2", "GFP_3"}  # censored rep dropped on phospho side
    assert pr_gfp == {"GFP_1", "GFP_2", "GFP_3"}  # protein side full

    occ = pd.read_csv(os.path.join(out, "BRSK2vsGFP", "relative_occupancy_BRSK2vsGFP.csv"))
    assert "GFP_1" not in occ.columns
    assert len(occ) == 2  # both sites survive, built from remaining phospho reps
