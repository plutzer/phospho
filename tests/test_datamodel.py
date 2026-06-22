import numpy as np
import pandas as pd
import pytest

from phospho import QuantData

pytestmark = pytest.mark.unit


def _sample_meta(short_names, conditions, assay_type):
    return pd.DataFrame(
        {
            "Run": [f"run_{s}" for s in short_names],
            "Type": [assay_type] * len(short_names),
            "Condition": conditions,
            "Replicate": list(range(1, len(short_names) + 1)),
            "short_name": short_names,
        },
        index=short_names,
    )


def make_protein(short_names=("GFP_A_1", "GFP_A_2", "GFP_B_1"),
                 conditions=("GFP_A", "GFP_A", "GFP_B")):
    quant = pd.DataFrame(np.arange(6.0).reshape(2, 3), columns=list(short_names))
    feature_meta = pd.DataFrame({"Protein.Group": ["P1;P2", "P3"]})
    sample_meta = _sample_meta(list(short_names), list(conditions), "Whole")
    return QuantData(quant, feature_meta, sample_meta, "Whole", "protein")


def make_phospho():
    quant = pd.DataFrame(np.arange(4.0).reshape(2, 2), columns=["GFP_A_1", "GFP_A_2"])
    feature_meta = pd.DataFrame({"Protein": ["P1", "P2"], "Residue": ["S", "T"], "Site": [10, 20]})
    sample_meta = _sample_meta(["GFP_A_1", "GFP_A_2"], ["GFP_A", "GFP_A"], "Phospho")
    return QuantData(quant, feature_meta, sample_meta, "Phospho", "phosphosite")


def test_valid_construction():
    qd = make_protein()
    assert qd.quant_cols == ["GFP_A_1", "GFP_A_2", "GFP_B_1"]


def test_condition_samples_is_exact_no_substring_crosstalk():
    qd = make_protein()
    assert qd.condition_samples("GFP_A") == ["GFP_A_1", "GFP_A_2"]
    assert qd.condition_samples("GFP_B") == ["GFP_B_1"]


def test_quant_columns_must_match_sample_meta():
    qd = make_protein()
    bad_quant = qd.quant.rename(columns={"GFP_A_1": "other"})
    with pytest.raises(ValueError, match="quant columns must match"):
        QuantData(bad_quant, qd.feature_meta, qd.sample_meta, "Whole", "protein")


def test_feature_meta_required_columns():
    qd = make_protein()
    with pytest.raises(ValueError, match="missing columns"):
        QuantData(qd.quant, pd.DataFrame({"Other": [1, 2]}), qd.sample_meta, "Whole", "protein")


def test_assay_type_mismatch_raises():
    qd = make_protein()
    bad = qd.sample_meta.copy()
    bad.loc[bad.index[0], "Type"] = "Phospho"
    with pytest.raises(ValueError, match="assay_type"):
        QuantData(qd.quant, qd.feature_meta, bad, "Whole", "protein")


def test_duplicate_short_name_raises():
    quant = pd.DataFrame(np.arange(4.0).reshape(2, 2), columns=["GFP_A_1", "GFP_A_1"])
    feature_meta = pd.DataFrame({"Protein.Group": ["P1", "P2"]})
    sample_meta = _sample_meta(["GFP_A_1", "GFP_A_1"], ["GFP_A", "GFP_A"], "Whole")
    with pytest.raises(ValueError, match="duplicate short_name|match sample_meta"):
        QuantData(quant, feature_meta, sample_meta, "Whole", "protein")


def test_unknown_feature_kind_raises():
    qd = make_protein()
    with pytest.raises(ValueError, match="feature_kind"):
        QuantData(qd.quant, qd.feature_meta, qd.sample_meta, "Whole", "widget")


def test_phospho_rejects_bad_residue():
    qd = make_phospho()
    bad = qd.feature_meta.copy()
    bad.loc[0, "Residue"] = "X"
    with pytest.raises(ValueError, match="residues"):
        QuantData(qd.quant, bad, qd.sample_meta, "Phospho", "phosphosite")


def test_phospho_rejects_multi_protein():
    qd = make_phospho()
    bad = qd.feature_meta.copy()
    bad.loc[0, "Protein"] = "P1;P2"
    with pytest.raises(ValueError, match="single accession"):
        QuantData(qd.quant, bad, qd.sample_meta, "Phospho", "phosphosite")


def test_with_quant_preserves_metadata():
    qd = make_protein()
    smaller = qd.quant.iloc[[0]]
    new = qd.with_quant(smaller)
    assert list(new.feature_meta.index) == [0]
    assert new.assay_type == "Whole"
