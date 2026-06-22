import numpy as np
import pandas as pd
import pytest

from phospho import build_pg_lookup, get_pg_id, relative_occupancy_quant

pytestmark = pytest.mark.unit


def test_get_pg_id_membership_and_absence():
    groups = ["P1;P2", "P3", "P4;P5;P6"]
    assert get_pg_id("P2", groups) == "P1;P2"
    assert get_pg_id("P4", groups) == "P4;P5;P6"
    assert get_pg_id("ZZ", groups) is None


def test_get_pg_id_first_match_wins():
    groups = ["P1;P2", "P2;P9"]
    assert get_pg_id("P2", groups) == "P1;P2"


def test_build_pg_lookup_matches_get_pg_id():
    groups = ["P1;P2", "P2;P9", "P3"]
    lookup = build_pg_lookup(groups)
    for accession in ["P1", "P2", "P9", "P3"]:
        assert lookup.get(accession) == get_pg_id(accession, groups)


def _frames():
    protein = pd.DataFrame(
        {
            "Protein.Group": ["P1;P2", "P3"],
            "c1": [1.0, 2.0],
            "c2": [1.0, 2.0],
            "t1": [3.0, 4.0],
            "t2": [3.0, 4.0],
        }
    )
    phospho = pd.DataFrame(
        {
            "Protein": ["P2", "P3", "PX", "P1"],
            "Residue": ["S", "T", "Y", "S"],
            "Site": [10, 20, 30, 40],
            "c1": [5.0, 6.0, 7.0, np.nan],
            "c2": [5.0, 6.0, 7.0, np.nan],
            "t1": [9.0, 10.0, 11.0, np.nan],
            "t2": [9.0, 10.0, 11.0, np.nan],
        }
    )
    return protein, phospho


def test_relative_occupancy_subtracts_matched_protein():
    protein, phospho = _frames()
    out = relative_occupancy_quant(
        protein, phospho, ["c1", "c2", "t1", "t2"], ["c1", "c2"], ["t1", "t2"], paired=True
    )
    # P2 matches group P1;P2 (protein 1.0/3.0): 5-1=4 control, 9-3=6 treated.
    row = out[out["Protein"] == "P2"].iloc[0]
    assert row["c1"] == 4.0 and row["t1"] == 6.0


def test_relative_occupancy_drops_unmatched_and_all_nan():
    protein, phospho = _frames()
    out = relative_occupancy_quant(
        protein, phospho, ["c1", "c2", "t1", "t2"], ["c1", "c2"], ["t1", "t2"], paired=True
    )
    proteins_out = set(out["Protein"])
    assert "PX" not in proteins_out  # unmatched protein group -> all NaN -> dropped
    assert "P1" not in proteins_out  # all quant values NaN -> dropped
    assert {"P2", "P3"} <= proteins_out
    assert {"log2FC", "p_value", "adj_p_value"} <= set(out.columns)
