import numpy as np
import pandas as pd
import pytest

from phospho import build_ptmsigdb_table, create_gct, write_gct

pytestmark = pytest.mark.unit


def _diff_frame():
    return pd.DataFrame(
        {
            "Protein": ["P1", "P2", "P3"],
            "Residue": ["S", "T", "Y"],
            "Site": [71, 100, 5],
            "log2FC": [2.0, -1.0, 0.5],
            "p_value": [0.01, 0.001, np.nan],
        }
    )


def test_ptmsigdb_id_uses_integer_site():
    table = build_ptmsigdb_table(_diff_frame())
    assert "P1;S71-p" in set(table["id"])
    assert "P2;T100-p" in set(table["id"])


def test_ptmsigdb_pvalue_is_signed_neg_log10():
    table = build_ptmsigdb_table(_diff_frame())
    up = table[table["id"] == "P1;S71-p"]["PValue"].iloc[0]
    down = table[table["id"] == "P2;T100-p"]["PValue"].iloc[0]
    np.testing.assert_allclose(up, -np.log10(0.01) * 1)
    np.testing.assert_allclose(down, -np.log10(0.001) * -1)


def test_ptmsigdb_drops_nan_rows():
    table = build_ptmsigdb_table(_diff_frame())
    assert "P3;Y5-p" not in set(table["id"])  # NaN p_value dropped
    assert len(table) == 2


def test_write_gct_header(tmp_path):
    table = build_ptmsigdb_table(_diff_frame())
    out = tmp_path / "out.gct"
    write_gct(table, str(out))
    lines = out.read_text().splitlines()
    assert lines[0] == "#1.3"
    assert lines[1] == f"{table.shape[0]}\t1\t0\t0"
    assert lines[2].split("\t") == ["id", "PValue"]


def test_create_gct_roundtrip(tmp_path):
    csv = tmp_path / "phospho.csv"
    _diff_frame().to_csv(csv, index=False)
    path = create_gct(str(csv), str(tmp_path))
    content = open(path).read()
    assert content.startswith("#1.3\n")
    assert "P1;S71-p" in content
