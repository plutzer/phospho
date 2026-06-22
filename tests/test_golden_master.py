"""End-to-end parity: the refactored pipeline must reproduce the committed
reference outputs from the original scripts on the golden-master subset.
"""

import os

import numpy as np
import pandas as pd
import pytest

from phospho import create_gct, run_pipeline

pytestmark = pytest.mark.golden

EXPERIMENTS = ["GSK3B_paired", "GSK3B_unpaired"]
PER_EXPERIMENT_CSVS = ["protein_{exp}.csv", "phospho_{exp}.csv", "relative_occupancy_{exp}.csv"]
TOP_LEVEL_CSVS = ["processed_proteins.csv", "processed_phospho.csv"]


@pytest.fixture(scope="module")
def new_output(subset_input, tmp_path_factory):
    out = tmp_path_factory.mktemp("new_output")
    run_pipeline(subset_input, str(out))
    return str(out)


def _assert_frames_match(new_df, ref_df):
    assert set(new_df.columns) == set(ref_df.columns)
    new_df = new_df[ref_df.columns].reset_index(drop=True)
    ref_df = ref_df.reset_index(drop=True)
    assert new_df.shape == ref_df.shape
    for col in ref_df.columns:
        if pd.api.types.is_numeric_dtype(ref_df[col]):
            np.testing.assert_allclose(
                new_df[col].to_numpy(dtype=float),
                ref_df[col].to_numpy(dtype=float),
                rtol=1e-9,
                atol=1e-12,
                equal_nan=True,
                err_msg=f"numeric mismatch in column {col}",
            )
        else:
            assert (
                new_df[col].fillna("<NA>").tolist() == ref_df[col].fillna("<NA>").tolist()
            ), f"value mismatch in column {col}"


@pytest.mark.parametrize("name", TOP_LEVEL_CSVS)
def test_top_level_tables_match(new_output, legacy_output, name):
    new_df = pd.read_csv(os.path.join(new_output, name))
    ref_df = pd.read_csv(os.path.join(legacy_output, name))
    _assert_frames_match(new_df, ref_df)


@pytest.mark.parametrize("exp", EXPERIMENTS)
@pytest.mark.parametrize("template", PER_EXPERIMENT_CSVS)
def test_per_experiment_tables_match(new_output, legacy_output, exp, template):
    name = template.format(exp=exp)
    new_df = pd.read_csv(os.path.join(new_output, exp, name))
    ref_df = pd.read_csv(os.path.join(legacy_output, exp, name))
    _assert_frames_match(new_df, ref_df)


def test_relative_occupancy_row_order_preserved(new_output, legacy_output):
    name = "relative_occupancy_GSK3B_paired.csv"
    new_df = pd.read_csv(os.path.join(new_output, "GSK3B_paired", name))
    ref_df = pd.read_csv(os.path.join(legacy_output, "GSK3B_paired", name))
    key = ["Protein", "Residue", "Site"]
    assert new_df[key].values.tolist() == ref_df[key].values.tolist()


def test_gct_export_matches_reference(new_output, legacy_output, tmp_path):
    phospho_csv = os.path.join(new_output, "GSK3B_paired", "phospho_GSK3B_paired.csv")
    create_gct(phospho_csv, str(tmp_path))
    new_lines = (tmp_path / "processed_ptm_sea_data.gct").read_text().splitlines()
    ref_path = os.path.join(legacy_output, "GSK3B_paired", "processed_ptm_sea_data.gct")
    ref_lines = open(ref_path).read().splitlines()

    assert new_lines[:3] == ref_lines[:3]  # version, dimensions, header
    new_body = sorted(new_lines[3:])
    ref_body = sorted(ref_lines[3:])
    assert len(new_body) == len(ref_body)
    for new_row, ref_row in zip(new_body, ref_body):
        new_id, new_val = new_row.split("\t")
        ref_id, ref_val = ref_row.split("\t")
        assert new_id == ref_id
        np.testing.assert_allclose(float(new_val), float(ref_val), rtol=1e-9, atol=1e-12)
