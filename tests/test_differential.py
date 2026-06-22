import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_ind, ttest_rel
from statsmodels.stats.multitest import multipletests

from phospho import bh_adjust, differential_stats, log2_fold_change, ttest_pvalues

pytestmark = pytest.mark.unit

LOG2_10 = np.log2(10)


def test_log2_fold_change_paired_is_mean_of_row_differences():
    con1 = pd.DataFrame({"x1": [2.0, 1.0], "x2": [3.0, 1.0]})
    con2 = pd.DataFrame({"y1": [1.0, 0.0], "y2": [1.0, 2.0]})
    expected = np.array([np.mean([1.0, 2.0]), np.mean([1.0, -1.0])]) * LOG2_10
    np.testing.assert_allclose(log2_fold_change(con1, con2, paired=True), expected)


def test_log2_fold_change_unpaired_is_difference_of_means():
    con1 = pd.DataFrame({"x1": [2.0], "x2": [4.0]})
    con2 = pd.DataFrame({"y1": [1.0], "y2": [1.0]})
    expected = np.array([(3.0 - 1.0)]) * LOG2_10
    np.testing.assert_allclose(log2_fold_change(con1, con2, paired=False), expected)


def test_log2_fold_change_paired_omits_nan():
    con1 = pd.DataFrame({"x1": [2.0], "x2": [np.nan]})
    con2 = pd.DataFrame({"y1": [1.0], "y2": [0.5]})
    # only the first replicate pair contributes: (2-1)=1
    np.testing.assert_allclose(log2_fold_change(con1, con2, paired=True), [1.0 * LOG2_10])


def test_ttest_pvalues_match_scipy_calls():
    con1 = pd.DataFrame(np.array([[2.0, 3.0, 2.5], [1.0, 1.2, 0.9]]))
    con2 = pd.DataFrame(np.array([[1.0, 1.1, 0.8], [1.0, 1.3, 1.1]]))
    np.testing.assert_allclose(
        ttest_pvalues(con1, con2, paired=True),
        ttest_rel(con1, con2, axis=1, nan_policy="omit").pvalue,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        ttest_pvalues(con1, con2, paired=False),
        ttest_ind(con1, con2, axis=1, equal_var=False, nan_policy="omit").pvalue,
        equal_nan=True,
    )


def test_bh_adjust_preserves_nan_and_matches_subset():
    pvals = np.array([0.01, np.nan, 0.04, 0.20, 0.005])
    out = bh_adjust(pvals)
    assert np.isnan(out[1])
    mask = ~np.isnan(pvals)
    expected = multipletests(pvals[mask], method="fdr_bh")[1]
    np.testing.assert_allclose(out[mask], expected)


def test_bh_adjust_all_nan_returns_all_nan():
    out = bh_adjust(np.array([np.nan, np.nan]))
    assert np.isnan(out).all()


def test_differential_stats_appends_three_columns():
    df = pd.DataFrame(
        {
            "g1": [2.0, 1.0, 5.0],
            "g2": [2.2, 1.1, 5.0],
            "h1": [1.0, 1.0, 5.0],
            "h2": [1.1, 0.9, 5.0],
        }
    )
    out = differential_stats(df, ["g1", "g2"], ["h1", "h2"], paired=False)
    assert {"log2FC", "p_value", "adj_p_value"} <= set(out.columns)
    # constant row (all 5.0) yields an undefined t-test -> NaN, which stays NaN.
    assert np.isnan(out["p_value"].iloc[2])
    assert np.isnan(out["adj_p_value"].iloc[2])
