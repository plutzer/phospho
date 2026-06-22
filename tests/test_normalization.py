import numpy as np
import pandas as pd
import pytest

from phospho import log10_normalize, total_int_normalize

pytestmark = pytest.mark.unit


def test_log10_maps_values_and_zero_to_nan():
    df = pd.DataFrame({"a": [1, 10, 100, 0], "b": [1000, 1, 0, 10]})
    out = log10_normalize(df, ["a", "b"])
    assert out["a"].tolist()[:3] == [0.0, 1.0, 2.0]
    assert np.isnan(out["a"].iloc[3])
    assert np.isnan(out["b"].iloc[2])


def test_log10_does_not_mutate_input():
    df = pd.DataFrame({"a": [1, 10]})
    log10_normalize(df, ["a"])
    assert df["a"].tolist() == [1, 10]


def test_log10_missing_column_raises():
    df = pd.DataFrame({"a": [1, 10]})
    with pytest.raises(KeyError):
        log10_normalize(df, ["a", "missing"])


def test_total_int_normalize_scales_columns_to_shared_median():
    # medians: a=2, b=4; mean(medians)=3; scaling a=1.5, b=0.75.
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
    out = total_int_normalize(df)
    np.testing.assert_allclose(out["a"], [1.5, 3.0, 4.5])
    np.testing.assert_allclose(out["b"], [1.5, 3.0, 4.5])
    # equal medians after scaling
    np.testing.assert_allclose(out.median(axis=0), [3.0, 3.0])
