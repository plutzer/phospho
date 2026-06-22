import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from phospho import intensity_boxplot, pca_plot, volcano_plot

pytestmark = pytest.mark.unit


@pytest.fixture
def quant():
    rng = np.random.default_rng(0)
    cols = [f"s{i}" for i in range(6)]
    return pd.DataFrame(rng.normal(size=(20, 6)), columns=cols)


def test_intensity_boxplot_returns_figure(quant):
    fig = intensity_boxplot(quant, list(quant.columns))
    assert isinstance(fig, Figure)


def test_pca_plot_returns_figure_with_labels(quant):
    fig = pca_plot(quant, list(quant.columns))
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert "PCA" in ax.get_title()
    assert ax.get_xlabel() == "Principal Component 1"


def test_volcano_plot_returns_figure_with_labels():
    df = pd.DataFrame({"log2FC": [-2.0, 0.0, 2.0], "p_value": [0.001, 0.5, 0.01]})
    fig = volcano_plot(df, "demo")
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Log2 Fold Change"
    assert ax.get_ylabel() == "-Log10 P-Value"
    assert "demo" in ax.get_title()
