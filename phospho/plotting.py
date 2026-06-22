"""Plotting helpers that return matplotlib Figures.

These functions never write to disk; callers (e.g. the CLI) save the returned
Figure. Set a non-interactive backend (``matplotlib.use("Agg")``) before import
for headless use.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from .processing import impute_norm


def intensity_boxplot(dataframe, quant_cols, title=None):
    """Boxplot of per-sample intensity distributions."""
    fig, ax = plt.subplots(figsize=(20, 10))
    dataframe[quant_cols].boxplot(rot=45, ax=ax)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def pca_plot(dataframe, quant_cols, title="PCA of Protein Quantification Data"):
    """PCA scatter (PC1 vs PC2) of samples, imputing missing values first.

    Missing values are imputed only to allow the decomposition; this does not
    affect the statistical pipeline.
    """
    data = impute_norm(dataframe[quant_cols].copy()).dropna(axis=0)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data.T)
    pca_df = pd.DataFrame(data=pca_result, columns=["PC1", "PC2"])
    pca_df["Condition"] = quant_cols

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", ax=ax)
    for i, txt in enumerate(pca_df["Condition"]):
        ax.annotate(txt, (pca_df["PC1"][i], pca_df["PC2"][i]), fontsize=8, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    fig.tight_layout()
    return fig


def volcano_plot(dataframe, title):
    """Volcano scatter of `log2FC` vs -log10(`p_value`) with reference lines."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        data=dataframe, x="log2FC", y=-np.log10(dataframe["p_value"]), alpha=0.7, ax=ax
    )
    ax.axhline(y=-np.log10(0.05), color="r", linestyle="--", label="p-value = 0.05")
    ax.axvline(x=1, color="g", linestyle="--", label="log2FC = 1")
    ax.axvline(x=-1, color="g", linestyle="--", label="log2FC = -1")
    ax.set_title(f"Volcano Plot - {title}")
    ax.set_xlabel("Log2 Fold Change")
    ax.set_ylabel("-Log10 P-Value")
    ax.legend()
    fig.tight_layout()
    return fig
