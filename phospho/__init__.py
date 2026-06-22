"""Phosphoproteomics analysis package for label-free DIA-NN data.

Public API grouped by stage:

- converters: :func:`load_conditions`, :func:`load_comparisons`,
  :func:`quantdata_from_diann`
- data model: :class:`QuantData`
- processing: :func:`log10_normalize`, :func:`total_int_normalize`,
  :func:`log2_fold_change`, :func:`ttest_pvalues`, :func:`bh_adjust`,
  :func:`differential_stats`, :func:`get_pg_id`, :func:`build_pg_lookup`,
  :func:`relative_occupancy_quant`
- exports: :func:`build_ptmsigdb_table`, :func:`write_gct`, :func:`create_gct`
- plotting: :func:`volcano_plot`, :func:`pca_plot`, :func:`intensity_boxplot`
- pipeline: :func:`run_pipeline`
"""

from .datamodel import QuantData
from .converters import load_conditions, load_comparisons, quantdata_from_diann
from .processing import (
    log10_normalize,
    total_int_normalize,
    impute_norm,
    log2_fold_change,
    ttest_pvalues,
    bh_adjust,
    differential_stats,
    get_pg_id,
    build_pg_lookup,
    relative_occupancy_quant,
)
from .exports import build_ptmsigdb_table, write_gct, create_gct
from .plotting import volcano_plot, pca_plot, intensity_boxplot
from .cli import run_pipeline

__all__ = [
    "QuantData",
    "load_conditions",
    "load_comparisons",
    "quantdata_from_diann",
    "log10_normalize",
    "total_int_normalize",
    "impute_norm",
    "log2_fold_change",
    "ttest_pvalues",
    "bh_adjust",
    "differential_stats",
    "get_pg_id",
    "build_pg_lookup",
    "relative_occupancy_quant",
    "build_ptmsigdb_table",
    "write_gct",
    "create_gct",
    "volcano_plot",
    "pca_plot",
    "intensity_boxplot",
    "run_pipeline",
]
