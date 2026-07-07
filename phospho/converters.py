"""Parse DIA-NN report matrices and design CSVs into :class:`QuantData`.

The two design CSVs drive everything: ``Conditions.csv`` (one row per run, typed
Whole or Phospho, optionally flagged ``Censor`` to exclude a run) and
``Comparisons.csv`` (one row per pairwise test). A DIA-NN matrix carries its runs
as columns; a converter keeps the uncensored runs of one assay type, renames them
from the raw ``Run`` path to ``short_name``, and treats the remaining columns as
per-feature metadata.
"""

import os

import pandas as pd

from .datamodel import QuantData

#: Assay type (Conditions ``Type``) -> feature kind for the resulting container.
ASSAY_FEATURE_KIND = {"Whole": "protein", "Phospho": "phosphosite"}

_TRUE_STRINGS = {"true", "t", "1", "yes"}
_FALSE_STRINGS = {"false", "f", "0", "no"}


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE_STRINGS:
        return True
    if text in _FALSE_STRINGS:
        return False
    raise ValueError(f"cannot interpret {value!r} as a boolean")


def load_conditions(path):
    """Load ``Conditions.csv`` and add the derived ``short_name`` column.

    ``short_name = Condition + '_' + Replicate``. Fails fast on missing columns
    or a ``Type`` outside {Whole, Phospho}. An optional ``Censor`` column marks
    runs to exclude from the analysis; it is coerced to bool and defaults to
    ``False`` for every run when the column is absent.
    """
    conditions = pd.read_csv(path)
    required = {"Run", "Type", "Condition", "Replicate"}
    missing = required - set(conditions.columns)
    if missing:
        raise ValueError(f"Conditions file missing columns: {sorted(missing)}")
    bad_types = set(conditions["Type"].unique()) - set(ASSAY_FEATURE_KIND)
    if bad_types:
        raise ValueError(f"Conditions Type must be Whole or Phospho; got {sorted(bad_types)}")
    conditions = conditions.copy()
    conditions["short_name"] = (
        conditions["Condition"] + "_" + conditions["Replicate"].astype(str)
    )
    if "Censor" in conditions.columns:
        # A blank cell means "not censored"; coerce the rest, rejecting garbage.
        conditions["Censor"] = conditions["Censor"].map(
            lambda v: False if pd.isna(v) else _coerce_bool(v)
        )
    else:
        conditions["Censor"] = False
    return conditions


def load_comparisons(path):
    """Load ``Comparisons.csv``, coercing ``Paired`` to bool.

    Fails fast on missing columns, duplicate ``Experiment`` names, or an
    uninterpretable ``Paired`` value.
    """
    comparisons = pd.read_csv(path)
    required = {"Condition1", "Condition2", "Experiment", "Paired"}
    missing = required - set(comparisons.columns)
    if missing:
        raise ValueError(f"Comparisons file missing columns: {sorted(missing)}")
    if comparisons["Experiment"].duplicated().any():
        dupes = comparisons.loc[comparisons["Experiment"].duplicated(), "Experiment"].tolist()
        raise ValueError(f"duplicate Experiment names in Comparisons: {dupes}")
    comparisons = comparisons.copy()
    comparisons["Paired"] = comparisons["Paired"].map(_coerce_bool)
    return comparisons


def quantdata_from_diann(matrix_path, conditions, assay_type):
    """Build a :class:`QuantData` from a DIA-NN matrix for one assay type.

    Keeps the runs whose Conditions ``Type`` equals `assay_type` and are not
    censored, renames them to ``short_name``, and keeps all other columns as
    feature metadata. Fails fast if a kept run is absent from the matrix (the
    legacy code dropped silently). A censored run's column, if present, is still
    recognized as a run and excluded from feature metadata; it simply never enters
    the quant matrix.
    """
    if assay_type not in ASSAY_FEATURE_KIND:
        raise ValueError(f"assay_type must be Whole or Phospho, got {assay_type!r}")
    feature_kind = ASSAY_FEATURE_KIND[assay_type]

    matrix = pd.read_csv(matrix_path, sep="\t")

    this_type = conditions[(conditions["Type"] == assay_type) & (~conditions["Censor"])]
    mapping = this_type.set_index("Run")["short_name"]
    absent = [run for run in mapping.index if run not in matrix.columns]
    if absent:
        raise ValueError(
            f"{assay_type} runs missing from {os.path.basename(matrix_path)}: {absent}"
        )

    run_cols = set(conditions["Run"])
    feature_cols = [c for c in matrix.columns if c not in run_cols]

    quant = matrix[list(mapping.index)].rename(columns=mapping.to_dict())
    feature_meta = matrix[feature_cols].copy()

    sample_meta = this_type.copy()
    sample_meta.index = sample_meta["short_name"]

    return QuantData(
        quant=quant,
        feature_meta=feature_meta,
        sample_meta=sample_meta,
        assay_type=assay_type,
        feature_kind=feature_kind,
    )
