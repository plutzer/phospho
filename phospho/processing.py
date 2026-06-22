"""Filtering, normalization, and statistics for quantitative proteomics data.

All quantification is carried in **log10** space once :func:`log10_normalize` has
run; fold changes are reported in log2 (`* log2(10)`). Functions here are pure:
they return new arrays/frames rather than mutating their inputs.
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel
from statsmodels.stats.multitest import multipletests

LOG2_OF_10 = np.log2(10)


def log10_normalize(dataframe, quant_cols):
    """Return a copy with `quant_cols` converted to log10, mapping 0 -> NaN.

    Zeros become NaN before the log so missing measurements do not collapse to
    -inf. Raises KeyError if a requested column is absent (fail-fast; the legacy
    code skipped silently).
    """
    missing = [c for c in quant_cols if c not in dataframe.columns]
    if missing:
        raise KeyError(f"quant columns not found for log10_normalize: {missing}")
    out = dataframe.copy()
    for col in quant_cols:
        out[col] = np.log10(out[col].replace(0, np.nan))
    return out


def total_int_normalize(values):
    """Scale each column so all columns share the mean per-column median.

    `scaling = median.mean() / median` per column, then multiply. Operates on
    whatever scale it is given; in this pipeline that is log10 values.
    """
    med_intensity = values.median(axis=0)
    scaling_factors = med_intensity.mean() / med_intensity
    return values * scaling_factors


def impute_norm(dataframe):
    """Impute missing values row-wise from N(row mean, row std).

    Used only to feed the PCA plot; the statistical tests operate on unimputed
    data. Rows with fewer than three non-missing values are left untouched.
    Mutates and returns `dataframe`.
    """
    for index, row in dataframe.iterrows():
        non_missing_values = row.dropna()
        if len(non_missing_values) < 3:
            continue
        mean = non_missing_values.mean()
        std = non_missing_values.std()
        missing_indices = row.index[row.isnull()]
        if not missing_indices.empty:
            imputed_values = np.random.normal(mean, std, size=len(missing_indices))
            dataframe.loc[index, missing_indices] = imputed_values
    return dataframe


def log2_fold_change(con1, con2, paired):
    """log2 fold change between two condition blocks of log10 values.

    Paired: mean of per-row differences. Unpaired: difference of per-row means.
    Both scaled by log2(10) to convert the log10 difference to log2.
    """
    con1 = np.asarray(con1, dtype=np.float64)
    con2 = np.asarray(con2, dtype=np.float64)
    if paired:
        return np.nanmean(con1 - con2, axis=1) * LOG2_OF_10
    return (np.nanmean(con1, axis=1) - np.nanmean(con2, axis=1)) * LOG2_OF_10


def ttest_pvalues(con1, con2, paired):
    """Row-wise t-test p-values; NaNs omitted per row.

    Paired -> related-samples t-test; unpaired -> Welch's t-test.
    """
    if paired:
        return ttest_rel(con1, con2, axis=1, nan_policy="omit").pvalue
    return ttest_ind(con1, con2, axis=1, equal_var=False, nan_policy="omit").pvalue


def bh_adjust(pvalues):
    """Benjamini-Hochberg adjust, leaving NaN p-values as NaN.

    Adjustment is computed over the non-NaN subset only so that untestable rows
    stay NaN rather than being coerced to a value.
    """
    pvalues = np.asarray(pvalues, dtype=np.float64)
    mask = ~np.isnan(pvalues)
    adjusted = np.full_like(pvalues, np.nan, dtype=np.float64)
    if mask.any():
        adjusted[mask] = multipletests(pvalues[mask], method="fdr_bh")[1]
    return adjusted


def differential_stats(dataframe, con1_cols, con2_cols, paired):
    """Append `log2FC`, `p_value`, and `adj_p_value` columns for one comparison.

    `dataframe` holds log10-normalized quantification; `con1_cols`/`con2_cols`
    name the columns for each condition. Returns a copy.
    """
    out = dataframe.copy()
    out["log2FC"] = log2_fold_change(out[con1_cols], out[con2_cols], paired)
    out["p_value"] = ttest_pvalues(out[con1_cols], out[con2_cols], paired)
    out["adj_p_value"] = bh_adjust(out["p_value"])
    return out


def get_pg_id(protein_id, protein_groups):
    """Return the protein group whose `;`-joined members include `protein_id`.

    First match wins; returns None when no group contains the accession.
    """
    for pg in protein_groups:
        if protein_id in pg.split(";"):
            return pg
    return None


def build_pg_lookup(protein_groups):
    """Map each member accession to its protein group (first occurrence wins).

    Vectorized equivalent of repeated :func:`get_pg_id` calls. With duplicate
    accessions across groups the first group seen is kept, matching the linear
    scan order of :func:`get_pg_id`.
    """
    lookup = {}
    for pg in protein_groups:
        for accession in pg.split(";"):
            if accession not in lookup:
                lookup[accession] = pg
    return lookup


def relative_occupancy_quant(protein_df, phospho_df, quant_cols, con1_cols, con2_cols, paired):
    """Phosphosite occupancy relative to protein abundance, with differential stats.

    Each phosphosite's `Protein` is matched to a protein group (see
    :func:`build_pg_lookup`); occupancy is ``phospho - protein`` per sample in
    log10 space. Sites whose protein group is absent become all-NaN and are
    dropped along with sites that are all-NaN across `quant_cols`. The surviving
    rows then receive `log2FC`, `p_value`, and `adj_p_value`.

    `quant_cols` is restricted to columns present in both frames, mirroring the
    per-comparison column subset the differential step produced upstream — this
    keeps the dropped-row set (and hence the BH denominator) faithful.
    """
    quant_cols = [c for c in quant_cols if c in protein_df.columns and c in phospho_df.columns]

    pgs = pd.unique(protein_df["Protein.Group"])
    lookup = build_pg_lookup(pgs)
    protein_by_pg = protein_df.drop_duplicates("Protein.Group").set_index("Protein.Group")[quant_cols]

    phospho_pg = phospho_df["Protein"].map(lookup.get)
    protein_aligned = protein_by_pg.reindex(phospho_pg).to_numpy()
    occupancy = phospho_df[quant_cols].to_numpy() - protein_aligned

    result = phospho_df.copy()
    result[quant_cols] = occupancy
    result = result.dropna(subset=quant_cols, how="all")

    return differential_stats(result, con1_cols, con2_cols, paired)
