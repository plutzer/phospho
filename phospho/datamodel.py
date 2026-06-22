"""The :class:`QuantData` container shared across the pipeline.

A ``QuantData`` couples a quantification matrix (features x samples) with sample
metadata (from ``Conditions.csv``) and per-feature metadata. Proteins and
phosphosites are each represented by one instance; relative occupancy links the
two. Construction is fail-fast: a malformed combination of matrix, samples, and
features raises rather than silently producing a degenerate object.
"""

from dataclasses import dataclass

import pandas as pd

#: Feature-metadata columns each feature kind must carry.
REQUIRED_FEATURE_COLS = {
    "protein": ["Protein.Group"],
    "phosphosite": ["Protein", "Residue", "Site"],
}

VALID_RESIDUES = {"S", "T", "Y"}


@dataclass
class QuantData:
    quant: pd.DataFrame
    feature_meta: pd.DataFrame
    sample_meta: pd.DataFrame
    assay_type: str  # "Whole" or "Phospho"
    feature_kind: str  # "protein" or "phosphosite"

    def __post_init__(self):
        if self.feature_kind not in REQUIRED_FEATURE_COLS:
            raise ValueError(
                f"feature_kind must be one of {sorted(REQUIRED_FEATURE_COLS)}, "
                f"got {self.feature_kind!r}"
            )

        if list(self.quant.columns) != list(self.sample_meta.index):
            raise ValueError(
                "quant columns must match sample_meta index exactly (order included)"
            )
        if list(self.quant.index) != list(self.feature_meta.index):
            raise ValueError("quant index must match feature_meta index exactly")

        missing = [
            c
            for c in REQUIRED_FEATURE_COLS[self.feature_kind]
            if c not in self.feature_meta.columns
        ]
        if missing:
            raise ValueError(
                f"feature_meta for {self.feature_kind!r} missing columns: {missing}"
            )

        bad_type = self.sample_meta["Type"][self.sample_meta["Type"] != self.assay_type]
        if not bad_type.empty:
            raise ValueError(
                f"sample_meta Type values {sorted(bad_type.unique())} do not all "
                f"equal assay_type {self.assay_type!r}"
            )

        duplicates = self.sample_meta.index[self.sample_meta.index.duplicated()]
        if len(duplicates):
            raise ValueError(f"duplicate short_name sample ids: {list(duplicates)}")

        if self.feature_kind == "phosphosite":
            self._validate_phosphosites()

    def _validate_phosphosites(self):
        residues = set(self.feature_meta["Residue"].dropna().unique())
        bad_residues = residues - VALID_RESIDUES
        if bad_residues:
            raise ValueError(f"unexpected phospho residues: {sorted(bad_residues)}")
        try:
            self.feature_meta["Site"].astype(int)
        except (ValueError, TypeError) as exc:
            raise ValueError("phospho Site column is not integer-castable") from exc
        has_semicolon = self.feature_meta["Protein"].astype(str).str.contains(";")
        if has_semicolon.any():
            raise ValueError(
                "phospho Protein column must be a single accession (no ';'); "
                "protein-group matching relies on this"
            )

    @property
    def quant_cols(self):
        """Sample column names (the short_names), in matrix order."""
        return list(self.sample_meta.index)

    def condition_samples(self, condition):
        """Exact, replicate-aware sample columns for one condition.

        Replaces substring matching on column names: only samples whose
        ``Condition`` equals `condition` are returned.
        """
        return self.sample_meta.index[self.sample_meta["Condition"] == condition].tolist()

    def copy(self):
        return QuantData(
            quant=self.quant.copy(),
            feature_meta=self.feature_meta.copy(),
            sample_meta=self.sample_meta.copy(),
            assay_type=self.assay_type,
            feature_kind=self.feature_kind,
        )

    def with_quant(self, new_quant):
        """Return a new container with a replacement quant matrix, same metadata."""
        return QuantData(
            quant=new_quant,
            feature_meta=self.feature_meta.loc[new_quant.index],
            sample_meta=self.sample_meta,
            assay_type=self.assay_type,
            feature_kind=self.feature_kind,
        )
