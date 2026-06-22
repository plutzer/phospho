"""Regenerate the golden-master subset and its reference outputs.

Provenance: the reference outputs under ``tests/reference/legacy_output/`` were
produced by the pre-refactor scripts (``DIANN_data_processing.py`` and
``phospho_to_ptmsigdb.py``) run on the committed subset in
``tests/reference/subset_input/``. The subset is a small, deterministic slice of
the (gitignored) ``ExampleData/`` directory built to exercise matched, unmatched,
and all-NaN phosphosite rows for relative occupancy.

This script needs ``ExampleData/`` present and is intended for manual reruns; the
generated outputs are committed so the test suite needs neither this script nor
the full dataset.

The committed ``legacy_output/`` is a frozen snapshot from the original
implementation. The root scripts are now thin shims over the ``phospho`` package,
so ``run_legacy()`` here exercises the refactored code; the snapshot's value as an
independent oracle comes from git history (the pre-shim scripts), not from
re-running this script.

Run from the repository root:
    python tests/reference/_generate_reference.py
"""

import os
import shutil
import subprocess
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLE = os.path.join(REPO, "ExampleData")
HERE = os.path.dirname(os.path.abspath(__file__))
SUBSET_INPUT = os.path.join(HERE, "subset_input")
LEGACY_OUTPUT = os.path.join(HERE, "legacy_output")

CONDITIONS = ["GSK3B-Y216D", "GFP_A"]
N_PHOSPHO = 400
N_PROTEIN_BULK = 120


def build_subset():
    conditions = pd.read_csv(os.path.join(EXAMPLE, "Conditions.csv"))
    conditions = conditions[conditions["Condition"].isin(CONDITIONS)]
    kept_runs = set(conditions["Run"])

    pg = pd.read_csv(os.path.join(EXAMPLE, "report.pg_matrix.tsv"), sep="\t")
    ph = pd.read_csv(os.path.join(EXAMPLE, "report.phosphosites_90.tsv"), sep="\t")

    pg_feature = [c for c in pg.columns if c not in set(conditions["Run"]) | _all_runs(EXAMPLE)]
    ph_feature = [c for c in ph.columns if c not in set(conditions["Run"]) | _all_runs(EXAMPLE)]
    pg_cols = pg_feature + [c for c in pg.columns if c in kept_runs]
    ph_cols = ph_feature + [c for c in ph.columns if c in kept_runs]

    ph_subset = ph.iloc[:N_PHOSPHO][ph_cols].copy()
    wanted_accessions = set(ph_subset["Protein"].dropna())

    # Protein groups that match a kept phosphosite (guarantees occupancy matches),
    # plus a deterministic bulk slice (some of which will be unmatched).
    matched_mask = pg["Protein.Group"].apply(
        lambda g: any(acc in wanted_accessions for acc in str(g).split(";"))
    )
    matched = pg[matched_mask]
    bulk = pg.iloc[:N_PROTEIN_BULK]
    pg_subset = pd.concat([matched, bulk]).drop_duplicates("Protein.Group")[pg_cols].copy()

    os.makedirs(SUBSET_INPUT, exist_ok=True)
    conditions.to_csv(os.path.join(SUBSET_INPUT, "Conditions.csv"), index=False)
    pg_subset.to_csv(os.path.join(SUBSET_INPUT, "report.pg_matrix.tsv"), sep="\t", index=False)
    ph_subset.to_csv(os.path.join(SUBSET_INPUT, "report.phosphosites_90.tsv"), sep="\t", index=False)

    comparisons = pd.DataFrame(
        {
            "Condition1": ["GSK3B-Y216D", "GSK3B-Y216D"],
            "Condition2": ["GFP_A", "GFP_A"],
            "Experiment": ["GSK3B_paired", "GSK3B_unpaired"],
            "Paired": [True, False],
        }
    )
    comparisons.to_csv(os.path.join(SUBSET_INPUT, "Comparisons.csv"), index=False)
    print(f"subset: {len(pg_subset)} protein groups, {len(ph_subset)} phosphosites")


def _all_runs(example_dir):
    return set(pd.read_csv(os.path.join(example_dir, "Conditions.csv"))["Run"])


def run_legacy():
    if os.path.isdir(LEGACY_OUTPUT):
        shutil.rmtree(LEGACY_OUTPUT)
    os.makedirs(LEGACY_OUTPUT)
    env = dict(os.environ, MPLBACKEND="Agg")
    subprocess.run(
        [sys.executable, os.path.join(REPO, "DIANN_data_processing.py"),
         "--input_dir", SUBSET_INPUT, "--output_dir", LEGACY_OUTPUT],
        check=True, env=env,
    )
    # PTM-SEA GCT from one phospho differential result.
    phospho_csv = os.path.join(LEGACY_OUTPUT, "GSK3B_paired", "phospho_GSK3B_paired.csv")
    subprocess.run(
        [sys.executable, os.path.join(REPO, "phospho_to_ptmsigdb.py"),
         "--input_file", phospho_csv, "--output_dir", os.path.join(LEGACY_OUTPUT, "GSK3B_paired")],
        check=True, env=env,
    )


if __name__ == "__main__":
    build_subset()
    run_legacy()
    print("reference generation complete")
