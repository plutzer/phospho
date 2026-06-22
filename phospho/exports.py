"""Export differential phospho results to downstream enrichment formats.

Each exporter is split into a pure ``build_*`` step that produces the table and a
``write_*`` step that serializes it. New targets (e.g. Kinase Library) follow the
same two-function pattern, reusing the shared differential result columns
(`Protein`, `Residue`, `Site`, `log2FC`, `p_value`).
"""

import numpy as np
import pandas as pd

GCT_VERSION = "#1.3"
GCT_FILENAME = "processed_ptm_sea_data.gct"


def build_ptmsigdb_table(data):
    """Build the two-column (id, PValue) table for PTM-SEA / PTMsigDB.

    `id` is ``<Protein>;<Residue><Site>-p`` and `PValue` is the signed
    -log10(p_value), signed by the direction of `log2FC`. Rows with any NaN in
    the two output columns are dropped. `Site` is formatted as an integer, so it
    must be integer-typed upstream (a float column yields ``S71.0-p``).
    """
    data = data.copy()
    data["id"] = data["Protein"] + ";" + data["Residue"] + data["Site"].astype(str) + "-p"
    sign = np.sign(data["log2FC"])
    data["PValue"] = -1 * np.log10(data["p_value"]) * sign
    return data[["id", "PValue"]].dropna().copy()


def write_gct(table, output_path):
    """Write a GCT 1.3 file: version line, dimension line, then the table.

    The dimension line is ``<n_rows>\\t<n_data_cols>\\t0\\t0`` where the row id
    column is excluded from the data-column count.
    """
    table.to_csv(output_path, sep="\t", index=False)
    header1 = GCT_VERSION
    header2 = f"{table.shape[0]}\t{table.shape[1] - 1}\t0\t0"
    with open(output_path, "r+") as f:
        content = f.read()
        f.seek(0, 0)
        f.write(header1 + "\n")
        f.write(header2 + "\n")
        f.write(content)


def create_gct(phospho_path, out_dir):
    """Read a phospho differential CSV and write a PTM-SEA GCT into `out_dir`.

    Returns the written path.
    """
    data = pd.read_csv(phospho_path)
    table = build_ptmsigdb_table(data)
    output_path = f"{out_dir}/{GCT_FILENAME}"
    write_gct(table, output_path)
    return output_path
