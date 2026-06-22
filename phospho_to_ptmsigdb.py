"""Backward-compatible entry point for the PTM-SEA / PTMsigDB GCT export.

The implementation now lives in ``phospho.exports``. This shim preserves the
original command-line interface:

    python phospho_to_ptmsigdb.py --input_file <phospho.csv> --output_dir <dir>
"""

import argparse

from phospho.exports import create_gct


def main():
    parser = argparse.ArgumentParser(description="Process phosphoproteomics data.")
    parser.add_argument("--input_file", required=True, help="Path to the input data file.")
    parser.add_argument("--output_dir", required=True, help="Path to the output folder.")
    args = parser.parse_args()
    create_gct(args.input_file, args.output_dir)


if __name__ == "__main__":
    main()
