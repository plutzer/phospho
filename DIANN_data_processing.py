"""Backward-compatible entry point for the DIA-NN processing pipeline.

The implementation now lives in the ``phospho`` package (see ``phospho.cli``).
This shim preserves the original command-line interface:

    python DIANN_data_processing.py --input_dir <dir> --output_dir <dir>
"""

from phospho.cli import main

if __name__ == "__main__":
    main()
