import argparse
import numpy as np
import pandas as pd

def create_gct(phospho_path, out_dir):
    data = pd.read_csv(phospho_path)
    # Perform any necessary processing on the data here
    # For example, you might want to filter or transform the data

    data['id'] = data['Protein'] + ';' + data['Residue'] + data['Site'].astype(str) + '-p'

    # Sign is the direction of the change -1 or 1 based on the log2FC column
    data['sign'] = np.sign(data['log2FC'])

    data['PValue'] = -1*np.log10(data['p_value']) * data['sign']

    out_data = data[['id', 'PValue']].dropna().copy()

    # Then save the processed data to the output directory
    output_path = f"{out_dir}/processed_ptm_sea_data.gct"
    out_data.to_csv(output_path, sep="\t", index=False)

    # Prepend the header for GCT format
    header1 = "#1.3"
    header2 = f"{out_data.shape[0]}\t{out_data.shape[1] - 1}\t0\t0"
    with open(output_path, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(header1 + "\n")
        f.write(header2 + "\n")
        f.write(content)

def main():
    parser = argparse.ArgumentParser(description="Process phosphoproteomics data.")
    parser.add_argument("--input_file", required=True, help="Path to the input data file.")
    parser.add_argument("--output_dir", required=True, help="Path to the output folder.")


    args = parser.parse_args()
    input_file = args.input_file
    output_dir = args.output_dir

    create_gct(input_file, output_dir)


if __name__ == "__main__":
    main()
