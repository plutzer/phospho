import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import time
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests


def pca_plot(dataframe, quant_cols):
    data = impute_norm(dataframe[quant_cols].copy()).dropna(axis=0)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data.T)
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    pca_df['Condition'] = quant_cols
    fig = plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2')

    for i, txt in enumerate(pca_df['Condition']):
        plt.annotate(txt, (pca_df['PC1'][i], pca_df['PC2'][i]), fontsize=8, alpha=0.7)

    plt.title('PCA of Protein Quantification Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig

def impute_norm(dataframe):
    """
    Impute missing values in the DataFrame by rows. Missing values are sampled from a normal distribution with the mean and standard deviation of the non-missing values in each row.
    Rows with less than 2 non-missing values are skipped, but retained in the dataframe.

    Parameters:
        dataframe (pd.DataFrame): DataFrame with missing values to be imputed.

    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
    """
    for index, row in dataframe.iterrows():
        non_missing_values = row.dropna()
        if len(non_missing_values) < 3:
            # print(f"Row {index} has less than 2 non-missing values. Skipping imputation for this row.")
            continue
        mean = non_missing_values.mean()
        std = non_missing_values.std()
        missing_indices = row.index[row.isnull()]
        if not missing_indices.empty:
            imputed_values = np.random.normal(mean, std, size=len(missing_indices))
            dataframe.loc[index, missing_indices] = imputed_values
            # print(f"Imputed {len(missing_indices)} missing values in row {index}.")
    return dataframe

def log10_normalize(dataframe, quant_cols):
    # Convert the quantification columns to log10 scale
    for col in quant_cols:
        if col in dataframe.columns:
            dataframe[col] = np.log10(dataframe[col].replace(0, np.nan))
            print(f"Converted column {col} to log10 scale.")
        else:
            print(f"Column {col} not found in  DataFrame. Skipping log10 conversion.")
    return dataframe

def total_int_normalize(linear_df):
    """
    Normalize the DataFrame by scaling each column so that all columns have the same total intensity (total intensity scaling).

    Parameters:
        linear_df (pd.DataFrame): DataFrame to be normalized.

    Returns:
        pd.DataFrame: Total intensity-scaled DataFrame.
    """
    total_intensity = linear_df.sum(axis=0)  # total per run (column)
    scaling_factors = total_intensity.median() / total_intensity
    scaled_df = linear_df * scaling_factors
    return scaled_df

def preprocess_proteins(proteins, conditions, output_dir):

    # Remove phospho columns from the proteins DataFrame
    phospho_cols = conditions[conditions['Type'] == 'Phospho']['Run'].values

    proteins = proteins.drop(columns=phospho_cols, errors='ignore')
    print(f"Removed phospho columns: {phospho_cols}")

    # Change the columns to the short names
    mapping = conditions[conditions['Type'] == 'Whole'].set_index('Run')['short_name'].to_dict()
    proteins.rename(columns=mapping, inplace=True)

    quant_cols = list(mapping.values())

    un_normalized_proteins = log10_normalize(proteins.copy(), quant_cols)
    un_normalized_proteins[quant_cols].boxplot(figsize=(20, 10), rot=45)
    plt.savefig(os.path.join(output_dir, "intensity_boxplot.png"))
    plt.close()

    # Mean-normalize
    scaled_proteins = total_int_normalize(proteins[quant_cols].copy())
    proteins[quant_cols] = scaled_proteins
    proteins = log10_normalize(proteins, quant_cols)
    
    proteins[quant_cols].boxplot(figsize=(20, 10), rot=45)
    plt.savefig(os.path.join(output_dir, "intensity_boxplot_normalized.png"))
    plt.close()
    
    # Create a PCA plot
    pca = pca_plot(proteins, quant_cols)
    pca.savefig(os.path.join(output_dir, "pca_plot_proteinlevel.png"))
    plt.close()

    return proteins

def preprocess_phospho(phospho, conditions, output_dir):
    whole_cols = conditions[conditions['Type'] == 'Whole']['Run'].values

    phospho = phospho.drop(columns=whole_cols, errors='ignore')
    print(f"Removed whole protein columns: {whole_cols}")

    # Change the columns to the short names
    mapping = conditions[conditions['Type'] == 'Phospho'].set_index('Run')['short_name'].to_dict()
    phospho.rename(columns=mapping, inplace=True)

    quant_cols = list(mapping.values())

    un_normalized_phospho = log10_normalize(phospho.copy(), quant_cols)
    un_normalized_phospho[quant_cols].boxplot(figsize=(20, 10), rot=45)
    plt.savefig(os.path.join(output_dir, "intensity_boxplot_phospho.png"))
    plt.close()

    # Mean-normalize
    scaled_phospho = total_int_normalize(phospho[quant_cols].copy())
    phospho[quant_cols] = scaled_phospho
    phospho = log10_normalize(phospho, quant_cols)
    phospho[quant_cols].boxplot(figsize=(20, 10), rot=45)
    plt.savefig(os.path.join(output_dir, "intensity_boxplot_phospho_normalized.png"))
    plt.close()

    # Create a PCA plot
    pca = pca_plot(phospho, quant_cols)
    pca.savefig(os.path.join(output_dir, "pca_plot_phospholevel.png"))
    plt.close()

    return phospho

def create_dirs(output_dir, comparisons):

    # Check to make sure there are no duplicate experiment names
    if comparisons['Experiment'].duplicated().any():
        raise ValueError("Duplicate experiment names found in Comparisons.csv. Please ensure all experiment names are unique.")

    for comparison in comparisons['Experiment']:
        if not os.path.exists(os.path.join(output_dir, comparison)):
            os.makedirs(os.path.join(output_dir, comparison))
            print(f"Created directory: {os.path.join(output_dir, comparison)}")
    return

def volcano_plot(dataframe, output_dir, title):
    fig = plt.figure(figsize=(10, 8))
    sns.scatterplot(data=dataframe, x='log2FC', y=-np.log10(dataframe['p_value']), alpha=0.7)
    plt.axhline(y=-np.log10(0.05), color='r', linestyle='--', label='p-value = 0.05')
    plt.axvline(x=1, color='g', linestyle='--', label='log2FC = 1')
    plt.axvline(x=-1, color='g', linestyle='--', label='log2FC = -1')
    plt.title(f'Volcano Plot - {title}')
    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-Log10 P-Value')
    plt.legend()
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"volcano_plot_{title}.png")
    fig.savefig(output_file)

def differential_analysis(data, comparisons, conditions, output_dir, type, prefix="", paired=False):
    conditions = conditions[conditions['Type'] == type].copy()

    quant_cols = list(conditions['short_name'].unique())
    
    for index, row in comparisons.iterrows():
        dataframe = data.copy()
        experiment_name = row['Experiment']
        if not os.path.exists(os.path.join(output_dir, experiment_name)):
            print(f"Directory {experiment_name} does not exist. Skipping comparison.")
            continue
        condition1 = row['Condition1']
        condition2 = row['Condition2']
        print(f"Running differential analysis for {experiment_name} between {condition1} and {condition2}...")

        comparison_cols = conditions[(conditions['Condition'] == condition1) | (conditions['Condition'] == condition2)]['short_name'].values

        cols_to_drop = [col for col in quant_cols if col not in comparison_cols]
        if cols_to_drop:
            print(f"Dropping columns not in comparison: {cols_to_drop}")
            dataframe = dataframe.drop(columns=cols_to_drop, errors='ignore')

        con1_cols = [col for col in comparison_cols if condition1 in col]
        con2_cols = [col for col in comparison_cols if condition2 in col]

        # Convert log10 values to log2 fold change: log2FC = (mean_log10_con1 - mean_log10_con2) * (log2(10))
        dataframe['log2FC'] = (dataframe[con1_cols].mean(axis=1) - dataframe[con2_cols].mean(axis=1)) * np.log2(10)

        # Calculate p-values using t-test
        if paired:
            raise  # TODO: Implement paired t-test logic
        else:
            dataframe['p_value'] = ttest_ind(dataframe[con1_cols], dataframe[con2_cols], axis=1, equal_var=False, nan_policy='omit').pvalue

        # Fill missing p-values with NaN with 1
        # dataframe['p_value'] = dataframe['p_value'].fillna(1.0)
        # Adjust p-values for multiple testing using Benjamini-Hochberg method, ignoring missing p-values
        mask = dataframe['p_value'].notna()
        adj_pvals = np.full_like(dataframe['p_value'], np.nan, dtype=np.float64)
        if mask.any():
            adj_pvals[mask] = multipletests(dataframe.loc[mask, 'p_value'], method='fdr_bh')[1]
        dataframe['adj_p_value'] = adj_pvals

        # Save the results to a CSV file
        output_file = os.path.join(output_dir, experiment_name, f"{prefix}{experiment_name}.csv")
        dataframe.to_csv(output_file, index=False)

        # Create a volcano plot
        volcano_plot(dataframe, os.path.join(output_dir, experiment_name), f"{experiment_name}_{type}")

def occupancy_all(conditions, comparisons, output_dir):
    # Run the relative occupancy analysis for all proteins and phosphosites
    for index, row in comparisons.iterrows():
        experiment_name = row['Experiment']
        if not os.path.exists(os.path.join(output_dir, experiment_name)):
            print(f"Directory {experiment_name} does not exist. Skipping occupancy analysis.")
            continue
        print(f"Running relative occupancy analysis for {experiment_name}...")
        protein_df = pd.read_csv(os.path.join(output_dir, experiment_name, f"protein_{experiment_name}.csv"))
        phospho_df = pd.read_csv(os.path.join(output_dir, experiment_name, f"phospho_{experiment_name}.csv"))
        quant_cols_all = list(conditions['short_name'].unique())
        condition1 = row['Condition1']
        condition2 = row['Condition2']
        relative_occupancy(protein_df,
                           phospho_df,
                           quant_cols_all,
                           output_dir=os.path.join(output_dir, experiment_name),
                           conditions = (condition1, condition2),
                           comparison_name=experiment_name,
                           paired=False
                           )

def get_pg_id(protein_id, protein_groups):
    """
    Get the Protein Group ID for a given protein ID.
    
    Parameters:
        protein_id (str): The protein ID to search for.
        protein_groups (list): List of protein group IDs.

    Returns:
        str: The Protein Group ID if found, otherwise returns the original protein ID.
    """
    for pg in protein_groups:
        if protein_id in pg.split(';'):
            return pg
    return None

def relative_occupancy(protein_df, phospho_df, quant_cols_all, output_dir, conditions, comparison_name, paired=False):
    """
    Calculates relative occupancy of phosphosites normalized to protein abundance and saves the results.

    Output:
        Writes a CSV file named 'relative_occupancy_{comparison_name}.csv' to the specified output_dir.
        The file contains all columns from the phospho_df, the normalized quantification columns, 
        and additional columns: 'log2FC', 'p_value', and 'adj_p_value'.
    """
    # Identify the paired conditions
    condition1, condition2 = conditions
    con1_cols = [col for col in quant_cols_all if condition1 in col]
    con2_cols = [col for col in quant_cols_all if condition2 in col]
    quant_cols = [col for col in quant_cols_all if col in protein_df.columns and col in phospho_df.columns]

    pgs = protein_df['Protein.Group'].unique()

    relative_occupancy_df = phospho_df.copy()
    for i, row in phospho_df.iterrows():
        protein_id = row['Protein']
        pg_id = get_pg_id(protein_id, pgs)
        phospho_vals = row[quant_cols].values.flatten()
        protein_vals = protein_df[protein_df['Protein.Group'] == pg_id][quant_cols].values.flatten()
        if len(protein_vals) == 0:
            # print(f"Warning: No protein values found for {pg_id}. Skipping row {i}.")
            normalized_phospho = np.full_like(phospho_vals, np.nan, dtype=np.float64)
            relative_occupancy_df.loc[i, quant_cols] = normalized_phospho
        else:
            normalized_phospho = phospho_vals - protein_vals
            relative_occupancy_df.loc[i, quant_cols] = normalized_phospho
        
    # drop rows from the relative occupancy DataFrame where all quant cols values are NaN
    relative_occupancy_df = relative_occupancy_df.dropna(subset=quant_cols, how='all')

    # Calculate the log2 fold change for the relative occupancy, and t-test p-values
    if paired:
        raise
    else:
        relative_occupancy_df['log2FC'] = (relative_occupancy_df[con1_cols].mean(axis=1) - relative_occupancy_df[con2_cols].mean(axis=1)) * np.log2(10)
        relative_occupancy_df['p_value'] = ttest_ind(relative_occupancy_df[con1_cols], relative_occupancy_df[con2_cols], axis=1, equal_var=False, nan_policy='omit').pvalue

    mask = relative_occupancy_df['p_value'].notna()
    adj_pvals = np.full_like(relative_occupancy_df['p_value'], np.nan, dtype=np.float64)
    if mask.any():
        adj_pvals[mask] = multipletests(relative_occupancy_df.loc[mask, 'p_value'], method='fdr_bh')[1]
    relative_occupancy_df['adj_p_value'] = adj_pvals

    # Save the relative occupancy DataFrame
    output_file = os.path.join(output_dir, f"relative_occupancy_{comparison_name}.csv")
    relative_occupancy_df.to_csv(output_file, index=False)
    
def main():

    parser = argparse.ArgumentParser(description="Process DIANN data.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files.")

    args = parser.parse_args()

    # Experimental Design
    print("Parsing experimental design...")

    comparisons = pd.read_csv(os.path.join(args.input_dir, "Comparisons.csv"))
    conditions = pd.read_csv(os.path.join(args.input_dir, "Conditions.csv"))
    conditions['short_name'] = conditions['Condition'] + '_' + conditions['Replicate'].astype(str)

    # Create directories
    print("Creating directories...")
    create_dirs(args.output_dir, comparisons)

    # Parse and process protein data
    print("Parsing Protein Data...")
    proteins = pd.read_csv(os.path.join(args.input_dir, "report.pg_matrix.tsv"), sep="\t")
    proteins = preprocess_proteins(proteins, conditions, args.output_dir)
    # Save the processed proteins data
    proteins.to_csv(os.path.join(args.output_dir, "processed_proteins.csv"), index=False)

    # Parse and process phospho data
    phospho = pd.read_csv(os.path.join(args.input_dir, "report.phosphosites_90.tsv"), sep="\t")
    phospho = preprocess_phospho(phospho, conditions, args.output_dir)
    phospho.to_csv(os.path.join(args.output_dir, "processed_phospho.csv"), index=False)

    # Normalize the phospho data to protein data
    # TODO

    # Run statistical tests for each comparison.
    differential_analysis(proteins, comparisons, conditions, args.output_dir, type = "Whole", prefix="protein_")
    
    # Here can filter the comparisons to only those that have phospho data
 
    differential_analysis(phospho, comparisons, conditions, args.output_dir, type = "Phospho", prefix="phospho_")
    # Run occupancy analysis for all proteins and phosphosites
    occupancy_all(conditions, comparisons, args.output_dir)
    

    print("DONE")

if __name__ == "__main__":
    main()

