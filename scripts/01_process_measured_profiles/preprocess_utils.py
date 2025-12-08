import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse
import os
import sys
from pathlib import Path
import logging

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config, setup_project
from utils import setup_logging_for_script, log_script_start, log_script_end, ensure_directories_exist

# Setup project and logging
setup_project()
logger = setup_logging_for_script(__file__)

# Set project directories using configuration
path = os.getcwd()
home_dir = str(config.PROJECT_ROOT)
data_dir = str(config.DATA_DIR)
resources_dir = str(config.RESOURCES_DIR)

# Ensure directories exist
ensure_directories_exist(home_dir,data_dir,resources_dir)

def get_gene_set():
    gene_list_df = pd.read_csv(os.path.join(resources_dir, 'OS_scRNA_gene_index.19264.tsv'), header=0, delimiter='\t')
    gene_list = list(gene_list_df['gene_name'])
    return gene_list


def main_gene_selection(X_df, gene_list):
    """
    Describe:
        rebuild the input adata to unified gene symbol
    """
    remove_columns = list(set(X_df.columns) - set(gene_list))
    X_df = X_df.drop(columns=remove_columns)
    to_fill_columns = list(set(gene_list) - set(X_df.columns))
    print(f'No. fill columns: {len(to_fill_columns)}')
    padding_df = pd.DataFrame(np.zeros((X_df.shape[0], len(to_fill_columns))), 
                              columns=to_fill_columns, 
                              index=X_df.index)
    X_df = pd.DataFrame(np.concatenate([df.values for df in [X_df, padding_df]], axis=1), 
                        index=X_df.index, 
                        columns=list(X_df.columns) + list(padding_df.columns))
    X_df = X_df[gene_list]
    
    var = pd.DataFrame(index=X_df.columns)
    var['mask'] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
    return X_df, to_fill_columns,var


def basic_filter(merged_df, qc_min_genes=200, qc_min_cells=0):
    sc.pp.filter_cells(merged_df, min_genes=qc_min_genes)
    sc.pp.filter_genes(merged_df, min_cells=qc_min_cells)
    return merged_df


def process_merged_df(merged_df):
    assert('gene_name' in merged_df.var.columns)
    merged_df = basic_filter(merged_df)
    qc_metrics = sc.pp.calculate_qc_metrics(merged_df)
    merged_df.obs['total_count'] = qc_metrics[0]['total_counts']
    merged_df.var.index = merged_df.var.index.str.replace(r'\.\d+', '', regex=True)
    merged_df.var['gene_name'] = merged_df.var.index

    return merged_df


def scfoundation_preprocessing(merged_df):
    obs_df = merged_df.obs
    gene_list = get_gene_set()

    merged_df_pd = merged_df.to_df()
    X_df, to_fill_columns, var = main_gene_selection(merged_df_pd, gene_list)
    X_df = X_df.astype('float32')

    adata = sc.AnnData(sparse.csr_matrix(X_df), obs=obs_df, var=X_df.columns.to_frame() )
    adata.var['gene_name'] = adata.var.index
    adata.var = adata.var.drop(columns=adata.var.columns[0])

    return(adata)