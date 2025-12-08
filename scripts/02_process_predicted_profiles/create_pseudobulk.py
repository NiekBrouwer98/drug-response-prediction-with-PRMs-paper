import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import scanpy as sc
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

def process_mcfarland_observations():
    all_cell_lines = sc.read_h5ad(os.path.join(data_dir, 'mcfarland_processed','all_cell_lines.h5ad'))

    input_data_df = pd.DataFrame.sparse.from_spmatrix(all_cell_lines.X, columns=all_cell_lines.var_names, index=all_cell_lines.obs_names)
    input_data_df['condition'] = all_cell_lines.obs['condition']
    input_data_df['cell_type'] = all_cell_lines.obs['cell_type']
    input_data_df = input_data_df.groupby(['cell_type','condition']).mean()
    input_data_df.reset_index(inplace=True)
    mean_pre_treatment = input_data_df[input_data_df['condition'] == 'ctrl']
    mean_post_treatment = input_data_df[input_data_df['condition'] != 'ctrl']
    mean_post_treatment.dropna(inplace=True)
    mean_pre_treatment = mean_pre_treatment.dropna(axis=0)
    mean_post_treatment = mean_post_treatment.dropna(axis=0)
    mean_pre_treatment[mean_pre_treatment.isnull().any(axis=1)]
    mean_post_treatment[mean_post_treatment.isnull().any(axis=1)]
    only_ctrl = set(mean_pre_treatment['cell_type'].unique()).difference(set(mean_post_treatment['cell_type'].unique()))

    mean_pre_treatment = mean_pre_treatment[~mean_pre_treatment['cell_type'].isin(only_ctrl)]
    only_pert = set(mean_post_treatment['cell_type'].unique()).difference(set(mean_pre_treatment['cell_type'].unique()))

    mean_post_treatment = mean_post_treatment[~mean_post_treatment['cell_type'].isin(only_pert)]
    all_pairs = mean_post_treatment[['cell_type','condition']].reset_index(drop=True).drop_duplicates()
    mean_ctrl_expression = mean_pre_treatment.copy()
    mean_ctrl_expression.drop(columns=['condition'], inplace=True)
    mean_ctrl_expression = pd.merge(mean_ctrl_expression.reset_index(drop=True),all_pairs, on=['cell_type'], how='left')

    mean_post_treatment['cell_type'] = mean_post_treatment['cell_type'].str.split('_').str[0]
    mean_ctrl_expression['cell_type'] = mean_ctrl_expression['cell_type'].str.split('_').str[0]
    mean_post_treatment.to_csv(os.path.join(data_dir,'observed_pseudobulk','mcfarland_mean_post_all_celllines.csv'))
    mean_ctrl_expression.to_csv(os.path.join(data_dir,'observed_pseudobulk','mcfarland_mean_pre_all_celllines.csv'))

    condition = mean_post_treatment['condition']
    mean_gene_expression = mean_post_treatment.set_index('cell_type').drop('condition', axis=1)
    mean_ctrl_expression = mean_ctrl_expression.set_index('cell_type').drop('condition', axis=1)
    mean_LFC = mean_gene_expression.subtract(mean_ctrl_expression).reset_index()
    mean_LFC['condition'] = condition.tolist()
    mean_LFC.to_csv(os.path.join(data_dir,'observed_pseudobulk','mcfarland_mean_LFC_all_celllines.csv'))


def process_sciplex_observations(adata, file_suffix):
    input_data_df = pd.DataFrame.sparse.from_spmatrix(adata.X, columns=adata.var_names, index=adata.obs_names)
    input_data_df['condition'] = adata.obs['condition']
    input_data_df['cell_type'] = adata.obs['cell_type']
    input_data_df = input_data_df.groupby(['cell_type','condition']).mean()
    input_data_df.reset_index(inplace=True)
    mean_pre_treatment = input_data_df[input_data_df['condition'] == 'ctrl']
    mean_post_treatment = input_data_df[input_data_df['condition'] != 'ctrl']
    all_pairs = mean_post_treatment[['cell_type','condition']].reset_index(drop=True).drop_duplicates()
    mean_ctrl_expression = mean_pre_treatment.copy()
    mean_ctrl_expression.drop(columns=['condition'], inplace=True)
    mean_ctrl_expression = pd.merge(mean_ctrl_expression.reset_index(drop=True),all_pairs, on=['cell_type'], how='left')

    mean_post_treatment.to_csv(os.path.join(data_dir,'observed_pseudobulk',f'sciplex_mean_post_{file_suffix}.csv'))
    mean_ctrl_expression.to_csv(os.path.join(data_dir,'observed_pseudobulk',f'sciplex_mean_pre_{file_suffix}.csv'))
    condition = mean_post_treatment['condition']
    mean_gene_expression = mean_post_treatment.set_index('cell_type').drop('condition', axis=1)
    mean_ctrl_expression = mean_pre_treatment.set_index('cell_type').drop('condition', axis=1)
    mean_LFC = mean_gene_expression.subtract(mean_ctrl_expression, level='cell_type').reset_index(drop=True)
    mean_LFC['condition'] = condition.tolist()

    mean_LFC.to_csv(os.path.join(data_dir,'observed_pseudobulk',f'sciplex_mean_LFC_{file_suffix}.csv'))


if __name__ == '__main__':
    log_script_start(__file__, logger)
    logger.info("Creating pseudobulks...")
    try:
        logger.info("Processing MCFARLAND observations...")
        process_mcfarland_observations()
        
        mcf7_adata = sc.read_h5ad(os.path.join(data_dir, 'sciplex_processed','sciplexmcf7.h5ad'))
        logger.info("Processing SCIPLEX MCF7 observations...")
        process_sciplex_observations(mcf7_adata, 'mcf7')
        a549_adata = sc.read_h5ad(os.path.join(data_dir, 'sciplex_processed','sciplexa549.h5ad'))
        logger.info("Processing SCIPLEX A549 observations...")
        process_sciplex_observations(a549_adata, 'a549')
        k562_adata = sc.read_h5ad(os.path.join(data_dir, 'sciplex_processed','sciplexk562.h5ad'))
        logger.info("Processing SCIPLEX K562 observations...")
        process_sciplex_observations(k562_adata, 'k562')
        logger.info("Pseudobulks created successfully")
    except Exception as e:
        logger.error(f"Error in create_pseudobulk: {str(e)}")
        raise
    log_script_end(__file__, logger)