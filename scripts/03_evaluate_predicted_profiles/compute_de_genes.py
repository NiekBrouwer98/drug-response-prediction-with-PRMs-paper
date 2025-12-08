import pandas as pd
import numpy as np
import scanpy as sc
import os
import sys
from pathlib import Path
import logging

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

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
results_dir = str(config.RESULTS_03_DIR)

# Ensure directories exist
ensure_directories_exist(home_dir,data_dir,results_dir)

def compute_de_genes(df, cell_line, groupby='product_name', reference='Vehicle'):
    sc.tl.rank_genes_groups(df, groupby=groupby, reference=reference, rankby_abs=True, n_genes=df.var.shape[0], use_raw=False)
    df = pd.DataFrame(df.uns['rank_genes_groups']['names']).transpose().reset_index().rename(columns={'index':'condition'})
    df['cell_type'] = cell_line
    
    return df

def main():
    for cl in ['mcf7', 'a549', 'k562']:
        df = sc.read_h5ad(os.path.join(data_dir, 'sciplex_processed', f'sciplex{cl}.h5ad'))

        de_genes = compute_de_genes(df, cl, groupby='product_name', reference='Vehicle')
        de_genes.to_csv(os.path.join(results_dir, f'sciplex{cl}_de_genes.csv'), index=False)


if __name__ == '__main__':
    log_script_start(__file__, logger)
    logger.info("Computing DE genes...")
    main()
    logger.info("DE genes computed successfully")
    log_script_end(__file__, logger)