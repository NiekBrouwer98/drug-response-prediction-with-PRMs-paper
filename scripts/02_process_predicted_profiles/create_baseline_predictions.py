import sys
import os
from pathlib import Path
import pandas as pd
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

# Ensure directories exist
ensure_directories_exist(home_dir,data_dir)

def create_baseline_predictions():
    for cell_line in ['mcf7','k562','a549']:
        observed_pre_treatment = pd.read_csv(os.path.join(data_dir, 'observed_pseudobulk', f'sciplex_mean_pre_{cell_line}.csv'), index_col=0)
        observed_pre_treatment['cell_type'] = cell_line.upper()
        mean_observed_pre_treatment = observed_pre_treatment.drop(columns=['condition']).groupby(['cell_type']).mean().reset_index()

        observed_post_treatment = pd.read_csv(os.path.join(data_dir, 'observed_pseudobulk', f'sciplex_mean_post_{cell_line}.csv'), index_col=0)
        observed_post_treatment['cell_type'] = cell_line.upper()
        mean_observed_post_treatment = observed_post_treatment.drop(columns=['condition']).groupby(['cell_type']).mean().reset_index()

        observed_lfc_treatment = pd.read_csv(os.path.join(data_dir, 'observed_pseudobulk', f'sciplex_mean_LFC_{cell_line}.csv'), index_col=0)
        observed_lfc_treatment['cell_type'] = cell_line.upper()
        mean_observed_lfc_treatment = observed_lfc_treatment.drop(columns=['condition']).groupby(['cell_type']).mean().reset_index()

        all_pairs = observed_post_treatment[['cell_type','condition']].reset_index(drop=True).drop_duplicates()

        mean_observed_pre_treatment = pd.merge(mean_observed_pre_treatment,all_pairs, on=['cell_type'], how='left')
        mean_observed_post_treatment = pd.merge(mean_observed_post_treatment,all_pairs, on=['cell_type'], how='left')
        mean_observed_lfc_treatment = pd.merge(mean_observed_lfc_treatment,all_pairs, on=['cell_type'], how='left')

        # No effect predictions are pre-treatment profiles
        mean_observed_pre_treatment.to_csv(os.path.join(data_dir, 'no_effect_predictions', f'sciplex_mean_post_{cell_line}.csv'), index=False)

        # Average effect predictions are post-treatment profiles
        mean_observed_post_treatment.to_csv(os.path.join(data_dir, 'average_effect_predictions', f'sciplex_mean_post_{cell_line}.csv'), index=False)
        mean_observed_lfc_treatment.to_csv(os.path.join(data_dir, 'average_effect_predictions', f'sciplex_mean_LFC_{cell_line}.csv'), index=False)
        logger.info(f"Baseline predictions created for {cell_line}")

if __name__ == '__main__':
    log_script_start(__file__, logger)
    try:
        for cell_line in ['mcf7','a549','k562']:
            _ = pd.read_csv(os.path.join(data_dir, 'observed_pseudobulk', f'sciplex_mean_post_{cell_line}.csv'), index_col=0)
    except FileNotFoundError:
        logger.error(f"First run create_pseudobulk.py to create the pseudobulks.")
        raise
    logger.info("Creating baseline predictions...")
    try:
        create_baseline_predictions()
    except Exception as e:
        logger.error(f"Error in create_baseline_predictions: {str(e)}")
        raise
    logger.info("Baseline predictions created successfully")
    log_script_end(__file__, logger)

    