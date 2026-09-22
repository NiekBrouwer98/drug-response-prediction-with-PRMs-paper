import sys
import os
from pathlib import Path
import pandas as pd
import logging
# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "03_evaluate_predicted_profiles"))
from config import config, setup_project
from prediction_io import load_sciplex_pair_grid
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
ensure_directories_exist(home_dir,data_dir)

def _save_baseline_profiles(
    observed_pre_treatment: pd.DataFrame,
    observed_post_treatment: pd.DataFrame,
    observed_lfc_treatment: pd.DataFrame,
    no_effect_path: str,
    average_post_path: str,
    average_lfc_path: str,
    *,
    all_pairs: pd.DataFrame | None = None,
) -> None:
    """Create no-effect (pre) and average-effect (mean post/LFC) baseline profiles."""
    pair_cols = ['cell_type', 'condition']
    if all_pairs is None:
        if 'tissue' in observed_post_treatment.columns:
            pair_cols.append('tissue')
        all_pairs = observed_post_treatment[pair_cols].drop_duplicates().reset_index(drop=True)
    else:
        all_pairs = all_pairs[pair_cols].drop_duplicates().reset_index(drop=True)

    meta_cols = ['condition']
    if 'tissue' in observed_pre_treatment.columns:
        meta_cols.append('tissue')

    mean_observed_pre_treatment = (
        observed_pre_treatment.drop(columns=meta_cols, errors='ignore')
        .groupby(['cell_type'])
        .mean()
        .reset_index()
    )
    mean_observed_post_treatment = (
        observed_post_treatment.drop(columns=meta_cols, errors='ignore')
        .groupby(['cell_type'])
        .mean()
        .reset_index()
    )
    mean_observed_lfc_treatment = (
        observed_lfc_treatment.drop(columns=meta_cols, errors='ignore')
        .groupby(['cell_type'])
        .mean()
        .reset_index()
    )

    mean_observed_pre_treatment = pd.merge(mean_observed_pre_treatment, all_pairs, on=['cell_type'], how='left')
    mean_observed_post_treatment = pd.merge(mean_observed_post_treatment, all_pairs, on=['cell_type'], how='left')
    mean_observed_lfc_treatment = pd.merge(mean_observed_lfc_treatment, all_pairs, on=['cell_type'], how='left')

    # No effect predictions are the pre-treatment profiles (same basal X as Measured Pre).
    # Prefer exact per-pair pre rows when the grid matches; otherwise tile cell-type means.
    pre_pairs = observed_pre_treatment[pair_cols].drop_duplicates()
    if (
        all_pairs.merge(pre_pairs, on=pair_cols, how='left', indicator=True)['_merge']
        .eq('both')
        .all()
        and len(observed_pre_treatment) >= len(all_pairs)
    ):
        no_effect = (
            observed_pre_treatment.merge(all_pairs, on=pair_cols, how='right')
            .drop_duplicates(subset=pair_cols)
        )
    else:
        no_effect = mean_observed_pre_treatment
    no_effect.to_csv(no_effect_path, index=False)

    # Average effect predictions are post-treatment profiles
    mean_observed_post_treatment.to_csv(average_post_path, index=False)
    mean_observed_lfc_treatment.to_csv(average_lfc_path, index=False)


def create_sciplex_baseline_predictions() -> None:
    all_pairs_by_line = {
        cell_line: load_sciplex_pair_grid(resources_dir)
        .loc[
            lambda df: df['cell_type'].astype(str).str.upper() == cell_line.upper(),
            ['cell_type', 'condition'],
        ]
        .drop_duplicates()
        .reset_index(drop=True)
        for cell_line in ['mcf7', 'k562', 'a549']
    }
    for cell_line in ['mcf7', 'k562', 'a549']:
        observed_pre_treatment = pd.read_csv(
            os.path.join(data_dir, 'observed_pseudobulk', f'sciplex_mean_pre_{cell_line}.csv'),
            index_col=0,
        )
        observed_pre_treatment['cell_type'] = cell_line.upper()

        observed_post_treatment = pd.read_csv(
            os.path.join(data_dir, 'observed_pseudobulk', f'sciplex_mean_post_{cell_line}.csv'),
            index_col=0,
        )
        observed_post_treatment['cell_type'] = cell_line.upper()

        observed_lfc_treatment = pd.read_csv(
            os.path.join(data_dir, 'observed_pseudobulk', f'sciplex_mean_LFC_{cell_line}.csv'),
            index_col=0,
        )
        observed_lfc_treatment['cell_type'] = cell_line.upper()

        all_pairs = all_pairs_by_line[cell_line]
        logger.info(
            "SciPlex %s baseline grid: %d (cell_type, condition) pairs",
            cell_line,
            len(all_pairs),
        )

        _save_baseline_profiles(
            observed_pre_treatment,
            observed_post_treatment,
            observed_lfc_treatment,
            os.path.join(data_dir, 'no_effect_predictions', f'sciplex_mean_post_{cell_line}.csv'),
            os.path.join(data_dir, 'average_effect_predictions', f'sciplex_mean_post_{cell_line}.csv'),
            os.path.join(data_dir, 'average_effect_predictions', f'sciplex_mean_LFC_{cell_line}.csv'),
            all_pairs=all_pairs,
        )
        logger.info(f"Baseline predictions created for sciplex {cell_line}")


def create_mcfarland_baseline_predictions() -> None:
    suffix = 'all_celllines'
    observed_pre_treatment = pd.read_csv(
        os.path.join(data_dir, 'observed_pseudobulk', f'mcfarland_mean_pre_{suffix}.csv'),
        index_col=0,
    )
    observed_post_treatment = pd.read_csv(
        os.path.join(data_dir, 'observed_pseudobulk', f'mcfarland_mean_post_{suffix}.csv'),
        index_col=0,
    )
    observed_lfc_treatment = pd.read_csv(
        os.path.join(data_dir, 'observed_pseudobulk', f'mcfarland_mean_LFC_{suffix}.csv'),
        index_col=0,
    )

    _save_baseline_profiles(
        observed_pre_treatment,
        observed_post_treatment,
        observed_lfc_treatment,
        os.path.join(data_dir, 'no_effect_predictions', f'mcfarland_mean_post_{suffix}.csv'),
        os.path.join(data_dir, 'average_effect_predictions', f'mcfarland_mean_post_{suffix}.csv'),
        os.path.join(data_dir, 'average_effect_predictions', f'mcfarland_mean_LFC_{suffix}.csv'),
    )
    logger.info("Baseline predictions created for mcfarland")


def create_baseline_predictions() -> None:
    create_sciplex_baseline_predictions()
    create_mcfarland_baseline_predictions()

if __name__ == '__main__':
    log_script_start(__file__, logger)
    required_pseudobulks = [
        *(f'sciplex_mean_post_{cell_line}.csv' for cell_line in ['mcf7', 'a549', 'k562']),
        'mcfarland_mean_post_all_celllines.csv',
    ]
    try:
        for filename in required_pseudobulks:
            _ = pd.read_csv(os.path.join(data_dir, 'observed_pseudobulk', filename), index_col=0)
    except FileNotFoundError:
        logger.error("First run create_pseudobulk.py to create the pseudobulks.")
        raise
    logger.info("Creating baseline predictions...")
    try:
        create_baseline_predictions()
    except Exception as e:
        logger.error(f"Error in create_baseline_predictions: {str(e)}")
        raise
    logger.info("Baseline predictions created successfully")
    log_script_end(__file__, logger)

    