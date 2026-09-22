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
sys.path.append(str(Path(__file__).parent.parent / "01_process_measured_profiles"))
from config import config, setup_project
from create_sciplex_splits import sciplex_condition_from_product
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

def _n_cells_table(obs, group_cols=("cell_type", "condition")):
    cols = [c for c in group_cols if c in obs.columns]
    return obs.groupby(cols, observed=True).size().rename("n_cells").reset_index()


def _atomic_to_csv(df: pd.DataFrame, path: str) -> None:
    """Write CSV via a temp file then replace, avoiding truncated network-drive writes."""
    import time

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    df.to_csv(tmp)
    last_err: Exception | None = None
    for _ in range(5):
        try:
            if target.exists():
                target.unlink()
            tmp.replace(target)
            return
        except PermissionError as exc:
            last_err = exc
            time.sleep(0.5)
    if last_err is not None:
        raise last_err
    raise RuntimeError(f'Failed to replace {target}')


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

    # Keep tissue as its own column before stripping cell_type to DepMap-style line names (required for
    # get_tissue_labels / LTO CV). Older CSVs only stored the stripped token and lost tissue entirely.
    def _split_cell_type_line_tissue(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        sp = out['cell_type'].astype(str).str.split('_', n=1, expand=True)
        out['tissue'] = sp[1] if 1 in sp.columns else pd.NA
        out['cell_type'] = sp[0]
        return out

    mean_post_treatment = _split_cell_type_line_tissue(mean_post_treatment)
    mean_ctrl_expression = _split_cell_type_line_tissue(mean_ctrl_expression)
    n_cells = _n_cells_table(all_cell_lines.obs)
    n_cells = _split_cell_type_line_tissue(n_cells).drop(columns=["tissue"], errors="ignore")
    mean_post_treatment = mean_post_treatment.merge(n_cells, on=["cell_type", "condition"], how="left")
    n_cells_ctrl = n_cells[n_cells["condition"] == "ctrl"].drop(columns=["condition"])
    mean_ctrl_expression = mean_ctrl_expression.merge(n_cells_ctrl, on=["cell_type"], how="left")
    _atomic_to_csv(mean_post_treatment, os.path.join(data_dir,'observed_pseudobulk','mcfarland_mean_post_all_celllines.csv'))
    _atomic_to_csv(mean_ctrl_expression, os.path.join(data_dir,'observed_pseudobulk','mcfarland_mean_pre_all_celllines.csv'))

    condition = mean_post_treatment['condition']
    drop_post = [c for c in ('condition', 'tissue', 'n_cells') if c in mean_post_treatment.columns]
    drop_pre = [c for c in ('condition', 'tissue', 'n_cells') if c in mean_ctrl_expression.columns]
    mean_gene_expression = mean_post_treatment.set_index('cell_type').drop(drop_post, axis=1)
    mean_ctrl_for_lfc = mean_ctrl_expression.set_index('cell_type').drop(drop_pre, axis=1)
    mean_LFC = mean_gene_expression.subtract(mean_ctrl_for_lfc).reset_index()
    mean_LFC['condition'] = condition.tolist()
    tissue_map = mean_post_treatment[['cell_type', 'tissue']].drop_duplicates()
    mean_LFC = mean_LFC.merge(tissue_map, on='cell_type', how='left')
    if 'n_cells' in mean_post_treatment.columns:
        mean_LFC['n_cells'] = mean_post_treatment['n_cells'].tolist()
    _atomic_to_csv(mean_LFC, os.path.join(data_dir,'observed_pseudobulk','mcfarland_mean_LFC_all_celllines.csv'))


def _groupby_mean_expression(adata: sc.AnnData, condition: pd.Series) -> pd.DataFrame:
    """Fast sparse-friendly mean expression per (cell_type, condition)."""
    from scipy import sparse

    cell_type = adata.obs['cell_type'].astype(str)
    keys = cell_type.to_numpy() + '||' + condition.astype(str).to_numpy()
    codes, uniques = pd.factorize(keys, sort=False)
    n_groups = len(uniques)
    n_cells = adata.n_obs
    X = adata.X
    if sparse.issparse(X):
        X = X.tocsr()
        indicator = sparse.csr_matrix(
            (np.ones(n_cells, dtype=np.float64), (codes, np.arange(n_cells))),
            shape=(n_groups, n_cells),
        )
        sums = indicator @ X
        means = sums.toarray() / np.maximum(np.bincount(codes, minlength=n_groups), 1)[:, None]
    else:
        X = np.asarray(X)
        means = np.zeros((n_groups, X.shape[1]), dtype=np.float64)
        counts = np.bincount(codes, minlength=n_groups)
        np.add.at(means, codes, X)
        means /= np.maximum(counts, 1)[:, None]

    parts = pd.Series(uniques).str.split(r'\|\|', n=1, expand=True)
    out = pd.DataFrame(means, columns=adata.var_names.astype(str))
    out.insert(0, 'cell_type', parts[0].to_numpy())
    out.insert(1, 'condition', parts[1].to_numpy())
    return out


def process_sciplex_observations(adata, file_suffix):
    condition = sciplex_condition_from_product(adata.obs)
    input_data_df = _groupby_mean_expression(adata, condition)
    mean_pre_treatment = input_data_df[input_data_df['condition'] == 'ctrl']
    mean_post_treatment = input_data_df[input_data_df['condition'] != 'ctrl']
    all_pairs = mean_post_treatment[['cell_type', 'condition']].reset_index(drop=True).drop_duplicates()
    mean_ctrl_expression = mean_pre_treatment.copy()
    mean_ctrl_expression.drop(columns=['condition'], inplace=True)
    mean_ctrl_expression = pd.merge(mean_ctrl_expression.reset_index(drop=True), all_pairs, on=['cell_type'], how='left')
    n_cells_obs = adata.obs.copy()
    n_cells_obs['condition'] = condition.to_numpy()
    n_cells = _n_cells_table(n_cells_obs)
    mean_post_treatment = mean_post_treatment.merge(n_cells, on=['cell_type', 'condition'], how='left')
    n_cells_ctrl = n_cells[n_cells['condition'] == 'ctrl'].drop(columns=['condition'])
    mean_ctrl_expression = mean_ctrl_expression.merge(n_cells_ctrl, on=['cell_type'], how='left')

    _atomic_to_csv(mean_post_treatment, os.path.join(data_dir, 'observed_pseudobulk', f'sciplex_mean_post_{file_suffix}.csv'))
    _atomic_to_csv(mean_ctrl_expression, os.path.join(data_dir, 'observed_pseudobulk', f'sciplex_mean_pre_{file_suffix}.csv'))
    condition_col = mean_post_treatment['condition']
    drop_cols = [c for c in ('condition', 'n_cells') if c in mean_post_treatment.columns]
    mean_gene_expression = mean_post_treatment.set_index('cell_type').drop(drop_cols, axis=1)
    drop_ctrl = [c for c in ('condition', 'n_cells') if c in mean_pre_treatment.columns]
    mean_ctrl_for_lfc = mean_pre_treatment.set_index('cell_type').drop(drop_ctrl, axis=1)
    mean_LFC = mean_gene_expression.subtract(mean_ctrl_for_lfc, level='cell_type').reset_index(drop=True)
    mean_LFC['condition'] = condition_col.tolist()
    if 'n_cells' in mean_post_treatment.columns:
        mean_LFC['n_cells'] = mean_post_treatment['n_cells'].tolist()

    _atomic_to_csv(mean_LFC, os.path.join(data_dir, 'observed_pseudobulk', f'sciplex_mean_LFC_{file_suffix}.csv'))
    logger.info(
        'Wrote SciPlex mean pseudobulks for %s (%d post conditions)',
        file_suffix,
        mean_post_treatment['condition'].nunique(),
    )


def process_sciplex_observations_from_full() -> None:
    """Build SciPlex mean pre/post/LFC from the full processed Srivatsan h5ad.

    Per-line ``sciplex{{a549,k562,mcf7}}.h5ad`` files are legacy Systema subsets
    (~77 drugs). The combined processed file has the full drug grid (~188).
    """
    processed_path = os.path.join(data_dir, 'sciplex_processed', 'Srivatsan_2019_raw_processed.h5ad')
    if not os.path.exists(processed_path):
        raise FileNotFoundError(
            f'Missing {processed_path}; run create_sciplex_splits.py first'
        )
    logger.info("Loading full SciPlex processed profiles from %s", processed_path)
    adata = sc.read_h5ad(processed_path)
    cell_types = adata.obs['cell_type'].astype(str).str.upper()
    for cell_line, suffix in (('A549', 'a549'), ('K562', 'k562'), ('MCF7', 'mcf7')):
        mask = cell_types.eq(cell_line).to_numpy()
        n = int(mask.sum())
        if n == 0:
            raise ValueError(f'No cells for cell_type={cell_line} in {processed_path}')
        logger.info("Processing SciPlex %s observations (%d cells)...", cell_line, n)
        process_sciplex_observations(adata[mask].copy(), suffix)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create observed mean pre/post/LFC pseudobulks')
    parser.add_argument(
        '--dataset',
        choices=['both', 'mcfarland', 'sciplex'],
        default='both',
        help='Which observed pseudobulks to rebuild (default: both).',
    )
    args = parser.parse_args()

    log_script_start(__file__, logger)
    logger.info("Creating pseudobulks (dataset=%s)...", args.dataset)
    try:
        if args.dataset in ('both', 'mcfarland'):
            logger.info("Processing MCFARLAND observations...")
            process_mcfarland_observations()
        if args.dataset in ('both', 'sciplex'):
            process_sciplex_observations_from_full()
        logger.info("Pseudobulks created successfully")
    except Exception as e:
        logger.error(f"Error in create_pseudobulk: {str(e)}")
        raise
    log_script_end(__file__, logger)