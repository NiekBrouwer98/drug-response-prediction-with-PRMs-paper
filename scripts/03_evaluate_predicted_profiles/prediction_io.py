"""Load CPA, chemCPA, and PRnet predicted post / LFC profiles."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from mcfarland_profile_metrics import normalize_mcfarland_profiles, normalize_sciplex_product_name

SCIPLEX_CELL_LINES = ('mcf7', 'a549', 'k562')


def _first_existing(paths: list[str]) -> str:
    for path in paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError('None of these prediction files exist:\n' + '\n'.join(paths))


def load_profile_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return normalize_mcfarland_profiles(df)


def load_cpa_post(data_dir: str, dataset: str) -> pd.DataFrame:
    """Combined CPA post files (new) with fallback to archived per-file names."""
    pred_dir = os.path.join(data_dir, 'CPA_predictions')
    if dataset == 'sciplex':
        path = _first_existing(
            [
                os.path.join(pred_dir, 'sciplex_mean_post.csv'),
                os.path.join(pred_dir, 'archived', 'sciplex_mean_post_a549.csv'),
            ]
        )
        if os.path.basename(path) == 'sciplex_mean_post.csv':
            return load_profile_csv(path)
        frames = []
        for cell_line in SCIPLEX_CELL_LINES:
            frames.append(
                load_profile_csv(os.path.join(os.path.dirname(path), f'sciplex_mean_post_{cell_line}.csv'))
            )
        return pd.concat(frames, ignore_index=True)

    if dataset != 'mcfarland':
        raise ValueError(f'Unknown dataset: {dataset}')
    path = _first_existing(
        [
            os.path.join(pred_dir, 'mcfarland_mean_post.csv'),
            os.path.join(pred_dir, 'mcfarland_mean_post_all.csv'),
            os.path.join(pred_dir, 'archived', 'mcfarland_mean_post_all.csv'),
        ]
    )
    return load_profile_csv(path)


def load_chemcpa_post(data_dir: str, dataset: str) -> pd.DataFrame:
    pred_dir = os.path.join(data_dir, 'chemCPA_predictions')
    filename = {
        'sciplex': 'sciplex_cv_pred_post.csv',
        'mcfarland': 'mcfarland_cv_pred_post.csv',
    }[dataset]
    path = os.path.join(pred_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f'chemCPA post file not found: {path}')
    return load_profile_csv(path)


def load_prnet_layer(path: str, layer: str) -> pd.DataFrame:
    import scanpy as sc

    adata = sc.read_h5ad(path)
    if layer not in adata.layers:
        raise KeyError(f'PRnet file {path} has no layer {layer!r}; found {list(adata.layers)}')
    if 'gene_name' in adata.var.columns:
        gene_names = adata.var['gene_name'].astype(str).tolist()
    else:
        gene_names = adata.var_names.astype(str).tolist()
    expr = pd.DataFrame(
        np.asarray(adata.layers[layer]),
        columns=gene_names,
    )
    obs = adata.obs.reset_index(drop=True)
    del adata
    df = pd.concat([obs, expr], axis=1)
    return normalize_mcfarland_profiles(df)


def load_prnet_post(data_dir: str, dataset: str) -> pd.DataFrame:
    pred_dir = os.path.join(data_dir, 'PRnet_predictions')
    filename = {
        'sciplex': 'sciplex_all_splits_predicted_pseudobulk.h5ad',
        'mcfarland': 'mcfarland_all_splits_predicted_pseudobulk.h5ad',
    }[dataset]
    path = os.path.join(pred_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f'PRnet file not found: {path}')
    return load_prnet_layer(path, layer='predicted_mean')


def filter_sciplex_cell_line(df: pd.DataFrame, cell_line: str) -> pd.DataFrame:
    col = 'cell_line' if 'cell_line' in df.columns else 'cell_type'
    return df[df[col].astype(str).str.upper() == cell_line.upper()].copy()


def align_sciplex_profile_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize SciPlex drug labels for joins (strip whitespace, collapse spaces)."""
    out = normalize_mcfarland_profiles(df)
    if 'condition' in out.columns:
        out = out.copy()
        out['condition'] = normalize_sciplex_product_name(out['condition'])
    return out


def load_sciplex_pair_grid(resources_dir: str) -> pd.DataFrame:
    """All 188 × 3 SciPlex (cell_line, drug) pairs from sensitivity metadata."""
    path = os.path.join(resources_dir, 'sciplex_sensitivity_info.csv')
    sens = pd.read_csv(path)
    out = sens[['cell_line', 'product_name']].drop_duplicates().copy()
    out['cell_line'] = out['cell_line'].astype(str).str.strip().str.upper()
    out = out.rename(columns={'cell_line': 'cell_type', 'product_name': 'condition'})
    return out


def load_sciplex_observed_post_all(data_dir: str) -> pd.DataFrame:
    """Concat observed post pseudobulks for all SciPlex cell lines."""
    frames = [load_sciplex_observed_post(data_dir, cell_line) for cell_line in SCIPLEX_CELL_LINES]
    return pd.concat(frames, ignore_index=True)


def load_sciplex_observed_post(data_dir: str, cell_line: str) -> pd.DataFrame:
    path = os.path.join(data_dir, 'observed_pseudobulk', f'sciplex_mean_post_{cell_line}.csv')
    return normalize_mcfarland_profiles(pd.read_csv(path, index_col=0))


def drop_control_rows(df: pd.DataFrame, control: str = 'ctrl') -> pd.DataFrame:
    if 'condition' not in df.columns:
        return df
    cond = df['condition'].astype(str)
    return df[~cond.str.lower().isin({control, 'dmso', 'vehicle'})].copy()
