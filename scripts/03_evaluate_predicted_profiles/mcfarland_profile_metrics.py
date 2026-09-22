"""Shared MSE / Pearson / Systema metrics for McFarland pseudobulk profiles."""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse

from systema_reference import (
    attach_split_from_template,
    normalize_split_values,
    reference_vector_all_non_control,
    reference_vectors_per_cell_line_non_control,
    resolve_cell_line_column,
    resolve_split_column,
)

N_DE_GENES = 20
META_COLUMNS = frozenset(
    {
        'cell_line',
        'cell_type',
        'condition',
        'tissue',
        'fold',
        'split',
        'n_cells',
        'sens',
        'target',
        'sens_label',
        'drug',
        'dose',
        'cpa_pert',
        'dataset',
        'perturbation',
        'product_name',
        'test_condition',
        'mask',
        'gene_name',
    }
)


def normalize_sciplex_product_name(names: pd.Series) -> pd.Series:
    return names.astype(str).str.strip().str.replace(r'\s+', '', regex=True)


def load_sciplex_de_genes(
    results_dir: str,
    cell_line: str,
    resources_dir: str | None = None,
    *,
    condition_key: str = 'product_name',
) -> pd.DataFrame:
    """Load SciPlex DE genes with ``condition`` aligned to profile ``condition`` keys."""
    de_genes = pd.read_csv(os.path.join(results_dir, f'sciplex{cell_line}_de_genes.csv')).copy()
    de_genes['condition'] = normalize_sciplex_product_name(de_genes['condition'])

    if condition_key == 'gene_target':
        if resources_dir is None:
            raise ValueError('resources_dir is required when condition_key="gene_target"')
        pert_to_drug = pd.read_csv(
            os.path.join(resources_dir, 'sciplex_drug_to_perturbation.csv'),
            index_col=0,
        )
        pert_to_drug['product_name_key'] = normalize_sciplex_product_name(pert_to_drug['product_name'])
        pert_to_drug = pert_to_drug.drop_duplicates(subset='product_name_key', keep='first')
        target_by_key = pert_to_drug.set_index('product_name_key')['target']
        de_genes['condition'] = de_genes['condition'].map(target_by_key)
        de_genes = de_genes.dropna(subset=['condition'])
    elif condition_key != 'product_name':
        raise ValueError(f'Unknown condition_key: {condition_key!r}')

    de_genes['cell_type'] = cell_line.upper()
    de_genes['cell_line'] = cell_line.upper()
    return de_genes


def map_sciplex_conditions_to_gene_targets(df: pd.DataFrame, resources_dir: str) -> pd.DataFrame:
    """Map SciPlex drug ``condition`` labels to GEARS/scFoundation gene-target keys.

    Predictions from GEARS/scFoundation use targets like ``ABL1+ctrl``, while observed
    SciPlex pseudobulks use product names. Without this remapping, pair matching finds
    zero overlapping conditions and Systema outcome tables are empty.
    """
    if 'condition' not in df.columns:
        raise ValueError('Expected a condition column to map to gene targets')

    pert_to_drug = pd.read_csv(
        os.path.join(resources_dir, 'sciplex_drug_to_perturbation.csv'),
        index_col=0,
    )
    pert_to_drug['product_name_key'] = normalize_sciplex_product_name(pert_to_drug['product_name'])
    target_by_key = (
        pert_to_drug.drop_duplicates(subset='product_name_key', keep='first')
        .set_index('product_name_key')['target']
    )

    out = df.copy()
    mapped = normalize_sciplex_product_name(out['condition']).map(target_by_key)
    n_unmapped = int(mapped.isna().sum())
    if n_unmapped:
        warnings.warn(
            f'Dropping {n_unmapped} SciPlex rows without a gene-target mapping '
            'in sciplex_drug_to_perturbation.csv',
            stacklevel=2,
        )
    out = out.loc[mapped.notna()].copy()
    out['condition'] = mapped.loc[mapped.notna()].to_numpy()

    cell_col = 'cell_line' if 'cell_line' in out.columns else (
        'cell_type' if 'cell_type' in out.columns else None
    )
    if cell_col is not None:
        before = len(out)
        out = out.drop_duplicates(subset=[cell_col, 'condition'], keep='first')
        dropped = before - len(out)
        if dropped:
            warnings.warn(
                f'Dropped {dropped} duplicate SciPlex (cell, gene-target) rows '
                'after drug→target mapping',
                stacklevel=2,
            )
    return out


def normalize_mcfarland_profiles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'cell_type' in out.columns and 'cell_line' not in out.columns:
        out = out.rename(columns={'cell_type': 'cell_line'})
    if 'cell_line' in out.columns:
        out['cell_line'] = (
            out['cell_line'].astype(str).str.strip().str.split('_').str[0].str.strip().str.upper()
        )
    if 'condition' in out.columns:
        out['condition'] = out['condition'].astype(str).str.strip()
    return normalize_split_values(out)


def _gene_mean_agg(df: pd.DataFrame, gene_columns: list[str], extra_first: tuple[str, ...] = ()) -> dict[str, str]:
    agg: dict[str, str] = {g: 'mean' for g in gene_columns if g in df.columns}
    for col in extra_first:
        if col in df.columns and col not in agg:
            agg[col] = 'first'
    return agg


def collapse_profile_folds(df: pd.DataFrame, gene_columns: list[str]) -> pd.DataFrame:
    """Average fold/split/drug-specific expression to one profile per (cell_line, condition)."""
    df = normalize_mcfarland_profiles(df)
    extra = ('tissue',) if 'tissue' in df.columns else ()
    return df.groupby(['cell_line', 'condition'], as_index=False).agg(
        _gene_mean_agg(df, gene_columns, extra_first=extra)
    )


def collapse_within_split(df: pd.DataFrame, gene_columns: list[str]) -> pd.DataFrame:
    """Average duplicate drugs (or rows) within each (cell_line, condition, split)."""
    df = normalize_mcfarland_profiles(df)
    split_col = resolve_split_column(df)
    keys = ['cell_line', 'condition']
    if split_col is not None:
        keys.append(split_col)
    if not df.duplicated(subset=keys).any():
        return df
    extra = ('tissue',) if 'tissue' in df.columns else ()
    return df.groupby(keys, as_index=False).agg(_gene_mean_agg(df, gene_columns, extra_first=extra))


def compute_mcfarland_de_genes_from_lfc(
    lfc_df: pd.DataFrame,
    n_genes: int = N_DE_GENES,
) -> pd.DataFrame:
    """
    Top ``n_genes`` genes by absolute LFC per (cell_line, condition).

    Output matches SciPlex DE-gene table shape: one row per pair, gene names in columns 0..n-1.
    """
    lfc_df = normalize_mcfarland_profiles(lfc_df)
    gene_cols = [c for c in lfc_df.columns if c not in META_COLUMNS]
    rows: list[dict] = []

    for (cell_line, condition), group in lfc_df.groupby(['cell_line', 'condition'], sort=False):
        if len(group) > 1 and 'fold' in group.columns:
            expr = group[gene_cols].mean(axis=0)
        else:
            expr = group[gene_cols].iloc[0]
        top_genes = expr.abs().nlargest(n_genes).index.tolist()
        row = {'cell_line': cell_line, 'condition': condition}
        for i, gene in enumerate(top_genes):
            row[str(i)] = gene
        rows.append(row)

    return pd.DataFrame(rows)


def _get_common_gene_columns(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
) -> list[str]:
    common = [
        c
        for c in predictions.columns
        if c in observations.columns and c not in META_COLUMNS
    ]
    if not common:
        raise ValueError('No common gene columns between predictions and observations.')
    return common


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        val = pearsonr(x, y)[0]
    return 0.0 if np.isnan(val) else float(val)


def _profiles_for_matching(
    df: pd.DataFrame,
    *,
    sciplex_normalize_condition: bool = False,
) -> pd.DataFrame:
    out = normalize_mcfarland_profiles(df)
    if sciplex_normalize_condition and 'condition' in out.columns:
        out = out.copy()
        out['condition'] = normalize_sciplex_product_name(out['condition'])
    return out


def _iter_matched_pairs(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    sciplex_normalize_condition: bool = False,
) -> list[tuple[str, str, pd.Series, pd.Series]]:
    predictions = _profiles_for_matching(
        predictions, sciplex_normalize_condition=sciplex_normalize_condition
    )
    observations = _profiles_for_matching(
        observations, sciplex_normalize_condition=sciplex_normalize_condition
    )
    keys = ['cell_line', 'condition']
    pred = predictions.drop_duplicates(subset=keys, keep='first')
    obs = observations.drop_duplicates(subset=keys, keep='first')
    merged = pred[keys].merge(obs[keys], on=keys, how='inner')
    pairs: list[tuple[str, str, pd.Series, pd.Series]] = []
    for cell_line, condition in merged.itertuples(index=False, name=None):
        pred_row = pred[(pred['cell_line'] == cell_line) & (pred['condition'] == condition)].iloc[0]
        obs_row = obs[(obs['cell_line'] == cell_line) & (obs['condition'] == condition)].iloc[0]
        pairs.append((cell_line, condition, pred_row, obs_row))
    return pairs


def _iter_matched_pairs_with_fold(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
    split_col: str,
    *,
    sciplex_normalize_condition: bool = False,
) -> list[tuple[str, str, int, pd.Series, pd.Series]]:
    predictions = _profiles_for_matching(
        predictions, sciplex_normalize_condition=sciplex_normalize_condition
    )
    observations = _profiles_for_matching(
        observations, sciplex_normalize_condition=sciplex_normalize_condition
    )
    obs = observations.drop_duplicates(subset=['cell_line', 'condition'], keep='first')
    pairs: list[tuple[str, str, int, pd.Series, pd.Series]] = []
    for _, pred_row in predictions.iterrows():
        cell_line = pred_row['cell_line']
        condition = pred_row['condition']
        split_value = int(pred_row[split_col])
        obs_match = obs[(obs['cell_line'] == cell_line) & (obs['condition'] == condition)]
        if obs_match.empty:
            continue
        pairs.append((cell_line, condition, split_value, pred_row, obs_match.iloc[0]))
    return pairs


def _de_genes_for_pair(
    de_genes: pd.DataFrame,
    cell_line: str,
    condition: str,
    gene_columns: list[str],
) -> list[str]:
    subset = de_genes[
        (de_genes['cell_line'] == cell_line) & (de_genes['condition'] == condition)
    ]
    if subset.empty:
        return []
    rank_cols = [c for c in subset.columns if c not in META_COLUMNS]
    genes = subset[rank_cols].iloc[0, :N_DE_GENES].dropna().astype(str).tolist()
    return [g for g in genes if g in gene_columns]


def compute_metrics_from_means_mcfarland(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
    de_genes: pd.DataFrame,
    *,
    sciplex_normalize_condition: bool = False,
) -> tuple[dict[str, float], dict[tuple[str, str], dict[str, float]]]:
    """MSE and Pearson per (cell_line, condition), plus DE-gene subsets."""
    metrics: dict[str, list[float]] = {'mse': [], 'pearson': [], 'mse_de': [], 'pearson_de': []}
    metrics_pair: dict[tuple[str, str], dict[str, float]] = {}

    predictions = _profiles_for_matching(
        predictions, sciplex_normalize_condition=sciplex_normalize_condition
    )
    observations = _profiles_for_matching(
        observations, sciplex_normalize_condition=sciplex_normalize_condition
    )
    de_genes = _profiles_for_matching(
        de_genes, sciplex_normalize_condition=sciplex_normalize_condition
    )
    gene_columns = _get_common_gene_columns(predictions, observations)
    pairs = _iter_matched_pairs(
        predictions,
        observations,
        sciplex_normalize_condition=False,
    )

    for cell_line, condition, pred_row, obs_row in pairs:
        pred_expr = pred_row[gene_columns].to_numpy(dtype=float)
        true_expr = obs_row[gene_columns].to_numpy(dtype=float)

        pair_metrics = {
            'mse': float(mse(pred_expr, true_expr)),
            'pearson': _safe_pearson(pred_expr, true_expr),
        }
        de_subset = _de_genes_for_pair(de_genes, cell_line, condition, gene_columns)
        if de_subset:
            pair_metrics['mse_de'] = float(mse(pred_row[de_subset], obs_row[de_subset]))
            pair_metrics['pearson_de'] = _safe_pearson(
                pred_row[de_subset].to_numpy(dtype=float),
                obs_row[de_subset].to_numpy(dtype=float),
            )
        else:
            pair_metrics['mse_de'] = 0.0
            pair_metrics['pearson_de'] = 0.0

        metrics_pair[(cell_line, condition)] = pair_metrics
        for key in metrics:
            metrics[key].append(pair_metrics[key])

    metrics_mean = {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}
    return metrics_mean, metrics_pair


def compute_metrics_with_systema_mcfarland(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
    de_genes: pd.DataFrame,
    split_template: pd.DataFrame | None = None,
    *,
    reference_observations: pd.DataFrame | None = None,
    sciplex_normalize_condition: bool = False,
) -> tuple[dict[str, float], dict[tuple, dict[str, float]]]:
    """
    MSE, Pearson, and Systema Pearson per (cell_line, condition, fold).

    Systema references are one mean vector over all non-control observations in the
    dataset (``pearson_systema*``) or within each cell line (``pearson_systema_cellline*``).
    The reference does not depend on fold or condition.
    """
    metrics: dict[str, list[float]] = {
        'mse': [],
        'pearson': [],
        'pearson_systema': [],
        'pearson_systema_cellline': [],
        'mse_de': [],
        'pearson_de': [],
        'pearson_systema_de': [],
        'pearson_systema_cellline_de': [],
    }
    metrics_pair: dict[tuple, dict[str, float]] = {}

    predictions = _profiles_for_matching(
        predictions, sciplex_normalize_condition=sciplex_normalize_condition
    )
    observations = _profiles_for_matching(
        observations, sciplex_normalize_condition=sciplex_normalize_condition
    )
    de_genes = _profiles_for_matching(
        de_genes, sciplex_normalize_condition=sciplex_normalize_condition
    )
    if split_template is not None:
        split_template = _profiles_for_matching(
            split_template, sciplex_normalize_condition=sciplex_normalize_condition
        )
        predictions = attach_split_from_template(predictions, split_template)

    split_col = resolve_split_column(predictions)
    if split_col is None:
        raise ValueError(
            'McFarland Systema evaluation requires a fold/split column on predictions '
            '(or a split_template such as CPA profiles with fold 0–4).'
        )

    ref_source = reference_observations if reference_observations is not None else observations
    ref_source = _profiles_for_matching(
        ref_source, sciplex_normalize_condition=sciplex_normalize_condition
    )

    gene_columns = _get_common_gene_columns(predictions, observations)
    cell_line_col = resolve_cell_line_column(ref_source)
    reference_vector = reference_vector_all_non_control(
        ref_source, gene_columns, cell_line_col=cell_line_col
    )
    reference_by_cellline = reference_vectors_per_cell_line_non_control(
        ref_source, gene_columns, cell_line_col=cell_line_col
    )
    ref_expr = reference_vector.to_numpy(dtype=float)
    pairs = _iter_matched_pairs_with_fold(
        predictions,
        observations,
        split_col,
        sciplex_normalize_condition=False,
    )

    for cell_line, condition, split_value, pred_row, obs_row in pairs:
        ref_expr_cl = (
            reference_by_cellline[str(cell_line)].to_numpy(dtype=float)
            if str(cell_line) in reference_by_cellline
            else None
        )
        pred_expr = pred_row[gene_columns].to_numpy(dtype=float)
        true_expr = obs_row[gene_columns].to_numpy(dtype=float)
        centered_pred = pred_expr - ref_expr
        centered_true = true_expr - ref_expr

        pair_metrics = {
            'mse': float(mse(pred_expr, true_expr)),
            'pearson': _safe_pearson(pred_expr, true_expr),
            'pearson_systema': _safe_pearson(centered_pred, centered_true),
        }
        if ref_expr_cl is not None:
            pair_metrics['pearson_systema_cellline'] = _safe_pearson(
                pred_expr - ref_expr_cl, true_expr - ref_expr_cl
            )
        else:
            pair_metrics['pearson_systema_cellline'] = 0.0

        de_subset = _de_genes_for_pair(de_genes, cell_line, condition, gene_columns)
        if de_subset:
            pred_de = pred_row[de_subset].to_numpy(dtype=float)
            true_de = obs_row[de_subset].to_numpy(dtype=float)
            ref_de = reference_vector[de_subset].to_numpy(dtype=float)
            pair_metrics['mse_de'] = float(mse(pred_de, true_de))
            pair_metrics['pearson_de'] = _safe_pearson(pred_de, true_de)
            pair_metrics['pearson_systema_de'] = _safe_pearson(
                pred_de - ref_de, true_de - ref_de
            )
            if ref_expr_cl is not None:
                ref_de_cl = reference_by_cellline[str(cell_line)][de_subset].to_numpy(dtype=float)
                pair_metrics['pearson_systema_cellline_de'] = _safe_pearson(
                    pred_de - ref_de_cl, true_de - ref_de_cl
                )
            else:
                pair_metrics['pearson_systema_cellline_de'] = 0.0
        else:
            pair_metrics['mse_de'] = 0.0
            pair_metrics['pearson_de'] = 0.0
            pair_metrics['pearson_systema_de'] = 0.0
            pair_metrics['pearson_systema_cellline_de'] = 0.0

        metrics_pair[(cell_line, condition, split_value)] = pair_metrics
        for key in metrics:
            metrics[key].append(pair_metrics[key])

    metrics_mean = {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}
    return metrics_mean, metrics_pair


def pair_metrics_to_long_df(
    pair_metrics: dict[tuple, dict[str, float]],
    model_name: str,
) -> pd.DataFrame:
    records = []
    for key, metric_dict in pair_metrics.items():
        if len(key) == 2:
            cell_line, condition = key
            fold = None
        elif len(key) == 3:
            cell_line, condition, fold = key
        else:
            raise ValueError(f'Unexpected pair_metrics key: {key!r}')
        for metric, value in metric_dict.items():
            record = {
                'cell_line': cell_line,
                'condition': condition,
                'metric': metric,
                'value': value,
                'model': model_name,
            }
            if fold is not None:
                record['fold'] = fold
            records.append(record)
    return pd.DataFrame(records)


def load_mcfarland_observed_post(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, 'observed_pseudobulk', 'mcfarland_mean_post_all_celllines.csv')
    return normalize_mcfarland_profiles(pd.read_csv(path, index_col=0))


def load_mcfarland_gears_post(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, 'GEARS_predictions', 'mcfarland_mean_post_all.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'McFarland GEARS post file not found: {path}')
    df = pd.read_csv(path)
    if 'perturbation' in df.columns:
        df = df.rename(columns={'perturbation': 'condition'})
    return normalize_mcfarland_profiles(df)


def load_mcfarland_scfoundation_post(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, 'scfoundation_predictions', 'mcfarland_mean_post_all.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'McFarland scFoundation post file not found: {path}')
    df = pd.read_csv(path)
    if 'perturbation' in df.columns:
        df = df.rename(columns={'perturbation': 'condition'})
    if 'test_condition' in df.columns:
        df = df.rename(columns={'test_condition': 'condition'})
    return normalize_mcfarland_profiles(df)


def load_mcfarland_observed_lfc(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, 'observed_pseudobulk', 'mcfarland_mean_LFC_all_celllines.csv')
    return normalize_mcfarland_profiles(pd.read_csv(path, index_col=0))


def prepare_predictions_for_evaluation(predictions: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    predictions = normalize_mcfarland_profiles(predictions)
    gene_columns = _get_common_gene_columns(
        predictions, normalize_mcfarland_profiles(observations)
    )
    return collapse_profile_folds(predictions, gene_columns)


def prepare_predictions_for_systema(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
    split_template: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Keep fold-specific rows; attach fold labels from ``split_template`` when missing."""
    predictions = normalize_mcfarland_profiles(predictions)
    if split_template is not None:
        predictions = attach_split_from_template(predictions, split_template)
    gene_columns = _get_common_gene_columns(
        predictions, normalize_mcfarland_profiles(observations)
    )
    return collapse_within_split(predictions, gene_columns)
