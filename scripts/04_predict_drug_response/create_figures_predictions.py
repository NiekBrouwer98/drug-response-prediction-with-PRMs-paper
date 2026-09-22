"""Drug-response figures from predicted-profile CV (existing boxplot formatting).

Produces SciPlex | McFarland figure sets from split-CV results:
- CPA, chemCPA, PRnet, average effect, no effect
- GEARS, scFoundation, average effect, no effect
- ± SMILES variants when ``*_split_cv_smiles_predictions.csv`` is present

Falls back to legacy GEARS/scFoundation CSVs only when split-CV lacks those models.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from statsmodels.stats.multitest import multipletests

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent))
from config import config, setup_project
from utils import ensure_directories_exist, setup_logging_for_script
import evaluate_predictions as ev

setup_project()
logger = setup_logging_for_script(__file__)

results_dir = str(config.RESULTS_04_DIR)
figures_dir = str(config.FIGURES_04_DIR)
ensure_directories_exist(results_dir, figures_dir)

pre_post_lfc_palette = ['#808080', '#FF0000', '#8B0000']
MEASURED_MODEL_ORDER = ['Pre', 'Pre+SMILES', 'Post', 'Post+SMILES', 'LFC', 'LFC+SMILES']
# SMILES modalities use the same Pre / Post / LFC colors.
MEASURED_MODEL_COLORS = {
    'Pre': '#808080',
    'Pre+SMILES': '#808080',
    'Post': '#FF0000',
    'Post+SMILES': '#FF0000',
    'LFC': '#8B0000',
    'LFC+SMILES': '#8B0000',
}
PREDICTED_MODEL_ORDER = ['Post', 'Post+SMILES', 'LFC', 'LFC+SMILES']

NEW_MODEL_CATEGORIES = ['CPA', 'chemCPA', 'PRnet', 'Average effect', 'No effect']
GEARS_MODEL_CATEGORIES = ['GEARS', 'scFoundation', 'Average effect', 'No effect']

STRATIFIED_FIGURE_SPECS = (
    ('per_drug', 'pearson', r'Within-drug Pearson r  $\rightarrow$'),
    ('per_drug', 'spearman', r'Within-drug Spearman $\rho$  $\rightarrow$'),
    ('per_drug', 'rmse', r'Within-drug RMSE $\leftarrow$'),
    ('per_drug', 'auroc', r'Within-drug ROC AUC  $\rightarrow$'),
    ('per_cell_line', 'pearson', r'Within-context Pearson r  $\rightarrow$'),
    ('per_cell_line', 'spearman', r'Within-context Spearman $\rho$  $\rightarrow$'),
    ('per_cell_line', 'rmse', r'Within-context RMSE $\leftarrow$'),
    ('per_cell_line', 'auroc', r'Within-context ROC AUC  $\rightarrow$'),
)

LABEL_MAP = {
    'post_treatment': 'Post',
    'pre_treatment': 'Pre',
    'pre_treatment_smiles': 'Pre+SMILES',
    'post_treatment_smiles': 'Post+SMILES',
    'LFC_smiles': 'LFC+SMILES',
    'observed': 'Measured',
    'CPA_predicted': 'CPA',
    'chemCPA_predicted': 'chemCPA',
    'PRnet_predicted': 'PRnet',
    'GEARS_predicted': 'GEARS',
    'scFoundation_predicted': 'scFoundation',
    'GEARS_noreg': 'GEARS\nOptimized',
    'no_effect': 'No effect',
    'average_effect': 'Average effect',
}


def _rename_labels(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(LABEL_MAP)


def compute_pearson(predictions: pd.DataFrame, groupby: list[str] | None = None) -> pd.DataFrame:
    if groupby is None:
        groupby = ['model', 'test_set', 'train_set', 'split']

    def _corr(group: pd.DataFrame) -> float:
        if group['true'].nunique() < 2 or group['pred'].nunique() < 2:
            return 0.0
        value = pearsonr(group['true'], group['pred'])[0]
        return 0.0 if np.isnan(value) else float(value)

    def _pval(group: pd.DataFrame) -> float:
        if group['true'].nunique() < 2 or group['pred'].nunique() < 2:
            return 1.0
        value = pearsonr(group['true'], group['pred'])[1]
        return 1.0 if np.isnan(value) else float(value)

    pearson_results = (
        predictions.groupby(groupby, group_keys=False)
        .apply(_corr, include_groups=False)
        .rename('pearson')
        .reset_index()
    )
    p_values = (
        predictions.groupby(groupby, group_keys=False)
        .apply(_pval, include_groups=False)
        .rename('p-value')
        .reset_index()
    )
    pearson_results = pearson_results.merge(p_values, on=groupby)
    pearson_results = _rename_labels(pearson_results)
    pearson_results['adjusted_p'] = multipletests(
        pearson_results['p-value'], alpha=0.05, method='fdr_bh'
    )[1]
    pearson_results['sig'] = np.where(pearson_results['adjusted_p'] < 0.05, '*', '')
    return pearson_results


def compute_mse(predictions: pd.DataFrame, groupby: list[str] | None = None) -> pd.DataFrame:
    if groupby is None:
        groupby = ['model', 'test_set', 'train_set', 'split']
    mse_results = (
        predictions.groupby(groupby, group_keys=False)
        .apply(lambda g: mean_squared_error(g['true'], g['pred']), include_groups=False)
        .rename('mse')
        .reset_index()
    )
    return _rename_labels(mse_results)


def _load_split_cv_predictions(dataset: str, suffix: str = '') -> pd.DataFrame | None:
    """Load split-CV predictions; optional ``suffix`` e.g. ``_smiles`` or ``smiles``.

    Preserves ``train_set`` when present (needed for train-Measured / test-predicted
    transfer CSVs where ``profile_source`` is a PRM but training used observed profiles).
    """
    if suffix and not suffix.startswith('_'):
        suffix = f'_{suffix}'
    path = os.path.join(results_dir, f'{dataset}_split_cv{suffix}_predictions.csv')
    if not os.path.exists(path):
        logger.warning('Missing split-CV predictions: %s', path)
        return None
    df = pd.read_csv(path, index_col=0)
    if 'train_set' in df.columns:
        df['train_set'] = (
            df['train_set']
            .astype(str)
            .replace(
                {
                    'observed': 'Measured',
                    'predicted': 'Predicted',
                    'Observed': 'Measured',
                    'Predicted': 'Predicted',
                    'Measured': 'Measured',
                }
            )
        )
    else:
        df['train_set'] = np.where(
            df['profile_source'].astype(str).eq('observed'), 'Measured', 'Predicted'
        )
    df = df.rename(columns={'profile_source': 'test_set'})
    return df


def attach_split_cv_measured(
    predictions: pd.DataFrame | None,
    dataset: str,
    *,
    model_keys: set[str] | frozenset[str] | None = None,
) -> pd.DataFrame:
    """Append Measured (observed) rows from the default ``*_split_cv_predictions.csv``.

    Fallback when a suffix job (e.g. ``_smiles``, transfer) was run without
    ``--include-observed``. Prefer in-file Measured from jobs that include it.
    """
    pred = (
        predictions.copy()
        if predictions is not None and len(predictions)
        else pd.DataFrame()
    )
    base = _load_split_cv_predictions(dataset, suffix='')
    if base is None or base.empty:
        return pred

    measured = base.loc[base['test_set'].astype(str).eq('observed')].copy()
    if measured.empty:
        # Already LABEL_MAP'd in some callers
        measured = base.loc[base['test_set'].astype(str).eq('Measured')].copy()
    if measured.empty:
        return pred

    if model_keys is not None:
        measured = measured.loc[measured['model'].astype(str).isin(model_keys)].copy()
    if measured.empty:
        return pred

    # Drop any existing Measured rows in ``pred`` before appending split-CV observed.
    if len(pred) and 'test_set' in pred.columns:
        pred = pred.loc[
            ~pred['test_set'].astype(str).isin(['observed', 'Measured'])
        ].copy()

    cols = list(dict.fromkeys([*pred.columns.tolist(), *measured.columns.tolist()]))
    out = pd.concat(
        [pred.reindex(columns=cols), measured.reindex(columns=cols)],
        ignore_index=True,
    )
    logger.info(
        'attach_split_cv_measured [%s]: +%d Measured rows (models=%s)',
        dataset,
        len(measured),
        sorted(measured['model'].astype(str).unique()),
    )
    return out


def predictions_have_measured(predictions: pd.DataFrame | None) -> bool:
    """True if prediction frame already has Measured / observed test-set rows."""
    if predictions is None or len(predictions) == 0 or 'test_set' not in predictions.columns:
        return False
    return predictions['test_set'].astype(str).isin(['observed', 'Measured']).any()


def _load_legacy_sciplex_predictions() -> pd.DataFrame | None:
    observed_path = os.path.join(results_dir, 'sciplex_regression_predictions_CV_twostep_AUC.csv')
    self_path = os.path.join(
        results_dir, 'sciplex_regression_predictions_selftrained_CV_twostep_AUC.csv'
    )
    frames = []
    if os.path.exists(observed_path):
        obs = pd.read_csv(observed_path, index_col=0)
        obs['trained_on_file'] = 'Measured'
        frames.append(ev.attach_group_columns(obs))
    if os.path.exists(self_path):
        pred = pd.read_csv(self_path, index_col=0)
        pred['trained_on_file'] = 'Predicted'
        frames.append(ev.attach_group_columns(pred))
    if not frames:
        logger.warning('Missing legacy SciPlex prediction CSVs for GEARS/scFoundation figures')
        return None
    return pd.concat(frames, ignore_index=True)


def _load_legacy_mcfarland_predictions() -> pd.DataFrame | None:
    path = os.path.join(results_dir, 'mcfarland_CPA_fold_cv_predictions.csv')
    if not os.path.exists(path):
        logger.warning('Missing legacy McFarland fold-CV predictions: %s', path)
        return None
    df = pd.read_csv(path, index_col=0)
    df['train_set'] = np.where(df['profile_source'] == 'observed', 'Measured', 'Predicted')
    df = df.rename(columns={'profile_source': 'test_set'})
    return df


def _metrics_from_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pearson_results = compute_pearson(predictions)
    mse_results = compute_mse(predictions)
    if 'trained_on_file' in predictions.columns:
        trained = (
            predictions[['model', 'test_set', 'train_set', 'split', 'trained_on_file']]
            .drop_duplicates()
            .replace(LABEL_MAP)
        )
        pearson_results = pearson_results.merge(
            trained, on=['model', 'test_set', 'train_set', 'split'], how='left'
        )
        mse_results = mse_results.merge(
            trained, on=['model', 'test_set', 'train_set', 'split'], how='left'
        )
        pearson_results['trained_on'] = pearson_results['trained_on_file']
        mse_results['trained_on'] = mse_results['trained_on_file']
    else:
        pearson_results['trained_on'] = np.where(
            pearson_results['test_set'] == 'Measured', 'Measured', 'Predicted'
        )
        mse_results['trained_on'] = np.where(
            mse_results['test_set'] == 'Measured', 'Measured', 'Predicted'
        )
    return pearson_results, mse_results


def _baseline_means(results: pd.DataFrame, metric: str) -> tuple[float, float, float]:
    mask = (
        (results['trained_on'] == 'Measured')
        & (results['test_set'] == 'Measured')
        & (results['model'].isin(MEASURED_MODEL_ORDER))
    )
    subset = results.loc[mask]
    return (
        subset.loc[subset['model'] == 'Pre', metric].mean(),
        subset.loc[subset['model'] == 'Post', metric].mean(),
        subset.loc[subset['model'] == 'LFC', metric].mean(),
    )


def _prepare_box_frames(pearson_results, predicted_categories=None):
    if predicted_categories is None:
        predicted_categories = NEW_MODEL_CATEGORIES
    box_data = pearson_results[~pearson_results['test_set'].isin(ev.END_TO_END_TEST_SETS)].copy()
    box_data = box_data[
        ~(
            ((box_data['test_set'] == 'No effect') | (box_data['train_set'] == 'No effect'))
            & (box_data['model'] == 'LFC')
        )
    ].copy()

    box0_data = box_data[
        (box_data['trained_on'] == 'Measured')
        & (box_data['test_set'].isin(['Measured']))
        & (box_data['model'].isin(MEASURED_MODEL_ORDER))
    ].copy()
    box2_data = box_data[
        (box_data['trained_on'] == 'Predicted') & (box_data['test_set'].isin(predicted_categories))
    ].copy()
    box2_data = box2_data[box2_data['model'].astype(str).isin(PREDICTED_MODEL_ORDER)].copy()

    box0_data['test_set'] = pd.Categorical(
        box0_data['test_set'], categories=['Measured'], ordered=True
    )
    box0_data['model'] = pd.Categorical(
        box0_data['model'], categories=MEASURED_MODEL_ORDER, ordered=True
    )
    box2_data['test_set'] = pd.Categorical(
        box2_data['test_set'],
        categories=list(predicted_categories),
        ordered=True,
    )
    box2_data['model'] = pd.Categorical(
        box2_data['model'],
        categories=[m for m in PREDICTED_MODEL_ORDER if m in set(box2_data['model'].astype(str))] or PREDICTED_MODEL_ORDER,
        ordered=True
    )
    return box0_data, box2_data


def _legend_display_label(model: str) -> str:
    """Strip ``+SMILES`` so legend shows Pre / Post / LFC only."""
    text = str(model)
    if text.endswith('+SMILES'):
        return text[: -len('+SMILES')]
    return text


def _drug_response_legend_handles(include_embedding: bool = False):
    """Legend for Pre/Post/LFC (SMILES stripped). Embedding is never shown."""
    del include_embedding  # kept for call-site compatibility
    handles: list[Patch] = []
    labels: list[str] = []
    seen: set[str] = set()
    for model in MEASURED_MODEL_ORDER:
        label = _legend_display_label(model)
        if label in seen:
            continue
        seen.add(label)
        handles.append(
            Patch(
                facecolor=MEASURED_MODEL_COLORS.get(model, '#000000'),
                edgecolor='black',
                label=label,
            )
        )
        labels.append(label)
    return handles, labels


def _add_axes_legend(ax, include_embedding: bool = False) -> None:
    """Place Pre/Post/LFC legend in the bottom-left of one panel."""
    handles, labels = _drug_response_legend_handles(include_embedding=include_embedding)
    if not handles:
        return
    ax.legend(
        handles,
        labels,
        loc='lower left',
        fontsize=14,
        frameon=False,
    )


def _add_figure_legend(fig, include_embedding=False):
    """Deprecated figure-level legend; prefer ``_add_axes_legend`` per panel."""
    handles, labels = _drug_response_legend_handles(include_embedding=include_embedding)
    fig.legend(
        handles,
        labels,
        loc='lower left',
        bbox_to_anchor=(0.01, 0.01),
        fontsize=18,
        frameon=False,
    )


def _add_boxplot_scatter_overlay(ax, data, metric, test_categories, model_categories, palette):
    x_locs = {cat: i for i, cat in enumerate(test_categories)}
    n_models = len(model_categories)
    box_width = 0.6

    offset_lookup = {
        model: (idx - (n_models - 1) / 2) * (box_width / n_models)
        for idx, model in enumerate(model_categories)
    }

    for test in test_categories:
        for j, model in enumerate(model_categories):
            group = data[(data['test_set'] == test) & (data['model'] == model)]
            if group.shape[0] == 0:
                continue
            y = pd.to_numeric(group[metric], errors='coerce').to_numpy(dtype=float)
            y = y[np.isfinite(y)]
            if len(y) == 0:
                continue
            x_center = x_locs[test] + offset_lookup[model]
            if len(y) == 1:
                x_vals = [x_center]
            else:
                x_span = (box_width / n_models) * 0.25
                x_vals = np.linspace(x_center - x_span / 2, x_center + x_span / 2, len(y))
            ax.scatter(
                x_vals,
                y,
                s=60,
                color=palette[j],
                edgecolor='black',
                linewidth=0.5,
                alpha=0.6,
                zorder=4,
                label=None,
            )


def _plot_box_panel(
    ax,
    data,
    metric,
    model_order,
    panel_idx,
    plot_legend,
    pre_obs=None,
    post_obs=None,
    lfc_obs=None,
    add_measured_lines=False,
    grid_lines_y=None,
    measured_lines_y=None,
    measured_lines_colors=None,
):
    data = data.dropna(subset=[metric]).copy()
    if isinstance(data.get('test_set', pd.Series(dtype=object)).dtype, pd.CategoricalDtype):
        data['test_set'] = data['test_set'].cat.remove_unused_categories()
    if isinstance(data.get('model', pd.Series(dtype=object)).dtype, pd.CategoricalDtype):
        data['model'] = data['model'].cat.remove_unused_categories()
    if data.empty:
        return
    test_categories = list(data['test_set'].cat.categories)
    model_categories = list(data['model'].cat.categories)
    palette = [MEASURED_MODEL_COLORS.get(m, '#000000') for m in model_categories]

    sns.boxplot(
        data=data,
        x='test_set',
        hue='model',
        y=metric,
        palette=palette,
        width=0.65,
        ax=ax,
    )
    _add_boxplot_scatter_overlay(ax, data, metric, test_categories, model_categories, palette)

    if grid_lines_y is not None:
        for y in grid_lines_y:
            ax.axhline(y=y, color='lightgrey', linewidth=1, linestyle='-', alpha=0.7, zorder=0)
    else:
        ax.yaxis.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)

    # if measured_lines_y is not None and measured_lines_colors is not None:
    #     for y, color in zip(measured_lines_y, measured_lines_colors):
    #         ax.axhline(y=y, color=color, linestyle='-', linewidth=2, zorder=2)
    # elif add_measured_lines and pre_obs is not None and post_obs is not None and lfc_obs is not None:
    #     pre_color = pre_post_lfc_palette[0]
    #     post_color = pre_post_lfc_palette[1]
    #     lfc_color = pre_post_lfc_palette[2]
    #     if panel_idx == 0:
    #         ax.axhline(y=pre_obs, color=pre_color, linestyle='-', linewidth=2, zorder=2)
    #         ax.axhline(y=post_obs, color=post_color, linestyle='-', linewidth=2, zorder=2)
    #         ax.axhline(y=lfc_obs, color=lfc_color, linestyle='-', linewidth=2, zorder=2)

    sns.despine(ax=ax, offset=10, trim=False)
    ax.set_xlabel('')
    ax.xaxis.label.set_visible(False)
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelsize=22, rotation=45)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        fontsize=22,
        rotation=45,
        ha='right',
        rotation_mode='anchor',
    )
    ax.yaxis.set_tick_params(labelsize=16)

    if hasattr(ax, 'legend_') and ax.legend_ is not None:
        ax.legend_.remove()

    n_cats = len(test_categories)
    if n_cats:
        ax.set_xlim(-0.5, n_cats - 0.5)


def _metric_ylabel(metric, stratum='pooled'):
    if metric == 'pearson':
        if stratum == 'per_drug':
            return r'Within-drug Pearson r  $\rightarrow$'
        if stratum == 'per_cell_line':
            return r'Within-line/tissue Pearson r  $\rightarrow$'
        return r'Pearson r  $\rightarrow$'
    if metric == 'top10_overlap':
        return r'Top-10% Retrieval Accuracy (%) $\rightarrow$'
    if metric == 'roc_auc':
        return r'ROC-AUC $\rightarrow$'
    if metric == 'mse':
        return r'MSE $\leftarrow$'
    if metric == 'rmse':
        return r'RMSE $\leftarrow$'
    if metric == 'spearman':
        return r'Spearman $\rho$  $\rightarrow$'
    if metric == 'auroc':
        return r'ROC AUC  $\rightarrow$'
    return metric


def _metric_ylim(metric, values, stratum='pooled'):
    """Data-driven y-limits with padding (no fixed correlation floors/ceilings)."""
    del stratum  # kept for call-site compatibility
    values = pd.to_numeric(values, errors='coerce').dropna()
    row_max = float(values.max()) if len(values) else 0.8
    row_min = float(values.min()) if len(values) else -0.25
    y_pad = max(0.05, 0.08 * (row_max - row_min))
    return (row_min - y_pad, row_max + y_pad)




def _sig_label(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return 'NS'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'NS'


def _metric_higher_is_better(metric: str) -> bool:
    return metric not in {'mse', 'rmse', 'mae'}



def _median_metric(frame, metric):
    if frame is None or frame.empty or metric not in frame.columns:
        return np.nan
    return float(np.nanmedian(pd.to_numeric(frame[metric], errors='coerce').to_numpy(dtype=float)))


def _format_delta_median(median_model: float, median_ref: float) -> str:
    """Median difference (model - reference) for annotation."""
    if np.isnan(median_model) or np.isnan(median_ref):
        return ''
    delta = median_model - median_ref
    return f'Δ{delta:+.2f}'


def _pre_smiles_frame(results):
    return results[
        (results['trained_on'] == 'Measured')
        & (results['test_set'] == 'Measured')
        & (results['model'] == 'Pre+SMILES')
    ]


def _lfc_model_label(model_categories=None):
    cats = list(model_categories) if model_categories is not None else list(PREDICTED_MODEL_ORDER)
    for cand in ('LFC+SMILES', 'LFC'):
        if cand in cats:
            return cand
    return None


def _hue_offsets(model_categories, box_width=0.65):
    n_models = max(len(model_categories), 1)
    return {
        model: (idx - (n_models - 1) / 2) * (box_width / n_models)
        for idx, model in enumerate(model_categories)
    }


def _subset_metric_frame(results, *, trained_on, test_set, model):
    return results[
        (results['trained_on'] == trained_on)
        & (results['test_set'] == test_set)
        & (results['model'] == model)
    ]


def _annotate_delta_at(ax, x, y, label, *, fontsize=14, zorder=20):
    if not label:
        return
    ax.text(
        x,
        y,
        label,
        ha='center',
        va='bottom',
        fontsize=fontsize,
        clip_on=False,
        zorder=zorder,
        transform=ax.transData,
    )


def _annotate_deltas_on_panel(
    ax,
    results,
    metric,
    *,
    ref_median,
    placements,
    model_categories,
    fontsize=14,
    y_rank_by_model=None,
):
    """Place Δ median labels.

    ``placements`` is a list of ``(x_center_category, trained_on, test_set, model)``.
    Optional ``y_rank_by_model`` staggers labels vertically (0 = highest).
    """
    if np.isnan(ref_median):
        return
    offsets = _hue_offsets(model_categories)
    y0, y1 = ax.get_ylim()
    y_span = y1 - y0
    y_base = y1 - 0.04 * y_span
    y_step = 0.07 * y_span
    y_rank_by_model = y_rank_by_model or {}
    for x_cat, trained_on, test_set, model in placements:
        if model not in offsets and model not in model_categories:
            x = float(x_cat)
        else:
            x = float(x_cat) + float(offsets.get(model, 0.0))
        frame = _subset_metric_frame(
            results, trained_on=trained_on, test_set=test_set, model=model
        )
        label = _format_delta_median(_median_metric(frame, metric), ref_median)
        rank = int(y_rank_by_model.get(model, 0))
        y_text = y_base - rank * y_step
        _annotate_delta_at(ax, x, y_text, label, fontsize=fontsize)


def _annotate_delta_vs_pre_smiles_e2e(ax, results, metric, end_to_end_label, *, ref_median=None, fontsize=14):
    """CPA end-to-end Δ median vs Pre+SMILES (same data-coord height/size as other Δ labels)."""
    if metric not in results.columns:
        return
    if ref_median is None:
        ref_median = _median_metric(_pre_smiles_frame(results), metric)
    e2e = _median_metric(results[results['test_set'] == end_to_end_label], metric)
    label = _format_delta_median(e2e, ref_median)
    if not label:
        return
    y0, y1 = ax.get_ylim()
    y_text = y1 - 0.04 * (y1 - y0)
    ax.text(
        0.0,
        y_text,
        label,
        ha='center',
        va='bottom',
        fontsize=fontsize,
        clip_on=False,
        zorder=30,
        transform=ax.transData,
    )


def _annotate_direct_vs_average_effect(ax, results, metric, end_to_end_label):
    """Backward-compatible alias for e2e Δ annotation."""
    _annotate_delta_vs_pre_smiles_e2e(ax, results, metric, end_to_end_label)


def _annotate_prm_vs_average_effect(ax, results, metric, predicted_categories):
    """Δ for each predicted source LFC(+SMILES), including baselines."""
    if metric not in results.columns:
        return
    lfc_label = _lfc_model_label(PREDICTED_MODEL_ORDER)
    if lfc_label is None:
        return
    ref = _median_metric(_pre_smiles_frame(results), metric)
    placements = [
        (i, 'Predicted', test_set, lfc_label)
        for i, test_set in enumerate(predicted_categories)
    ]
    _annotate_deltas_on_panel(
        ax,
        results,
        metric,
        ref_median=ref,
        placements=placements,
        model_categories=list(PREDICTED_MODEL_ORDER),
        fontsize=14,
    )


def _add_x_group_labels(ax, categories, groups, *, y_line=-0.26, y_text=-0.28, fontsize=20):
    """Draw a short bar and label under contiguous x-tick groups (axes y coords)."""
    from matplotlib.transforms import blended_transform_factory

    if not categories:
        return
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for label, members in groups:
        idxs = [i for i, cat in enumerate(categories) if cat in members]
        if not idxs:
            continue
        # only label contiguous runs
        runs = []
        run = [idxs[0]]
        for i in idxs[1:]:
            if i == run[-1] + 1:
                run.append(i)
            else:
                runs.append(run)
                run = [i]
        runs.append(run)
        for run in runs:
            x0 = run[0] - 0.35
            x1 = run[-1] + 0.35
            ax.plot(
                [x0, x1],
                [y_line, y_line],
                transform=trans,
                color='black',
                linewidth=1.0,
                solid_capstyle='butt',
                clip_on=False,
            )
            ax.text(
                (x0 + x1) / 2,
                y_text,
                label,
                transform=trans,
                ha='center',
                va='top',
                fontsize=fontsize,
                clip_on=False,
            )


def _plot_single_dataset_panels(fig, gs_slot, results, metric, plot_text, plot_end_to_end, show_ylabel, show_legend, predicted_categories=None, end_to_end_label=None, stratum='pooled', ylabel=None):
    """One dataset: measured | [optional E2E] | predicted profile sources."""
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    if predicted_categories is None:
        predicted_categories = NEW_MODEL_CATEGORIES
    if end_to_end_label is None:
        end_to_end_label = ev.CPA_E2E_TEST_SET
    n_pred = max(len(predicted_categories), 1)
    # Panel widths proportional to individual box count so each box has equal physical width.
    # Measured: one x-tick with len(MEASURED_MODEL_ORDER) hue boxes.
    # End-to-end: one x-tick with one box.
    # Predicted: n_pred x-ticks with len(PREDICTED_MODEL_ORDER) hue boxes each.
    n_measured_boxes = max(len(MEASURED_MODEL_ORDER), 1)
    n_mid_boxes = 1  # CPA end-to-end only
    n_pred_hues = max(len(PREDICTED_MODEL_ORDER), 1)
    n_pred_boxes = n_pred * n_pred_hues
    reduced_wspace = 0.01

    if plot_end_to_end:
        inner = GridSpecFromSubplotSpec(
            1, 3, gs_slot,
            width_ratios=[n_measured_boxes, n_mid_boxes, n_pred_boxes],
            wspace=reduced_wspace,
        )
        axes = [fig.add_subplot(inner[0, i]) for i in range(3)]
        box_target_axes = [0, 2]
        e2e_ax_idx = 1
    else:
        inner = GridSpecFromSubplotSpec(
            1, 2, gs_slot,
            width_ratios=[n_measured_boxes, n_pred_boxes],
            wspace=reduced_wspace,
        )
        axes = [fig.add_subplot(inner[0, i]) for i in range(2)]
        box_target_axes = [0, 1]
        e2e_ax_idx = None

    box0_data, box2_data = _prepare_box_frames(results, predicted_categories=predicted_categories)
    model_orders = [MEASURED_MODEL_ORDER, PREDICTED_MODEL_ORDER]

    pre_obs, post_obs, lfc_obs = _baseline_means(results, metric)

    ylim_parts = [box0_data, box2_data]
    if plot_end_to_end:
        e2e_ylim = results[results['test_set'] == end_to_end_label]
        if len(e2e_ylim):
            ylim_parts.append(e2e_ylim)
        elastic_ylim = results[
            (results['trained_on'] == 'Measured')
            & (results['test_set'] == 'Measured')
            & (results['model'] == 'Pre+SMILES')
        ]
        if len(elastic_ylim):
            ylim_parts.append(elastic_ylim)
    box_data_concat = pd.concat(ylim_parts)
    ylim = _metric_ylim(metric, box_data_concat[metric], stratum=stratum)
    num_gridlines = 6
    grid_lines_y = np.linspace(ylim[0], ylim[1], num=num_gridlines)

    measured_lines_y = [pre_obs, post_obs, lfc_obs]
    measured_lines_colors = [pre_post_lfc_palette[0], pre_post_lfc_palette[1], pre_post_lfc_palette[2]]

    for i, (plot_box_data, ax_idx) in enumerate(zip([box0_data, box2_data], box_target_axes)):
        _plot_box_panel(
            axes[ax_idx],
            plot_box_data,
            metric,
            model_orders[i],
            i,
            False,
            pre_obs=pre_obs,
            post_obs=post_obs,
            lfc_obs=lfc_obs,
            add_measured_lines=False,
            grid_lines_y=grid_lines_y,
            measured_lines_y=measured_lines_y,
            measured_lines_colors=measured_lines_colors,
        )

        if plot_text and ax_idx == box_target_axes[-1]:
            axes[ax_idx].text(
                x=4.7, y=pre_obs * 0.99, s='Pre', color=pre_post_lfc_palette[0], fontsize=24, ha='left'
            )
            if post_obs > lfc_obs:
                axes[ax_idx].text(
                    x=4.7, y=post_obs * 1, s='Post', color=pre_post_lfc_palette[1], fontsize=24, ha='left'
                )
                axes[ax_idx].text(
                    x=4.7, y=lfc_obs * 0.9, s='LFC', color=pre_post_lfc_palette[2], fontsize=24, ha='left'
                )
            else:
                axes[ax_idx].text(
                    x=4.7, y=post_obs * 0.9, s='Post', color=pre_post_lfc_palette[1], fontsize=24, ha='left'
                )
                axes[ax_idx].text(
                    x=4.7, y=lfc_obs * 1.05, s='LFC', color=pre_post_lfc_palette[2], fontsize=24, ha='left'
                )

    if plot_end_to_end and e2e_ax_idx is not None:
        e2e_rows = results[results['test_set'] == end_to_end_label].copy()
        e2e_rows['model'] = 'Embedding'
        mid_categories = [end_to_end_label]
        model_categories = ['Embedding']
        box4_data = e2e_rows.copy()
        box4_data['test_set'] = pd.Categorical(
            box4_data['test_set'], categories=mid_categories, ordered=True
        )
        box4_data['model'] = pd.Categorical(
            box4_data['model'], categories=model_categories, ordered=True
        )
        box4_data = box4_data.dropna(subset=[metric])

        e2e_ax = axes[e2e_ax_idx]
        if len(box4_data):
            # Same geometry as measured/predicted: hue boxes with width=0.65.
            sns.boxplot(
                data=box4_data,
                x='test_set',
                hue='model',
                y=metric,
                palette=['black'],
                width=0.65,
                ax=e2e_ax,
                legend=False,
            )
            _add_boxplot_scatter_overlay(
                e2e_ax,
                box4_data,
                metric,
                mid_categories,
                model_categories,
                ['black'],
            )
        e2e_ax.set_xlim(-0.5, 0.5)
        for y in grid_lines_y:
            e2e_ax.axhline(y=y, color='lightgrey', linewidth=1, linestyle='-', alpha=0.7, zorder=0)
        e2e_ax.yaxis.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)
        e2e_ax.set_axisbelow(True)
        sns.despine(ax=e2e_ax, offset=10, trim=False)
        e2e_ax.set_xticks(range(len(mid_categories)))
        e2e_ax.set_xticklabels(
            mid_categories,
            fontsize=22,
            rotation=45,
            ha='right',
            rotation_mode='anchor',
        )
        e2e_ax.set_xlabel('')
        e2e_ax.xaxis.label.set_visible(False)
        e2e_ax.set_ylabel('')
        e2e_ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        e2e_ax.set_ylim(*ylim)
        if hasattr(e2e_ax, 'legend_') and e2e_ax.legend_ is not None:
            e2e_ax.legend_.remove()

    # Shared ylim; hide duplicate y-axes.
    for ax_idx, ax in enumerate(axes):
        ax.set_ylim(*ylim)
        if ax_idx != box_target_axes[0]:
            ax.set_ylabel('')
            ax.yaxis.label.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)

    if show_ylabel:
        axes[box_target_axes[0]].set_ylabel(
            ylabel or _metric_ylabel(metric, stratum=stratum), fontsize=22
        )

    pred_ax = axes[box_target_axes[-1]]
    pred_cats = list(predicted_categories)
    pred_set = set(pred_cats)
    if {'CPA', 'chemCPA', 'PRnet', 'GEARS', 'scFoundation'} <= pred_set:
        _add_x_group_labels(
            pred_ax,
            pred_cats,
            (
                ('chemical PRMs', {'CPA', 'chemCPA', 'PRnet'}),
                ('genetic PRMs', {'GEARS', 'scFoundation'}),
                ('baselines', {'Average effect', 'No effect'}),
            ),
        )
    elif {'CPA', 'chemCPA', 'PRnet'} <= pred_set:
        _add_x_group_labels(
            pred_ax,
            pred_cats,
            (
                ('chemical PRMs', {'CPA', 'chemCPA', 'PRnet'}),
                ('baselines', {'Average effect', 'No effect'}),
            ),
        )
    elif {'GEARS', 'scFoundation'} <= pred_set:
        _add_x_group_labels(
            pred_ax,
            pred_cats,
            (
                ('genetic PRMs', {'GEARS', 'scFoundation'}),
                ('baselines', {'Average effect', 'No effect'}),
            ),
        )

    if plot_end_to_end and e2e_ax_idx is not None:
        _add_x_group_labels(
            axes[e2e_ax_idx],
            [end_to_end_label],
            (('direct\npredictor', {end_to_end_label}),),
        )

    return axes


def plot_sensitvity_predictions(
    pearson_results=None,
    metric='pearson',
    plot_text=True,
    plot_end_to_end=True,
    plot_legend=True,
    datasets=None,
    predicted_categories=None,
    stratum='pooled',
    ylabel=None,
):
    """Plot drug-response prediction metrics.

    Pass ``pearson_results`` for one dataset, or ``datasets`` for SciPlex and McFarland
    as side-by-side columns.
    """
    from matplotlib.gridspec import GridSpec

    if datasets is None:
        if pearson_results is None:
            raise ValueError('Provide pearson_results or datasets.')
        datasets = [
            {
                'results': pearson_results,
                'label': None,
                'plot_end_to_end': plot_end_to_end,
            }
        ]

    multi_dataset = len(datasets) > 1

    reduced_outer_wspace = 0.16

    if not multi_dataset:
        ds = datasets[0]
        if ds.get('label'):
            print(f"{ds['label']}:", end=' ')
        pre_obs, post_obs, lfc_obs = _baseline_means(ds['results'], metric)
        print(f'Pre: {pre_obs}, Post: {post_obs}, LFC: {lfc_obs}')

        use_e2e = ds['plot_end_to_end']
        fig = plt.figure(figsize=(14 if use_e2e else 12, 12), dpi=300)
        gs = GridSpec(1, 1, figure=fig)
        axes = _plot_single_dataset_panels(
            fig,
            gs[0],
            ds['results'],
            metric,
            plot_text,
            use_e2e,
            show_ylabel=True,
            show_legend=plot_legend,
            predicted_categories=ds.get('predicted_categories', predicted_categories),
            end_to_end_label=ds.get('end_to_end_label', ev.CPA_E2E_TEST_SET),
            stratum=ds.get('stratum', stratum),
            ylabel=ds.get('ylabel', ylabel),
        )
        if plot_legend and len(axes):
            _add_axes_legend(axes[0], include_embedding=use_e2e)

        fig.tight_layout(pad=0.01)
        fig.subplots_adjust(bottom=0.22)
        return fig, axes

    n_ds = len(datasets)
    # Match observations stratified figure height (12); width ~42 for 3 panels.
    fig = plt.figure(figsize=(14 * n_ds, 12), dpi=300)
    gs_outer = GridSpec(1, n_ds, figure=fig, wspace=reduced_outer_wspace)

    all_axes = []
    for ds_idx, ds in enumerate(datasets):
        label = ds.get('label') or f'dataset_{ds_idx}'
        if label:
            print(f'{label}:', end=' ')
        pre_obs, post_obs, lfc_obs = _baseline_means(ds['results'], metric)
        print(f'Pre: {pre_obs}, Post: {post_obs}, LFC: {lfc_obs}')

        use_e2e = ds.get('plot_end_to_end', plot_end_to_end)
        ds_stratum = ds.get('stratum', stratum)
        axes_ds = _plot_single_dataset_panels(
            fig,
            gs_outer[ds_idx],
            ds['results'],
            metric,
            plot_text,
            use_e2e,
            show_ylabel=ds.get('show_ylabel', ds_idx == 0),
            show_legend=False,
            predicted_categories=ds.get('predicted_categories', predicted_categories),
            end_to_end_label=ds.get('end_to_end_label', ev.CPA_E2E_TEST_SET),
            stratum=ds_stratum,
            ylabel=ds.get('ylabel'),
        )
        if plot_legend and len(axes_ds):
            _add_axes_legend(axes_ds[0], include_embedding=use_e2e)

        if label:
            ax = axes_ds[0]
            ax.set_title("")
            left = axes_ds[0].get_position()
            right = axes_ds[-1].get_position()
            xcenter = (left.x0 + right.x1) / 2
            ytop = left.y1 + 0.03
            fig.text(
                xcenter,
                ytop,
                label,
                fontsize=34,
                fontweight='bold',
                ha='center',
                va='bottom',
            )

        all_axes.extend(axes_ds)

    fig.tight_layout(pad=0.01)
    fig.subplots_adjust(bottom=0.22)
    return fig, np.atleast_1d(all_axes)


def _pearson_box_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    """Fold-wise Pearson in the boxplot schema, keeping undefined correlations as NaN."""
    preds = _add_trained_on(ev.normalize_prediction_frame(predictions))
    fold = ev.compute_fold_metrics(preds)
    if 'trained_on' not in fold.columns:
        fold['trained_on'] = np.where(fold['test_set'].eq('Measured'), 'Measured', 'Predicted')
    return fold


def print_head_pearson_comparison(
    continuous: pd.DataFrame,
    twopart: pd.DataFrame,
    dataset_label: str,
    predicted_categories: list[str] | None = None,
) -> None:
    """Print fold-wise Pearson for continuous vs two-stage, with paired deltas."""
    from scipy import stats

    predicted_categories = predicted_categories or NEW_MODEL_CATEGORIES
    keys = ['test_set', 'model', 'split']
    a = continuous[keys + ['pearson']].rename(columns={'pearson': 'continuous'})
    b = twopart[keys + ['pearson']].rename(columns={'pearson': 'twopart'})
    merged = a.merge(b, on=keys, how='inner')
    merged['delta'] = merged['twopart'] - merged['continuous']
    order = ['Measured'] + list(predicted_categories)
    print('=' * 88)
    print(f'{dataset_label}: two-stage (t=0.02) minus continuous ElasticNet')
    print('=' * 88)
    for test_set in order:
        sub_t = merged[merged['test_set'] == test_set]
        if sub_t.empty:
            continue
        print(f'  {test_set}')
        models = MEASURED_MODEL_ORDER if test_set == 'Measured' else PREDICTED_MODEL_ORDER
        for model in models:
            sub = sub_t[sub_t['model'] == model].sort_values('split')
            if sub.empty:
                continue
            cont = sub['continuous'].to_numpy(dtype=float)
            tp = sub['twopart'].to_numpy(dtype=float)
            delta = sub['delta'].to_numpy(dtype=float)
            paired = delta[np.isfinite(delta)]
            extra = ''
            if len(paired) >= 2:
                mean_d = float(np.mean(paired))
                try:
                    _, pval = stats.ttest_rel(
                        tp[np.isfinite(delta)],
                        cont[np.isfinite(delta)],
                    )
                except Exception:
                    pval = np.nan
                extra = f'   delta {mean_d:+.3f}  p={pval:.2g}' if np.isfinite(pval) else f'   delta {mean_d:+.3f}'
            print(
                f'    {model:<4}  continuous {ev._fmt_mean_sd(cont)}   '
                f'two-stage {ev._fmt_mean_sd(tp)}{extra}'
            )
    print()


def plot_continuous_vs_twopart(
    sciplex_continuous: pd.DataFrame,
    sciplex_twopart: pd.DataFrame,
    mcfarland_continuous: pd.DataFrame,
    mcfarland_twopart: pd.DataFrame,
    predicted_categories: list[str] | None = None,
    outfile: str | None = None,
    metric: str = 'pearson',
):
    """Side-by-side Pearson: columns = continuous vs two-stage, rows = SciPlex / McFarland."""
    predicted_categories = predicted_categories or NEW_MODEL_CATEGORIES
    rows = [
        {
            'label': 'SciPlex3',
            'continuous': sciplex_continuous,
            'twopart': sciplex_twopart,
        },
        {
            'label': 'McFarland',
            'continuous': mcfarland_continuous,
            'twopart': mcfarland_twopart,
        },
    ]
    col_specs = (
        ('continuous', 'Continuous ElasticNet'),
        ('twopart', 'Two-stage (t = 0.02)'),
    )
    all_vals = pd.concat(
        [row[key][metric] for row in rows for key, _ in col_specs if metric in row[key].columns],
        ignore_index=True,
    )
    ylim_vals = pd.to_numeric(all_vals, errors='coerce').dropna()
    y_pad = max(0.05, 0.08 * (float(ylim_vals.max()) - float(ylim_vals.min()))) if len(ylim_vals) else 0.05
    ylim = (-0.25, max(0.8, float(ylim_vals.max()) + y_pad)) if len(ylim_vals) else (-0.25, 0.8)

    fig = plt.figure(figsize=(20, 12), dpi=300)
    gs = GridSpec(2, 2, figure=fig, wspace=0.10, hspace=0.42)
    panel_axes: list[list] = []
    for row_idx, row in enumerate(rows):
        row_axes = []
        for col_idx, (key, _) in enumerate(col_specs):
            axes_ds = _plot_single_dataset_panels(
                fig,
                gs[row_idx, col_idx],
                row[key],
                metric,
                predicted_categories,
                show_ylabel=(col_idx == 0),
                ylabel=r'Pearson r  $\rightarrow$',
                ylim=ylim,
            )
            row_axes.append(axes_ds)
        panel_axes.append(row_axes)

    _add_figure_legend(fig, include_embedding=False)
    fig.tight_layout(pad=0.01)
    fig.subplots_adjust(bottom=0.18, right=0.90, left=0.07, top=0.90)

    for col_idx, (_, title) in enumerate(col_specs):
        left = panel_axes[0][col_idx][0].get_position()
        right = panel_axes[0][col_idx][-1].get_position()
        fig.text(
            (left.x0 + right.x1) / 2,
            0.94,
            title,
            fontsize=18,
            fontweight='bold',
            ha='center',
            va='bottom',
        )
    for row_idx, row in enumerate(rows):
        left = panel_axes[row_idx][0][0].get_position()
        fig.text(
            0.015,
            (left.y0 + left.y1) / 2,
            row['label'],
            fontsize=18,
            fontweight='bold',
            ha='center',
            va='center',
            rotation=90,
        )

    if outfile:
        fig.savefig(outfile, bbox_inches='tight')
        logger.info('Saved %s', outfile)
    return fig, panel_axes


def _save_combined_figures(
    sciplex_results: dict[str, pd.DataFrame],
    mcfarland_results: dict[str, pd.DataFrame],
    predicted_categories: list[str],
    stem: str,
    metric: str = 'pearson',
    ylabel: str | None = None,
    *,
    plot_end_to_end: bool = True,
) -> str | None:
    datasets = []
    if metric in sciplex_results:
        datasets.append(
            {
                'results': sciplex_results[metric],
                'label': 'SciPlex3 dataset',
                'plot_end_to_end': plot_end_to_end,
            }
        )
    if metric in mcfarland_results:
        datasets.append(
            {
                'results': mcfarland_results[metric],
                'label': 'McFarland dataset',
                'plot_end_to_end': plot_end_to_end,
            }
        )
    if not datasets:
        logger.warning('No results available for %s %s', stem, metric)
        return None

    fig, _ = plot_sensitvity_predictions(
        metric=metric,
        datasets=datasets,
        predicted_categories=predicted_categories,
        ylabel=ylabel,
        plot_end_to_end=plot_end_to_end,
    )
    path = os.path.join(figures_dir, f'{stem}_{metric}_combined.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    logger.info('Saved %s', path)
    return path


def _prepared_metric_dicts(
    sciplex_preds: pd.DataFrame | None,
    mcfarland_preds: pd.DataFrame | None,
    *,
    plot_end_to_end: bool = True,
    inject_measured_t1: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Fold-metric frames with optional Measured T1 + CPA end-to-end inject."""

    def _one(preds: pd.DataFrame | None, dataset: str) -> dict[str, pd.DataFrame]:
        if preds is None or preds.empty:
            return {}
        frame = prepare_sensitivity_results(
            preds,
            dataset=dataset,
            stratum='pooled',
            plot_end_to_end=plot_end_to_end,
            inject_measured_t1=inject_measured_t1,
        )
        out: dict[str, pd.DataFrame] = {}
        for metric in ('pearson', 'mse', 'auroc'):
            if metric in frame.columns:
                out[metric] = frame
        return out

    return _one(sciplex_preds, 'sciplex'), _one(mcfarland_preds, 'mcfarland')


def _add_trained_on(predictions: pd.DataFrame) -> pd.DataFrame:
    df = predictions.copy()
    if 'trained_on' in df.columns:
        return df
    if 'trained_on_file' in df.columns:
        df['trained_on'] = df['trained_on_file']
        return df
    test = df['test_set'].replace(LABEL_MAP) if 'test_set' in df.columns else None
    if test is not None:
        df['trained_on'] = np.where(test.eq('Measured') | test.eq('observed'), 'Measured', 'Predicted')
    return df


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.duplicated().any():
        return df
    return df.loc[:, ~df.columns.duplicated()].copy()


def _measured_t1_boxplot_from_task_cv(
    dataset: str,
    group_col: str,
) -> pd.DataFrame:
    """Score measured-task T1 for predicted-profile boxplots.

    ``group_col='fold'`` (pooled panels): one point per PRM fold — aligns Measured
    Pre with split-CV No-effect Post. ``group_col`` = cell line / tissue matches
    the observations stratified T1 figure.
    """
    # Import here so notebook kernels pick up edits after reload.
    import create_figures_tasks as taskfig

    ds_key = str(dataset).strip().lower()
    ds_label = 'SciPlex3' if 'sciplex' in ds_key else 'McFarland'
    preds = taskfig._load_prediction_frames(
        include_two_stage=False, t1_cv_scheme='predefined_fold'
    )
    preds = preds[
        preds['dataset'].astype(str).eq(ds_label)
        & preds['task'].astype(str).eq('T1')
        & preds['model'].astype(str).isin(taskfig.SMILES_MODEL_ORDER)
    ].copy()
    if preds.empty:
        raise ValueError(f'No measured-task T1 predictions for dataset={dataset!r}')

    scores = taskfig.score_predictions_by_group(
        preds, group_col_by_dataset={ds_label: group_col}
    )
    if scores.empty:
        raise ValueError(
            f'No scored measured-task T1 groups for dataset={dataset!r}, '
            f'group_col={group_col!r}'
        )

    out = scores.copy()
    out['test_set'] = 'Measured'
    out['train_set'] = 'Measured'
    out['trained_on'] = 'Measured'
    out['split'] = out['group'].astype(str)
    # Schema expected by plot_sensitvity_predictions / paired helpers.
    if 'spearman' not in out.columns:
        out['spearman'] = out['pearson']
    logger.info(
        'Measured T1 from task-CV (%s by %s): %d groups, models=%s, mean Pearson=%s',
        ds_label,
        group_col,
        len(out),
        sorted(out['model'].astype(str).unique()),
        {
            m: round(float(g['pearson'].mean()), 4)
            for m, g in out.groupby(out['model'].astype(str))
        },
    )
    return out


def load_t1_measured_predictions(
    dataset: str,
    *,
    cv_scheme: str = 'predefined_fold',
) -> pd.DataFrame:
    """Raw T1 continuous predictions from measured-task CV (Measured profiles).

    Default ``cv_scheme='predefined_fold'`` matches the observations T1 figure
    (5-fold / ``*_t1_5fold_predictions.csv``). Use ``cv_scheme='exhaustive'`` for
    leave-one-(drug, context) T1.
    """
    if cv_scheme not in {'exhaustive', 'predefined_fold'}:
        raise ValueError("cv_scheme must be 'exhaustive' or 'predefined_fold'")

    prefix = str(dataset).strip().lower()
    if prefix.startswith('sciplex'):
        prefix = 'sciplex'
    elif prefix.startswith('mcfarland'):
        prefix = 'mcfarland'

    path = os.path.join(results_dir, f'{prefix}_measured_tasks_predictions.csv')
    smiles_path = os.path.join(results_dir, f'{prefix}_smiles_measured_tasks_predictions.csv')
    nosmiles_path = os.path.join(
        results_dir, f'{prefix}_nosmiles_measured_tasks_predictions.csv'
    )
    t1_5fold_path = os.path.join(results_dir, f'{prefix}_measured_tasks_t1_5fold_predictions.csv')
    t1_5fold_smiles_path = os.path.join(
        results_dir, f'{prefix}_smiles_measured_tasks_t1_5fold_predictions.csv'
    )
    t1_5fold_nosmiles_path = os.path.join(
        results_dir, f'{prefix}_nosmiles_measured_tasks_t1_5fold_predictions.csv'
    )

    frames: list[pd.DataFrame] = []
    if cv_scheme == 'predefined_fold':
        for candidate in (
            t1_5fold_path,
            t1_5fold_smiles_path,
            t1_5fold_nosmiles_path,
        ):
            if os.path.exists(candidate):
                frames.append(_dedupe_columns(pd.read_csv(candidate)))
        if not frames:
            for candidate in (path, smiles_path, nosmiles_path):
                if os.path.exists(candidate):
                    frames.append(_dedupe_columns(pd.read_csv(candidate)))
    else:
        # Prefer the same loader as observations (predefined-fold T1 + SMILES merge).
        try:
            import create_figures_tasks as taskfig

            ds_label = 'SciPlex3' if prefix == 'sciplex' else 'McFarland'
            preds = taskfig._load_prediction_frames(include_two_stage=False)
            preds = preds[
                preds['dataset'].astype(str).eq(ds_label)
                & preds['task'].astype(str).eq('T1')
            ].copy()
            if not preds.empty:
                frames.append(preds)
        except Exception as exc:
            logger.warning('task-CV loader failed (%s); falling back to CSV merge', exc)

        if not frames:
            base = None
            for candidate in (path, smiles_path, nosmiles_path):
                if not os.path.exists(candidate):
                    continue
                extra = _dedupe_columns(pd.read_csv(candidate))
                base = extra if base is None else _merge_smiles_prediction_frames(base, extra)
            if base is not None:
                frames.append(base)

    if not frames:
        raise FileNotFoundError(
            f'Missing T1 measured-task predictions for {prefix!r} '
            f'(looked for {path}, {smiles_path}, {nosmiles_path}, and optional *_t1_5fold_*). '
            'Run: python run_measured_task_cv.py --mode continuous'
        )

    preds = frames[0]
    for extra in frames[1:]:
        preds = _merge_smiles_prediction_frames(preds, extra)

    preds = preds[preds['task'].astype(str).eq('T1')].copy()
    if cv_scheme == 'predefined_fold' and 'cv_scheme' in preds.columns:
        wanted = preds['cv_scheme'].astype(str).eq('predefined_fold')
        if wanted.any():
            preds = preds.loc[wanted].copy()
    elif cv_scheme == 'exhaustive' and 'cv_scheme' in preds.columns:
        wanted = preds['cv_scheme'].astype(str).eq('exhaustive')
        if wanted.any():
            preds = preds.loc[wanted].copy()
        else:
            logger.warning(
                'No cv_scheme=exhaustive T1 rows for %s; using available T1.',
                prefix,
            )

    if 'head' in preds.columns:
        preds = preds[preds['head'].fillna('continuous').astype(str).eq('continuous')].copy()
    if preds.empty:
        raise ValueError(f'No continuous T1 {cv_scheme} rows for dataset={dataset!r}')

    preds['test_set'] = 'Measured'
    preds['train_set'] = 'Measured'
    preds['trained_on'] = 'Measured'
    preds['split'] = preds['fold'] if 'fold' in preds.columns else 0
    return ev.normalize_prediction_frame(preds)


def _merge_smiles_prediction_frames(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """Prefer ``extra`` for matching (task, model[, head, threshold]) keys."""
    extra = extra.copy()
    base = base.copy()
    key_cols = [
        c for c in ('task', 'model', 'head', 'threshold')
        if c in extra.columns and c in base.columns
    ]
    if 'task' not in key_cols or 'model' not in key_cols:
        models_in_extra = set(extra['model'].astype(str).unique())
        base_keep = base[~base['model'].astype(str).isin(models_in_extra)].copy()
        return pd.concat([base_keep, extra], ignore_index=True)
    keys = extra[key_cols].drop_duplicates()
    merged = base.merge(keys.assign(_from_extra=1), on=key_cols, how='left')
    base_keep = merged[merged['_from_extra'].isna()].drop(columns=['_from_extra'])
    return pd.concat([base_keep, extra], ignore_index=True)


def load_t1_measured_boxplot_frame(
    dataset: str,
    *,
    cv_scheme: str = 'predefined_fold',
) -> pd.DataFrame:
    """T1 measured-task metrics as Measured-profile boxplot rows."""
    if cv_scheme == 'predefined_fold':
        return _measured_t1_boxplot_from_task_cv(
            dataset, ev.context_group_column(dataset)
        )
    preds = load_t1_measured_predictions(dataset, cv_scheme=cv_scheme)
    return ev.grouped_metrics_boxplot_frame(preds, ev.context_group_column(dataset))


def inject_t1_measured_metrics(results: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Fill missing Measured modalities from measured-task T1 (5-fold + SMILES).

    Pooled predicted-profile plots use one box per **PRM fold**, so T1 is scored
    by ``fold`` (not by cell line / tissue). That keeps Measured Pre aligned with
    No-effect Post from split-CV (both train/test on Pre features under the same
    fold IDs). Existing Measured rows (e.g. from split-CV observed) are kept; T1
    only supplies modalities not already present.
    """
    t1 = _measured_t1_boxplot_from_task_cv(dataset, 'fold')
    existing = results[results['test_set'].astype(str).eq('Measured')].copy()
    if len(existing):
        have = set(existing['model'].astype(str))
        t1 = t1.loc[~t1['model'].astype(str).isin(have)].copy()
    if t1.empty:
        return results
    cols = list(dict.fromkeys([*results.columns.tolist(), *t1.columns.tolist()]))
    return pd.concat(
        [results.reindex(columns=cols), t1.reindex(columns=cols)],
        ignore_index=True,
    )


def inject_t1_measured_grouped_metrics(
    results: pd.DataFrame, dataset: str, group_col: str
) -> pd.DataFrame:
    """Fill missing Measured modalities from group-pooled predefined-fold T1 (+SMILES)."""
    t1 = _measured_t1_boxplot_from_task_cv(dataset, group_col)
    existing = results[results['test_set'].astype(str).eq('Measured')].copy()
    if len(existing):
        have = set(existing['model'].astype(str))
        t1 = t1.loc[~t1['model'].astype(str).isin(have)].copy()
    if t1.empty:
        return results
    cols = list(dict.fromkeys([*results.columns.tolist(), *t1.columns.tolist()]))
    return pd.concat(
        [results.reindex(columns=cols), t1.reindex(columns=cols)],
        ignore_index=True,
    )


def fill_measured_pre_from_no_effect_post(results: pd.DataFrame) -> pd.DataFrame:
    """Replace Measured Pre(+SMILES) boxes with No-effect Post(+SMILES) metrics.

    Thin wrapper around ``evaluate_predictions.fill_measured_pre_from_no_effect_post``
    (shared with stratified summary tables).
    """
    return ev.fill_measured_pre_from_no_effect_post(results)


def fold_metric_boxplot_frame(
    predictions: pd.DataFrame,
    *,
    stratum: str = 'pooled',
    dataset: str | None = None,
) -> pd.DataFrame:
    """Fold- or group-level metric rows for sensitivity boxplots."""
    if stratum == 'pooled':
        return ev.fold_metrics_boxplot_frame(predictions)
    if stratum == 'per_drug':
        group_col = 'condition'
    elif stratum == 'per_cell_line':
        group_col = ev.context_group_column(dataset or 'sciplex')
    else:
        raise ValueError(f'Unknown stratum: {stratum!r}')
    return ev.grouped_metrics_boxplot_frame(predictions, group_col)


def prepare_sensitivity_results(
    predictions: pd.DataFrame,
    *,
    dataset: str,
    stratum: str = 'pooled',
    plot_end_to_end: bool = False,
    inject_measured_t1: bool = True,
    data_dir: str | None = None,
    align_common_groups: bool = False,
) -> pd.DataFrame:
    """Build boxplot metric frame for one dataset/stratum.

    Default: Measured Post(+SMILES) uses predefined-fold measured-task T1.
    For SciPlex within-line panels, Measured Pre(+SMILES) is filled from
    No-effect Post(+SMILES) so it matches the split-CV Pre/no-effect baseline.
    Stratified panels keep each ``(test_set, model)``'s own eligible groups by
    default (CPA-family vs GEARS-family may differ in *n*). Set
    ``align_common_groups=True`` to intersect groups across panels.
    """
    data_dir = data_dir or str(config.DATA_DIR)
    results = fold_metric_boxplot_frame(predictions, stratum=stratum, dataset=dataset)
    if inject_measured_t1 and stratum == 'pooled':
        results = inject_t1_measured_metrics(results, dataset)
        if plot_end_to_end:
            results = ev.append_cpa_end_to_end_fold_metrics(results, dataset, data_dir)
    elif inject_measured_t1:
        group_col = (
            'condition' if stratum == 'per_drug' else ev.context_group_column(dataset)
        )
        results = inject_t1_measured_grouped_metrics(results, dataset, group_col)
        if plot_end_to_end:
            results = ev.append_cpa_end_to_end_grouped_metrics(
                results, dataset, group_col, data_dir
            )
    elif plot_end_to_end:
        if stratum == 'pooled':
            results = ev.append_cpa_end_to_end_fold_metrics(results, dataset, data_dir)
        elif stratum in {'per_drug', 'per_cell_line'}:
            group_col = (
                'condition' if stratum == 'per_drug' else ev.context_group_column(dataset)
            )
            results = ev.append_cpa_end_to_end_grouped_metrics(
                results, dataset, group_col, data_dir
            )
    # SciPlex within-line: Measured Pre ≡ No-effect Post under the same splits.
    if stratum == 'per_cell_line' and 'sciplex' in str(dataset).strip().lower():
        results = fill_measured_pre_from_no_effect_post(results)
    if align_common_groups and stratum in {'per_drug', 'per_cell_line'}:
        results = ev.align_stratified_metric_groups(results)
    return results


def plot_family_custom_panels(
    stem: str,
    categories: list[str],
    panel_specs: list[dict],
    *,
    metric: str = 'pearson',
    plot_end_to_end: bool = False,
    inject_measured_t1: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot an arbitrary list of dataset/stratum panels side by side."""
    datasets = []
    for i, spec in enumerate(panel_specs):
        dataset = spec['dataset']
        stratum = spec.get('stratum', 'pooled')
        use_e2e = spec.get('plot_end_to_end', plot_end_to_end)
        results = prepare_sensitivity_results(
            spec['predictions'],
            dataset=dataset,
            stratum=stratum,
            plot_end_to_end=use_e2e,
            inject_measured_t1=spec.get('inject_measured_t1', inject_measured_t1),
            align_common_groups=spec.get('align_common_groups', False),
        )
        datasets.append(
            {
                'results': results,
                'label': spec.get('label'),
                'plot_end_to_end': use_e2e,
                'end_to_end_label': spec.get('end_to_end_label', ev.CPA_E2E_TEST_SET),
                'stratum': stratum,
                'ylabel': spec.get('ylabel'),
                'show_ylabel': spec.get('show_ylabel', i == 0),
                'predicted_categories': spec.get('predicted_categories', categories),
            }
        )

    fig, axes = plot_sensitvity_predictions(
        metric=metric,
        plot_text=False,
        plot_legend=True,
        plot_end_to_end=plot_end_to_end,
        predicted_categories=categories,
        datasets=datasets,
    )
    out = os.path.join(figures_dir, f'{stem}_{metric}_combined.pdf')
    fig.savefig(out, bbox_inches='tight')
    logger.info('Saved %s', out)
    return fig, axes


def combine_prm_nosmiles_baseline_smiles(
    nosmiles: pd.DataFrame | None,
    smiles: pd.DataFrame | None,
    *,
    prm_sources: tuple[str, ...] | None = None,
    baseline_sources: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Hybrid frame: PRM gene-only Post/LFC + baseline Post/LFC+SMILES.

    Baseline SMILES heads are remapped to ``post_treatment`` / ``LFC`` so they
    share the gene-only hue slots in stratified plots (Measured stays +SMILES via
    ``inject_measured_t1``). Raw ``profile_source`` / ``test_set`` tokens are
    accepted before or after ``LABEL_MAP`` renaming.
    """
    if prm_sources is None:
        prm_sources = (
            'CPA_predicted',
            'chemCPA_predicted',
            'PRnet_predicted',
            'GEARS_predicted',
            'scFoundation_predicted',
            'CPA',
            'chemCPA',
            'PRnet',
            'GEARS',
            'scFoundation',
        )
    if baseline_sources is None:
        baseline_sources = (
            'average_effect',
            'no_effect',
            'Average effect',
            'No effect',
        )

    prm_keys = set(prm_sources)
    baseline_keys = set(baseline_sources)
    gene_models = {'post_treatment', 'LFC', 'Post', 'LFC'}
    smiles_models = {
        'post_treatment_smiles',
        'LFC_smiles',
        'Post+SMILES',
        'LFC+SMILES',
    }
    smiles_to_gene = {
        'post_treatment_smiles': 'post_treatment',
        'Post+SMILES': 'Post',
        'LFC_smiles': 'LFC',
        'LFC+SMILES': 'LFC',
    }

    parts: list[pd.DataFrame] = []
    if nosmiles is not None and len(nosmiles):
        df = nosmiles.copy()
        src_col = 'test_set' if 'test_set' in df.columns else 'profile_source'
        keep = df[src_col].astype(str).isin(prm_keys) & df['model'].astype(str).isin(
            gene_models
        )
        parts.append(df.loc[keep].copy())

    if smiles is not None and len(smiles):
        df = smiles.copy()
        src_col = 'test_set' if 'test_set' in df.columns else 'profile_source'
        keep = df[src_col].astype(str).isin(baseline_keys) & df['model'].astype(str).isin(
            smiles_models
        )
        base = df.loc[keep].copy()
        if len(base):
            base['model'] = base['model'].astype(str).replace(smiles_to_gene)
            parts.append(base)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def replace_measured_predictions_with_t1(predictions: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Ensure Measured SMILES modalities exist; keep split-CV Measured when present.

    T1 task-CV rows only fill modalities missing from ``predictions`` (e.g. Post+SMILES /
    LFC+SMILES when the split-CV file only has Pre+SMILES observed).
    """
    preds = ev.normalize_prediction_frame(predictions)
    t1 = load_t1_measured_predictions(dataset, cv_scheme='predefined_fold')
    smiles = {'Pre+SMILES', 'Post+SMILES', 'LFC+SMILES'}
    if 'model' in t1.columns:
        t1 = t1[t1['model'].astype(str).isin(smiles)].copy()
    existing = preds[preds['test_set'].astype(str).eq('Measured')].copy()
    if len(existing):
        have = set(existing['model'].astype(str))
        t1 = t1.loc[~t1['model'].astype(str).isin(have)].copy()
    if t1.empty:
        return preds
    cols = list(dict.fromkeys([*preds.columns.tolist(), *t1.columns.tolist()]))
    return pd.concat(
        [preds.reindex(columns=cols), t1.reindex(columns=cols)],
        ignore_index=True,
    )


def _run_family(
    sciplex_preds: pd.DataFrame | None,
    mcfarland_preds: pd.DataFrame | None,
    predicted_categories: list[str],
    stem: str,
    *,
    plot_end_to_end: bool = True,
    inject_measured_t1: bool = True,
) -> None:
    sciplex_preds = _add_trained_on(sciplex_preds) if sciplex_preds is not None else None
    mcfarland_preds = _add_trained_on(mcfarland_preds) if mcfarland_preds is not None else None

    sciplex_prepared, mcfarland_prepared = _prepared_metric_dicts(
        sciplex_preds,
        mcfarland_preds,
        plot_end_to_end=plot_end_to_end,
        inject_measured_t1=inject_measured_t1,
    )
    _save_combined_figures(
        sciplex_prepared,
        mcfarland_prepared,
        predicted_categories,
        stem,
        metric='pearson',
        plot_end_to_end=plot_end_to_end,
    )
    _save_combined_figures(
        sciplex_prepared,
        mcfarland_prepared,
        predicted_categories,
        stem,
        metric='mse',
        plot_end_to_end=plot_end_to_end,
    )

    dataset_frames = []
    if sciplex_preds is not None:
        dataset_frames.append(('SciPlex3', sciplex_preds))
    if mcfarland_preds is not None:
        dataset_frames.append(('McFarland', mcfarland_preds))

    fold_tables = []
    comparison_tables = []
    per_drug_tables = []
    per_line_tables = []
    stratified_results = {'sciplex': {}, 'mcfarland': {}}

    for label, preds in dataset_frames:
        fold = ev.compute_fold_metrics(preds)
        fold = fold.assign(dataset=label, family=stem)
        fold_tables.append(fold)
        grouped_drug = ev.compute_grouped_metrics(preds, 'condition')
        context_col = ev.context_group_column(label)
        grouped_line = ev.compute_grouped_metrics(preds, context_col)
        per_drug = ev.grouped_metrics_boxplot_frame(
            preds, 'condition', align_common_groups=False
        )
        per_line = ev.grouped_metrics_boxplot_frame(
            preds, context_col, align_common_groups=False
        )
        if not per_drug.empty:
            per_drug_tables.append(per_drug.assign(dataset=label, family=stem))
        if not per_line.empty:
            per_line_tables.append(per_line.assign(dataset=label, family=stem))

        key = 'sciplex' if label == 'SciPlex3' else 'mcfarland'
        stratified_results[key]['per_drug'] = per_drug
        stratified_results[key]['per_cell_line'] = per_line

        pearson_comps = ev.print_metric_report(
            figure_name=f'{stem}_pearson_combined.pdf',
            dataset_label=label,
            predictions=preds,
            fold_metrics=fold,
            metric='pearson',
            predicted_categories=predicted_categories,
            higher_is_better=True,
            extra_metrics=['spearman'],
        )
        mse_comps = ev.print_metric_report(
            figure_name=f'{stem}_mse_combined.pdf',
            dataset_label=label,
            predictions=preds,
            fold_metrics=fold,
            metric='mse',
            predicted_categories=predicted_categories,
            higher_is_better=False,
            extra_metrics=['mae', 'rmse'],
        )
        ev.print_stratified_report(
            dataset_label=label,
            predictions=preds,
            predicted_categories=predicted_categories,
            metric='pearson',
        )
        extra = ev.print_supplement_report(
            dataset_label=label,
            fold_metrics=fold,
            predicted_categories=predicted_categories,
        )
        for comps in (pearson_comps, mse_comps, extra.get('comparisons')):
            if comps is not None and not comps.empty:
                comparison_tables.append(comps.assign(dataset=label, family=stem))

        if not grouped_drug.empty:
            grouped_drug.assign(dataset=label, family=stem).to_csv(
                os.path.join(results_dir, f'{stem}_{label.lower()}_per_drug_metrics.csv'),
                index=False,
            )
        if not grouped_line.empty:
            grouped_line.assign(dataset=label, family=stem).to_csv(
                os.path.join(results_dir, f'{stem}_{label.lower()}_per_cell_line_metrics.csv'),
                index=False,
            )

        print_path = os.path.join(results_dir, f'{stem}_{label.lower()}_fold_metrics.csv')
        fold.to_csv(print_path, index=False)
        logger.info('Wrote %s', print_path)

    _save_combined_figures(
        sciplex_prepared,
        mcfarland_prepared,
        predicted_categories,
        stem,
        metric='auroc',
        ylabel=r'ROC AUC  $\rightarrow$',
        plot_end_to_end=plot_end_to_end,
    )

    stratified_prepared = {
        'sciplex': {
            key: (
                prepare_sensitivity_results(
                    sciplex_preds,
                    dataset='sciplex',
                    stratum=key,
                    plot_end_to_end=plot_end_to_end,
                    inject_measured_t1=inject_measured_t1,
                )
                if sciplex_preds is not None
                else pd.DataFrame()
            )
            for key in ('per_drug', 'per_cell_line')
        },
        'mcfarland': {
            key: (
                prepare_sensitivity_results(
                    mcfarland_preds,
                    dataset='mcfarland',
                    stratum=key,
                    plot_end_to_end=plot_end_to_end,
                    inject_measured_t1=inject_measured_t1,
                )
                if mcfarland_preds is not None
                else pd.DataFrame()
            )
            for key in ('per_drug', 'per_cell_line')
        },
    }

    for key, metric, ylabel in STRATIFIED_FIGURE_SPECS:
        _save_stratified_figure(
            stratified_prepared,
            predicted_categories,
            f'{stem}_{metric}_{key}',
            metric,
            ylabel,
            key=key,
            plot_end_to_end=plot_end_to_end,
        )

    if fold_tables:
        combined_fold = pd.concat(fold_tables, ignore_index=True)
        combined_fold.to_csv(os.path.join(results_dir, f'{stem}_fold_metrics.csv'), index=False)
    if comparison_tables:
        pd.concat(comparison_tables, ignore_index=True).to_csv(
            os.path.join(results_dir, f'{stem}_paired_comparisons.csv'), index=False
        )
    if per_drug_tables:
        pd.concat(per_drug_tables, ignore_index=True).to_csv(
            os.path.join(results_dir, f'{stem}_per_drug_fold_pearson.csv'), index=False
        )
    if per_line_tables:
        pd.concat(per_line_tables, ignore_index=True).to_csv(
            os.path.join(results_dir, f'{stem}_per_cell_line_fold_pearson.csv'), index=False
        )


def _save_stratified_figure(
    stratified_results: dict,
    predicted_categories: list[str],
    stem: str,
    metric: str,
    ylabel: str,
    key: str = 'pearson_per_drug',
    *,
    plot_end_to_end: bool = True,
) -> None:
    sciplex_df = stratified_results['sciplex'].get(key, pd.DataFrame())
    mcfarland_df = stratified_results['mcfarland'].get(key, pd.DataFrame())
    datasets = []
    if sciplex_df is not None and not sciplex_df.empty:
        if metric in sciplex_df.columns:
            sciplex_df = sciplex_df.copy()
            if metric in {'pearson', 'spearman'}:
                sciplex_df[metric] = sciplex_df[metric].fillna(0.0)
            else:
                sciplex_df = sciplex_df.dropna(subset=[metric])
        if not sciplex_df.empty:
            datasets.append(
                {
                    'results': sciplex_df,
                    'label': 'SciPlex3 dataset',
                    'plot_end_to_end': plot_end_to_end,
                }
            )
    if mcfarland_df is not None and not mcfarland_df.empty:
        if metric in mcfarland_df.columns:
            mcfarland_df = mcfarland_df.copy()
            if metric in {'pearson', 'spearman'}:
                mcfarland_df[metric] = mcfarland_df[metric].fillna(0.0)
            else:
                mcfarland_df = mcfarland_df.dropna(subset=[metric])
        if not mcfarland_df.empty:
            datasets.append(
                {
                    'results': mcfarland_df,
                    'label': 'McFarland dataset',
                    'plot_end_to_end': plot_end_to_end,
                }
            )
    if not datasets:
        logger.warning('No stratified results for %s', stem)
        return
    fig, _ = plot_sensitvity_predictions(
        metric=metric,
        datasets=datasets,
        predicted_categories=predicted_categories,
        ylabel=ylabel,
        plot_end_to_end=plot_end_to_end,
    )
    path = os.path.join(figures_dir, f'{stem}_combined.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    logger.info('Saved %s', path)


SWEEP_LINE_MODELS = ['Measured', 'CPA', 'chemCPA', 'PRnet', 'Average effect', 'No effect']
# Per-panel styles (Post and LFC are separate axes):
#   solid        = two-stage
#   small dotted = continuous ElasticNet
SWEEP_TWOPART_STYLE = '-'
SWEEP_CONTINUOUS_STYLE = (0, (1, 2))
DEFAULT_HURDLE = 0.02


def _modality_legend_label(model: str) -> str:
    """Short modality label for legends / panel titles (drop +SMILES)."""
    text = str(model)
    if text.endswith('+SMILES'):
        return text[: -len('+SMILES')]
    return text


def _threshold_sensitivity_legend_handles(
    *,
    model_names: list[str],
    palette: dict[str, str],
    has_continuous: bool,
) -> tuple[list, list]:
    """Color = profile source; line style = two-stage vs continuous."""
    from matplotlib.lines import Line2D

    handles: list = []
    labels: list = []

    handles.append(Line2D([], [], linestyle='none', marker='', color='none'))
    labels.append('Model')
    for name in model_names:
        handles.append(
            Line2D(
                [],
                [],
                color=palette.get(name, '#333333'),
                linestyle='-',
                lw=2.0,
                marker='o',
                markersize=4,
            )
        )
        labels.append(name)

    handles.append(Line2D([], [], linestyle='none', marker='', color='none'))
    labels.append('Setup')
    handles.append(
        Line2D([], [], color='black', linestyle=SWEEP_TWOPART_STYLE, lw=1.8, marker='o', markersize=4)
    )
    labels.append('Two-stage')
    if has_continuous:
        handles.append(
            Line2D([], [], color='black', linestyle=SWEEP_CONTINUOUS_STYLE, lw=1.4)
        )
        labels.append('Continuous')
    return handles, labels


def _resolve_twopart_sweep_suffix(preferred: str = '_smiles') -> str:
    """Prefer SMILES sweep CSVs when present; else gene-only (no suffix)."""
    for suffix in (preferred, ''):
        paths = [
            Path(results_dir) / f'{ds}_twopart_threshold_sweep{suffix}_predictions.csv'
            for ds in ('sciplex', 'mcfarland')
        ]
        if any(p.exists() for p in paths):
            return suffix
    return preferred


def load_twopart_threshold_sweep_fold_metrics(
    suffix: str | None = None,
) -> pd.DataFrame:
    """Fold-wise Pearson for two-stage hurdle threshold sweeps.

    Reads ``{dataset}_twopart_threshold_sweep{suffix}_predictions.csv`` and
    scores per fold × threshold. ``suffix=None`` auto-selects ``_smiles`` if
    present, otherwise the unsuffixed gene-only sweep.
    """
    if suffix is None:
        suffix = _resolve_twopart_sweep_suffix()
    frames: list[pd.DataFrame] = []
    for dataset in ('sciplex', 'mcfarland'):
        path = Path(results_dir) / f'{dataset}_twopart_threshold_sweep{suffix}_predictions.csv'
        if not path.exists():
            logger.warning('Missing threshold-sweep predictions: %s', path)
            continue
        preds = pd.read_csv(path)
        fold = ev.compute_fold_metrics(preds)
        if 'dataset' not in fold.columns:
            fold['dataset'] = dataset
        frames.append(fold)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out_path = Path(results_dir) / f'twopart_threshold_sweep{suffix}_fold_metrics.csv'
    out.to_csv(out_path, index=False)
    logger.info('Wrote %s (%d rows)', out_path, len(out))
    return out


def load_continuous_fold_metrics_for_sweep(
    suffix: str | None = None,
) -> pd.DataFrame:
    """Continuous ElasticNet fold metrics for threshold-sweep overlays."""
    if suffix is None:
        suffix = _resolve_twopart_sweep_suffix()
    # Continuous main results use *_split_cv{suffix}_predictions.csv
    frames: list[pd.DataFrame] = []
    for dataset in ('sciplex', 'mcfarland'):
        path = Path(results_dir) / f'{dataset}_split_cv{suffix}_predictions.csv'
        if not path.exists() and suffix:
            path = Path(results_dir) / f'{dataset}_split_cv_predictions.csv'
        if not path.exists():
            logger.warning('Missing continuous split-CV predictions: %s', path)
            continue
        preds = pd.read_csv(path)
        if 'head' in preds.columns:
            preds = preds[preds['head'].astype(str).eq('continuous')].copy()
        fold = ev.compute_fold_metrics(preds)
        if 'dataset' not in fold.columns:
            fold['dataset'] = dataset
        frames.append(fold)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# Stratified panels matching the main predicted-profile figure.
SWEEP_STRATIFIED_PANEL_SPECS: list[dict[str, str]] = [
    {
        'key': 'sciplex_cell_line',
        'dataset': 'sciplex',
        'group_col': 'cell_line',
        'title': 'SciPlex3\n(within cell line)',
        'ylabel': r'Within-line Pearson $r$',
    },
    {
        'key': 'mcfarland_tissue',
        'dataset': 'mcfarland',
        'group_col': 'tissue',
        'title': 'McFarland\n(within tissue)',
        'ylabel': r'Within-tissue Pearson $r$',
    },
    {
        'key': 'mcfarland_drug',
        'dataset': 'mcfarland',
        'group_col': 'condition',
        'title': 'McFarland\n(within drug)',
        'ylabel': r'Within-drug Pearson $r$',
    },
]


def _read_sweep_predictions(dataset: str, *, suffix: str, kind: str) -> pd.DataFrame:
    """Load twopart-threshold-sweep or continuous split-CV prediction CSV."""
    if kind == 'twopart':
        path = Path(results_dir) / f'{dataset}_twopart_threshold_sweep{suffix}_predictions.csv'
    elif kind == 'continuous':
        path = Path(results_dir) / f'{dataset}_split_cv{suffix}_predictions.csv'
        if not path.exists() and suffix:
            path = Path(results_dir) / f'{dataset}_split_cv_predictions.csv'
    else:
        raise ValueError(f'Unknown sweep prediction kind: {kind!r}')
    if not path.exists():
        return pd.DataFrame()
    preds = pd.read_csv(path)
    if kind == 'continuous' and 'head' in preds.columns:
        preds = preds[preds['head'].astype(str).eq('continuous')].copy()
    return preds


def load_stratified_threshold_sensitivity_panels(
    suffix: str | None = None,
    *,
    panel_specs: list[dict[str, str]] | None = None,
    align_common_groups: bool = True,
) -> list[dict]:
    """Build stratified twopart + continuous metric tables for sweep figures.

    Matches the main figure strata: SciPlex within cell line, McFarland within
    tissue, McFarland within drug. Each panel dict has ``title``, ``ylabel``,
    ``twopart``, and ``continuous`` DataFrames (one row per group × setting).
    """
    if suffix is None:
        suffix = _resolve_twopart_sweep_suffix()
    specs = list(panel_specs or SWEEP_STRATIFIED_PANEL_SPECS)
    panels: list[dict] = []
    for spec in specs:
        dataset = str(spec['dataset'])
        group_col = str(spec['group_col'])
        twopart_preds = _read_sweep_predictions(dataset, suffix=suffix, kind='twopart')
        cont_preds = _read_sweep_predictions(dataset, suffix=suffix, kind='continuous')
        twopart = (
            ev.grouped_metrics_boxplot_frame(
                twopart_preds,
                group_col,
                align_common_groups=align_common_groups,
            )
            if len(twopart_preds)
            else pd.DataFrame()
        )
        continuous = (
            ev.grouped_metrics_boxplot_frame(
                cont_preds,
                group_col,
                align_common_groups=align_common_groups,
            )
            if len(cont_preds)
            else pd.DataFrame()
        )
        for frame in (twopart, continuous):
            if frame is None or frame.empty:
                continue
            if 'dataset' not in frame.columns:
                frame['dataset'] = dataset
            frame['panel_key'] = spec['key']
            frame['stratum'] = group_col
        if twopart.empty:
            logger.warning(
                'No stratified twopart metrics for %s (%s)',
                spec['key'],
                group_col,
            )
            continue
        panels.append(
            {
                'key': spec['key'],
                'title': spec['title'],
                'ylabel': spec.get('ylabel', r'Pearson $r$'),
                'dataset': dataset,
                'group_col': group_col,
                'twopart': twopart,
                'continuous': continuous if not continuous.empty else None,
            }
        )
        logger.info(
            'Sweep panel %s: twopart rows=%d continuous rows=%d group=%s',
            spec['key'],
            len(twopart),
            0 if continuous is None or continuous.empty else len(continuous),
            group_col,
        )
    return panels


def plot_threshold_sensitivity(
    twopart_fold: pd.DataFrame | None,
    continuous_fold: pd.DataFrame | None,
    outfile: str,
    predicted_categories: list[str] | None = None,
    *,
    panel_frames: list[dict] | None = None,
    sharey: str | bool = 'row',
) -> plt.Figure:
    """Line plots of Pearson r versus two-stage hurdle threshold.

    Separate rows for Post and LFC. Colour = profile source; solid = two-stage;
    small dotted = continuous ElasticNet reference.

    Pass ``panel_frames`` for custom columns (e.g. stratified SciPlex line /
    McFarland tissue / McFarland drug). Each item needs ``title``, ``twopart``,
    and optional ``continuous``. When ``panel_frames`` is None, one column is
    built per dataset in ``twopart_fold`` (pooled fold metrics).
    """
    predicted_categories = predicted_categories or NEW_MODEL_CATEGORIES
    label_map = {'sciplex': 'SciPlex3', 'mcfarland': 'McFarland'}

    if panel_frames is None:
        if twopart_fold is None or twopart_fold.empty:
            raise ValueError('twopart_fold is required when panel_frames is None')
        twopart_fold = twopart_fold.copy()
        twopart_fold['test_set'] = twopart_fold['test_set'].replace(LABEL_MAP)
        twopart_fold['model'] = twopart_fold['model'].replace(LABEL_MAP)
        if continuous_fold is not None and len(continuous_fold):
            continuous_fold = continuous_fold.copy()
            continuous_fold['test_set'] = continuous_fold['test_set'].replace(LABEL_MAP)
            continuous_fold['model'] = continuous_fold['model'].replace(LABEL_MAP)
        datasets = [
            ds
            for ds in ('sciplex', 'mcfarland', 'SciPlex3', 'McFarland')
            if ds in set(twopart_fold.get('dataset', pd.Series(dtype=str)).astype(str))
        ]
        if not datasets:
            datasets = (
                sorted(twopart_fold['dataset'].dropna().unique().tolist())
                if 'dataset' in twopart_fold.columns
                else [None]
            )
        panel_frames = []
        for dataset in datasets:
            title = label_map.get(str(dataset), dataset) if dataset is not None else ''
            tp = twopart_fold
            if dataset is not None and 'dataset' in twopart_fold.columns:
                tp = twopart_fold[
                    twopart_fold['dataset'].astype(str).isin(
                        {dataset, label_map.get(str(dataset), str(dataset))}
                    )
                ]
            cont = None
            if continuous_fold is not None and len(continuous_fold):
                cont = continuous_fold
                if dataset is not None and 'dataset' in continuous_fold.columns:
                    cont = continuous_fold[
                        continuous_fold['dataset'].astype(str).isin(
                            {dataset, label_map.get(str(dataset), str(dataset))}
                        )
                    ]
            panel_frames.append(
                {
                    'title': title,
                    'ylabel': r'Pearson $r$',
                    'twopart': tp,
                    'continuous': cont,
                }
            )

    # Normalize labels inside panels.
    normalized_panels: list[dict] = []
    for panel in panel_frames:
        tp = panel['twopart'].copy()
        tp['test_set'] = tp['test_set'].replace(LABEL_MAP)
        tp['model'] = tp['model'].replace(LABEL_MAP)
        cont = panel.get('continuous')
        if cont is not None and len(cont):
            cont = cont.copy()
            cont['test_set'] = cont['test_set'].replace(LABEL_MAP)
            cont['model'] = cont['model'].replace(LABEL_MAP)
        else:
            cont = None
        normalized_panels.append({**panel, 'twopart': tp, 'continuous': cont})

    modalities = [
        m
        for m in PREDICTED_MODEL_ORDER
        if any(m in set(p['twopart']['model'].astype(str)) for p in normalized_panels)
        or any(
            p.get('continuous') is not None
            and len(p['continuous'])
            and m in set(p['continuous']['model'].astype(str))
            for p in normalized_panels
        )
    ]
    if not modalities:
        modalities = [m for m in PREDICTED_MODEL_ORDER if m]

    n_col = max(len(normalized_panels), 1)
    n_mod = max(len(modalities), 1)
    fig, axes = plt.subplots(
        n_mod,
        n_col,
        figsize=(5.4 * n_col, 3.6 * n_mod),
        dpi=300,
        sharex=True,
        sharey=sharey,
        squeeze=False,
    )

    palette = {
        'Measured': '#8B0000',
        'CPA': '#0072B2',
        'chemCPA': '#009E73',
        'PRnet': '#E69F00',
        'GEARS': '#56B4E9',
        'scFoundation': '#CC79A7',
        'Average effect': '#4d4d4d',
        'No effect': '#7f7f7f',
    }

    metric = 'pearson'
    models_seen: list[str] = []
    for row, modality in enumerate(modalities):
        for col, panel in enumerate(normalized_panels):
            ax = axes[row, col]
            tp = panel['twopart']
            tp = tp[tp['model'].astype(str).eq(str(modality))]
            tp = tp[tp['test_set'].isin(predicted_categories + ['Measured'])]
            tp = tp.dropna(subset=[metric, 'threshold']) if 'threshold' in tp.columns else tp
            if tp.empty:
                ax.set_visible(False)
                continue
            summary = (
                tp.groupby(['test_set', 'threshold'], as_index=False)
                .agg(mean=(metric, 'mean'), std=(metric, 'std'))
            )
            for test_set in ['Measured'] + predicted_categories:
                line = summary[summary['test_set'] == test_set]
                if line.empty:
                    continue
                if test_set not in models_seen:
                    models_seen.append(test_set)
                ax.plot(
                    line['threshold'],
                    line['mean'],
                    color=palette.get(test_set, '#333333'),
                    linestyle=SWEEP_TWOPART_STYLE,
                    marker='o',
                    markersize=4,
                    lw=1.8,
                )

            cont = panel.get('continuous')
            if cont is not None and len(cont):
                csub = cont[cont['model'].astype(str).eq(str(modality))]
                csub = csub[csub['test_set'].isin(predicted_categories + ['Measured'])]
                csub = csub.dropna(subset=[metric])
                for test_set in ['Measured'] + predicted_categories:
                    vals = csub[csub['test_set'] == test_set][metric]
                    if vals.empty or not np.isfinite(vals.mean()):
                        continue
                    if test_set not in models_seen:
                        models_seen.append(test_set)
                    ax.axhline(
                        float(vals.mean()),
                        color=palette.get(test_set, '#333333'),
                        linestyle=SWEEP_CONTINUOUS_STYLE,
                        lw=1.2,
                        alpha=0.85,
                    )

            ax.axvline(DEFAULT_HURDLE, color='0.35', ls='-.', lw=1.0, alpha=0.85)
            if col == 0:
                ylab = panel.get('ylabel') or r'Pearson $r$'
                ax.set_ylabel(
                    f'{_modality_legend_label(modality)}\n{ylab}',
                    fontsize=16,
                )
            else:
                ax.set_ylabel('')
            if row == n_mod - 1:
                ax.set_xlabel('Hurdle threshold', fontsize=16)
            else:
                ax.set_xlabel('')
            if row == 0 and panel.get('title'):
                ax.set_title(panel['title'], fontsize=16, fontweight='bold')
            ax.tick_params(axis='both', labelsize=13)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    has_continuous = any(
        p.get('continuous') is not None and len(p['continuous']) for p in normalized_panels
    )
    handles, labels = _threshold_sensitivity_legend_handles(
        model_names=models_seen,
        palette=palette,
        has_continuous=has_continuous,
    )
    if handles:
        fig.legend(
            handles,
            labels,
            loc='center left',
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            fontsize=14,
            handlelength=2.8,
            labelspacing=0.45,
        )
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches='tight', dpi=300)
    logger.info('Saved %s', outfile)
    return fig


def plot_threshold_sensitivity_stratified(
    outfile: str,
    *,
    suffix: str | None = None,
    predicted_categories: list[str] | None = None,
    align_common_groups: bool = True,
) -> plt.Figure:
    """Threshold sensitivity with main-figure strata (line / tissue / drug)."""
    panels = load_stratified_threshold_sensitivity_panels(
        suffix=suffix,
        align_common_groups=align_common_groups,
    )
    if not panels:
        raise ValueError('No stratified threshold-sweep panels available')
    return plot_threshold_sensitivity(
        None,
        None,
        outfile,
        predicted_categories=predicted_categories,
        panel_frames=panels,
        sharey=False,
    )


def _format_threshold_setting_label(threshold: float | None) -> str:
    """Column label for continuous vs a hurdle threshold."""
    if threshold is None or (isinstance(threshold, float) and not np.isfinite(threshold)):
        return 'Continuous'
    t = float(threshold)
    if abs(t - round(t)) < 1e-12:
        return f't={int(round(t))}'
    text = f'{t:.2f}'.rstrip('0').rstrip('.')
    return f't={text}'


def build_threshold_rank_tables(
    twopart_fold: pd.DataFrame,
    continuous_fold: pd.DataFrame | None,
    *,
    predicted_categories: list[str] | None = None,
    modalities: list[str] | None = None,
    metric: str = 'pearson',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean metric + within-column ranks for continuous and two-stage thresholds.

    Ranks are 1 = best (highest ``metric``) among ``predicted_categories`` within
    each (dataset, modality, setting) column. Returns ``(means_long, ranks_long)``.
    """
    predicted_categories = list(predicted_categories or NEW_MODEL_CATEGORIES)
    modalities = list(modalities or [m for m in PREDICTED_MODEL_ORDER if m])
    label_map = {'sciplex': 'SciPlex3', 'mcfarland': 'McFarland'}

    frames: list[pd.DataFrame] = []
    for source, frame, setting_kind in (
        ('continuous', continuous_fold, 'continuous'),
        ('twopart', twopart_fold, 'twopart'),
    ):
        if frame is None or frame.empty:
            continue
        df = frame.copy()
        df['test_set'] = df['test_set'].replace(LABEL_MAP)
        df['model'] = df['model'].replace(LABEL_MAP)
        if 'dataset' in df.columns:
            df['dataset'] = (
                df['dataset']
                .astype(str)
                .replace(label_map)
            )
        df = df[df['model'].astype(str).isin(modalities)]
        df = df[df['test_set'].astype(str).isin(predicted_categories)]
        df[metric] = pd.to_numeric(df[metric], errors='coerce')
        df = df.dropna(subset=[metric])
        if df.empty:
            continue
        if setting_kind == 'continuous':
            df = df.copy()
            df['setting'] = 'Continuous'
            df['threshold'] = np.nan
            df['setting_order'] = 0
        else:
            df = df.dropna(subset=['threshold']).copy()
            df['setting'] = df['threshold'].map(_format_threshold_setting_label)
            # Continuous first, then thresholds ascending.
            thr_rank = {t: i + 1 for i, t in enumerate(sorted(df['threshold'].unique()))}
            df['setting_order'] = df['threshold'].map(thr_rank)
        frames.append(df)

    if not frames:
        empty = pd.DataFrame(
            columns=[
                'dataset',
                'modality',
                'model',
                'setting',
                'setting_order',
                'threshold',
                'mean',
                'rank',
            ]
        )
        return empty, empty

    long = pd.concat(frames, ignore_index=True)
    means = (
        long.groupby(
            ['dataset', 'model', 'test_set', 'setting', 'setting_order'],
            as_index=False,
        )
        .agg(mean=(metric, 'mean'), threshold=('threshold', 'first'))
        .rename(columns={'model': 'modality', 'test_set': 'model'})
    )
    means['rank'] = (
        means.groupby(['dataset', 'modality', 'setting'])['mean']
        .rank(ascending=False, method='min')
        .astype(int)
    )
    means = means.sort_values(
        ['dataset', 'modality', 'setting_order', 'rank', 'model']
    ).reset_index(drop=True)
    ranks = means[
        ['dataset', 'modality', 'model', 'setting', 'setting_order', 'threshold', 'rank', 'mean']
    ].copy()
    return means, ranks


def plot_threshold_rank_heatmap(
    twopart_fold: pd.DataFrame,
    continuous_fold: pd.DataFrame | None,
    outfile: str,
    *,
    predicted_categories: list[str] | None = None,
    modalities: list[str] | None = None,
    metric: str = 'pearson',
    panel_order: list[str] | None = None,
) -> plt.Figure:
    """Heatmap of within-column ranks (1 = best) across continuous + thresholds.

    Rows = profile sources (PRMs / baselines); columns = Continuous and each
    two-stage hurdle threshold. Faceted by dataset/panel × modality (Post/LFC).
    """
    predicted_categories = list(predicted_categories or NEW_MODEL_CATEGORIES)
    modalities = list(
        modalities
        or [m for m in PREDICTED_MODEL_ORDER if m in {'Post+SMILES', 'LFC+SMILES', 'Post', 'LFC'}]
    )
    means, ranks = build_threshold_rank_tables(
        twopart_fold,
        continuous_fold,
        predicted_categories=predicted_categories,
        modalities=modalities,
        metric=metric,
    )
    if ranks.empty:
        raise ValueError('No fold metrics available for threshold rank heatmap')

    csv_path = Path(results_dir) / f'{Path(outfile).stem}_ranks.csv'
    ranks.to_csv(csv_path, index=False)
    logger.info('Wrote %s (%d rows)', csv_path, len(ranks))

    if panel_order is not None:
        datasets = [p for p in panel_order if p in set(ranks['dataset'].astype(str))]
    else:
        datasets = [
            ds
            for ds in ('SciPlex3', 'McFarland')
            if ds in set(ranks['dataset'].astype(str))
        ]
    if not datasets:
        datasets = sorted(ranks['dataset'].astype(str).unique())
    modalities_present = [m for m in modalities if m in set(ranks['modality'].astype(str))]
    if not modalities_present:
        modalities_present = sorted(ranks['modality'].astype(str).unique())

    n_row = len(modalities_present)
    n_col = len(datasets)
    fig, axes = plt.subplots(
        n_row,
        n_col,
        figsize=(4.0 * n_col + 1.2, 0.45 * max(len(predicted_categories), 1) * n_row + 1.8),
        dpi=300,
        squeeze=False,
        sharex=False,
        sharey=True,
    )

    vmax = float(ranks['rank'].max()) if len(ranks) else 1.0
    cmap = sns.color_palette('YlGn_r', as_cmap=True)

    for i, modality in enumerate(modalities_present):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            sub = ranks[
                (ranks['dataset'].astype(str) == dataset)
                & (ranks['modality'].astype(str) == modality)
            ]
            if sub.empty:
                ax.set_visible(False)
                continue
            settings = (
                sub[['setting', 'setting_order']]
                .drop_duplicates()
                .sort_values('setting_order')['setting']
                .tolist()
            )
            models = [m for m in predicted_categories if m in set(sub['model'].astype(str))]
            mat = (
                sub.pivot(index='model', columns='setting', values='rank')
                .reindex(index=models, columns=settings)
            )
            sns.heatmap(
                mat,
                ax=ax,
                annot=True,
                fmt='.0f',
                cmap=cmap,
                vmin=1,
                vmax=vmax,
                cbar=j == n_col - 1,
                cbar_kws={'label': 'Rank (1 = best)', 'shrink': 0.8} if j == n_col - 1 else None,
                linewidths=0.4,
                linecolor='white',
                square=False,
            )
            ax.set_xlabel('')
            ax.set_ylabel(modality if j == 0 else '')
            title = dataset if i == 0 else ''
            if title:
                ax.set_title(title, fontsize=11, fontweight='bold')
            ax.tick_params(axis='x', labelrotation=45, labelsize=8)
            ax.tick_params(axis='y', labelsize=9)

    fig.suptitle(
        f'Within-setting {metric} rank across profile sources',
        fontsize=13,
        fontweight='bold',
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches='tight', dpi=300)
    logger.info('Saved %s', outfile)
    return fig


def plot_threshold_rank_heatmap_stratified(
    outfile: str,
    *,
    suffix: str | None = None,
    predicted_categories: list[str] | None = None,
    align_common_groups: bool = True,
) -> plt.Figure:
    """Rank heatmap for stratified sweep panels (line / tissue / drug)."""
    panels = load_stratified_threshold_sensitivity_panels(
        suffix=suffix,
        align_common_groups=align_common_groups,
    )
    if not panels:
        raise ValueError('No stratified threshold-sweep panels available')

    twopart_parts: list[pd.DataFrame] = []
    cont_parts: list[pd.DataFrame] = []
    panel_order: list[str] = []
    for panel in panels:
        label = str(panel['title']).replace('\n', ' ')
        panel_order.append(label)
        tp = panel['twopart'].copy()
        tp['dataset'] = label
        twopart_parts.append(tp)
        cont = panel.get('continuous')
        if cont is not None and len(cont):
            c = cont.copy()
            c['dataset'] = label
            cont_parts.append(c)

    return plot_threshold_rank_heatmap(
        pd.concat(twopart_parts, ignore_index=True),
        pd.concat(cont_parts, ignore_index=True) if cont_parts else None,
        outfile,
        predicted_categories=predicted_categories,
        panel_order=panel_order,
    )


def _latex_escape(value: object) -> str:
    text = str(value).replace('\n', ' ')
    replacements = {
        '\\': r'\textbackslash{}',
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    out = []
    for ch in text:
        out.append(replacements.get(ch, ch))
    return ''.join(out)


# Display names for stratified summary / comparison LaTeX tables.
_TABLE_DISPLAY_NAMES = {
    'Pre+SMILES': 'ElasticNet',
}


def _table_display_name(value: object) -> str:
    text = str(value).replace('\n', ' ').strip()
    return _TABLE_DISPLAY_NAMES.get(text, text)


def _normalize_table_token(value: object) -> str:
    return str(value).replace('\n', ' ').strip()


def _table_display_modality(modality: str) -> str:
    """Map internal *+SMILES modality labels to *+Morgan for tables."""
    mapping = {
        'Pre+SMILES': 'Pre+Morgan',
        'Post+SMILES': 'Post+Morgan',
        'LFC+SMILES': 'LFC+Morgan',
    }
    return mapping.get(modality, modality)


def _table_model_modality(test_set: object, model: object) -> tuple[str, str]:
    """Map (test_set, model) to LaTeX columns (Model, Modality).

    Column rename relative to the CSV: Source→Model, Model→Modality, with
    Measured / CPA End-to-End display overrides.
    """
    source = _normalize_table_token(test_set)
    modality = _normalize_table_token(model)

    if source in {'CPA End-to-End', 'CPA\\nEnd-to-End'} or (
        'CPA' in source and 'End-to-End' in source
    ):
        return 'CPA End-to-End', '-'

    if source == 'Measured':
        if modality in {'Pre+SMILES', 'Pre+Morgan', 'ElasticNet'}:
            return 'ElasticNet', 'Pre+Morgan'
        if modality in {'Post+SMILES', 'Post+Morgan'}:
            return 'ElasticNet', 'Post+Morgan'
        if modality in {'LFC+SMILES', 'LFC+Morgan'}:
            return 'ElasticNet', 'LFC+Morgan'
        return 'ElasticNet', _table_display_modality(modality)

    return source, _table_display_modality(modality)


def _fmt_mean_sd(mean: float, sd: float, digits: int = 3, *, bold: bool = False) -> str:
    if pd.isna(mean):
        return '--'
    if pd.isna(sd):
        text = f'{mean:.{digits}f}'
    else:
        text = f'{mean:.{digits}f} $\\pm$ {sd:.{digits}f}'
    return rf'\textbf{{{text}}}' if bold else text


def _fmt_ci(
    mean_diff: float,
    lo: float,
    hi: float,
    digits: int = 3,
    *,
    bold: bool = False,
    stacked: bool = False,
) -> str:
    if pd.isna(mean_diff):
        return '--'
    if pd.isna(lo) or pd.isna(hi):
        text = f'{mean_diff:+.{digits}f}'
    elif stacked:
        text = (
            f'{mean_diff:+.{digits}f}'
            f'\\\\{{[}}{lo:.{digits}f}, {hi:.{digits}f}{{]}}'
        )
        text = rf'\shortstack[r]{{{text}}}'
    else:
        text = (
            f'{mean_diff:+.{digits}f} '
            f'[{lo:.{digits}f}, {hi:.{digits}f}]'
        )
    return rf'\textbf{{{text}}}' if bold else text


def _bold_best_mask(
    values: pd.Series,
    *,
    higher_is_better: bool,
) -> pd.Series:
    """True for rows achieving the best finite value (ties all marked)."""
    vals = pd.to_numeric(values, errors='coerce')
    mask = pd.Series(False, index=values.index)
    finite = vals.dropna()
    if finite.empty:
        return mask
    best = finite.max() if higher_is_better else finite.min()
    mask.loc[finite.index[np.isclose(finite.to_numpy(dtype=float), float(best))]] = True
    return mask


STRATUM_CAPTIONS = {
    'pooled': 'Pooled fold-level',
    'per_drug': 'Within-drug (McFarland)',
    'per_cell_line': 'Within-context (tissue / cell line)',
}

# Dataset-specific panels when writing per_cell_line tables (SciPlex vs McFarland).
PER_CELL_LINE_DATASET_PANELS = (
    ('SciPlex3', 'sciplex_per_cell_line', 'Within cell line (SciPlex3)'),
    ('McFarland', 'mcfarland_per_tissue', 'Within tissue (McFarland)'),
)

SUMMARY_METRIC_SPECS = [
    ('pearson', 'Pearson $r$', True),
    ('spearman', r'Spearman $\rho$', True),
    ('rmse', 'RMSE', False),
]
COMPARISON_METRICS = ('pearson', 'spearman', 'rmse')


def _family_slug(family: str) -> str:
    """Filesystem-safe slug for stratified summary TeX / CSV stems."""
    import re

    s = str(family).strip().lower().replace(' ', '_').replace('/', '_').replace('→', 'to')
    s = re.sub(r'[^a-z0-9_]+', '_', s)
    return re.sub(r'_+', '_', s).strip('_') or 'family'


def _fmt_sig_tex(sig: object) -> str:
    text = str(sig) if sig is not None and not (isinstance(sig, float) and np.isnan(sig)) else 'NS'
    if text == '**':
        return r'$^{**}$'
    if text == '*':
        return r'$^{*}$'
    return 'NS'


def _fmt_pvalue(p: object, digits: int = 3) -> str:
    """Format a two-sided p-value for supplement tables."""
    if p is None or (isinstance(p, float) and (np.isnan(p) or not np.isfinite(p))):
        return '--'
    try:
        val = float(p)
    except (TypeError, ValueError):
        return '--'
    if val < 10 ** (-digits):
        return f'{val:.1e}'
    return f'{val:.{digits}g}'


def _fmt_pvalue_with_sig(p: object, sig: object, digits: int = 3) -> str:
    """Format p-value with significance stars attached (tables 8–10 style)."""
    p_tex = _fmt_pvalue(p, digits=digits)
    if p_tex == '--':
        return '--'
    sig_tex = _fmt_sig_tex(sig)
    if sig_tex != 'NS':
        return f'{p_tex}{sig_tex}'
    return p_tex


def _delta_ci_from_summary_row(
    row: pd.Series, metric: str, slug: str
) -> tuple[float, float, float, float, str]:
    return (
        row.get(f'{metric}_vs_{slug}_delta', np.nan),
        row.get(f'{metric}_vs_{slug}_ci_low', np.nan),
        row.get(f'{metric}_vs_{slug}_ci_high', np.nan),
        row.get(f'{metric}_vs_{slug}_pvalue', np.nan),
        str(row.get(f'{metric}_vs_{slug}_sig', 'NS')),
    )


def write_summary_tables_tex(
    summary: pd.DataFrame,
    comparisons: pd.DataFrame | None = None,
    *,
    outdir: str | Path | None = None,
    family: str = 'CPA family',
    wrap_landscape: bool = False,
) -> list[Path]:
    """Write supplement longtables: mean±sd plus paired Δ [95% CI] and $p$/$q$.

    For each stratum and metric (Pearson / Spearman / RMSE), writes one table with
    all models/modalities and paired differences vs Pre+Morgan, Average effect, and
    No effect. Also writes a long-form comparison table per stratum.

    Primary family (CPA Pearson vs Pre+Morgan / Average effect) reports BH-FDR $q$
    within each stratum; all other comparisons report uncorrected exploratory $p$.

    Landscape wrapping is off by default so the supplement can place section
    headings on the same page as the first table.
    """
    outdir = Path(outdir) if outdir is not None else Path(figures_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    slug = _family_slug(family)

    summary = summary[summary['family'].astype(str).eq(family)].copy()
    if summary.empty:
        logger.warning('No summary rows for family=%s; skipping LaTeX export', family)
        return written

    strata = [s for s in ('pooled', 'per_drug', 'per_cell_line') if s in set(summary['stratum'])]
    for extra in sorted(set(summary['stratum']) - set(strata)):
        strata.append(extra)

    def _longtable_block(
        *,
        colspec: str,
        n_cols: int,
        caption: str,
        label: str,
        header: str,
        body_rows: list[str],
        source_comment: str,
        tabcolsep_pt: int = 2,
    ) -> list[str]:
        # If the remaining page fragment is too short for this table (or a clean
        # start), break to a new page first.
        need = min(max(14, 8 + len(body_rows)), 42)
        return [
            source_comment,
            rf'\Needspace{{{need}\baselineskip}}',
            r'\begingroup',
            r'\tiny',
            rf'\setlength{{\tabcolsep}}{{{tabcolsep_pt}pt}}',
            r'\setlength{\LTleft}{\fill}',
            r'\setlength{\LTright}{\fill}',
            rf'\begin{{longtable}}{{{colspec}}}',
            rf'\caption{{{caption}}}',
            rf'\label{{{label}}}\\',
            r'\toprule',
            header + r' \\',
            r'\midrule',
            r'\endfirsthead',
            rf'\caption[]{{{caption} (continued)}}\\',
            r'\toprule',
            header + r' \\',
            r'\midrule',
            r'\endhead',
            r'\midrule',
            rf'\multicolumn{{{n_cols}}}{{r}}{{\textit{{Continued on next page}}}}\\',
            r'\endfoot',
            r'\bottomrule',
            r'\endlastfoot',
            *body_rows,
            r'\end{longtable}',
            r'\endgroup',
            '',
        ]

    # Portrait-friendly widths (stacked Δ/CI; leave room for tabcolsep).
    _ENRICHED_COLSPEC = (
        r'>{\raggedright\arraybackslash}p{0.09\textwidth}'
        r'>{\raggedright\arraybackslash}p{0.11\textwidth}'
        r'>{\raggedright\arraybackslash}p{0.10\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.10\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.11\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.06\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.11\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.06\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.11\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.06\textwidth}'
    )
    _CMP_COLSPEC = (
        r'>{\raggedright\arraybackslash}p{0.08\textwidth}'
        r'>{\raggedright\arraybackslash}p{0.07\textwidth}'
        r'>{\raggedright\arraybackslash}p{0.09\textwidth}'
        r'>{\raggedright\arraybackslash}p{0.08\textwidth}'
        r'>{\raggedright\arraybackslash}p{0.10\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.18\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.08\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.04\textwidth}'
    )

    baseline_specs = (
        ('pre_smiles', r'$\Delta$ vs Pre [95\% CI]', r'$p$/$q$'),
        ('avg_effect', r'$\Delta$ vs Avg [95\% CI]', r'$p$/$q$'),
        ('no_effect', r'$\Delta$ vs Noeff [95\% CI]', r'$p$/$q$'),
    )

    for stratum in strata:
        sub = summary[summary['stratum'] == stratum].copy()
        # SciPlex3 has too few cell lines per drug for stable within-drug metrics.
        if stratum == 'per_drug' and 'dataset' in sub.columns:
            sub = sub[
                ~sub['dataset'].astype(str).str.lower().str.startswith('sciplex')
            ].copy()
        if sub.empty:
            continue

        if stratum == 'per_cell_line' and 'dataset' in sub.columns:
            panels: list[tuple[str, str, pd.DataFrame]] = []
            for ds_name, file_tag, caption_prefix in PER_CELL_LINE_DATASET_PANELS:
                ds_sub = sub[sub['dataset'].astype(str).eq(ds_name)].copy()
                if not ds_sub.empty:
                    panels.append((file_tag, caption_prefix, ds_sub))
            if not panels:
                continue
        else:
            panels = [(stratum, STRATUM_CAPTIONS.get(stratum, stratum), sub)]

        for file_tag, caption_prefix, panel_sub in panels:
            sort_cols = [
                c for c in ('dataset', 'test_set', 'model') if c in panel_sub.columns
            ]
            panel_sub = panel_sub.sort_values(sort_cols).reset_index(drop=True)
            if {'test_set', 'model'}.issubset(panel_sub.columns):
                mapped = panel_sub.apply(
                    lambda r: _table_model_modality(r['test_set'], r['model']),
                    axis=1,
                    result_type='expand',
                )
                mapped.columns = ['_disp_model', '_disp_modality']
                panel_sub = pd.concat([panel_sub, mapped], axis=1)
                panel_sub = panel_sub.sort_values(
                    ['dataset', '_disp_model', '_disp_modality']
                ).reset_index(drop=True)

            for metric, metric_label, _higher in SUMMARY_METRIC_SPECS:
                digits = 2 if metric in ('pearson', 'spearman') else 3
                body: list[str] = []
                for _, row in panel_sub.iterrows():
                    disp_model, disp_modality = _table_model_modality(
                        row.get('test_set', ''), row.get('model', '')
                    )
                    cells = [
                        _latex_escape(row.get('dataset', '')),
                        _latex_escape(disp_model),
                        _latex_escape(disp_modality),
                        _fmt_mean_sd(
                            row.get(f'{metric}_mean'),
                            row.get(f'{metric}_sd'),
                            digits=digits,
                        ),
                    ]
                    for slug_ref, _, _ in baseline_specs:
                        delta, lo, hi, pval, sig = _delta_ci_from_summary_row(
                            row, metric, slug_ref
                        )
                        cells.append(
                            _fmt_ci(delta, lo, hi, digits=digits, stacked=True)
                        )
                        cells.append(_fmt_pvalue_with_sig(pval, sig))
                    body.append(' & '.join(cells) + r' \\')

                header = (
                    'Dataset & Model & Modality & Mean $\\pm$ s.d. & '
                    + ' & '.join(
                        f'{lab} & {pval}' for _, lab, pval in baseline_specs
                    )
                )
                block = _longtable_block(
                    colspec=_ENRICHED_COLSPEC,
                    n_cols=10,
                    caption=(
                        f'{caption_prefix}: {metric_label} '
                        r'(mean $\pm$ s.d.; paired $\Delta$ [95\% CI]. '
                        r'Primary: CPA vs Pre+Morgan / Average effect (Pearson) '
                        r'reports BH-FDR $q$ within stratum; all other $p$ are '
                        r'uncorrected exploratory. '
                        r'Sig: $^{**}$ $<0.01$, $^{*}$ $<0.05$).'
                    ),
                    label=f'tab:predictions-summary-{slug}-{file_tag}-{metric}',
                    header=header,
                    body_rows=body,
                    source_comment=(
                        '% Auto-generated from predictions_stratified_summary_tables.csv '
                        '(primary CPA Pearson vs Pre/Avg = BH-FDR q; else exploratory p)'
                    ),
                    tabcolsep_pt=2,
                )
                lines = (
                    [r'\begin{landscape}', *block, r'\end{landscape}', '']
                    if wrap_landscape
                    else [*block, '']
                )
                path = outdir / (
                    f'predictions_stratified_summary_{slug}_{file_tag}_{metric}.tex'
                )
                path.write_text('\n'.join(lines), encoding='utf-8')
                written.append(path)
                logger.info('Wrote %s (%d rows)', path, len(panel_sub))

            # Backward-compatible alias: pooled multi-metric mean±sd only (no deltas).
            legacy_body: list[str] = []
            for _, row in panel_sub.iterrows():
                disp_model, disp_modality = _table_model_modality(
                    row.get('test_set', ''), row.get('model', '')
                )
                cells = [
                    _latex_escape(row.get('dataset', '')),
                    _latex_escape(disp_model),
                    _latex_escape(disp_modality),
                ]
                for metric, _, _ in SUMMARY_METRIC_SPECS:
                    dig = 2 if metric in ('pearson', 'spearman') else 3
                    cells.append(
                        _fmt_mean_sd(
                            row.get(f'{metric}_mean'),
                            row.get(f'{metric}_sd'),
                            digits=dig,
                        )
                    )
                legacy_body.append(' & '.join(cells) + r' \\')
            legacy_header = (
                'Dataset & Model & Modality & '
                + ' & '.join(label for _, label, _ in SUMMARY_METRIC_SPECS)
            )
            legacy_lines = _longtable_block(
                colspec=(
                    r'>{\raggedright\arraybackslash}p{0.11\textwidth}'
                    r'>{\raggedright\arraybackslash}p{0.17\textwidth}'
                    r'>{\raggedright\arraybackslash}p{0.13\textwidth}'
                    r'>{\raggedleft\arraybackslash}p{0.17\textwidth}'
                    r'>{\raggedleft\arraybackslash}p{0.17\textwidth}'
                    r'>{\raggedleft\arraybackslash}p{0.15\textwidth}'
                ),
                n_cols=6,
                caption=(
                    f'{caption_prefix} '
                    r'(mean $\pm$ s.d.\ across folds).'
                ),
                label=f'tab:predictions-summary-{slug}-{file_tag}',
                header=legacy_header,
                body_rows=legacy_body,
                source_comment=(
                    '% Auto-generated mean±sd overview (see per-metric Δ tables for CIs).'
                ),
                tabcolsep_pt=3,
            )
            legacy_path = (
                outdir / f'predictions_stratified_summary_{slug}_{file_tag}.tex'
            )
            legacy_path.write_text('\n'.join(legacy_lines), encoding='utf-8')
            written.append(legacy_path)

        if comparisons is None or comparisons.empty or 'stratum' not in comparisons.columns:
            continue
        comp = comparisons[
            comparisons['stratum'].eq(stratum)
            & comparisons['family'].astype(str).eq(family)
            & comparisons['metric'].isin(COMPARISON_METRICS)
        ].copy()
        if stratum == 'per_drug' and 'dataset' in comp.columns:
            comp = comp[
                ~comp['dataset'].astype(str).str.lower().str.startswith('sciplex')
            ].copy()
        if comp.empty:
            continue
        sort_comp = [
            c for c in ('dataset', 'metric', 'test_set', 'model', 'baseline')
            if c in comp.columns
        ]
        comp = comp.sort_values(sort_comp).reset_index(drop=True)
        # Display BH-FDR q for primary rows; raw p otherwise (exploratory).
        raw_p = pd.to_numeric(comp.get('t_pvalue', np.nan), errors='coerce')
        fdr_p = (
            pd.to_numeric(comp['t_pvalue_fdr'], errors='coerce')
            if 't_pvalue_fdr' in comp.columns
            else pd.Series(np.nan, index=comp.index)
        )
        if 'is_primary' in comp.columns:
            use_fdr = comp['is_primary'].fillna(False) & fdr_p.notna()
        else:
            use_fdr = fdr_p.notna()
        comp['_p_for_table'] = np.where(use_fdr, fdr_p, raw_p)
        comp['_sig_for_table'] = pd.Series(comp['_p_for_table'], index=comp.index).map(
            lambda p: (
                'NS'
                if p is None or (isinstance(p, float) and np.isnan(p))
                else ('**' if p < 0.01 else ('*' if p < 0.05 else 'NS'))
            )
        )
        p_col = '_p_for_table'
        sig_col = '_sig_for_table'

        cbody: list[str] = []
        for _, row in comp.iterrows():
            n_folds = row.get('n_folds')
            n_str = '--' if pd.isna(n_folds) else str(int(n_folds))
            disp_model, disp_modality = _table_model_modality(
                row.get('test_set', ''), row.get('model', '')
            )
            cells = [
                _latex_escape(row.get('dataset', '')),
                _latex_escape(row.get('metric', '')),
                _latex_escape(disp_model),
                _latex_escape(disp_modality),
                _latex_escape(_normalize_table_token(row.get('baseline', ''))),
                _fmt_ci(row.get('mean_diff'), row.get('ci_low'), row.get('ci_high')),
                _fmt_pvalue_with_sig(row.get(p_col), row.get(sig_col, 'NS')),
                n_str,
            ]
            cbody.append(' & '.join(cells) + r' \\')

        clines = _longtable_block(
            colspec=_CMP_COLSPEC,
            n_cols=8,
            caption=(
                f'Paired fold comparisons vs Pre+Morgan / Average effect / No effect '
                f'({STRATUM_CAPTIONS.get(stratum, stratum).lower()}; '
                r'paired $\Delta$ [95\% CI]. Primary CPA Pearson vs Pre+Morgan / '
                r'Average effect = BH-FDR $q$ within stratum; remaining $p$ are '
                r'uncorrected exploratory; sig: $^{**}$ $<0.01$, $^{*}$ $<0.05$).'
            ),
            label=f'tab:predictions-comparisons-{slug}-{stratum}',
            header=(
                r'Dataset & Metric & Model & Modality & Baseline & '
                r'Mean $\Delta$ [95\% CI] & $p$/$q$ & $n$'
            ),
            body_rows=cbody,
            source_comment=(
                '% Auto-generated from predictions_stratified_paired_comparisons.csv '
                '(primary CPA Pearson vs Pre/Avg = BH-FDR q; else exploratory p)'
            ),
            tabcolsep_pt=2,
        )
        cpath = outdir / f'predictions_stratified_comparisons_{slug}_{stratum}.tex'
        cpath.write_text('\n'.join(clines), encoding='utf-8')
        written.append(cpath)
        logger.info('Wrote %s (%d rows)', cpath, len(comp))

    master = outdir / f'predictions_stratified_summary_tables_{slug}.tex'
    # Prefer enriched per-metric tables in the master input order.
    preferred = [
        p
        for p in written
        if p.name.startswith(f'predictions_stratified_summary_{slug}_')
        and any(p.name.endswith(f'_{m}.tex') for m in COMPARISON_METRICS)
    ]
    preferred += [
        p
        for p in written
        if p.name.startswith(f'predictions_stratified_comparisons_{slug}_')
    ]
    master_lines = [
        '% Auto-generated stratified summary / comparison tables for Supplementary Data.',
        f'% Family={family}; metrics=Pearson/Spearman/RMSE with paired Δ / p vs Pre+Morgan / baselines.',
        '% Requires: booktabs, longtable, pdflscape (landscape).',
        '',
    ]
    for path in preferred:
        master_lines.append(rf'\input{{{path.name}}}')
    master_lines.append('')
    master.write_text('\n'.join(master_lines), encoding='utf-8')
    written.append(master)
    logger.info('Wrote master %s', master)
    return written


def plot_pred_vs_true_by_modality(
    predictions: pd.DataFrame,
    *,
    models: tuple[str, ...] = ('Post+SMILES', 'LFC+SMILES'),
    sources: list[str] | None = None,
    color_by: str = 'test_set',
    outfile: str | Path | None = None,
    title: str | None = None,
    point_size: float = 14,
) -> tuple[plt.Figure, np.ndarray]:
    """True vs predicted sensitivity scatters, one panel per response modality.

    Intended for train-and-test-on-predicted OOF rows (``true`` / ``pred``).
    Points are coloured by ``color_by`` (default: PRM / baseline ``test_set``).
    """
    if predictions is None or predictions.empty:
        raise ValueError('predictions frame is empty')
    need = {'true', 'pred', 'model', color_by}
    missing = need - set(predictions.columns)
    if missing:
        raise ValueError(f'predictions missing columns: {sorted(missing)}')

    df = _rename_labels(predictions.copy())
    if 'train_set' in df.columns:
        df = df.loc[df['train_set'].astype(str).eq('Predicted')].copy()
    model_keys = set(models) | {
        k for k, v in LABEL_MAP.items() if v in models
    }
    df = df.loc[df['model'].astype(str).isin(model_keys)].copy()
    df['model'] = df['model'].astype(str).replace(LABEL_MAP)
    df = df.loc[df['model'].astype(str).isin(models)].copy()
    df = df.dropna(subset=['true', 'pred'])
    if df.empty:
        raise ValueError('No rows left after filtering models / train_set')

    if sources is None:
        sources = [
            s
            for s in (
                'CPA',
                'chemCPA',
                'PRnet',
                'GEARS',
                'scFoundation',
                'Average effect',
                'No effect',
            )
            if s in set(df[color_by].astype(str))
        ]
        extra = sorted(set(df[color_by].astype(str)) - set(sources))
        sources = sources + extra
    else:
        sources = [s for s in sources if s in set(df[color_by].astype(str))]

    palette = dict(
        zip(sources, sns.color_palette('colorblind', n_colors=max(len(sources), 1)))
    )

    n = len(models)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(4.8 * n, 4.6),
        dpi=300,
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, model in zip(axes, models):
        sub = df.loc[df['model'].astype(str).eq(model)].copy()
        for source in sources:
            pts = sub.loc[sub[color_by].astype(str).eq(source)]
            if pts.empty:
                continue
            ax.scatter(
                pts['true'],
                pts['pred'],
                s=point_size,
                alpha=0.7,
                color=palette[source],
                edgecolors='none',
                label=source,
                zorder=2,
            )

        lo = float(min(sub['true'].min(), sub['pred'].min(), 0.0))
        hi = float(max(sub['true'].max(), sub['pred'].max(), 1.0))
        pad = 0.03 * (hi - lo if hi > lo else 1.0)
        lim = (lo - pad, hi + pad)
        ax.plot(lim, lim, color='0.4', lw=1.0, ls='--', zorder=1)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect('equal', adjustable='box')

        r = pearsonr(sub['true'], sub['pred'])[0]
        ax.text(
            0.04,
            0.96,
            f'r = {r:.2f}\nn = {len(sub)}',
            transform=ax.transAxes,
            va='top',
            ha='left',
            fontsize=11,
            bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'none', 'pad': 2},
        )
        ax.set_xlabel('True sensitivity')
        ax.set_ylabel('Predicted sensitivity' if ax is axes[0] else '')
        ax.set_title(model, fontsize=14, fontweight='bold')
        sns.despine(ax=ax)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title='Profile source',
            loc='center left',
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            fontsize=9,
            title_fontsize=10,
            markerscale=1.4,
        )
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    if outfile is not None:
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, bbox_inches='tight')
        logger.info('Saved %s', outfile)
    return fig, axes


def plot_cpa_end_to_end_scatter(
    data_dir: str | Path | None = None,
    *,
    outfile: str | None = None,
    datasets: tuple[str, ...] = ('sciplex', 'mcfarland'),
) -> tuple[plt.Figure, np.ndarray]:
    """True vs predicted CPA end-to-end sensitivity, colored by tissue.

    Side-by-side SciPlex | McFarland panels. Points are OOF labeled pairs;
    identity line and pooled Pearson r are annotated per panel.
    """
    if data_dir is None:
        data_dir = Path(config.DATA_DIR)
    data_dir = Path(data_dir)

    n = len(datasets)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(5.2 * n, 4.8),
        dpi=300,
        squeeze=False,
    )
    axes = axes.ravel()
    labels = {'sciplex': 'SciPlex3', 'mcfarland': 'McFarland'}

    for ax, dataset in zip(axes, datasets):
        preds = ev.load_cpa_end_to_end_predictions(dataset, data_dir)
        preds = preds.dropna(subset=['true', 'pred']).copy()
        if 'tissue' not in preds.columns or preds['tissue'].isna().all():
            preds = ev.attach_tissue_column(preds, dataset)

        tissues = sorted(preds['tissue'].astype(str).unique())
        # Colorblind-friendly cycle; husl for many McFarland tissues.
        if len(tissues) <= 8:
            palette = dict(zip(tissues, sns.color_palette('colorblind', n_colors=len(tissues))))
        else:
            palette = dict(zip(tissues, sns.color_palette('husl', n_colors=len(tissues))))

        for tissue in tissues:
            sub = preds[preds['tissue'].astype(str) == tissue]
            ax.scatter(
                sub['true'],
                sub['pred'],
                s=18 if dataset == 'sciplex' else 12,
                alpha=0.75,
                color=palette[tissue],
                edgecolors='none',
                label=tissue,
                zorder=2,
            )

        lo = float(min(preds['true'].min(), preds['pred'].min(), 0.0))
        hi = float(max(preds['true'].max(), preds['pred'].max(), 1.0))
        pad = 0.03 * (hi - lo if hi > lo else 1.0)
        lim = (lo - pad, hi + pad)
        ax.plot(lim, lim, color='0.4', lw=1.0, ls='--', zorder=1)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect('equal', adjustable='box')

        r = pearsonr(preds['true'], preds['pred'])[0]
        ax.text(
            0.04,
            0.96,
            f'r = {r:.2f}\nn = {len(preds)}',
            transform=ax.transAxes,
            va='top',
            ha='left',
            fontsize=11,
            bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none', 'pad': 2},
        )
        ax.set_xlabel('True sensitivity')
        ax.set_ylabel('Predicted sensitivity' if ax is axes[0] else '')
        ax.set_title(labels.get(dataset, dataset), fontsize=14, fontweight='bold')
        sns.despine(ax=ax)

        ncol = 1 if len(tissues) <= 6 else 2
        ax.legend(
            title='Tissue',
            fontsize=7 if len(tissues) > 8 else 9,
            title_fontsize=9,
            frameon=False,
            loc='upper left',
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            ncol=ncol,
            markerscale=1.2,
            handletextpad=0.3,
            columnspacing=0.8,
        )

    fig.tight_layout()
    if outfile is None:
        outfile = os.path.join(figures_dir, 'cpa_end_to_end_scatter_by_tissue.pdf')
    fig.savefig(outfile, bbox_inches='tight')
    logger.info('Saved %s', outfile)
    return fig, axes


def _predictions_have_models(df: pd.DataFrame | None, models: list[str]) -> bool:
    if df is None or df.empty or 'test_set' not in df.columns:
        return False
    labeled = set(_rename_labels(df[['test_set']].copy())['test_set'].astype(str))
    return any(m in labeled for m in models)


def _gears_family_predictions(
    split_cv: pd.DataFrame | None,
    legacy: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Prefer split-CV GEARS/scFoundation; fall back to legacy self-trained tables."""
    if _predictions_have_models(split_cv, ['GEARS', 'scFoundation']):
        return split_cv
    return legacy


def main() -> None:
    split_sciplex = _load_split_cv_predictions('sciplex')
    split_mcfarland = _load_split_cv_predictions('mcfarland')
    _run_family(
        split_sciplex,
        split_mcfarland,
        NEW_MODEL_CATEGORIES,
        'predictions_cpa_chemcpa_prnet',
    )
    _run_family(
        _gears_family_predictions(split_sciplex, _load_legacy_sciplex_predictions()),
        _gears_family_predictions(split_mcfarland, _load_legacy_mcfarland_predictions()),
        GEARS_MODEL_CATEGORIES,
        'predictions_gears_scfoundation',
    )

    smiles_sciplex = _load_split_cv_predictions('sciplex', suffix='_smiles')
    smiles_mcfarland = _load_split_cv_predictions('mcfarland', suffix='_smiles')
    if smiles_sciplex is not None or smiles_mcfarland is not None:
        logger.info(
            'Train/test-on-predicted figures from *_split_cv_smiles_predictions.csv'
        )
        _run_family(
            smiles_sciplex,
            smiles_mcfarland,
            NEW_MODEL_CATEGORIES,
            'predictions_cpa_chemcpa_prnet_smiles',
            plot_end_to_end=True,
            inject_measured_t1=True,
        )
        if _predictions_have_models(smiles_sciplex, ['GEARS', 'scFoundation']) or _predictions_have_models(
            smiles_mcfarland, ['GEARS', 'scFoundation']
        ):
            _run_family(
                smiles_sciplex,
                smiles_mcfarland,
                GEARS_MODEL_CATEGORIES,
                'predictions_gears_scfoundation_smiles',
                plot_end_to_end=True,
                inject_measured_t1=True,
            )
        else:
            logger.warning(
                'Skipping GEARS-family ± SMILES figure; split-CV smiles outputs lack '
                'GEARS/scFoundation (rerun with PREDICTED_MODELS including them)'
            )
    else:
        logger.warning(
            'Skipping predicted-profile ± SMILES figure; no *_split_cv_smiles_predictions.csv found'
        )

    transfer_sciplex = _load_split_cv_predictions(
        'sciplex', suffix='_train_measured_test_predicted'
    )
    transfer_mcfarland = _load_split_cv_predictions(
        'mcfarland', suffix='_train_measured_test_predicted'
    )
    if transfer_sciplex is not None or transfer_mcfarland is not None:
        logger.info(
            'Train-Measured/test-predicted figures from '
            '*_split_cv_train_measured_test_predicted_predictions.csv'
        )
        _run_family(
            transfer_sciplex,
            transfer_mcfarland,
            NEW_MODEL_CATEGORIES,
            'predictions_train_measured_test_predicted',
            plot_end_to_end=True,
            # Transfer CSVs already include Measured→Measured; keep those and
            # only fill missing modalities from measured-task T1.
            inject_measured_t1=True,
        )
        if _predictions_have_models(transfer_sciplex, ['GEARS', 'scFoundation']) or _predictions_have_models(
            transfer_mcfarland, ['GEARS', 'scFoundation']
        ):
            _run_family(
                transfer_sciplex,
                transfer_mcfarland,
                GEARS_MODEL_CATEGORIES,
                'predictions_train_measured_test_predicted_gears_scfoundation',
                plot_end_to_end=True,
                inject_measured_t1=True,
            )
    else:
        logger.warning(
            'Skipping train-Measured/test-predicted figure; '
            'no *_split_cv_train_measured_test_predicted_predictions.csv found'
        )

    plot_cpa_end_to_end_scatter()

    # Re-export portrait supplement tables from existing summary CSVs when present.
    summary_csv = Path(results_dir) / 'predictions_stratified_summary_tables.csv'
    if summary_csv.exists():
        summary = pd.read_csv(summary_csv)
        comparisons_csv = Path(results_dir) / 'predictions_stratified_paired_comparisons.csv'
        comparisons = (
            pd.read_csv(comparisons_csv) if comparisons_csv.exists() else None
        )
        for family in sorted(summary['family'].astype(str).unique()):
            write_summary_tables_tex(
                summary,
                comparisons,
                outdir=figures_dir,
                family=family,
                wrap_landscape=False,
            )
        logger.info(
            'Wrote portrait stratified summary TeX for families in %s', summary_csv
        )


if __name__ == '__main__':
    main()
