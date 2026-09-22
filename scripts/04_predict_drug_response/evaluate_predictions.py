"""Stratified drug-response metrics, paired CIs, and formal comparisons.

Pooled Pearson/MSE can be driven by average drug potency or cell-line baseline
shifts. This module reports within-drug and within-context ranking (cell line for
SciPlex, tissue for McFarland), plus Spearman, MAE/RMSE, calibration, and
classification metrics. Stratified boxplots use one point per group with OOF
predictions pooled across CV folds (not one point per fold).
"""

from __future__ import annotations

import logging
import warnings
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

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

CPA_E2E_TEST_SET = 'CPA\nEnd-to-End'
GEARS_E2E_TEST_SET = 'GEARS\nEnd-to-End'
END_TO_END_TEST_SETS = frozenset({CPA_E2E_TEST_SET, GEARS_E2E_TEST_SET})
CPA_E2E_MODEL = 'Embedding'

SENSITIVE_THRESHOLD = 0.02
GROUP_KEYS = ['dataset', 'model', 'test_set', 'train_set', 'split', 'head', 'threshold']
POOL_KEYS = ['dataset', 'model', 'test_set', 'train_set', 'head', 'threshold']
MIN_GROUP_N = 3
BASELINES = ('Average effect', 'No effect')
# Genetic PRMs are omitted when intersecting stratified groups so chemical-family
# models (Measured / CPA / chemCPA / PRnet / baselines) share one group set.
# GEARS / scFoundation cover far fewer McFarland cell lines; including them in
# the intersection would shrink every panel to that sparse subset.
GENETIC_PRM_TEST_SETS = frozenset({
    'GEARS',
    'scFoundation',
    'GEARS\nOptimized',
    GEARS_E2E_TEST_SET,
})



def context_group_column(dataset: str) -> str:
    """Column for within-context metrics: ``tissue`` (McFarland) or ``cell_line`` (SciPlex)."""
    label = str(dataset).strip().lower()
    if 'mcfarland' in label:
        return 'tissue'
    return 'cell_line'


def _as_float(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _finite_pairs(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep only labeled pairs; unlabeled McFarland e2e rows have NaN ``sens_true``."""
    y = _as_float(y)
    p = _as_float(p)
    mask = np.isfinite(y) & np.isfinite(p)
    return y[mask], p[mask]


def _parse_cond_line_index(idx: pd.Index | pd.Series) -> tuple[pd.Series, pd.Series] | None:
    labels = idx.astype(str)
    if not labels.str.contains('_').any() or labels.str.match(r'^\d+$').all():
        return None
    parts = labels.str.rsplit('_', n=1)
    return parts.str[0], parts.str[-1].str.upper()


def attach_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure cell_line and condition exist; parse SciPlex ``cond_LINE`` index if needed."""
    out = df.copy()
    has_line = 'cell_line' in out.columns and out['cell_line'].notna().any()
    has_cond = 'condition' in out.columns and out['condition'].notna().any()
    if has_line and has_cond:
        return out
    if 'index' in out.columns:
        parsed = _parse_cond_line_index(out['index'])
        if parsed is not None:
            out['condition'], out['cell_line'] = parsed
            return out
    parsed = _parse_cond_line_index(out.index)
    if parsed is not None:
        out['condition'], out['cell_line'] = parsed
    return out


def normalize_prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonize split-CV and legacy GEARS/scFoundation prediction tables."""
    out = df.copy()
    if 'profile_source' in out.columns and 'test_set' not in out.columns:
        out['test_set'] = out['profile_source']
    out = attach_group_columns(out)
    out = out.replace(LABEL_MAP)
    if 'train_set' not in out.columns:
        out['train_set'] = np.where(out['test_set'].eq('Measured'), 'Measured', 'Predicted')
    else:
        out['train_set'] = out['train_set'].replace(LABEL_MAP)
    if 'trained_on' not in out.columns:
        if 'trained_on_file' in out.columns:
            out['trained_on'] = out['trained_on_file']
        else:
            out['trained_on'] = np.where(out['test_set'].eq('Measured'), 'Measured', 'Predicted')
    out['model'] = out['model'].replace(LABEL_MAP)
    out['test_set'] = out['test_set'].replace(LABEL_MAP)
    return out


def _safe_corr(y: np.ndarray, p: np.ndarray, method: str) -> float:
    y, p = _finite_pairs(y, p)
    if len(y) < 2 or np.nanstd(y) < 1e-12 or np.nanstd(p) < 1e-12:
        return 0.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with np.errstate(invalid='ignore'):
                if method == 'pearson':
                    val = stats.pearsonr(y, p)[0]
                else:
                    val = stats.spearmanr(y, p)[0]
    except Exception:
        return 0.0
    return 0.0 if val is None or np.isnan(val) else float(val)


def _binary_labels(y: np.ndarray, threshold: float = SENSITIVE_THRESHOLD) -> np.ndarray:
    """Sensitive = true sensitivity above ``threshold`` (default 0.02)."""
    return (y > threshold).astype(int)


def _safe_auroc(y: np.ndarray, p: np.ndarray, threshold: float = SENSITIVE_THRESHOLD) -> float:
    y, p = _finite_pairs(y, p)
    if len(y) < MIN_GROUP_N or np.nanstd(p) < 1e-12:
        return np.nan
    y_bin = _binary_labels(y, threshold=threshold)
    if y_bin.min() == y_bin.max():
        return np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return float(roc_auc_score(y_bin, p))
    except ValueError:
        return np.nan


def _group_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y, p = _finite_pairs(y, p)
    n = int(len(y))
    mae = float(mean_absolute_error(y, p)) if n else np.nan
    mse = float(mean_squared_error(y, p)) if n else np.nan
    rmse = float(np.sqrt(mse)) if n else np.nan
    if n >= 3 and np.std(p) > 0:
        try:
            slope, intercept, _, _, _ = stats.linregress(p, y)
        except ValueError:
            slope, intercept = np.nan, np.nan
    else:
        slope, intercept = np.nan, np.nan

    y_bin = _binary_labels(y)
    auroc = auprc = f1 = acc = bacc = np.nan
    if n >= MIN_GROUP_N and y_bin.min() != y_bin.max() and np.std(p) > 0:
        auroc = _safe_auroc(y, p)
        try:
            auprc = float(average_precision_score(y_bin, p))
        except ValueError:
            auprc = np.nan
        p_bin = (p > SENSITIVE_THRESHOLD).astype(int)
        f1 = float(f1_score(y_bin, p_bin, zero_division=0))
        acc = float(accuracy_score(y_bin, p_bin))
        bacc = float(balanced_accuracy_score(y_bin, p_bin))

    calib_mae = np.nan
    if n >= 10 and np.std(p) > 0:
        try:
            bins = pd.qcut(p, q=5, duplicates='drop')
            calib = pd.DataFrame({'y': y, 'p': p, 'bin': bins}).groupby('bin', observed=True).mean()
            if len(calib) >= 2:
                calib_mae = float(mean_absolute_error(calib['y'], calib['p']))
        except (ValueError, TypeError):
            calib_mae = np.nan

    return {
        'n': n,
        'pearson': _safe_corr(y, p, 'pearson'),
        'spearman': _safe_corr(y, p, 'spearman'),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'calibration_slope': float(slope),
        'calibration_intercept': float(intercept),
        'calibration_mae': calib_mae,
        'auroc': auroc,
        'auprc': auprc,
        'f1': f1,
        'accuracy': acc,
        'balanced_accuracy': bacc,
    }


def _present_keys(df: pd.DataFrame, extra: Iterable[str] = ()) -> list[str]:
    keys = [k for k in GROUP_KEYS if k in df.columns]
    for col in extra:
        if col in df.columns and col not in keys:
            keys.append(col)
    return keys


def compute_fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, test_set, train_set, split) with pooled metrics."""
    df = normalize_prediction_frame(predictions)
    keys = _present_keys(df)

    rows = []
    for key, group in df.groupby(keys, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        record = dict(zip(keys, key))
        record.update(_group_metrics(_as_float(group['true']), _as_float(group['pred'])))
        if 'trained_on' in group.columns:
            record['trained_on'] = group['trained_on'].iloc[0]
        rows.append(record)
    return pd.DataFrame(rows)


def compute_grouped_metrics(
    predictions: pd.DataFrame,
    group_col: str,
    *,
    pool_folds: bool = True,
) -> pd.DataFrame:
    """Metrics within each drug, cell line, or tissue.

    When ``pool_folds`` is True (default), OOF predictions are pooled across CV
    splits so each group contributes a single metric value (e.g. three SciPlex
    cell lines → three points). Set ``pool_folds=False`` to keep the legacy
    per-fold × per-group table.
    """
    df = normalize_prediction_frame(predictions)
    if group_col not in df.columns:
        return pd.DataFrame()
    if pool_folds:
        keys = [k for k in POOL_KEYS if k in df.columns] + [group_col]
    else:
        keys = _present_keys(df, extra=[group_col])
    rows = []
    for key, group in df.groupby(keys, dropna=False):
        if len(group) < MIN_GROUP_N:
            continue
        key = key if isinstance(key, tuple) else (key,)
        record = dict(zip(keys, key))
        record.update(_group_metrics(_as_float(group['true']), _as_float(group['pred'])))
        if 'trained_on' in group.columns:
            record['trained_on'] = group['trained_on'].iloc[0]
        record['stratum'] = group_col
        rows.append(record)
    return pd.DataFrame(rows)


def compute_grouped_auroc(
    predictions: pd.DataFrame,
    group_col: str,
    *,
    pool_folds: bool = True,
) -> pd.DataFrame:
    """Within-drug or within-context AUROC using sensitivity threshold 0.02."""
    df = normalize_prediction_frame(predictions)
    if group_col not in df.columns:
        return pd.DataFrame()
    if pool_folds:
        keys = [k for k in POOL_KEYS if k in df.columns] + [group_col]
    else:
        keys = _present_keys(df, extra=[group_col])
    rows = []
    for key, group in df.groupby(keys, dropna=False):
        if len(group) < MIN_GROUP_N:
            continue
        key = key if isinstance(key, tuple) else (key,)
        record = dict(zip(keys, key))
        record['auroc'] = _safe_auroc(_as_float(group['true']), _as_float(group['pred']))
        if 'trained_on' in group.columns:
            record['trained_on'] = group['trained_on'].iloc[0]
        record['stratum'] = group_col
        rows.append(record)
    return pd.DataFrame(rows)


def fold_mean_from_grouped_table(
    grouped: pd.DataFrame,
    metrics: tuple[str, ...] = ('pearson', 'spearman', 'rmse', 'auroc'),
) -> pd.DataFrame:
    """Average within-drug or within-line metrics to one row per fold."""
    if grouped is None or grouped.empty:
        return pd.DataFrame()
    keys = _present_keys(grouped)
    present = [m for m in metrics if m in grouped.columns]
    if not present:
        return pd.DataFrame()
    out = grouped.groupby(keys, dropna=False)[present].mean().reset_index()
    n_groups = grouped.groupby(keys, dropna=False).size().reset_index(name='n_groups')
    out = out.merge(n_groups, on=keys, how='left')
    if 'trained_on' in grouped.columns:
        trained = grouped.groupby(keys, dropna=False)['trained_on'].first().reset_index()
        out = out.merge(trained, on=keys, how='left')
    # Undefined within-group correlations (constant preds / too few points) -> 0.
    for metric in ('pearson', 'spearman'):
        if metric in out.columns:
            out[metric] = out[metric].fillna(0.0)
    return out


def fold_mean_grouped_metric(
    predictions: pd.DataFrame,
    group_col: str,
    metric: str = 'pearson',
) -> pd.DataFrame:
    """Mean within-drug or within-line metric, one value per fold (boxplot-ready)."""
    return fold_mean_from_grouped_table(
        compute_grouped_metrics(predictions, group_col),
        metrics=(metric,),
    )


def fold_mean_grouped_metrics(
    predictions: pd.DataFrame,
    group_col: str,
    metrics: tuple[str, ...] = ('pearson', 'spearman', 'rmse', 'auroc'),
) -> pd.DataFrame:
    """Legacy helper: within-group metrics computed per fold, then averaged across groups."""
    return fold_mean_from_grouped_table(
        compute_grouped_metrics(predictions, group_col, pool_folds=False),
        metrics=metrics,
    )


def grouped_metrics_boxplot_frame(
    predictions: pd.DataFrame,
    group_col: str,
    *,
    align_common_groups: bool = False,
    exclude_test_sets: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Within-drug / within-context metrics with one boxplot point per group.

    OOF predictions are pooled across CV folds. ``split`` is set to the group
    label so paired comparisons align on the same drug/line/tissue. Sources with
    no eligible groups (``< MIN_GROUP_N`` samples) get a single zero-filled row
    per (model, test_set) so boxes still appear.

    When ``align_common_groups`` is True, keep only groups that are eligible for
    every (test_set, model) panel except those in ``exclude_test_sets``
    (defaults to genetic PRMs: GEARS / scFoundation).
    """
    df = normalize_prediction_frame(predictions)
    pool_keys = [k for k in POOL_KEYS if k in df.columns]
    skeleton_cols = list(pool_keys)
    if 'trained_on' in df.columns:
        skeleton_cols.append('trained_on')
    skeleton = df[skeleton_cols].drop_duplicates(subset=pool_keys).copy()

    grouped = compute_grouped_metrics(df, group_col, pool_folds=True)
    if grouped.empty:
        out = skeleton.copy()
        out['split'] = 'none'
        out[group_col] = 'none'
        out['pearson'] = 0.0
        out['spearman'] = 0.0
        out['n_groups'] = 0
    else:
        out = grouped.copy()
        out['split'] = out[group_col].astype(str)
        out['n_groups'] = 1

    if 'trained_on' not in out.columns:
        out['trained_on'] = np.where(out['test_set'].eq('Measured'), 'Measured', 'Predicted')
    if 'pearson' not in out.columns:
        out['pearson'] = 0.0
    else:
        out['pearson'] = out['pearson'].fillna(0.0)
    if 'spearman' not in out.columns:
        out['spearman'] = 0.0
    else:
        out['spearman'] = out['spearman'].fillna(0.0)
    if 'n_groups' in out.columns:
        out['n_groups'] = out['n_groups'].fillna(0).astype(int)
    if align_common_groups:
        out = align_stratified_metric_groups(
            out, exclude_test_sets=exclude_test_sets
        )
    return out


def align_stratified_metric_groups(
    results: pd.DataFrame,
    *,
    exclude_test_sets: Iterable[str] | None = None,
    split_col: str = 'split',
) -> pd.DataFrame:
    """Restrict stratified metric rows to groups shared across chemical-family panels.

    For every ``(test_set, model)`` present in ``results`` whose ``test_set`` is
    not in ``exclude_test_sets`` (default: GEARS / scFoundation), collect the
    set of group labels in ``split_col``, then keep only the intersection.
    Excluded genetic-PRM panels are also filtered to that common set when they
    appear on the same figure (they may still have fewer points if coverage is
    incomplete).
    """
    if results is None or results.empty:
        return results
    if split_col not in results.columns or 'test_set' not in results.columns:
        return results

    exclude = {
        str(x) for x in (
            GENETIC_PRM_TEST_SETS if exclude_test_sets is None else exclude_test_sets
        )
    }
    frame = results.copy()
    split = frame[split_col].astype(str)
    usable = frame.loc[split.ne('none') & split.ne('nan') & split.notna()].copy()
    if usable.empty:
        return frame

    panels = usable.loc[~usable['test_set'].astype(str).isin(exclude)]
    if panels.empty:
        logger.warning(
            'align_stratified_metric_groups: no non-excluded panels; leaving unaligned'
        )
        return frame

    group_sets: list[set[str]] = []
    panel_keys: list[tuple[str, str]] = []
    model_col = 'model' if 'model' in panels.columns else None
    group_cols = ['test_set'] + ([model_col] if model_col else [])
    for keys, sub in panels.groupby(group_cols, dropna=False):
        groups = set(sub[split_col].astype(str))
        if not groups:
            continue
        group_sets.append(groups)
        if model_col:
            ts, model = keys if isinstance(keys, tuple) else (keys, '')
            panel_keys.append((str(ts), str(model)))
        else:
            panel_keys.append((str(keys), ''))

    if not group_sets:
        return frame

    common = set.intersection(*group_sets)
    logger.info(
        'align_stratified_metric_groups: %d common groups across %d panels '
        '(excluded test_sets=%s); example panels=%s',
        len(common),
        len(group_sets),
        sorted(exclude),
        panel_keys[:6],
    )
    if not common:
        logger.warning(
            'align_stratified_metric_groups: empty intersection across %d panels; '
            'leaving unaligned',
            len(group_sets),
        )
        return frame

    keep = frame[split_col].astype(str).isin(common)
    return frame.loc[keep].reset_index(drop=True).copy()


def _mean_sd(values: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def _fmt_mean_sd(values: Iterable[float], digits: int = 3) -> str:
    mean, sd = _mean_sd(np.asarray(list(values), dtype=float))
    if np.isnan(mean):
        return 'NA'
    return f'{mean:.{digits}f} +/- {sd:.{digits}f}'


def _fmt_folds(values: Iterable[float], digits: int = 3) -> str:
    vals = [v for v in values if not (v is None or (isinstance(v, float) and np.isnan(v)))]
    if not vals:
        return 'NA'
    return ', '.join(f'{float(v):.{digits}f}' for v in vals)


def paired_difference(
    model_vals: np.ndarray,
    baseline_vals: np.ndarray,
    higher_is_better: bool,
) -> dict[str, float]:
    """Paired fold difference (model − baseline) with t-based 95% CI."""
    a = np.asarray(model_vals, dtype=float)
    b = np.asarray(baseline_vals, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    d = a[mask] - b[mask]
    n = int(len(d))
    out = {
        'n_folds': n,
        'mean_diff': np.nan,
        'ci_low': np.nan,
        'ci_high': np.nan,
        't_pvalue': np.nan,
        'wilcoxon_pvalue': np.nan,
        'higher_is_better': higher_is_better,
    }
    if n == 1:
        out['mean_diff'] = float(d[0])
        return out
    if n < 2:
        return out
    mean = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(n))
    tcrit = float(stats.t.ppf(0.975, n - 1))
    out['mean_diff'] = mean
    out['ci_low'] = mean - tcrit * se
    out['ci_high'] = mean + tcrit * se
    try:
        out['t_pvalue'] = float(stats.ttest_rel(a[mask], b[mask]).pvalue)
    except ValueError:
        out['t_pvalue'] = np.nan
    if np.allclose(d, 0):
        out['wilcoxon_pvalue'] = 1.0
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                out['wilcoxon_pvalue'] = float(stats.wilcoxon(d, zero_method='wilcox').pvalue)
        except ValueError:
            out['wilcoxon_pvalue'] = np.nan
    return out


def significance_stars(p: float | None) -> str:
    """Map a p-value to ``**`` / ``*`` / ``NS`` (``**`` if p < 0.01, ``*`` if p < 0.05)."""
    if p is None or (isinstance(p, (float, np.floating)) and np.isnan(p)):
        return 'NS'
    if float(p) < 0.01:
        return '**'
    if float(p) < 0.05:
        return '*'
    return 'NS'


def _panel_rows(fold_metrics: pd.DataFrame, predicted_categories: list[str]) -> pd.DataFrame:
    if fold_metrics is None or fold_metrics.empty:
        return pd.DataFrame()
    frame = fold_metrics.copy()
    if 'trained_on' not in frame.columns:
        frame['trained_on'] = np.where(frame['test_set'].eq('Measured'), 'Measured', 'Predicted')
    measured = frame[
        (frame['trained_on'] == 'Measured')
        & (frame['test_set'] == 'Measured')
        & (frame['model'].isin(['Pre', 'Pre+SMILES', 'Post', 'Post+SMILES', 'LFC', 'LFC+SMILES']))
    ]
    predicted = frame[
        (frame['trained_on'] == 'Predicted')
        & (frame['test_set'].isin(predicted_categories))
        & (frame['model'].isin(['Post', 'Post+SMILES', 'LFC', 'LFC+SMILES']))
    ]
    e2e = frame[frame['test_set'].isin(END_TO_END_TEST_SETS)]
    parts = [measured, predicted]
    if not e2e.empty:
        parts.append(e2e)
    return pd.concat(parts, ignore_index=True)


def _align_folds(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot metric to one column per test_set, rows = model × split."""
    cols = ['model', 'split', 'test_set', metric]
    sub = frame[cols].dropna(subset=[metric])
    return sub.pivot_table(index=['model', 'split'], columns='test_set', values=metric)


def _metric_by_split(frame: pd.DataFrame, metric: str) -> pd.Series:
    """One metric value per ``split`` (mean if duplicates)."""
    if frame is None or frame.empty or metric not in frame.columns or 'split' not in frame.columns:
        return pd.Series(dtype=float)
    sub = frame.copy()
    sub[metric] = pd.to_numeric(sub[metric], errors='coerce')
    return sub.groupby(sub['split'].astype(str), dropna=False)[metric].mean()


def _pre_smiles_metric_frame(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    frame = fold_metrics
    if 'trained_on' in frame.columns:
        return frame[
            (frame['trained_on'] == 'Measured')
            & (frame['test_set'] == 'Measured')
            & (frame['model'] == 'Pre+SMILES')
        ]
    return frame[(frame['test_set'] == 'Measured') & (frame['model'] == 'Pre+SMILES')]


def _average_effect_metric_frame(fold_metrics: pd.DataFrame, model: str) -> pd.DataFrame:
    """Average-effect rows for ``model``, with SMILES→gene modality fallback."""
    return _effect_baseline_metric_frame(fold_metrics, 'Average effect', model)


def _effect_baseline_metric_frame(
    fold_metrics: pd.DataFrame,
    baseline_test_set: str,
    model: str,
) -> pd.DataFrame:
    """Baseline source rows (Average / No effect) for ``model``, with modality fallback."""
    base = fold_metrics[fold_metrics['test_set'] == baseline_test_set]
    if base.empty:
        return base
    if model in set(base['model'].astype(str)):
        return base[base['model'] == model]
    fallback = {'LFC+SMILES': 'LFC', 'Post+SMILES': 'Post'}.get(str(model))
    if fallback is not None and fallback in set(base['model'].astype(str)):
        return base[base['model'] == fallback]
    for cand in ('LFC+SMILES', 'LFC', 'Post+SMILES', 'Post'):
        if cand in set(base['model'].astype(str)):
            return base[base['model'] == cand]
    return base.iloc[0:0]


def _paired_stats_vs_frame(
    model_frame: pd.DataFrame,
    ref_frame: pd.DataFrame,
    metric: str,
    *,
    higher_is_better: bool,
) -> dict:
    a = _metric_by_split(model_frame, metric)
    b = _metric_by_split(ref_frame, metric)
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return paired_difference(np.array([]), np.array([]), higher_is_better)
    return paired_difference(
        a.loc[common].to_numpy(dtype=float),
        b.loc[common].to_numpy(dtype=float),
        higher_is_better=higher_is_better,
    )


def _add_significance_columns(comparisons: pd.DataFrame) -> pd.DataFrame:
    if comparisons is None or comparisons.empty:
        return comparisons if comparisons is not None else pd.DataFrame()
    out = comparisons.copy()
    if 't_pvalue' in out.columns:
        out['sig'] = out['t_pvalue'].map(significance_stars)
    else:
        out['sig'] = 'NS'
    if 't_pvalue_fdr' in out.columns:
        out['sig_fdr'] = out['t_pvalue_fdr'].map(significance_stars)
    return out


# Primary inference family for predicted-profile claims (Figure 2): CPA Pearson vs
# Pre+Morgan and Average effect. Correct BH within each stratum call; all other
# PRMs / baselines / metrics are exploratory (uncorrected).
PRIMARY_PREDICTED_TEST_SETS = frozenset({'CPA'})
PRIMARY_PREDICTED_BASELINES = frozenset({'Pre+SMILES', 'Average effect'})
PRIMARY_PREDICTED_METRIC = 'pearson'


def _mark_primary_predicted_comparisons(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Flag primary CPA Pearson vs Pre / Average-effect rows."""
    out = comparisons.copy()
    if out.empty:
        out['is_primary'] = pd.Series(dtype=bool)
        return out
    out['is_primary'] = (
        out['metric'].astype(str).eq(PRIMARY_PREDICTED_METRIC)
        & out['test_set'].astype(str).isin(PRIMARY_PREDICTED_TEST_SETS)
        & out['baseline'].astype(str).isin(PRIMARY_PREDICTED_BASELINES)
    )
    return out


def _apply_primary_bh_fdr(comparisons: pd.DataFrame) -> pd.DataFrame:
    """BH-FDR only on ``is_primary`` rows (one family per comparisons frame)."""
    out = comparisons.copy()
    out['t_pvalue_fdr'] = np.nan
    if out.empty or 'is_primary' not in out.columns or 't_pvalue' not in out.columns:
        return out
    mask = out['is_primary'].fillna(False) & out['t_pvalue'].notna()
    if mask.any():
        out.loc[mask, 't_pvalue_fdr'] = multipletests(
            out.loc[mask, 't_pvalue'], alpha=0.05, method='fdr_bh'
        )[1]
    return out


def compute_paired_comparisons_vs_references(
    fold_metrics: pd.DataFrame,
    metric: str,
    predicted_categories: list[str],
    higher_is_better: bool,
    *,
    references: tuple[str, ...] = ('Pre+SMILES', 'Average effect', 'No effect'),
) -> pd.DataFrame:
    """Paired mean Δ vs Pre+SMILES / Average effect / No effect (matched on ``split``).

    Candidates are measured modalities (except Pre+SMILES), predicted-profile
    modalities for ``predicted_categories``, and any end-to-end rows present.
    Effect baselines use the same modality when available.

    Multiple-testing: BH-FDR is applied only to the primary family
    (CPA × {Pre+SMILES, Average effect} × Pearson) within this call (one
    stratum). All other rows keep uncorrected ``t_pvalue`` for exploratory use.
    """
    panel = _panel_rows(fold_metrics, predicted_categories)
    if panel.empty or metric not in panel.columns:
        return pd.DataFrame()

    pre_ref = _pre_smiles_metric_frame(fold_metrics)
    rows: list[dict] = []
    keys = panel[['test_set', 'model']].drop_duplicates()
    for _, key in keys.iterrows():
        test_set = key['test_set']
        model = key['model']
        if test_set == 'Measured' and model == 'Pre+SMILES':
            continue
        model_frame = panel[(panel['test_set'] == test_set) & (panel['model'] == model)]
        for baseline in references:
            if baseline == 'Pre+SMILES':
                ref_frame = pre_ref
            elif baseline in {'Average effect', 'No effect'}:
                if test_set == baseline:
                    continue
                ref_model = model if test_set != CPA_E2E_TEST_SET else 'LFC+SMILES'
                ref_frame = _effect_baseline_metric_frame(
                    fold_metrics, baseline, str(ref_model)
                )
            else:
                continue
            if ref_frame is None or ref_frame.empty:
                continue
            stats_row = _paired_stats_vs_frame(
                model_frame, ref_frame, metric, higher_is_better=higher_is_better
            )
            stats_row.update(
                {
                    'model': model,
                    'test_set': test_set,
                    'baseline': baseline,
                    'metric': metric,
                }
            )
            rows.append(stats_row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = _mark_primary_predicted_comparisons(out)
    out = _apply_primary_bh_fdr(out)
    return _add_significance_columns(out)


def compute_paired_comparisons(
    fold_metrics: pd.DataFrame,
    metric: str,
    predicted_categories: list[str],
    higher_is_better: bool,
) -> pd.DataFrame:
    """Paired comparisons vs Pre+SMILES and Average / No-effect baselines."""
    return compute_paired_comparisons_vs_references(
        fold_metrics,
        metric,
        predicted_categories,
        higher_is_better,
        references=('Pre+SMILES', 'Average effect', 'No effect'),
    )


def attach_paired_deltas_to_summary(
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = ('pearson', 'spearman', 'rmse'),
    baselines: tuple[tuple[str, str], ...] = (
        ('Pre+SMILES', 'pre_smiles'),
        ('Average effect', 'avg_effect'),
        ('No effect', 'no_effect'),
    ),
) -> pd.DataFrame:
    """Widen paired Δ / CI / p / sig columns onto the summary mean±sd table.

    Primary family rows (CPA Pearson vs Pre+Morgan / Average effect) use BH-FDR
    ``t_pvalue_fdr`` and stars from $q$; all other comparisons use uncorrected
    ``t_pvalue`` (exploratory).
    """
    if summary is None or summary.empty:
        return summary if summary is not None else pd.DataFrame()
    out = summary.copy()
    if comparisons is None or comparisons.empty:
        return out

    join_keys = [c for c in ('dataset', 'family', 'stratum', 'test_set', 'model') if c in out.columns]
    for baseline, slug in baselines:
        for metric in metrics:
            sub = comparisons[
                (comparisons['baseline'] == baseline) & (comparisons['metric'] == metric)
            ].copy()
            if sub.empty:
                continue
            raw_p = pd.to_numeric(sub.get('t_pvalue', np.nan), errors='coerce')
            fdr_p = (
                pd.to_numeric(sub['t_pvalue_fdr'], errors='coerce')
                if 't_pvalue_fdr' in sub.columns
                else pd.Series(np.nan, index=sub.index)
            )
            if 'is_primary' in sub.columns:
                use_fdr = sub['is_primary'].fillna(False) & fdr_p.notna()
            else:
                use_fdr = fdr_p.notna()
            sub['_pvalue_for_table'] = np.where(use_fdr, fdr_p, raw_p)
            sub['_sig_for_table'] = pd.Series(
                sub['_pvalue_for_table'], index=sub.index
            ).map(significance_stars)
            keep = join_keys + [
                'mean_diff',
                'ci_low',
                'ci_high',
                '_pvalue_for_table',
                '_sig_for_table',
                'n_folds',
            ]
            keep = [c for c in keep if c in sub.columns]
            sub = sub[keep].rename(
                columns={
                    'mean_diff': f'{metric}_vs_{slug}_delta',
                    'ci_low': f'{metric}_vs_{slug}_ci_low',
                    'ci_high': f'{metric}_vs_{slug}_ci_high',
                    '_pvalue_for_table': f'{metric}_vs_{slug}_pvalue',
                    '_sig_for_table': f'{metric}_vs_{slug}_sig',
                    'n_folds': f'{metric}_vs_{slug}_n',
                }
            )
            out = out.merge(sub, on=join_keys, how='left')
    return out


def _print_paired_line(comp: pd.DataFrame, test_set: str, model: str, digits: int = 3) -> str:
    bits = []
    sub = comp[(comp['test_set'] == test_set) & (comp['model'] == model)]
    for _, row in sub.iterrows():
        if np.isnan(row['mean_diff']):
            bits.append(f"delta {row['baseline']}=NA")
            continue
        p = row['t_pvalue']
        p_txt = 'NA' if np.isnan(p) else (f'{p:.1e}' if p < 0.001 else f'{p:.3f}')
        fdr = row.get('t_pvalue_fdr', np.nan)
        fdr_txt = '' if pd.isna(fdr) else f', FDR={fdr:.3f}'
        bits.append(
            f"delta {row['baseline']} {row['mean_diff']:+.{digits}f} "
            f"(95% CI {row['ci_low']:+.{digits}f} to {row['ci_high']:+.{digits}f}, p={p_txt}{fdr_txt})"
        )
    return '   '.join(bits)


def _stratum_note(predictions: pd.DataFrame, dataset_label: str) -> str:
    df = normalize_prediction_frame(predictions)
    n_line = df['cell_line'].nunique() if 'cell_line' in df.columns else 0
    n_drug = df['condition'].nunique() if 'condition' in df.columns else 0
    return f'{dataset_label}  ({n_drug} drugs x {n_line} cell lines)'


def print_metric_report(
    *,
    figure_name: str,
    dataset_label: str,
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    metric: str,
    predicted_categories: list[str],
    higher_is_better: bool,
    extra_metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Print fold-wise or mean±SD values and paired diffs for one figure panel."""
    extra_metrics = extra_metrics or []
    panel = _panel_rows(fold_metrics, predicted_categories)
    comparisons = compute_paired_comparisons(
        fold_metrics, metric, predicted_categories, higher_is_better=higher_is_better
    )
    direction = 'higher better' if higher_is_better else 'lower better'
    print('=' * 88)
    print(f'Figure: {figure_name}')
    print(f'{_stratum_note(predictions, dataset_label)}   metric={metric} ({direction})')
    print('=' * 88)

    measured_order = ['Pre', 'Pre+SMILES', 'Post', 'Post+SMILES', 'LFC', 'LFC+SMILES']
    pred_model_order = [m for m in predicted_categories]
    profile_order = ['Post', 'Post+SMILES', 'LFC', 'LFC+SMILES']

    def _block(title: str, rows: pd.DataFrame, model_order: list[str], test_order: list[str]) -> None:
        print(f'\n{title}')
        for test_set in test_order:
            sub_t = rows[rows['test_set'] == test_set]
            if sub_t.empty:
                continue
            print(f'  {test_set}')
            for model in model_order:
                sub = sub_t[sub_t['model'] == model].sort_values('split')
                if sub.empty:
                    continue
                vals = sub[metric].to_numpy(dtype=float)
                line = f'    {model:<4}  {_fmt_mean_sd(vals)}   folds: {_fmt_folds(vals)}'
                extras = []
                for extra in extra_metrics:
                    if extra in sub.columns:
                        extras.append(f'{extra} {_fmt_mean_sd(sub[extra])}')
                if extras:
                    line += '   | ' + '   '.join(extras)
                if test_set in [m for m in predicted_categories if m not in BASELINES] and not comparisons.empty:
                    paired = _print_paired_line(comparisons, test_set, model)
                    if paired:
                        line += f'\n          {paired}'
                print(line)

    _block('Measured profiles', panel[panel['test_set'] == 'Measured'], measured_order, ['Measured'])
    _block(
        'Predicted profiles (delta = model - baseline, paired by fold)',
        panel[panel['test_set'] != 'Measured'],
        profile_order,
        pred_model_order,
    )
    print()
    return comparisons


def print_stratified_report(
    *,
    dataset_label: str,
    predictions: pd.DataFrame,
    predicted_categories: list[str],
    metric: str = 'pearson',
) -> None:
    """Per-drug and within-context mean±SD across groups (folds pooled)."""
    df = normalize_prediction_frame(predictions)
    context_col = context_group_column(dataset_label)
    if context_col == 'tissue':
        context_title = 'Per-tissue (within-tissue ranking across drugs)'
        context_note = '>=3 drugs'
    else:
        context_title = 'Per-cell-line (within-line ranking across drugs)'
        context_note = '>=3 drugs'
    for group_col, title, min_note in (
        ('condition', 'Per-drug (within-drug ranking across cell lines)', '>=3 cell lines'),
        (context_col, context_title, context_note),
    ):
        if group_col not in df.columns:
            print(f'\n{title}: skipped (no {group_col} column)')
            continue
        grouped = compute_grouped_metrics(df, group_col, pool_folds=True)
        if grouped.empty:
            print(f'\n{title}: no groups with {min_note}')
            continue
        # ``split`` = group label for paired comparisons across groups
        box = grouped.copy()
        box['split'] = box[group_col].astype(str)
        panel = _panel_rows(box, predicted_categories)
        n_groups = int(grouped[group_col].nunique())
        print(f'\n{title}  [{metric}; {n_groups} groups with {min_note}]')
        print(
            f'  {dataset_label}: OOF preds pooled across folds; '
            f'values are mean +/- SD across {group_col} groups'
        )
        comparisons = compute_paired_comparisons(
            box,
            metric,
            predicted_categories,
            higher_is_better=True,
        )
        for test_set in ['Measured'] + list(predicted_categories):
            sub_t = panel[panel['test_set'] == test_set]
            if sub_t.empty:
                continue
            models = (
                ['Pre', 'Pre+SMILES', 'Post', 'Post+SMILES', 'LFC', 'LFC+SMILES']
                if test_set == 'Measured'
                else ['Post', 'Post+SMILES', 'LFC', 'LFC+SMILES']
            )
            print(f'  {test_set}')
            for model in models:
                sub = sub_t[sub_t['model'] == model].sort_values('split')
                if sub.empty:
                    continue
                vals = sub[metric].to_numpy(dtype=float)
                line = f'    {model:<4}  {_fmt_mean_sd(vals)}   groups: {_fmt_folds(vals)}'
                if test_set not in BASELINES and test_set != 'Measured' and not comparisons.empty:
                    paired = _print_paired_line(comparisons, test_set, model)
                    if paired:
                        line += f'\n          {paired}'
                print(line)

        print('  Group-level distribution (median [IQR]):')
        dist_panel = _panel_rows(box, predicted_categories)
        for test_set in ['Measured'] + [m for m in predicted_categories if m not in BASELINES]:
            for model in (
                ['Pre', 'Pre+SMILES', 'Post', 'Post+SMILES', 'LFC', 'LFC+SMILES']
                if test_set == 'Measured'
                else ['Post', 'Post+SMILES', 'LFC', 'LFC+SMILES']
            ):
                vals = dist_panel.loc[
                    (dist_panel['test_set'] == test_set) & (dist_panel['model'] == model), metric
                ].dropna()
                if vals.empty:
                    continue
                q1, med, q3 = np.percentile(vals, [25, 50, 75])
                print(
                    f'    {test_set:<16} {model:<4}  median {med:.3f}  '
                    f'IQR [{q1:.3f}, {q3:.3f}]  n={len(vals)}'
                )


def print_supplement_report(
    *,
    dataset_label: str,
    fold_metrics: pd.DataFrame,
    predicted_categories: list[str],
) -> dict[str, pd.DataFrame]:
    """Spearman, MAE/RMSE, calibration, classification, and formal comparisons."""
    panel = _panel_rows(fold_metrics, predicted_categories)
    print('\n' + '-' * 88)
    print(f'Supplementary metrics  {dataset_label}')
    print('-' * 88)

    metric_specs = [
        ('pearson', True, 'Pearson r'),
        ('spearman', True, 'Spearman ρ'),
        ('mae', False, 'MAE'),
        ('rmse', False, 'RMSE'),
        ('calibration_slope', True, 'Calibration slope (true ~ pred; 1 = perfect)'),
        ('calibration_intercept', True, 'Calibration intercept (0 = perfect)'),
        ('auroc', True, 'AUROC (sensitive = true > 0.02)'),
        ('auprc', True, 'AUPRC'),
        ('balanced_accuracy', True, 'Balanced accuracy'),
        ('f1', True, 'F1'),
    ]
    all_comparisons = []
    for metric, hib, title in metric_specs:
        if metric not in panel.columns:
            continue
        print(f'\n{title}')
        comps = compute_paired_comparisons(
            fold_metrics, metric, predicted_categories, higher_is_better=hib
        )
        comps = comps.assign(dataset=dataset_label) if not comps.empty else comps
        if not comps.empty:
            all_comparisons.append(comps)
        for test_set in ['Measured'] + list(predicted_categories):
            sub_t = panel[panel['test_set'] == test_set]
            if sub_t.empty:
                continue
            models = (
                ['Pre', 'Pre+SMILES', 'Post', 'Post+SMILES', 'LFC', 'LFC+SMILES']
                if test_set == 'Measured'
                else ['Post', 'Post+SMILES', 'LFC', 'LFC+SMILES']
            )
            for model in models:
                sub = sub_t[sub_t['model'] == model].sort_values('split')
                if sub.empty:
                    continue
                vals = sub[metric].to_numpy(dtype=float)
                line = f'  {test_set:<16} {model:<4}  {_fmt_mean_sd(vals)}   folds: {_fmt_folds(vals)}'
                if (
                    test_set not in BASELINES
                    and test_set != 'Measured'
                    and not comps.empty
                    and metric in ('pearson', 'spearman', 'mae', 'rmse', 'auroc')
                ):
                    paired = _print_paired_line(comps, test_set, model)
                    if paired:
                        line += f'\n                    {paired}'
                print(line)

    comparisons = pd.concat(all_comparisons, ignore_index=True) if all_comparisons else pd.DataFrame()
    return {'panel': panel, 'comparisons': comparisons}


def boxplot_frame_from_fold_metric(
    fold_metrics: pd.DataFrame,
    metric: str,
    predicted_categories: list[str],
) -> pd.DataFrame:
    """Rename ``metric`` to a column the existing boxplot helper can plot."""
    panel = _panel_rows(fold_metrics, predicted_categories).copy()
    if metric not in panel.columns:
        return panel
    return panel


def fold_metrics_boxplot_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    """Fold-wise metrics in the schema expected by ``plot_sensitvity_predictions``."""
    df = normalize_prediction_frame(predictions)
    fold = compute_fold_metrics(df)
    if 'trained_on' not in fold.columns:
        fold['trained_on'] = np.where(fold['test_set'].eq('Measured'), 'Measured', 'Predicted')
    return fold


def load_cpa_end_to_end_predictions(
    dataset: str,
    data_dir: str | Path | None = None,
    *,
    variant: str | None = None,
) -> pd.DataFrame:
    """Load CPA end-to-end sensitivity predictions from ``data/CPA_end_to_end``.

    By default uses the top-level CSVs
    (``CPA_end_to_end/{dataset}_sensitivity_predictions.csv``). Pass ``variant``
    (e.g. ``'with_post'``) to load from a subdirectory instead; falls back to the
    top-level file if that path is missing.
    """
    from pathlib import Path as _Path

    if data_dir is None:
        data_dir = _Path(__file__).resolve().parents[2] / 'data'
    root = _Path(data_dir) / 'CPA_end_to_end'
    fname = f'{dataset}_sensitivity_predictions.csv'
    candidates = []
    if variant:
        candidates.append(root / str(variant) / fname)
    candidates.append(root / fname)

    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f'Missing CPA end-to-end predictions for {dataset!r} '
            f'(tried: {", ".join(str(p) for p in candidates)})'
        )

    df = pd.read_csv(path)
    rename = {'sens_true': 'true', 'sens_pred': 'pred', 'cell_type': 'cell_line'}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df = df.dropna(subset=['true', 'pred'])
    df['model'] = CPA_E2E_MODEL
    df['test_set'] = CPA_E2E_TEST_SET
    df['train_set'] = 'Predicted'
    df['trained_on'] = 'Predicted'
    df = attach_group_columns(df)
    return attach_tissue_column(df, dataset)


def attach_tissue_column(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Ensure a ``tissue`` column (SciPlex: cell line; McFarland: DepMap/name suffix)."""
    out = df.copy()
    dataset = str(dataset).strip().lower()
    if 'tissue' in out.columns and out['tissue'].notna().any():
        if dataset.startswith('sciplex'):
            # SciPlex tissue labels are the cell lines themselves.
            out['tissue'] = out['cell_line'].astype(str).str.strip().str.upper()
        return out

    if 'cell_line' not in out.columns:
        raise KeyError('Expected cell_line to derive tissue')

    cl = out['cell_line'].astype(str).str.strip()
    if dataset.startswith('sciplex'):
        out['tissue'] = cl.str.upper()
        return out

    # McFarland: parse LINE_TISSUE when present; fill remaining via DepMap map.
    has_suffix = cl.str.contains('_', regex=False)
    parsed_tissue = cl.str.rsplit('_', n=1).str[-1].where(has_suffix)
    base = cl.str.split('_').str[0].str.upper()
    out['tissue'] = parsed_tissue
    try:
        from prediction_utils import get_tissue_labels

        tissues = get_tissue_labels()[['cell_line', 'tissue']].drop_duplicates(
            subset='cell_line', keep='first'
        )
        tissues = tissues.assign(cell_line=tissues['cell_line'].astype(str).str.upper())
        mapped = base.map(tissues.set_index('cell_line')['tissue'])
        out['tissue'] = out['tissue'].fillna(mapped)
    except Exception:
        pass
    out['tissue'] = out['tissue'].fillna('UNKNOWN').astype(str).str.upper()
    return out


def cpa_end_to_end_fold_metrics(
    dataset: str,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Fold-wise metrics for CPA end-to-end sensitivity predictions."""
    return compute_fold_metrics(load_cpa_end_to_end_predictions(dataset, data_dir))


def append_cpa_end_to_end_fold_metrics(
    fold_df: pd.DataFrame,
    dataset: str,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Append CPA end-to-end fold rows to a boxplot/summary metrics frame."""
    e2e = cpa_end_to_end_fold_metrics(dataset, data_dir)
    if fold_df is None or fold_df.empty:
        return e2e
    return pd.concat([fold_df, e2e], ignore_index=True)


def append_cpa_end_to_end_grouped_metrics(
    fold_df: pd.DataFrame,
    dataset: str,
    group_col: str,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Append within-drug or within-line CPA end-to-end fold-mean metrics."""
    e2e = grouped_metrics_boxplot_frame(
        load_cpa_end_to_end_predictions(dataset, data_dir), group_col
    )
    if fold_df is None or fold_df.empty:
        return e2e
    if e2e.empty:
        return fold_df
    return pd.concat([fold_df, e2e], ignore_index=True)


def build_cpa_end_to_end_summary_tables(
    dataset_label: str,
    dataset: str,
    *,
    family: str = 'CPA family',
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Summary rows for CPA end-to-end (pooled + per-drug + per-cell-line)."""
    preds = load_cpa_end_to_end_predictions(dataset, data_dir)
    fold = compute_fold_metrics(preds)
    metrics = ('pearson', 'spearman', 'mae', 'rmse', 'auroc')

    def _e2e_summary(fold_metrics: pd.DataFrame, stratum: str) -> pd.DataFrame:
        row = {'test_set': CPA_E2E_TEST_SET, 'model': CPA_E2E_MODEL}
        for metric in metrics:
            if metric not in fold_metrics.columns:
                continue
            mean, sd = _mean_sd(fold_metrics[metric].to_numpy(dtype=float))
            row[f'{metric}_mean'] = mean
            row[f'{metric}_sd'] = sd
        return pd.DataFrame([row]).assign(dataset=dataset_label, family=family, stratum=stratum)

    parts = [_e2e_summary(fold, 'pooled')]
    for stratum, group_col in (
        ('per_drug', 'condition'),
        ('per_cell_line', context_group_column(dataset)),
    ):
        grouped = compute_grouped_metrics(preds, group_col, pool_folds=True)
        if not grouped.empty:
            parts.append(_e2e_summary(grouped, stratum))
    return pd.concat(parts, ignore_index=True)


def _summarize_panel_metrics(
    panel: pd.DataFrame,
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for (test_set, model), sub in panel.groupby(['test_set', 'model'], sort=False):
        row = {'test_set': test_set, 'model': model}
        for metric in metrics:
            if metric not in sub.columns:
                continue
            mean, sd = _mean_sd(sub[metric].to_numpy(dtype=float))
            row[f'{metric}_mean'] = mean
            row[f'{metric}_sd'] = sd
        rows.append(row)
    return pd.DataFrame(rows)


def fill_measured_pre_from_no_effect_post(results: pd.DataFrame) -> pd.DataFrame:
    """Replace Measured Pre(+SMILES) rows with No-effect Post(+SMILES) metrics.

    No-effect Post is trained/tested on Pre expression under the predicted-profile
    CV splits. Using those scores for Measured Pre keeps the Pre box / summary
    row aligned with the No-effect baseline on the same groups (SciPlex
    within-line panels).
    """
    if results is None or results.empty:
        return results
    if 'test_set' not in results.columns or 'model' not in results.columns:
        return results

    post_to_pre = {'Post': 'Pre', 'Post+SMILES': 'Pre+SMILES'}
    ne_post = results[
        results['test_set'].astype(str).eq('No effect')
        & results['model'].astype(str).isin(post_to_pre)
    ].copy()
    if ne_post.empty:
        logger.warning(
            'fill_measured_pre_from_no_effect_post: no No-effect Post rows; '
            'leaving Measured Pre unchanged'
        )
        return results

    out = results[
        ~(
            results['test_set'].astype(str).eq('Measured')
            & results['model'].astype(str).isin(['Pre', 'Pre+SMILES'])
        )
    ].copy()
    filled = ne_post.copy()
    filled['model'] = filled['model'].astype(str).map(post_to_pre)
    filled['test_set'] = 'Measured'
    if 'train_set' in filled.columns:
        filled['train_set'] = 'Measured'
    if 'trained_on' in filled.columns:
        filled['trained_on'] = 'Measured'
    cols = list(dict.fromkeys([*out.columns.tolist(), *filled.columns.tolist()]))
    logger.info(
        'fill_measured_pre_from_no_effect_post: replaced Measured Pre with '
        'No-effect Post (%d rows, models=%s)',
        len(filled),
        sorted(filled['model'].astype(str).unique()),
    )
    return pd.concat(
        [out.reindex(columns=cols), filled.reindex(columns=cols)],
        ignore_index=True,
    )


def build_stratified_summary_tables(
    predictions: pd.DataFrame,
    *,
    dataset_label: str,
    family: str,
    predicted_categories: list[str],
    dataset: str | None = None,
    data_dir: str | Path | None = None,
    include_cpa_end_to_end: bool = False,
    comparison_metrics: tuple[str, ...] = ('pearson', 'spearman', 'rmse'),
) -> dict[str, pd.DataFrame]:
    """Build pooled / per-drug / per-cell-line summaries and paired Δ tables.

    Paired comparisons are vs Pre+SMILES and Average effect (same modality), with
    mean Δ, 95% CI, p-value, and ``sig`` stars (``**`` / ``*`` / ``NS``). Summary
    rows also receive widened ``*_vs_pre_smiles_*`` / ``*_vs_avg_effect_*`` columns
    for Pearson, Spearman, and RMSE.

    For SciPlex within-line summaries, Measured Pre(+SMILES) is filled from
    No-effect Post(+SMILES) (same as the stratified boxplot panels).
    """
    df = normalize_prediction_frame(predictions)
    fold = compute_fold_metrics(df)
    dataset_key = dataset or (
        'mcfarland' if 'mcfarland' in str(dataset_label).lower() else 'sciplex'
    )
    if include_cpa_end_to_end:
        try:
            fold = append_cpa_end_to_end_fold_metrics(fold, dataset_key, data_dir)
        except FileNotFoundError:
            pass

    metrics = ('pearson', 'spearman', 'mae', 'rmse', 'auroc')
    panel = _panel_rows(fold, predicted_categories)

    pooled = _summarize_panel_metrics(panel, metrics).assign(
        dataset=dataset_label,
        family=family,
        stratum='pooled',
    )

    # Keep each (test_set, model) on its own eligible groups (CPA vs GEARS coverage
    # may differ); do not intersect across panels.
    per_drug_groups = grouped_metrics_boxplot_frame(
        df, 'condition', align_common_groups=False
    )
    per_line_groups = grouped_metrics_boxplot_frame(
        df, context_group_column(dataset_label), align_common_groups=False
    )
    if include_cpa_end_to_end:
        try:
            per_drug_groups = append_cpa_end_to_end_grouped_metrics(
                per_drug_groups, dataset_key, 'condition', data_dir
            )
            per_line_groups = append_cpa_end_to_end_grouped_metrics(
                per_line_groups,
                dataset_key,
                context_group_column(dataset_key),
                data_dir,
            )
        except FileNotFoundError:
            pass

    # SciPlex within-line: Measured Pre ≡ No-effect Post (matches boxplot panels).
    if 'sciplex' in str(dataset_key).strip().lower():
        per_line_groups = fill_measured_pre_from_no_effect_post(per_line_groups)

    per_drug_panel = _panel_rows(per_drug_groups, predicted_categories)
    per_line_panel = _panel_rows(per_line_groups, predicted_categories)
    per_drug = (
        _summarize_panel_metrics(per_drug_panel, metrics).assign(
            dataset=dataset_label, family=family, stratum='per_drug'
        )
        if not per_drug_panel.empty
        else pd.DataFrame()
    )
    per_cell_line = (
        _summarize_panel_metrics(per_line_panel, metrics).assign(
            dataset=dataset_label, family=family, stratum='per_cell_line'
        )
        if not per_line_panel.empty
        else pd.DataFrame()
    )

    metric_higher = {
        'pearson': True,
        'spearman': True,
        'rmse': False,
        'mae': False,
        'auroc': True,
    }
    comparison_frames = []
    for stratum_name, frame in (
        ('pooled', fold),
        ('per_drug', per_drug_groups),
        ('per_cell_line', per_line_groups),
    ):
        for metric in comparison_metrics:
            if metric not in frame.columns:
                continue
            comps = compute_paired_comparisons_vs_references(
                frame,
                metric,
                predicted_categories,
                higher_is_better=metric_higher.get(metric, True),
            )
            if not comps.empty:
                comparison_frames.append(
                    comps.assign(dataset=dataset_label, family=family, stratum=stratum_name)
                )

    comparisons = (
        pd.concat(comparison_frames, ignore_index=True) if comparison_frames else pd.DataFrame()
    )
    if not comparisons.empty and 'sig' not in comparisons.columns:
        comparisons = _add_significance_columns(comparisons)

    summary = pd.concat(
        [part for part in (pooled, per_drug, per_cell_line) if part is not None and not part.empty],
        ignore_index=True,
    )
    summary = attach_paired_deltas_to_summary(
        summary,
        comparisons,
        metrics=tuple(m for m in comparison_metrics if m in {'pearson', 'spearman', 'rmse'}),
    )
    return {
        'summary': summary,
        'comparisons': comparisons,
        'fold_metrics': fold,
        'per_drug_fold': per_drug_groups,
        'per_cell_line_fold': per_line_groups,
    }


def summarize_hurdle_sweep(predictions: pd.DataFrame) -> pd.DataFrame:
    """Fold-wise Pearson (hurdle predictions) and AUROC (classifier score vs true > threshold)."""
    out = normalize_prediction_frame(predictions)
    if 'score' not in out.columns:
        out['score'] = out['pred']
    keys = [c for c in ('dataset', 'model', 'test_set', 'split', 'threshold', 'head') if c in out.columns]
    rows = []
    for key, group in out.groupby(keys, dropna=False):
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        y = _as_float(group['true'])
        pred = _as_float(group['pred'])
        score = _as_float(group['score'])
        threshold = float(rec['threshold']) if 'threshold' in rec and pd.notna(rec['threshold']) else SENSITIVE_THRESHOLD
        rec['pearson'] = _safe_corr(y, pred, 'pearson')
        rec['auroc'] = _safe_auroc(y, score, threshold=threshold)
        rec['n'] = int(len(group))
        rec['n_sensitive'] = int((y > threshold).sum())
        rec['frac_sensitive'] = rec['n_sensitive'] / rec['n'] if rec['n'] else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_label_threshold_sweep(
    predictions: pd.DataFrame,
    thresholds: Iterable[float],
) -> pd.DataFrame:
    """AUROC of existing continuous scores vs labels defined at each threshold. Pearson is independent of t."""
    out = normalize_prediction_frame(predictions)
    score_col = 'score' if 'score' in out.columns else 'pred'
    keys = [c for c in ('dataset', 'model', 'test_set', 'split') if c in out.columns]
    rows = []
    for threshold in thresholds:
        for key, group in out.groupby(keys, dropna=False):
            rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
            rec['threshold'] = float(threshold)
            rec['head'] = 'continuous'
            y = _as_float(group['true'])
            pred = _as_float(group['pred'])
            score = _as_float(group[score_col])
            rec['pearson'] = _safe_corr(y, pred, 'pearson')
            rec['auroc'] = _safe_auroc(y, score, threshold=float(threshold))
            rec['n'] = int(len(group))
            rec['n_sensitive'] = int((y > float(threshold)).sum())
            rec['frac_sensitive'] = rec['n_sensitive'] / rec['n'] if rec['n'] else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)
