"""Plot reconstruction metrics: SciPlex | McFarland, one figure per metric.

Model families:
- CPA family: CPA, chemCPA, PRnet, Average effect, No effect
- GEARS family: GEARS, scFoundation, Average effect, No effect
- All models: CPA family + GEARS/scFoundation (shared baselines once)
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from collections.abc import Collection
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.transforms import blended_transform_factory
from scipy import stats
from statsmodels.stats.multitest import multipletests

_eval_dir = Path(__file__).parent
sys.path.insert(0, str(_eval_dir))
sys.path.append(str(_eval_dir.parent.parent))
from config import config, setup_project
from utils import ensure_directories_exist, setup_logging_for_script

setup_project()
logger = setup_logging_for_script(__file__)

results_dir = str(config.RESULTS_03_DIR)
figures_dir = str(config.FIGURES_03_DIR)
ensure_directories_exist(results_dir, figures_dir)

SCIPLEX_CELL_LINES = ('mcf7', 'a549', 'k562')
DATASET_ORDER = ('SciPlex3 dataset', 'McFarland dataset')
DATASET_TITLES = {
    'SciPlex3 dataset': 'SciPlex3',
    'McFarland dataset': 'McFarland',
}
BOX_WIDTH = 0.5
MAX_SWARM_PER_GROUP = 1000
# Match Figure2 layout: row-2 panels are each half of the 3-panel stratified
# figure (14 * 3, 12), so two side-by-side PDFs share its width and height.
METRIC_FIGSIZE = (21, 12)
ALL_FAMILY_FIGSIZE = (21, 12)

CPA_FAMILY = ('CPA', 'chemCPA', 'PRnet', 'Average effect', 'No effect')
GEARS_FAMILY = ('GEARS', 'scFoundation', 'Average effect', 'No effect')
ALL_FAMILY = (
    'CPA',
    'chemCPA',
    'PRnet',
    'GEARS',
    'scFoundation',
    'Average effect',
    'No effect',
)
PRM_MODELS = frozenset({'CPA', 'chemCPA', 'PRnet', 'GEARS', 'scFoundation'})
AVERAGE_EFFECT = 'Average effect'
NO_EFFECT = 'No effect'

SCIPLEX_STEMS = {
    'CPA': 'CPA_outcomes',
    'chemCPA': 'chemCPA_outcomes',
    'PRnet': 'PRnet_outcomes',
    'GEARS': 'GEARS_outcomes',
    'scFoundation': 'scfoundation_outcomes',
    'No effect': 'no_effect_outcomes',
    'Average effect': 'average_effect_outcomes',
}

MCFARLAND_FILES = {
    'CPA': 'mcfarland_CPA{suffix}_outcomes.csv',
    'chemCPA': 'mcfarland_chemCPA{suffix}_outcomes.csv',
    'PRnet': 'mcfarland_PRnet{suffix}_outcomes.csv',
    'GEARS': 'mcfarland_GEARS{suffix}_outcomes.csv',
    'scFoundation': 'mcfarland_scFoundation{suffix}_outcomes.csv',
    'No effect': 'mcfarland_no_effect{suffix}_outcomes.csv',
    'Average effect': 'mcfarland_average_effect{suffix}_outcomes.csv',
}

MODEL_COLORS = {
    'CPA': '#1f77b4',
    'chemCPA': '#d62728',
    'PRnet': '#17becf',
    'GEARS': '#2ca02c',
    'scFoundation': '#ff7f0e',
    'Average effect': '#9467bd',
    'No effect': '#7f7f7f',
}

MSE_PEARSON_METRICS = ['mse', 'pearson', 'mse_de', 'pearson_de']
MSE_PEARSON_LABELS = {
    'mse': r'MSE (all genes) $\leftarrow$',
    'pearson': r'Pearson r (all genes) $\rightarrow$',
    'mse_de': r'MSE (top20 DE) $\leftarrow$',
    'pearson_de': r'Pearson r (top20 DE) $\rightarrow$',
}
SYSTEMA_METRICS = [
    'mse',
    'pearson_systema',
    'pearson_systema_cellline',
    'pearson_systema_cellline_de',
]
SYSTEMA_LABELS = {
    'mse': r'MSE (all genes) $\leftarrow$',
    'pearson_systema': r'Systema Pearson r $\rightarrow$',
    'pearson_systema_cellline': r'Systema Pearson r (per cell line) $\rightarrow$',
    'pearson_systema_cellline_de': r'Systema Pearson r (per cell line, top20 DE) $\rightarrow$',
}

# Combined supplement table: standard metrics + Systema Pearson (per cell line),
# matching performance_boxplot_systema_all_models_pearson_systema_cellline.
COMBINED_SUMMARY_SPECS: tuple[tuple[bool, str], ...] = (
    *((False, m) for m in MSE_PEARSON_METRICS),
    (True, 'pearson_systema_cellline'),
)


def _load_sciplex_outcomes_csv(cell_line: str, stem: str) -> pd.DataFrame | None:
    candidates = [stem]
    if 'CPA' in stem:
        candidates.append(stem.replace('CPA', 'cpa'))
    for name in candidates:
        fpath = os.path.join(results_dir, f'sciplex{cell_line}_{name}.csv')
        if os.path.exists(fpath):
            return pd.read_csv(fpath)
    logger.warning('Missing SciPlex file: sciplex%s_%s.csv', cell_line, stem)
    return None


def process_sciplex_performances(suffix: str = '') -> pd.DataFrame:
    frames = []
    for cell_line in SCIPLEX_CELL_LINES:
        for model, stem in SCIPLEX_STEMS.items():
            file_stem = stem if not suffix else stem.replace('_outcomes', f'_{suffix}_outcomes')
            df = _load_sciplex_outcomes_csv(cell_line, file_stem)
            if df is None:
                continue
            df['model'] = model
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f'No SciPlex outcome files found in {results_dir}')
    outcomes = pd.concat(frames, axis=0, ignore_index=True).dropna(subset=['value'])
    outcomes['cell_type_condition'] = outcomes['cell_type'] + '_' + outcomes['condition']
    outcomes['dataset'] = 'SciPlex3 dataset'
    return outcomes


def process_mcfarland_performances(suffix: str = '') -> pd.DataFrame:
    frames = []
    for model, filename_tmpl in MCFARLAND_FILES.items():
        filename = filename_tmpl.format(suffix=suffix)
        fpath = os.path.join(results_dir, filename)
        if not os.path.exists(fpath):
            logger.warning('Skipping missing McFarland file: %s', filename)
            continue
        df = pd.read_csv(fpath)
        df['model'] = model
        frames.append(df)
    if not frames:
        raise FileNotFoundError('No McFarland outcome files found in results_dir')
    outcomes = pd.concat(frames, axis=0, ignore_index=True).dropna(subset=['value'])
    outcomes['dataset'] = 'McFarland dataset'
    outcomes['cell_type'] = outcomes['cell_line']
    outcomes['cell_type_condition'] = outcomes['cell_line'] + '_' + outcomes['condition']
    return outcomes


def load_all_outcomes(systema: bool = False) -> pd.DataFrame:
    suffix = 'systema' if systema else ''
    mcfarland_suffix = '_systema' if systema else ''
    sciplex = process_sciplex_performances(suffix=suffix)
    mcfarland = process_mcfarland_performances(suffix=mcfarland_suffix)
    return pd.concat([sciplex, mcfarland], axis=0, ignore_index=True)


def _prepare_plot_frame(df: pd.DataFrame, dataset: str, model_order: tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    out['model'] = out['model'].astype(str)
    return out[(out['dataset'] == dataset) & (out['model'].isin(model_order))]


def _models_with_data(df: pd.DataFrame, model_order: tuple[str, ...]) -> list[str]:
    return [m for m in model_order if (df['model'] == m).any()]


def _subsample_groups(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, grp in df.groupby(['model', 'metric'], observed=True):
        if len(grp) > MAX_SWARM_PER_GROUP:
            grp = grp.sample(MAX_SWARM_PER_GROUP, random_state=0)
        parts.append(grp)
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]


def _prepare_datasets(
    all_outcomes: pd.DataFrame,
    model_order: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    by_dataset: dict[str, pd.DataFrame] = {}
    for dataset in DATASET_ORDER:
        data = _subsample_groups(_prepare_plot_frame(all_outcomes, dataset, model_order))
        if not data.empty:
            by_dataset[dataset] = data
    return by_dataset


def _add_x_positions(df: pd.DataFrame, model_order: tuple[str, ...]) -> tuple[pd.DataFrame, list[str]]:
    """Map models onto a fixed x-order so missing models leave empty slots."""
    models = list(model_order)
    model_to_idx = {m: i for i, m in enumerate(models)}
    df = df.copy()
    df['x_pos'] = df['model'].map(model_to_idx)
    df = df.dropna(subset=['x_pos'])
    df['x_pos'] = df['x_pos'].astype(int)
    return df, models


def _style_model_xticks(ax) -> None:
    ax.tick_params(axis='x', labelsize=20, rotation=45)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
        label.set_rotation_mode('anchor')


MODEL_X_GROUPS: tuple[tuple[str, set[str]], ...] = (
    ('chemical PRMs', {'CPA', 'chemCPA', 'PRnet'}),
    ('genetic PRMs', {'GEARS', 'scFoundation'}),
    ('baselines', {'Average effect', 'No effect'}),
)


def _add_x_group_labels(
    ax,
    categories: list[str],
    groups: tuple[tuple[str, set[str]], ...] = MODEL_X_GROUPS,
    *,
    y_line: float = -0.24,
    y_text: float = -0.26,
    fontsize: int = 20,
) -> None:
    """Draw a short bar and label under contiguous x-tick groups (axes y coords)."""
    if not categories:
        return
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for label, members in groups:
        idxs = [i for i, cat in enumerate(categories) if cat in members]
        if not idxs:
            continue
        runs: list[list[int]] = []
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


def _sig_label(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return 'NS'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'NS'


def _metric_higher_is_better(metric: str) -> bool:
    return 'mse' not in metric


def _paired_or_unpaired_vs_baseline(
    data_subset: pd.DataFrame,
    model: str,
    *,
    baseline: str,
) -> tuple[float, float, int, str]:
    """Return (median_diff, p_value, n, test) for PRM vs ``baseline``.

    Prefers paired Wilcoxon on cell×condition; falls back to Mann-Whitney.
    ``median_diff`` is PRM − baseline.
    """
    prm = data_subset[data_subset['model'] == model][['cell_type', 'condition', 'value']].dropna(
        subset=['value']
    ).copy()
    base = data_subset[data_subset['model'] == baseline][
        ['cell_type', 'condition', 'value']
    ].dropna(subset=['value']).copy()
    if prm.empty or base.empty:
        return np.nan, np.nan, 0, 'none'

    def _keys(frame: pd.DataFrame) -> pd.Series:
        return frame['cell_type'].astype(str) + '_' + frame['condition'].astype(str)

    prm = prm.assign(pair_key=_keys(prm))
    base = base.assign(pair_key=_keys(base))
    merged = prm.merge(base, on='pair_key', suffixes=('_prm', '_base'))

    # SciPlex chemical PRMs use drug names; baselines / GEARS often use gene targets.
    if len(merged) < 5:
        try:
            from mcfarland_profile_metrics import map_sciplex_conditions_to_gene_targets

            resources_dir = str(config.RESOURCES_DIR)
            prm_mapped = map_sciplex_conditions_to_gene_targets(prm.copy(), resources_dir)
            prm_mapped = prm_mapped.assign(
                pair_key=prm_mapped['cell_type'].astype(str)
                + '_'
                + prm_mapped['condition'].astype(str)
            )
            merged = prm_mapped.merge(base, on='pair_key', suffixes=('_prm', '_base'))
        except Exception:
            merged = merged

    if len(merged) >= 5:
        d = merged['value_prm'].to_numpy(dtype=float) - merged['value_base'].to_numpy(dtype=float)
        mean_diff = float(np.nanmedian(d))
        n = int(np.isfinite(d).sum())
        if np.allclose(d, 0.0, equal_nan=False):
            return mean_diff, 1.0, n, 'wilcoxon_paired'
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                p = float(stats.wilcoxon(d, zero_method='wilcox').pvalue)
        except ValueError:
            p = np.nan
        return mean_diff, p, n, 'wilcoxon_paired'

    a = prm['value'].to_numpy(dtype=float)
    b = base['value'].to_numpy(dtype=float)
    mean_diff = float(np.nanmedian(a) - np.nanmedian(b))
    n = min(len(a), len(b))
    if len(a) < 3 or len(b) < 3:
        return mean_diff, np.nan, n, 'mannwhitneyu'
    try:
        p = float(stats.mannwhitneyu(a, b, alternative='two-sided').pvalue)
    except ValueError:
        p = np.nan
    return mean_diff, p, n, 'mannwhitneyu'


def _prm_vs_baseline_stats(
    data_subset: pd.DataFrame,
    model_order: tuple[str, ...],
    metric: str,
    *,
    baseline: str = AVERAGE_EFFECT,
) -> pd.DataFrame:
    """PRM vs baseline tests for one dataset×metric panel.

    FDR (BH) is corrected across PRMs within this panel. Significance is
    ``**/ *`` only when the PRM is better than baseline and FDR < 0.05.
    """
    empty = pd.DataFrame(
        columns=[
            'model',
            'baseline',
            'metric',
            'median_diff',
            'n',
            'test',
            'higher_is_better',
            'better_than_baseline',
            'p_raw',
            'p_fdr',
            'significance',
        ]
    )
    if baseline not in model_order or not (data_subset['model'] == baseline).any():
        return empty

    prm_cats = [m for m in model_order if m in PRM_MODELS]
    if not prm_cats:
        return empty

    higher = _metric_higher_is_better(metric)
    rows: list[dict] = []
    for model in prm_cats:
        if not (data_subset['model'] == model).any():
            continue
        median_diff, p, n, test = _paired_or_unpaired_vs_baseline(
            data_subset, model, baseline=baseline
        )
        better = bool(np.isfinite(median_diff) and ((median_diff > 0) if higher else (median_diff < 0)))
        rows.append(
            {
                'model': model,
                'baseline': baseline,
                'metric': metric,
                'median_diff': median_diff,
                'n': n,
                'test': test,
                'higher_is_better': higher,
                'better_than_baseline': better,
                'p_raw': p,
            }
        )
    if not rows:
        return empty

    out = pd.DataFrame(rows)
    pvals = out['p_raw'].to_numpy(dtype=float)
    fdr = np.full_like(pvals, np.nan)
    mask = np.isfinite(pvals)
    if mask.any():
        fdr[mask] = multipletests(pvals[mask], alpha=0.05, method='fdr_bh')[1]
    out['p_fdr'] = fdr
    out['significance'] = [
        (
            'NS'
            if (not better) or not np.isfinite(p_fdr) or p_fdr >= 0.05
            else _sig_label(float(p_fdr))
        )
        for better, p_fdr in zip(out['better_than_baseline'], out['p_fdr'])
    ]
    return out


def compute_prm_vs_baseline_table(
    all_outcomes: pd.DataFrame,
    model_order: tuple[str, ...],
    metrics: list[str],
    *,
    baseline: str = AVERAGE_EFFECT,
    comparison_set: str = 'all_models',
) -> pd.DataFrame:
    """Build supplementary significance table for one model family.

    Uses the full outcome table (not the swarm subsample used for plotting).
    """
    frames: list[pd.DataFrame] = []
    for dataset in DATASET_ORDER:
        data = _prepare_plot_frame(all_outcomes, dataset, model_order)
        if data.empty:
            continue
        for metric in metrics:
            subset = data[data['metric'] == metric].dropna(subset=['value'])
            if subset.empty:
                continue
            stats_df = _prm_vs_baseline_stats(
                subset, model_order, metric, baseline=baseline
            )
            if stats_df.empty:
                continue
            frames.append(
                stats_df.assign(dataset=dataset, comparison_set=comparison_set)
            )
    if not frames:
        return pd.DataFrame()
    cols = [
        'comparison_set',
        'dataset',
        'metric',
        'model',
        'baseline',
        'median_diff',
        'n',
        'test',
        'higher_is_better',
        'better_than_baseline',
        'p_raw',
        'p_fdr',
        'significance',
    ]
    return pd.concat(frames, ignore_index=True)[cols]


def save_prm_vs_baseline_significance(
    all_outcomes: pd.DataFrame,
    *,
    systema: bool = False,
) -> pd.DataFrame:
    """Write PRM-vs-baseline tests for all families to results (supplementary).

    Each metric is compared against both Average effect and No effect.
    FDR (BH) is corrected across PRMs within dataset × metric × baseline × family.
    """
    metrics = SYSTEMA_METRICS if systema else MSE_PEARSON_METRICS
    frames: list[pd.DataFrame] = []
    for comparison_set, model_order in (
        ('cpa_family', CPA_FAMILY),
        ('gears_family', GEARS_FAMILY),
        ('all_models', ALL_FAMILY),
    ):
        for baseline in (AVERAGE_EFFECT, NO_EFFECT):
            frames.append(
                compute_prm_vs_baseline_table(
                    all_outcomes,
                    model_order=model_order,
                    metrics=metrics,
                    baseline=baseline,
                    comparison_set=comparison_set,
                )
            )
    frames = [f for f in frames if not f.empty]
    if not frames:
        logger.warning('No PRM-vs-baseline significance rows to save')
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    tag = '_systema' if systema else ''
    path = os.path.join(results_dir, f'prm_vs_baseline_significance{tag}.csv')
    out.to_csv(path, index=False)
    logger.info('Saved supplementary significance table %s (%d rows)', path, len(out))
    return out


def save_reconstruction_metrics_summary(
    all_outcomes: pd.DataFrame,
    *,
    systema: bool = False,
) -> pd.DataFrame:
    """Mean±s.d. / median reconstruction metrics per dataset × model × metric."""
    metrics = SYSTEMA_METRICS if systema else MSE_PEARSON_METRICS
    rows: list[dict] = []
    for dataset in DATASET_ORDER:
        data = _prepare_plot_frame(all_outcomes, dataset, ALL_FAMILY)
        if data.empty:
            continue
        for model in ALL_FAMILY:
            for metric in metrics:
                vals = data.loc[
                    (data['model'] == model) & (data['metric'] == metric), 'value'
                ].dropna()
                if vals.empty:
                    continue
                rows.append(
                    {
                        'dataset': dataset,
                        'model': model,
                        'metric': metric,
                        'n': int(len(vals)),
                        'mean': float(vals.mean()),
                        'std': float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                        'median': float(vals.median()),
                        'q25': float(vals.quantile(0.25)),
                        'q75': float(vals.quantile(0.75)),
                    }
                )
    out = pd.DataFrame(rows)
    tag = '_systema' if systema else ''
    path = os.path.join(results_dir, f'reconstruction_metrics_summary{tag}.csv')
    out.to_csv(path, index=False)
    logger.info('Saved reconstruction metrics summary %s (%d rows)', path, len(out))
    return out


def _fmt_mean_sd(mean: float, std: float, metric: str) -> str:
    if not np.isfinite(mean):
        return '--'
    if 'mse' in metric:
        return f'{mean:.2e} $\\pm$ {std:.2e}'
    return f'{mean:.3f} $\\pm$ {std:.3f}'


def _metric_cell(summary: pd.DataFrame, dataset: str, model: str, metric: str) -> str:
    row = summary[
        (summary['dataset'] == dataset)
        & (summary['model'] == model)
        & (summary['metric'] == metric)
    ]
    if row.empty:
        return '--'
    mean = float(row.iloc[0]['mean'])
    std = float(row.iloc[0]['std'])
    if not np.isfinite(mean):
        return '--'
    if 'mse' in metric:
        return rf'\shortstack[c]{{{mean:.2e}\\$\pm$\,{std:.2e}}}'
    return rf'\shortstack[c]{{{mean:.3f}\\$\pm$\,{std:.3f}}}'


def _metric_label(metric: str, *, systema: bool) -> str:
    label_dict = SYSTEMA_LABELS if systema else MSE_PEARSON_LABELS
    return label_dict.get(metric, metric).replace('$\\leftarrow$', '').replace('$\\rightarrow$', '').strip()


def _fmt_mean_sd_recon(mean: float, std: float, metric: str) -> str:
    if not np.isfinite(mean):
        return '--'
    if 'mse' in metric:
        if not np.isfinite(std):
            return f'{mean:.2e}'
        return f'{mean:.2e} $\\pm$ {std:.2e}'
    # Correlation / Pearson-style metrics: 2 decimals.
    if not np.isfinite(std):
        return f'{mean:.2f}'
    return f'{mean:.2f} $\\pm$ {std:.2f}'


def _fmt_median_delta(delta: float, metric: str) -> str:
    if not np.isfinite(delta):
        return '--'
    if 'mse' in metric:
        return f'{delta:+.2e}'
    return f'{delta:+.2f}'


def _fmt_fdr_with_sig(p_fdr: float, significance: object) -> str:
    if not np.isfinite(p_fdr):
        return '--'
    text = f'{p_fdr:.2e}'
    sig = str(significance) if significance is not None else 'NS'
    if sig == '**':
        return text + r'$^{**}$'
    if sig == '*':
        return text + r'$^{*}$'
    return text


def _lookup_summary_row(
    summary: pd.DataFrame, dataset: str, model: str, metric: str
) -> pd.Series | None:
    row = summary[
        (summary['dataset'] == dataset)
        & (summary['model'] == model)
        & (summary['metric'] == metric)
    ]
    if row.empty:
        return None
    return row.iloc[0]


def _lookup_sig_row(
    significance: pd.DataFrame | None,
    *,
    dataset: str,
    model: str,
    metric: str,
    baseline: str,
) -> pd.Series | None:
    if significance is None or significance.empty:
        return None
    src = significance
    if 'comparison_set' in src.columns:
        src = src[src['comparison_set'] == 'all_models']
    row = src[
        (src['dataset'] == dataset)
        & (src['model'] == model)
        & (src['metric'] == metric)
        & (src['baseline'] == baseline)
    ]
    if row.empty:
        return None
    return row.iloc[0]


def write_reconstruction_metrics_summary_tex(
    summary: pd.DataFrame,
    *,
    systema: bool = False,
    summary_systema: pd.DataFrame | None = None,
    significance: pd.DataFrame | None = None,
    significance_systema: pd.DataFrame | None = None,
    out_name: str | None = None,
) -> Path:
    """Write reconstruction longtables in drug-response-table style.

    One longtable per metric: Dataset / Model / Mean±s.d. / median Δ vs Average
    effect + FDR / median Δ vs No effect + FDR. When ``summary_systema`` is set,
    emit the combined metric set (standard + Systema Pearson per cell line).
    """
    combined = summary_systema is not None
    if combined:
        models = [
            m for m in ALL_FAMILY
            if m in set(summary['model']) or m in set(summary_systema['model'])
        ]
        specs = COMBINED_SUMMARY_SPECS
        out_name = out_name or 'reconstruction_metrics_summary_combined.tex'
    else:
        models = [m for m in ALL_FAMILY if m in set(summary['model'])]
        metrics = SYSTEMA_METRICS if systema else MSE_PEARSON_METRICS
        specs = tuple((systema, m) for m in metrics)
        tag = '_systema' if systema else ''
        out_name = out_name or f'reconstruction_metrics_summary{tag}.tex'

    summary_path = Path(figures_dir) / out_name
    colspec = (
        r'>{\raggedright\arraybackslash}p{0.11\textwidth}'
        r'>{\raggedright\arraybackslash}p{0.13\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.13\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.15\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.09\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.15\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.09\textwidth}'
    )
    header = (
        r'Dataset & Model & Mean $\pm$ s.d. & '
        r'Median $\Delta$ vs Avg & FDR & '
        r'Median $\Delta$ vs Noeff & FDR \\'
    )

    lines: list[str] = [
        '% Auto-generated reconstruction accuracy (mean±sd + median Δ vs baselines).',
        '% Requires: booktabs, longtable.',
        '% Page packing: MSE+Pearson; MSE_DE+Pearson_DE; Systema alone.',
        '',
    ]

    # Pair standard metrics two-per-page; Systema on its own page.
    page_breaks_after = {1, 3}  # after pearson, after pearson_de

    for i_spec, (systema_flag, metric) in enumerate(specs):
        src = summary_systema if (combined and systema_flag) else summary
        sig_src = (
            significance_systema if (combined and systema_flag) else significance
        )
        label = _metric_label(metric, systema=systema_flag)
        caption = (
            f'Reconstruction: {label} '
            r'(mean $\pm$ s.d.; median $\Delta$ vs Avg / Noeff; BH-FDR; '
            r'$^{**}$ FDR$<0.01$, $^{*}$ FDR$<0.05$).'
        )
        body: list[str] = []
        for dataset in DATASET_ORDER:
            ds_lab = DATASET_TITLES.get(dataset, dataset)
            for model in models:
                srow = _lookup_summary_row(src, dataset, model, metric)
                if srow is None:
                    mean_tex = '--'
                else:
                    mean_tex = _fmt_mean_sd_recon(
                        float(srow['mean']), float(srow['std']), metric
                    )
                # Baselines: no self-comparison deltas.
                delta_cells: list[str] = []
                for baseline in (AVERAGE_EFFECT, NO_EFFECT):
                    if model == baseline:
                        delta_cells.extend(['--', '--'])
                        continue
                    rsig = _lookup_sig_row(
                        sig_src,
                        dataset=dataset,
                        model=model,
                        metric=metric,
                        baseline=baseline,
                    )
                    if rsig is None:
                        delta_cells.extend(['--', '--'])
                    else:
                        delta_cells.append(
                            _fmt_median_delta(float(rsig['median_diff']), metric)
                        )
                        delta_cells.append(
                            _fmt_fdr_with_sig(
                                float(rsig['p_fdr']), rsig.get('significance', 'NS')
                            )
                        )
                cells = [
                    ds_lab,
                    model,
                    mean_tex,
                    *delta_cells,
                ]
                body.append(' & '.join(cells) + r' \\')

        block: list[str] = [
            r'\begingroup',
            r'\tiny',
            r'\setlength{\tabcolsep}{2.5pt}',
            r'\setlength{\abovecaptionskip}{2pt}',
            r'\setlength{\belowcaptionskip}{1pt}',
            r'\renewcommand{\arraystretch}{0.92}',
            r'\setlength{\LTleft}{\fill}',
            r'\setlength{\LTright}{\fill}',
            rf'\begin{{longtable}}{{{colspec}}}',
            rf'\caption{{{caption}}}',
            rf'\label{{tab:reconstruction-{metric}}}\\',
            r'\toprule',
            header,
            r'\midrule',
            r'\endfirsthead',
            rf'\caption[]{{{caption} (continued)}}\\',
            r'\toprule',
            header,
            r'\midrule',
            r'\endhead',
            r'\midrule',
            r'\multicolumn{7}{r}{\textit{Continued on next page}}\\',
            r'\endfoot',
            r'\bottomrule',
            r'\endlastfoot',
            *body,
            r'\end{longtable}',
            r'\endgroup',
        ]
        if i_spec in page_breaks_after:
            block.extend(['', r'\clearpage', ''])
        elif i_spec < len(specs) - 1:
            block.append(r'\vspace{0.4em}')
            block.append('')
        else:
            block.append('')
        lines.extend(block)

    summary_path.write_text('\n'.join(lines), encoding='utf-8')
    logger.info('Wrote %s (%d metrics)', summary_path, len(specs))
    return summary_path


def write_prm_vs_baseline_significance_tex(
    significance: pd.DataFrame,
    *,
    systema: bool = False,
    significance_systema: pd.DataFrame | None = None,
    out_name: str | None = None,
) -> Path:
    """Write PRM-vs-baseline significance longtable.

    If ``significance_systema`` is provided, emit one combined table matching
    ``COMBINED_SUMMARY_SPECS`` (standard metrics + Systema Pearson per cell line).
    """
    combined = significance_systema is not None
    if combined:
        specs = COMBINED_SUMMARY_SPECS
        caption = (
            r'PRM vs baseline reconstruction: median $\Delta$ (PRM $-$ baseline); '
            r'paired Wilcoxon (else Mann--Whitney); BH-FDR within panel; '
            r'stars require better-than-baseline and FDR $<0.05$ '
            r'($^{**}$ FDR$<0.01$, $^{*}$ FDR$<0.05$).'
        )
        out_name = out_name or 'prm_vs_baseline_significance_combined.tex'
    else:
        metrics = SYSTEMA_METRICS if systema else MSE_PEARSON_METRICS
        specs = tuple((systema, m) for m in metrics)
        caption = (
            r'PRM vs baseline reconstruction'
            + (' (Systema)' if systema else '')
            + r': median $\Delta$ (PRM $-$ baseline); paired Wilcoxon (else Mann--Whitney); '
            r'BH-FDR within panel; stars require better-than-baseline and FDR $<0.05$ '
            r'($^{**}$ FDR$<0.01$, $^{*}$ FDR$<0.05$).'
        )
        tag = '_systema' if systema else ''
        out_name = out_name or f'prm_vs_baseline_significance{tag}.tex'

    sig_path = os.path.join(figures_dir, out_name)
    prm_models = [m for m in ALL_FAMILY if m in PRM_MODELS]
    sig_std = significance[significance['comparison_set'] == 'all_models'].copy()
    sig_sys = (
        significance_systema[significance_systema['comparison_set'] == 'all_models'].copy()
        if combined
        else None
    )

    sig_lines = [
        '% Auto-generated PRM vs Average effect / No effect significance.',
        r'\begingroup',
        r'\scriptsize',
        r'\setlength{\tabcolsep}{3pt}',
        r'\setlength{\LTleft}{\fill}',
        r'\setlength{\LTright}{\fill}',
        r'\begin{longtable}{@{}lllcccc@{}}',
        rf'\caption{{{caption}}}\\',
        r'\toprule',
        r'Dataset & Metric & Baseline & Model & Median $\Delta$ & $n$ & FDR \\',
        r'\midrule',
        r'\endfirsthead',
        rf'\caption[]{{{caption} (continued)}}\\',
        r'\toprule',
        r'Dataset & Metric & Baseline & Model & Median $\Delta$ & $n$ & FDR \\',
        r'\midrule',
        r'\endhead',
        r'\midrule',
        r'\multicolumn{7}{r}{\textit{Continued on next page}}\\',
        r'\endfoot',
        r'\bottomrule',
        r'\endlastfoot',
    ]
    for dataset in DATASET_ORDER:
        for systema_flag, metric in specs:
            src = sig_sys if (combined and systema_flag) else sig_std
            label = _metric_label(metric, systema=systema_flag)
            for baseline in (AVERAGE_EFFECT, NO_EFFECT):
                for model in prm_models:
                    row = src[
                        (src['dataset'] == dataset)
                        & (src['metric'] == metric)
                        & (src['baseline'] == baseline)
                        & (src['model'] == model)
                    ]
                    if row.empty:
                        continue
                    r0 = row.iloc[0]
                    delta = r0['median_diff']
                    p_fdr = r0['p_fdr']
                    if not np.isfinite(delta):
                        delta_s = '--'
                    elif 'mse' in metric:
                        delta_s = f'{delta:+.2e}'
                    else:
                        delta_s = f'{delta:+.3f}'
                    if not np.isfinite(p_fdr):
                        fdr_s = '--'
                    else:
                        fdr_s = f'{p_fdr:.2e}'
                        sig_lab = str(r0['significance'])
                        if sig_lab == '**':
                            fdr_s = fdr_s + r'$^{**}$'
                        elif sig_lab == '*':
                            fdr_s = fdr_s + r'$^{*}$'
                    sig_lines.append(
                        f'{dataset} & {label} & {baseline} & {model} & '
                        f'{delta_s} & {int(r0["n"])} & {fdr_s} \\\\'
                    )
    sig_lines.extend([r'\end{longtable}', r'\endgroup', ''])
    Path(sig_path).write_text('\n'.join(sig_lines), encoding='utf-8')
    logger.info('Wrote %s', sig_path)
    return Path(sig_path)


def write_reconstruction_supplement_tex(
    summary: pd.DataFrame,
    significance: pd.DataFrame,
    *,
    systema: bool = False,
    summary_systema: pd.DataFrame | None = None,
    significance_systema: pd.DataFrame | None = None,
) -> None:
    """Write TeX tables for reconstruction summary (+ optional PRM-vs-baseline Δ)."""
    write_reconstruction_metrics_summary_tex(
        summary,
        systema=systema,
        summary_systema=summary_systema,
        significance=significance,
        significance_systema=significance_systema,
    )
    # Standalone significance longtable kept for optional use / debugging.
    write_prm_vs_baseline_significance_tex(
        significance,
        systema=systema,
        significance_systema=significance_systema,
    )


def _draw_metric_panel(
    ax, data_subset: pd.DataFrame, model_order: tuple[str, ...], hide_xticklabels: bool = False
) -> None:
    data_subset, models = _add_x_positions(data_subset, model_order)
    rng = np.random.default_rng(0)

    for model, grp in data_subset.groupby('model', observed=True):
        x = grp['x_pos'].iloc[0]
        color = MODEL_COLORS[model]
        ax.boxplot(
            [grp['value'].dropna().values],
            positions=[x],
            widths=BOX_WIDTH,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
            medianprops={'color': 'black', 'linewidth': 1.1},
            boxprops={'facecolor': 'lightgrey', 'edgecolor': '#333333', 'linewidth': 0.9},
            whiskerprops={'color': '#444444', 'linewidth': 0.8},
            capprops={'color': '#444444', 'linewidth': 0.8},
            zorder=2,
        )
        jitter = rng.uniform(-0.1, 0.1, size=len(grp))
        ax.scatter(
            grp['x_pos'].values + jitter,
            grp['value'].values,
            c=[color],
            s=20,
            alpha=1,
            linewidths=0,
            zorder=3,
        )

    ax.set_xticks(range(len(models)))
    if hide_xticklabels:
        ax.tick_params(axis='x', labelbottom=False)
    else:
        ax.set_xticklabels(models, fontsize=12)
    ax.set_xlim(-0.55, len(models) - 0.45)
    if not hide_xticklabels:
        _style_model_xticks(ax)
    ax.grid(True, which='major', axis='y', linestyle='--', color='#cccccc', zorder=0, alpha=0.7)
    ax.grid(False, axis='x')


def _metric_ylim(metric: str, values: pd.Series) -> tuple[float, float]:
    if values.empty:
        return (-1.05, 1.05) if 'mse' not in metric else (1e-5, 1.0)
    if 'mse' in metric:
        ymax = max(values.max() * 1.55, 1e-4)
        return 1e-6, ymax
    return -1.05, 1.20


def _hide_for_metric(flag: bool | Collection[str], metric: str) -> bool:
    if isinstance(flag, (str, bytes)):
        raise TypeError('pass a set of metric names, not a string')
    if isinstance(flag, bool):
        return flag
    return metric in flag


def _plot_metric_side_by_side(
    by_dataset: dict[str, pd.DataFrame],
    metric: str,
    ylabel: str,
    model_order: tuple[str, ...],
    figsize: tuple[float, float] | None = None,
    hide_xticklabels: bool = False,
    hide_x_group_labels: bool = False,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=figsize or METRIC_FIGSIZE, dpi=300)

    y_values = []
    for dataset in DATASET_ORDER:
        if dataset in by_dataset:
            sub = by_dataset[dataset]
            sub = sub[(sub['metric'] == metric)].dropna(subset=['value'])
            if not sub.empty:
                y_values.append(sub['value'])

    ylim = _metric_ylim(metric, pd.concat(y_values) if y_values else pd.Series(dtype=float))
    yscale = 'log' if 'mse' in metric else 'linear'

    for ax, dataset in zip(axes, DATASET_ORDER):
        if dataset not in by_dataset:
            ax.set_visible(False)
            continue
        data_subset = by_dataset[dataset][by_dataset[dataset]['metric'] == metric].dropna(subset=['value'])
        if data_subset.empty:
            ax.set_visible(False)
            continue
        _draw_metric_panel(ax, data_subset, model_order, hide_xticklabels=hide_xticklabels)
        if not hide_x_group_labels:
            _add_x_group_labels(ax, list(model_order))
        ax.set_title(
            DATASET_TITLES.get(dataset, dataset),
            fontsize=34,
            fontweight='bold',
            pad=10,
            loc='center',
        )
        ax.set_xlabel('')
        ax.set_ylabel(ylabel, fontsize=22)
        ax.tick_params(axis='y', labelsize=16)
   
        ax.set_yscale(yscale)
        ax.set_ylim(ylim)
        sns.despine(ax=ax, offset=8, trim=False)

    for ax in axes:
        if ax.get_visible():
            if hide_xticklabels:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                _style_model_xticks(ax)

    # Room for rotated tick labels + category bars; leftover is cropped by
    # bbox_inches='tight'. Extra wspace so both panel y-labels fit.
    fig.subplots_adjust(bottom=0.18, top=0.88, wspace=0.22)
    return fig


def plot_all_metrics(
    all_outcomes: pd.DataFrame,
    model_order: tuple[str, ...],
    metrics: list[str],
    label_dict: dict[str, str],
    file_prefix: str,
    hide_xticklabels: bool | Collection[str],
    figsize: tuple[float, float] | None = None,
    hide_x_group_labels: bool | Collection[str] = False,
) -> dict[str, plt.Figure]:
    by_dataset = _prepare_datasets(all_outcomes, model_order)
    if not by_dataset:
        raise ValueError(f'No data to plot for models {model_order}')

    figures: dict[str, plt.Figure] = {}
    for metric in metrics:
        if not any((df['metric'] == metric).any() for df in by_dataset.values()):
            logger.info('Skipping %s: no rows', metric)
            continue
        fig = _plot_metric_side_by_side(
            by_dataset,
            metric=metric,
            ylabel=label_dict[metric],
            model_order=model_order,
            figsize=figsize,
            hide_xticklabels=_hide_for_metric(hide_xticklabels, metric),
            hide_x_group_labels=_hide_for_metric(hide_x_group_labels, metric),
        )
        out_path = os.path.join(figures_dir, f'{file_prefix}_{metric}.pdf')
        last_error: Exception | None = None
        tmp_path = out_path + '.tmp.pdf'
        for attempt in range(5):
            try:
                fig.savefig(tmp_path, bbox_inches='tight', dpi=300)
                os.replace(tmp_path, out_path)
                last_error = None
                break
            except PermissionError as exc:
                last_error = exc
                time.sleep(1.5)
        if last_error is not None:
            fallback = out_path.replace('.pdf', '_new.pdf')
            logger.warning('Could not overwrite %s (%s); writing %s', out_path, last_error, fallback)
            fig.savefig(fallback, bbox_inches='tight', dpi=300)
        else:
            logger.info('Saved %s', out_path)
        figures[metric] = fig
        plt.close(fig)
    return figures


def save_combined_outcomes(all_outcomes: pd.DataFrame, systema: bool = False) -> None:
    tag = '_systema' if systema else ''
    sciplex = all_outcomes[all_outcomes['dataset'] == 'SciPlex3 dataset']
    mcfarland = all_outcomes[all_outcomes['dataset'] == 'McFarland dataset']
    sciplex.to_csv(os.path.join(results_dir, f'sciplex_all_outcomes{tag}.csv'), index=False)
    mcfarland.to_csv(os.path.join(results_dir, f'mcfarland_all_outcomes{tag}.csv'), index=False)
    all_outcomes.to_csv(
        os.path.join(results_dir, f'sciplex_mcfarland_all_outcomes{tag}.csv'), index=False
    )


def main() -> None:
    outcomes = load_all_outcomes(systema=False)
    save_combined_outcomes(outcomes, systema=False)
    logger.info('Rows per dataset / model:\n%s', outcomes.groupby(['dataset', 'model']).size().unstack(fill_value=0))
    outcomes_clean = outcomes.dropna(subset=['value'])
    recon_summary = save_reconstruction_metrics_summary(outcomes_clean, systema=False)
    sig_table = save_prm_vs_baseline_significance(outcomes_clean, systema=False)
    write_reconstruction_supplement_tex(recon_summary, sig_table, systema=False)

    plot_all_metrics(
        outcomes.dropna(subset=['value']),
        model_order=CPA_FAMILY,
        metrics=MSE_PEARSON_METRICS,
        label_dict=MSE_PEARSON_LABELS,
        file_prefix='performance_boxplot_cpa_family',
        hide_xticklabels=False,
    )
    plot_all_metrics(
        outcomes.dropna(subset=['value']),
        model_order=GEARS_FAMILY,
        metrics=MSE_PEARSON_METRICS,
        label_dict=MSE_PEARSON_LABELS,
        file_prefix='performance_boxplot_gears_family',
        hide_xticklabels=False,
    )
    plot_all_metrics(
        outcomes.dropna(subset=['value']),
        model_order=ALL_FAMILY,
        metrics=MSE_PEARSON_METRICS,
        label_dict=MSE_PEARSON_LABELS,
        file_prefix='performance_boxplot_all_models',
        hide_xticklabels=False,
        figsize=ALL_FAMILY_FIGSIZE,
    )

    outcomes_systema = load_all_outcomes(systema=True)
    save_combined_outcomes(outcomes_systema, systema=True)
    logger.info(
        'Systema rows per dataset / model:\n%s',
        outcomes_systema.groupby(['dataset', 'model']).size().unstack(fill_value=0),
    )
    outcomes_systema_clean = outcomes_systema.dropna(subset=['value'])
    recon_summary_s = save_reconstruction_metrics_summary(outcomes_systema_clean, systema=True)
    sig_table_s = save_prm_vs_baseline_significance(outcomes_systema_clean, systema=True)
    write_reconstruction_supplement_tex(recon_summary_s, sig_table_s, systema=True)
    write_reconstruction_metrics_summary_tex(
        recon_summary,
        summary_systema=recon_summary_s,
        significance=sig_table,
        significance_systema=sig_table_s,
    )
    write_prm_vs_baseline_significance_tex(
        sig_table, significance_systema=sig_table_s
    )
    plot_all_metrics(
        outcomes_systema.dropna(subset=['value']),
        model_order=CPA_FAMILY,
        metrics=SYSTEMA_METRICS,
        label_dict=SYSTEMA_LABELS,
        file_prefix='performance_boxplot_systema_cpa_family',
        hide_xticklabels=False,
    )
    plot_all_metrics(
        outcomes_systema.dropna(subset=['value']),
        model_order=GEARS_FAMILY,
        metrics=SYSTEMA_METRICS,
        label_dict=SYSTEMA_LABELS,
        file_prefix='performance_boxplot_systema_gears_family',
        hide_xticklabels=False,
    )
    plot_all_metrics(
        outcomes_systema.dropna(subset=['value']),
        model_order=ALL_FAMILY,
        metrics=SYSTEMA_METRICS,
        label_dict=SYSTEMA_LABELS,
        file_prefix='performance_boxplot_systema_all_models',
        hide_xticklabels=False,
        figsize=ALL_FAMILY_FIGSIZE,
    )


if __name__ == '__main__':
    main()
