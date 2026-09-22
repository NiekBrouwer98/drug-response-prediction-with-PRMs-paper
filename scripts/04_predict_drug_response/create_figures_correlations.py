"""Gene-gene correlation and sensitivity figures for all PRMs + baselines.

Models: measured, CPA, chemCPA, PRnet, GEARS, scFoundation, average effect, no effect.
Requires step-04 split-CV outputs (``sciplex_split_cv_smiles_feature_importance.pkl``;
gene features only for ElasticNet weights).

Gene–gene heatmaps use the top-1000 highest-variance genes from measured pre
profiles (same selection as the drug-response models), intersected with genes
shared across profile tables. Untreated controls are excluded; the Measured
panel uses observations and other panels use each model's predictions.

The SciPlex gene–sensitivity clustermap uses the 100 genes with largest
|ElasticNet weight| (with MKI67 / UBE2H annotations) and companion
gene–sensitivity scatter panels for concordant-weight genes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent))

from config import config, setup_project
from compute_correlations import (
    ALL_FAMILY_DISPLAY_NAMES,
    ALL_FAMILY_ORDER,
    build_gene_gene_corr_dict,
    common_genes,
    compute_correlation_with_y,
    gene_gene_cosine_sims_per_cell_line,
    get_mcfarland_predictions,
    get_observations,
    get_predictions,
    get_sciplex_post_feature_importance,
    plot_gene_gene_cosine_sim_boxplot,
    plot_gene_gene_heatmaps_by_cell_line,
    plot_pooled_gene_gene_heatmaps,
)
from feature_importance_io import DEFAULT_SCIPLEX_FI_PREFIX
from prediction_utils import (
    feature_selection,
    get_McFarland_mean_data,
    get_sciplex_AUCs,
    get_sciplex_mean_data,
    merge_sciplex_sensitivity,
)
from utils import ensure_directories_exist, setup_logging_for_script

setup_project()
logger = setup_logging_for_script(__file__)

FIGURE_STEM = 'all_models'
RESULTS_PREFIX = DEFAULT_SCIPLEX_FI_PREFIX
HVG_N_TOP = 1000
figures_dir = str(config.FIGURES_04_DIR)
resources_dir = str(config.RESOURCES_DIR)
ensure_directories_exist(figures_dir)

_FEATURE_SELECTION_META = frozenset({'tissue', 'condition', 'cell_line'})


def top_n_pre_hvgs(
    pre_df: pd.DataFrame,
    n: int = HVG_N_TOP,
    allowed: list[str] | None = None,
) -> list[str]:
    """Top-n highest-variance genes from measured pre profiles (same as response models).

    When ``allowed`` is set, variance ranking is restricted to that gene universe
    (typically genes shared across measured + predicted profile tables).
    """
    df = pre_df
    if allowed is not None:
        meta = [c for c in ('cell_line', 'condition', 'tissue') if c in pre_df.columns]
        keep = [g for g in allowed if g in pre_df.columns] + meta
        df = pre_df.loc[:, keep]
    selected = feature_selection(df, n_features=n)
    return [g for g in selected if g not in _FEATURE_SELECTION_META]


def _write_gene_gene_heatmap_suite(
    predictions: dict[str, pd.DataFrame],
    genes: list[str],
    *,
    stem: str,
    dataset: str,
    ylabel: str,
    heatmap_prefix: str = '',
    per_cell_line_heatmaps: bool = False,
    sciplex_cell_lines: tuple[str, ...] = ('MCF7', 'A549', 'K562'),
    single_cell_lines: tuple[str, ...] = (),
) -> None:
    """Pooled (+ optional per-line) gene–gene heatmaps and cosine-sim boxplot."""
    logger.info(
        '%s gene–gene suite (%s): %d genes → stem=%s',
        dataset, ylabel, len(genes), stem,
    )
    corrs = build_gene_gene_corr_dict(predictions, genes, per_cell_line=True)
    pooled_corrs = build_gene_gene_corr_dict(predictions, genes, per_cell_line=False)

    plot_pooled_gene_gene_heatmaps(
        pooled_corrs,
        outfile=os.path.join(
            figures_dir, f'gene_gene_correlations_{heatmap_prefix}{stem}.pdf'
        ),
        ylabel=ylabel,
    )
    plt.close('all')

    lines_to_plot: list[str] = []
    if per_cell_line_heatmaps:
        plot_gene_gene_heatmaps_by_cell_line(
            corrs,
            outfile=os.path.join(
                figures_dir, f'gene_gene_correlations_per_cell_line_{heatmap_prefix}{stem}.pdf'
            ),
            cell_lines=sciplex_cell_lines,
        )
        plt.close('all')
        lines_to_plot.extend(sciplex_cell_lines)
    lines_to_plot.extend(single_cell_lines)

    seen: set[str] = set()
    for cell_line in lines_to_plot:
        if cell_line in seen:
            continue
        seen.add(cell_line)
        plot_gene_gene_heatmaps_by_cell_line(
            corrs,
            cell_lines=(cell_line,),
            outfile=os.path.join(
                figures_dir,
                # Keep dataset prefix before cell line (e.g. mcfarland_caov3_…).
                f'gene_gene_correlations_{heatmap_prefix}{cell_line.lower()}_{stem}.pdf',
            ),
            figsize=(3.5 * len(ALL_FAMILY_ORDER), 3.2),
        )
        plt.close('all')

    sims = gene_gene_cosine_sims_per_cell_line(corrs)
    plot_gene_gene_cosine_sim_boxplot(
        sims,
        outfile=os.path.join(
            figures_dir, f'gene_gene_cosine_sim_{dataset}_boxplot_{stem}.pdf'
        ),
        title=ylabel,
    )
    plt.close('all')


def _sensitivity_merged_frames(all_predictions: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    sens = get_sciplex_AUCs().rename(columns={'y': 'sensitivity'})
    out: dict[str, pd.DataFrame] = {}
    for key, frame in all_predictions.items():
        merged = merge_sciplex_sensitivity(frame, sens, value_col='sensitivity')
        out[key] = merged
    return out


def top_n_genes_by_abs_importance(
    feature_importance: pd.DataFrame,
    n: int,
    allowed: list[str] | None = None,
) -> list[str]:
    """Return ``n`` genes with the largest |ElasticNet weight|, preserving abs rank."""
    fi = feature_importance
    if allowed is not None:
        fi = fi[fi['gene'].isin(allowed)]
    ranked = fi['importance'].abs().sort_values(ascending=False)
    return fi.loc[ranked.index[:n], 'gene'].tolist()


def elasticnet_active_genes(
    feature_importance: pd.DataFrame,
    allowed: list[str] | None = None,
    *,
    abs_weight_quantile: float = 0.9,
) -> list[str]:
    """Genes with |ElasticNet weight| at/above a quantile within ``allowed`` (default top decile).

    Among the allowed set (typically HVGs), keep genes whose |mean weight| is at least
    the ``abs_weight_quantile`` of |weights| in that set. Ordered by descending |weight|.
    """
    fi = feature_importance
    if allowed is not None:
        fi = fi[fi['gene'].isin(allowed)]
    if fi.empty:
        return []
    abs_w = fi['importance'].abs()
    threshold = float(abs_w.quantile(abs_weight_quantile))
    selected = fi.loc[abs_w >= threshold].copy()
    selected = selected.assign(_abs=selected['importance'].abs()).sort_values(
        '_abs', ascending=False
    )
    logger.info(
        'ElasticNet weight filter: |w| >= %.3g (q=%.2f) → %d / %d genes',
        threshold, abs_weight_quantile, len(selected), len(fi),
    )
    return selected['gene'].tolist()


def select_concordant_weight_genes(
    all_predictions: dict[str, pd.DataFrame],
    feature_importance: pd.DataFrame,
    allowed: list[str] | None = None,
    n: int = 3,
) -> list[str]:
    """Top-n +/− ElasticNet weights whose CPA–sensitivity Spearman matches the weight sign."""
    fi = feature_importance
    if allowed is not None:
        fi = fi[fi['gene'].isin(allowed)]
    genes = fi['gene'].tolist()
    cpa_corr = compute_correlation_with_y(
        all_predictions['CPA'],
        subset=genes + ['cell_line', 'condition'],
    ).rename('cpa_corr')
    merged = fi.merge(cpa_corr.rename_axis('gene').reset_index(), on='gene', how='inner')

    pos = merged.loc[(merged['importance'] > 0) & (merged['cpa_corr'] > 0)]
    neg = merged.loc[(merged['importance'] < 0) & (merged['cpa_corr'] < 0)]
    if pos.empty or neg.empty:
        raise ValueError(
            'No genes with matching ElasticNet sign and CPA–sensitivity correlation '
            f'(n_pos={len(pos)}, n_neg={len(neg)})'
        )
    pos_rows = pos.sort_values('importance', ascending=False).head(n)
    neg_rows = neg.sort_values('importance', ascending=True).head(n)
    selected = pd.concat([neg_rows, pos_rows], ignore_index=True)
    for _, row in selected.iterrows():
        logger.info(
            'Sensitivity-panel gene: %s (w=%.3g, CPA rho=%.2f)',
            row['gene'], float(row['importance']), float(row['cpa_corr']),
        )
    return selected['gene'].astype(str).tolist()


def _contrasting_text_color(rgba: tuple[float, ...]) -> str:
    r, g, b = rgba[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return 'black' if luminance > 0.55 else 'white'


def plot_sensitivity_clustermap(
    all_predictions: dict[str, pd.DataFrame],
    post_feature_importance: pd.DataFrame,
    important_features: list[str],
    outfile: str,
    highlight_genes: list[str] | None = None,
    model_order: list[str] | None = None,
) -> None:
    model_order = model_order or ALL_FAMILY_ORDER
    results: dict[str, pd.Series] = {}
    subset_cols = important_features + ['cell_line', 'condition', 'y']
    for model in model_order:
        if model == 'importance' or model not in all_predictions:
            continue
        results[model] = compute_correlation_with_y(
            all_predictions[model],
            subset=[c for c in subset_cols if c in all_predictions[model].columns],
        )

    results_df = pd.DataFrame(results)
    ordered_cols = [m for m in model_order if m in results_df.columns and m != 'importance']
    results_df = results_df[ordered_cols]
    results_with_fi = results_df.merge(post_feature_importance, left_index=True, right_on='gene', how='left')
    results_with_fi = results_with_fi.set_index('gene')
    results_with_fi = results_with_fi[ordered_cols + ['importance']]
    results_with_fi = results_with_fi.loc[results_with_fi.index.isin(important_features)]
    results_with_fi = results_with_fi.rename(columns=ALL_FAMILY_DISPLAY_NAMES)

    weights = results_with_fi['importance']
    n_neg = int((weights < 0).sum())
    n_pos = int((weights > 0).sum())
    logger.info(
        'Clustermap genes: %d (abs-selected); %d negative, %d positive; |w| %.3g–%.3g',
        len(weights), n_neg, n_pos, float(weights.abs().min()), float(weights.abs().max()),
    )

    # Feature-weight strip: independent of heatmap ±0.5; narrower than full |w|
    # range so mid weights keep contrast (extremes saturate). No weight colorbar.
    abs_w = np.abs(weights.to_numpy(dtype=float))
    vmax_w = float(np.nanpercentile(abs_w, 75))
    vmax_max = float(np.nanmax(abs_w)) if np.isfinite(np.nanmax(abs_w)) else 0.0
    if not np.isfinite(vmax_w) or vmax_w <= 0:
        vmax_w = vmax_max if vmax_max > 0 else 1.0
    logger.info('Feature-weight strip scale ±%.3g (75th |w|; max |w|=%.3g)', vmax_w, vmax_max)
    weight_norm = mcolors.TwoSlopeNorm(vmin=-vmax_w, vcenter=0.0, vmax=vmax_w)
    weight_cmap = cm.get_cmap('coolwarm')
    column_values = results_with_fi.sort_values(by='importance')['importance']
    row_annotation = pd.DataFrame(
        {
            'feature weight': column_values.map(
                lambda x: weight_cmap(weight_norm(float(np.clip(x, -vmax_w, vmax_w))))
            ),
        },
        index=column_values.index,
    )
    heatmap_df = results_with_fi.sort_values(by='importance').drop(columns=['importance'])
    heatmap_df.index.name = None
    target_vector = heatmap_df['Measured'].values.reshape(-1, 1).T
    similarities = cosine_similarity(heatmap_df.T, target_vector).flatten()
    cosine_sim_series = pd.Series(similarities, index=heatmap_df.columns)
    col_norm = mcolors.Normalize(vmin=cosine_sim_series.min(), vmax=cosine_sim_series.max())
    col_cmap = cm.get_cmap('Reds')
    col_colors = cosine_sim_series.map(lambda x: col_cmap(col_norm(x)))
    column_annotation = pd.DataFrame({'cos. sim.': col_colors}, index=heatmap_df.columns)

    corr_norm = mcolors.Normalize(vmin=-0.5, vmax=0.5)
    corr_cmap = cm.get_cmap('coolwarm')

    mpl.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
    g = sns.clustermap(
        heatmap_df,
        cmap=corr_cmap,
        center=0,
        vmax=0.5,
        vmin=-0.5,
        method='average',
        row_cluster=False,
        col_cluster=False,
        row_colors=row_annotation,
        col_colors=column_annotation,
        yticklabels=False,
        xticklabels=True,
        figsize=(12, 10),
        cbar_pos=None,
        dendrogram_ratio=(0.16, 0.04),
        colors_ratio=(0.06, 0.06),
    )

    if hasattr(g, 'ax_col_dendrogram'):
        g.ax_col_dendrogram.set_visible(False)
        g.ax_row_dendrogram.set_visible(False)

    g.ax_row_colors.set_ylabel('')
    g.ax_row_colors.set_xticklabels(
        g.ax_row_colors.get_xticklabels(),
        rotation=45, ha='right', rotation_mode='anchor', fontsize=16,
    )
    g.ax_row_colors.tick_params(axis='x', length=0)

    for x, col in enumerate(heatmap_df.columns):
        g.ax_col_colors.text(
            x + 0.5, 0.5, f'{cosine_sim_series[col]:.2f}',
            ha='center', va='center', fontsize=16,
            color=_contrasting_text_color(col_colors.iloc[x]),
            clip_on=True,
        )
    for x in range(len(heatmap_df.columns)):
        g.ax_col_colors.add_patch(Rectangle(
            (x, 0), 1, 1, fill=False, edgecolor='0.25', linewidth=0.6, zorder=3,
        ))
    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor', fontsize=22,
    )
    g.ax_heatmap.set_ylabel('')
    g.ax_col_colors.tick_params(axis='y', labelsize=14)

    heatmap_pos = g.ax_heatmap.get_position()
    cbar_ax = g.fig.add_axes([
        heatmap_pos.x1 + 0.04,
        heatmap_pos.y0,
        0.025,
        heatmap_pos.height,
    ])
    corr_cb = ColorbarBase(
        cbar_ax,
        cmap=corr_cmap,
        norm=corr_norm,
        orientation='vertical',
    )
    corr_cb.set_ticks([-0.5, -0.25, 0.0, 0.25, 0.5])
    corr_cb.set_label('Correlation with sensitivity (1-AUDRC)', fontsize=18, rotation=90, labelpad=12)
    corr_cb.ax.tick_params(labelsize=16)

    if highlight_genes:
        gene_order = list(heatmap_df.index)
        for gene in highlight_genes:
            if gene not in gene_order:
                logger.warning('Clustermap highlight gene %s is not in the plotted set', gene)
                continue
            y = gene_order.index(gene) + 0.5
            g.ax_row_colors.annotate(
                gene,
                xy=(0.0, y),
                xycoords=('axes fraction', 'data'),
                xytext=(-18, 0),
                textcoords='offset points',
                ha='right',
                va='center',
                fontsize=16,
                fontweight='bold',
                arrowprops=dict(arrowstyle='-|>', color='black', lw=1.4, mutation_scale=14),
                annotation_clip=False,
                clip_on=False,
            )

    g.savefig(outfile, bbox_inches='tight', dpi=300)
    plt.close(g.fig)


def _spearman_trend_line(x: np.ndarray, y: np.ndarray, rho: float) -> tuple[np.ndarray, np.ndarray]:
    """Line through the means whose slope matches Spearman rho (flat when rho is 0)."""
    sx = float(np.std(x, ddof=1))
    sy = float(np.std(y, ddof=1))
    slope = 0.0 if sx == 0 or not np.isfinite(rho) else rho * sy / sx
    intercept = float(np.mean(y) - slope * np.mean(x))
    x_line = np.array([np.min(x), np.max(x)], dtype=float)
    return x_line, intercept + slope * x_line


def plot_gene_sensitivity_panels(
    sens_frames: dict[str, pd.DataFrame],
    genes: list[str],
    figures_dir_path: str,
    stem: str | None = None,
) -> None:
    stem = stem or FIGURE_STEM
    display_order = [
        ('observations', 'Measured', '#8B0000'),
        ('CPA', 'CPA', '#1f77b4'),
        ('average_effect', 'Average effect', '#4d4d4d'),
    ]

    for gene in genes:
        fig, ax = plt.subplots(figsize=(5, 4.4), dpi=300)
        text_y = 0.95
        obs = sens_frames['observations']
        for key, label, color in display_order:
            df = sens_frames[key]
            valid = df[['sensitivity', gene]].dropna()
            x = valid['sensitivity'].to_numpy()
            y = valid[gene].to_numpy()
            r, _ = spearmanr(x, y)
            ax.scatter(x, y, s=20, alpha=0.7, color=color, linewidths=0)
            x_line, y_line = _spearman_trend_line(x, y, r)
            ax.plot(x_line, y_line, color=color, lw=3)
            ax.text(
                0.08, text_y, f'{label}: ρ = {r:.2f}',
                transform=ax.transAxes, fontsize=14, fontweight='bold', color=color,
                ha='left', va='top',
            )
            text_y -= 0.07
        ax.set_ylabel(f'{gene} expression\n(log-normalized counts)', fontsize=16)
        ax.set_xlabel('Sensitivity (1-AUDRC)', fontsize=16)
        ax.set_ylim([obs[gene].min() - 0.02, obs[gene].max() + 0.15])
        sns.despine(ax=ax, offset=10, trim=False)
        fig.savefig(
            os.path.join(figures_dir_path, f'sciplex_{gene}_vs_sensitivity_{stem}.pdf'),
            bbox_inches='tight', dpi=300,
        )
        plt.close(fig)


def main() -> None:
    logger.info('Loading all-model profiles for correlation figures...')
    all_predictions = get_predictions()
    mcfarland_predictions = get_mcfarland_predictions()

    # New filenames only (do not overwrite legacy ElasticNet top-|w| PDFs).
    hvg_stem = f'hvg{HVG_N_TOP}_{FIGURE_STEM}'
    shared_genes = common_genes(all_predictions)
    sciplex_pre, _, _ = get_sciplex_mean_data()
    sciplex_hvgs = top_n_pre_hvgs(sciplex_pre, n=HVG_N_TOP, allowed=shared_genes)
    logger.info('%d SciPlex pre HVGs (n=%d) shared across profiles', len(sciplex_hvgs), HVG_N_TOP)
    _write_gene_gene_heatmap_suite(
        all_predictions,
        sciplex_hvgs,
        stem=hvg_stem,
        dataset='sciplex',
        ylabel='SciPlex3',
        per_cell_line_heatmaps=True,
    )

    post_fi = get_sciplex_post_feature_importance(RESULTS_PREFIX)
    # Legacy gene–sensitivity clustermap + gene–sensitivity scatters (top-100 |w|).
    post_important = top_n_genes_by_abs_importance(post_fi, 100, allowed=shared_genes)
    logger.info(
        '%d ElasticNet top-|w| genes for sensitivity clustermap / panels',
        len(post_important),
    )
    plot_sensitivity_clustermap(
        all_predictions,
        post_fi,
        post_important,
        os.path.join(figures_dir, f'sciplex_feature_importance_clustermap_{FIGURE_STEM}.pdf'),
        highlight_genes=['MKI67', 'UBE2H'],
    )

    sens_frames = _sensitivity_merged_frames(all_predictions)
    panel_genes = select_concordant_weight_genes(
        all_predictions, post_fi, allowed=post_important, n=3,
    )
    plot_gene_sensitivity_panels(
        sens_frames,
        genes=panel_genes,
        figures_dir_path=figures_dir,
        stem=FIGURE_STEM,
    )

    mcfarland_pre, _, _ = get_McFarland_mean_data()
    mcfarland_shared = common_genes(mcfarland_predictions)
    mcfarland_hvgs = top_n_pre_hvgs(mcfarland_pre, n=HVG_N_TOP, allowed=mcfarland_shared)
    logger.info(
        '%d McFarland pre HVGs (n=%d) shared across profiles', len(mcfarland_hvgs), HVG_N_TOP,
    )
    _write_gene_gene_heatmap_suite(
        mcfarland_predictions,
        mcfarland_hvgs,
        stem=hvg_stem,
        dataset='mcfarland',
        ylabel='McFarland',
        heatmap_prefix='mcfarland_',
        per_cell_line_heatmaps=False,
        single_cell_lines=('CAOV3',),
    )
    logger.info('Correlation figures written to %s (stem=%s)', figures_dir, hvg_stem)


if __name__ == '__main__':
    main()
