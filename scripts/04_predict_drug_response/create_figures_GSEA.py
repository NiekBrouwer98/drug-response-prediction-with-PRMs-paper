"""GSEA dotplot figures for all PRMs (chemical + genetic) on SciPlex and McFarland."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent))

from config import config, setup_project
from feature_importance_io import (
    GSEA_MODEL_ORDER_LFC,
    GSEA_MODEL_ORDER_POST,
    filter_sciplex_gsea_dotplot_models,
    gsea_model_column_name,
    sciplex_dotplot_model_label,
)
from utils import ensure_directories_exist, setup_logging_for_script

setup_project()
logger = setup_logging_for_script(__file__)

results_dir = str(config.RESULTS_04_DIR)
figures_dir = str(config.FIGURES_04_DIR)
ensure_directories_exist(figures_dir)


def _load_cpa_family_gsea_results(dataset: str) -> pd.DataFrame:
    """Load GSEA CSVs for ``dataset``.

    Discovers ``GSEA_{dataset}_*.csv`` on disk so genetic PRMs (GEARS / scFoundation)
    are included even when the feature-importance pickle is CPA-family only.
    """
    pattern = f'GSEA_{dataset}_*.csv'
    paths = sorted(Path(results_dir).glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f'No {dataset} GSEA CSV files found ({pattern}); run compute_GSEA.main() first.'
        )
    frames = []
    seen_models: set[str] = set()
    # Prefer canonical Measured keys (``post`` / ``lfc``) over legacy ``*_Observed``.
    ranked = sorted(
        paths,
        key=lambda p: (
            0 if p.name in {f'GSEA_{dataset}_post.csv', f'GSEA_{dataset}_lfc.csv'} else 1,
            p.name,
        ),
    )
    for path in ranked:
        raw_key = path.name.removeprefix(f'GSEA_{dataset}_').removesuffix('.csv')
        # Skip legacy / alternate labels we do not plot.
        if 'Optimized' in raw_key or 'noreg' in raw_key.lower():
            continue
        if raw_key.endswith('_Observed') or raw_key == 'Observed':
            continue
        model = gsea_model_column_name(raw_key.replace('\n', '_'))
        if model in seen_models:
            continue
        seen_models.add(model)
        res = pd.read_csv(path, index_col=0)[['Term', 'NES', 'FDR q-val']]
        res = res.assign(model=model)
        frames.append(res)
    if not frames:
        raise FileNotFoundError(f'No usable {dataset} GSEA CSV files after filtering.')
    return pd.concat(frames, ignore_index=True)


def _load_sciplex_gsea_results() -> pd.DataFrame:
    return _load_cpa_family_gsea_results('sciplex')


def _complete_gsea_term_model_grid(
    data: pd.DataFrame,
    terms: list[str],
    model_order: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    """Fill term–model pairs with no GSEA NES as NES=0, FDR q-val=1."""
    grid = pd.MultiIndex.from_product(
        [list(terms), list(model_order)], names=['Term', 'model'],
    ).to_frame(index=False)
    src = data.copy()
    src['model'] = src['model'].astype(str)
    out = grid.merge(src, on=['Term', 'model'], how='left')
    missing = out['NES'].isna()
    out['nes_missing'] = missing
    out.loc[missing, 'NES'] = 0.0
    out.loc[missing, 'FDR q-val'] = 1.0
    return out


def _gsea_neglog10_fdr(fdr: pd.Series | np.ndarray) -> np.ndarray:
    return -np.log10(np.clip(np.asarray(fdr, dtype=float), 1e-5, 1.0))


def _gsea_size_from_fdr(
    fdr: pd.Series | np.ndarray,
    log_min: float,
    log_max: float,
    min_size: float,
    max_size: float,
) -> np.ndarray:
    logv = _gsea_neglog10_fdr(fdr)
    return (logv - log_min) / (log_max - log_min + 1e-9) * (max_size - min_size) + min_size


def _add_gsea_size_legend(
    fig: plt.Figure,
    *,
    log_min: float,
    log_max: float,
    min_size: float,
    max_size: float,
    bbox_to_anchor: tuple[float, float] = (1.02, 0.48),
) -> None:
    fdr_min = float(10 ** (-log_max))
    fdr_ticks = [fdr for fdr in (1.0, 0.1, 0.05, 0.01) if fdr >= fdr_min * 0.5]
    if 1.0 not in fdr_ticks:
        fdr_ticks = [1.0, *fdr_ticks]
    handles = []
    labels = []
    for fdr in fdr_ticks:
        size = float(np.clip(
            _gsea_size_from_fdr([fdr], log_min, log_max, min_size, max_size)[0],
            min_size,
            max_size,
        ))
        handles.append(
            Line2D(
                [0], [0],
                linestyle='None',
                marker='o',
                markersize=np.sqrt(size),
                markerfacecolor='0.85',
                markeredgecolor='black',
                markeredgewidth=0.6,
            )
        )
        labels.append(f'{fdr:g}')
    fig.legend(
        handles,
        labels,
        title='FDR q-val',
        loc='upper left',
        bbox_to_anchor=bbox_to_anchor,
        frameon=False,
        labelspacing=1.5,
        handletextpad=0.8,
        borderaxespad=0.0,
    )


GSEA_DOTPLOT_PATHWAYS: tuple[str, ...] = (
    'E2F Targets',
    'G2-M Checkpoint',
    'Myc Targets V1',
    'Apoptosis',
    'Interferon Gamma Response',
    'Allograft Rejection',
    'TNF-alpha Signaling via NF-kB',
    'Unfolded Protein Response',
    'p53 Pathway',
    'UV Response Up',
    'UV Response Dn',
    'Reactive Oxygen Species Pathway',
)


def _prepare_lfc_gsea_panel(
    dotplot_data: pd.DataFrame,
    *,
    model_order: tuple[str, ...] | list[str],
    row_order: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], tuple[str, ...]]:
    """Filter to LFC models / pathways; return (panel_df, row_order, model_order)."""
    data = dotplot_data[dotplot_data['Term'].isin(GSEA_DOTPLOT_PATHWAYS)].copy()
    available = {
        sciplex_dotplot_model_label(m) for m in data['model'].astype(str).unique()
    }
    order = tuple(m for m in model_order if m in available)
    if not order:
        raise ValueError(f'No LFC GSEA models left; available={sorted(available)}')
    panel = filter_sciplex_gsea_dotplot_models(
        data, profile_kind='lfc', model_order=order,
    )
    if row_order is None:
        nes = panel.pivot(index='Term', columns='model', values='NES').fillna(0.0001)
        sort_col = 'Measured' if 'Measured' in nes.columns else nes.columns[0]
        row_order = nes.sort_values(by=sort_col, ascending=False).index.tolist()
    panel = _complete_gsea_term_model_grid(panel, row_order, order)
    return panel, row_order, order


def _scatter_gsea_panel(
    ax: plt.Axes,
    sub_data: pd.DataFrame,
    *,
    title: str,
    model_order: tuple[str, ...] | list[str],
    row_order: list[str],
    norm,
) -> object:
    """Draw one GSEA dotplot panel; return the scatter collection."""
    nes_matrix_sub = sub_data.pivot(index='Term', columns='model', values='NES').fillna(0)
    model_to_x = {model: i for i, model in enumerate(model_order)}
    term_to_y = {term: i for i, term in enumerate(row_order)}
    sub_data = sub_data.copy()
    if 'Measured' in nes_matrix_sub.columns:
        obs_sign = np.sign(nes_matrix_sub['Measured']).replace(0, np.nan)
        sub_data['sign_agree'] = [
            np.sign(row['NES']) == obs_sign.get(row['Term'], np.nan)
            for _, row in sub_data.iterrows()
        ]
    else:
        sub_data['sign_agree'] = False

    alpha = np.where(
        sub_data['nes_missing'].to_numpy(),
        1.0,
        np.where(sub_data['sign_agree'].to_numpy(), 1.0, 0.3),
    )
    sc = ax.scatter(
        x=[model_to_x[m] for m in sub_data['model'].astype(str)],
        y=[term_to_y[t] for t in sub_data['Term']],
        s=sub_data['SizeNorm'],
        c=sub_data['NES'],
        cmap='coolwarm',
        norm=norm,
        edgecolors='black',
        alpha=alpha,
    )
    ax.set_xticks(list(model_to_x.values()))
    ax.set_xticklabels(model_order, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_title(title)
    ax.set_xlim(-0.5, len(model_order) - 0.5)
    ax.yaxis.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    return sc


def create_cpa_family_gsea_dotplot(
    dotplot_data: pd.DataFrame,
    outfile: str,
    figure_size: tuple[float, float] = (12, 4),
) -> None:
    from matplotlib.colors import PowerNorm

    dotplot_data = dotplot_data[dotplot_data['Term'].isin(GSEA_DOTPLOT_PATHWAYS)].copy()

    # Display labels present in the raw GSEA table (before post/lfc stripping).
    available_labels = {
        sciplex_dotplot_model_label(m)
        for m in dotplot_data['model'].astype(str).unique()
    }
    post_order = tuple(m for m in GSEA_MODEL_ORDER_POST if m in available_labels)
    lfc_order = tuple(m for m in GSEA_MODEL_ORDER_LFC if m in available_labels)
    if not post_order or not lfc_order:
        raise ValueError(
            f'No GSEA models left after filtering; available={sorted(available_labels)}'
        )

    nes_matrix_all = dotplot_data.pivot(index='Term', columns='model', values='NES').fillna(0.0001)
    obs_cols = [c for c in nes_matrix_all.columns if 'Measured' in c or c == 'Measured']
    sort_col = obs_cols[0] if obs_cols else nes_matrix_all.columns[0]
    row_order = nes_matrix_all.sort_values(by=sort_col, ascending=False).index.tolist()

    max_nes = dotplot_data['NES'].abs().quantile(0.95)
    norm = PowerNorm(gamma=0.5, vmin=-max_nes, vmax=max_nes)
    min_size, max_size = 60, 400

    data_post = filter_sciplex_gsea_dotplot_models(
        dotplot_data, profile_kind='post', model_order=post_order,
    )
    data_lfc = filter_sciplex_gsea_dotplot_models(
        dotplot_data, profile_kind='lfc', model_order=lfc_order,
    )
    data_post = _complete_gsea_term_model_grid(data_post, row_order, post_order)
    data_lfc = _complete_gsea_term_model_grid(data_lfc, row_order, lfc_order)

    log_scores = np.concatenate([
        _gsea_neglog10_fdr(data_post['FDR q-val']),
        _gsea_neglog10_fdr(data_lfc['FDR q-val']),
    ])
    log_min, log_max = float(np.min(log_scores)), float(np.max(log_scores))
    data_post['SizeNorm'] = _gsea_size_from_fdr(
        data_post['FDR q-val'], log_min, log_max, min_size, max_size,
    )
    data_lfc['SizeNorm'] = _gsea_size_from_fdr(
        data_lfc['FDR q-val'], log_min, log_max, min_size, max_size,
    )

    fig, axes = plt.subplots(
        1, 2, figsize=figure_size, dpi=300, sharey=True, gridspec_kw={'width_ratios': [8, 7]},
    )
    sc = None
    for ax, sub_data, title, model_order in (
        (axes[0], data_post, 'Trained on post profiles', post_order),
        (axes[1], data_lfc, 'Trained on LFC profiles', lfc_order),
    ):
        sc = _scatter_gsea_panel(
            ax, sub_data, title=title, model_order=model_order, row_order=row_order, norm=norm,
        )

    axes[0].set_yticks(range(len(row_order)))
    axes[0].set_yticklabels(row_order)
    plt.tight_layout()
    if sc is not None:
        cbar_ax = fig.add_axes([1.02, 0.55, 0.03, 0.35])
        cbar = fig.colorbar(sc, cax=cbar_ax)
        cbar.set_label('NES')
        _add_gsea_size_legend(
            fig,
            log_min=log_min,
            log_max=log_max,
            min_size=min_size,
            max_size=max_size,
            bbox_to_anchor=(1.02, 0.48),
        )
    fig.savefig(outfile, bbox_inches='tight', dpi=300)
    plt.close(fig)


def create_lfc_cross_dataset_gsea_dotplot(
    sciplex_data: pd.DataFrame,
    mcfarland_data: pd.DataFrame,
    outfile: str,
    figure_size: tuple[float, float] = (12, 4),
) -> None:
    """Two-panel LFC GSEA: SciPlex (left) | McFarland (right)."""
    from matplotlib.colors import PowerNorm

    sciplex_lfc, row_order, sciplex_order = _prepare_lfc_gsea_panel(
        sciplex_data, model_order=GSEA_MODEL_ORDER_LFC,
    )
    mcfarland_lfc, _, mcfarland_order = _prepare_lfc_gsea_panel(
        mcfarland_data, model_order=GSEA_MODEL_ORDER_LFC, row_order=row_order,
    )

    combined = pd.concat([sciplex_lfc, mcfarland_lfc], ignore_index=True)
    max_nes = float(combined['NES'].abs().quantile(0.95))
    if not np.isfinite(max_nes) or max_nes <= 0:
        max_nes = 1.0
    norm = PowerNorm(gamma=0.5, vmin=-max_nes, vmax=max_nes)
    min_size, max_size = 60, 400
    log_scores = _gsea_neglog10_fdr(combined['FDR q-val'])
    log_min, log_max = float(np.min(log_scores)), float(np.max(log_scores))
    sciplex_lfc = sciplex_lfc.copy()
    mcfarland_lfc = mcfarland_lfc.copy()
    sciplex_lfc['SizeNorm'] = _gsea_size_from_fdr(
        sciplex_lfc['FDR q-val'], log_min, log_max, min_size, max_size,
    )
    mcfarland_lfc['SizeNorm'] = _gsea_size_from_fdr(
        mcfarland_lfc['FDR q-val'], log_min, log_max, min_size, max_size,
    )

    fig, axes = plt.subplots(
        1, 2, figsize=figure_size, dpi=300, sharey=True, gridspec_kw={'width_ratios': [1, 1]},
    )
    sc = _scatter_gsea_panel(
        axes[0],
        sciplex_lfc,
        title='SciPlex3 (LFC)',
        model_order=sciplex_order,
        row_order=row_order,
        norm=norm,
    )
    _scatter_gsea_panel(
        axes[1],
        mcfarland_lfc,
        title='McFarland (LFC)',
        model_order=mcfarland_order,
        row_order=row_order,
        norm=norm,
    )
    axes[0].set_yticks(range(len(row_order)))
    axes[0].set_yticklabels(row_order)
    plt.tight_layout()
    cbar_ax = fig.add_axes([1.02, 0.55, 0.03, 0.35])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('NES')
    _add_gsea_size_legend(
        fig,
        log_min=log_min,
        log_max=log_max,
        min_size=min_size,
        max_size=max_size,
        bbox_to_anchor=(1.02, 0.48),
    )
    fig.savefig(outfile, bbox_inches='tight', dpi=300)
    plt.close(fig)


def create_sciplex_gsea_dotplot(
    dotplot_data: pd.DataFrame,
    outfile: str,
    figure_size: tuple[float, float] = (12, 4),
) -> None:
    create_cpa_family_gsea_dotplot(dotplot_data, outfile, figure_size=figure_size)


def main() -> None:
    logger.info('Writing all-model GSEA dotplots for SciPlex and McFarland...')
    sciplex = _load_cpa_family_gsea_results('sciplex')
    mcfarland = _load_cpa_family_gsea_results('mcfarland')
    for dataset, data, outfile, figsize in (
        ('sciplex', sciplex, 'GSEA_dotplot_sciplex.pdf', (12, 4)),
        ('mcfarland', mcfarland, 'GSEA_dotplot.pdf', (12, 4)),
    ):
        create_cpa_family_gsea_dotplot(
            data,
            os.path.join(figures_dir, outfile),
            figure_size=figsize,
        )
    create_lfc_cross_dataset_gsea_dotplot(
        sciplex,
        mcfarland,
        os.path.join(figures_dir, 'GSEA_dotplot_lfc_sciplex_mcfarland.pdf'),
        figure_size=(12, 4),
    )
    logger.info('GSEA figures written to %s', figures_dir)


if __name__ == '__main__':
    main()
