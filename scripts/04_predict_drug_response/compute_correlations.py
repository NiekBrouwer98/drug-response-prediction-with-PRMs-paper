import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import os

import pickle
from matplotlib.transforms import blended_transform_factory

from config import config
from prediction_utils import (
    PROFILE_META_COLS,
    _normalize_sciplex_product_name,
    get_CPA_predictions,
    get_chemCPA_predictions,
    get_PRnet_predictions,
    get_GEARS_predictions,
    get_scfoundation_predictions,
    get_sciplex_AUCs,
    merge_sciplex_sensitivity,
    get_average_effect_predictions,
    get_no_effect_predictions,
    get_McFarland_mean_data,
    get_McFarland_CPA_predictions,
    get_McFarland_chemCPA_predictions,
    get_McFarland_PRnet_predictions,
    get_McFarland_GEARS_predictions,
    get_McFarland_scFoundation_predictions,
    get_McFarland_average_effect_predictions,
    get_McFarland_no_effect_predictions,
)

# Set project directories using configuration
home_dir = str(config.PROJECT_ROOT)
data_dir = str(config.DATA_DIR)
resources_dir = str(config.RESOURCES_DIR)
results_dir = str(config.RESULTS_04_DIR)
figures_dir = str(config.FIGURES_04_DIR)

# Chemical + genetic PRMs with shared baselines (matches prediction / profile figures).
ALL_FAMILY_ORDER = [
    'observations',
    'CPA',
    'chemCPA',
    'PRnet',
    'GEARS',
    'scFoundation',
    'no_effect',
    'average_effect',
]
ALL_FAMILY_TITLE_MAP = {
    'observations': 'Measured\n ',
    'CPA': 'CPA',
    'chemCPA': 'chemCPA',
    'PRnet': 'PRnet',
    'GEARS': 'GEARS',
    'scFoundation': 'scFoundation',
    'no_effect': 'No effect',
    'average_effect': 'Average effect',
}
ALL_FAMILY_DISPLAY_NAMES = {
    'observations': 'Measured',
    'CPA': 'CPA',
    'chemCPA': 'chemCPA',
    'PRnet': 'PRnet',
    'GEARS': 'GEARS',
    'scFoundation': 'scFoundation',
    'no_effect': 'No effect',
    'average_effect': 'Average effect',
}

# Back-compat aliases used by older imports / CPA-only notebooks.
CPA_FAMILY_ORDER = [
    'observations', 'CPA', 'chemCPA', 'PRnet', 'no_effect', 'average_effect',
]
CPA_FAMILY_TITLE_MAP = {
    k: ALL_FAMILY_TITLE_MAP[k] for k in CPA_FAMILY_ORDER
}
CPA_FAMILY_DISPLAY_NAMES = {
    k: ALL_FAMILY_DISPLAY_NAMES[k] for k in CPA_FAMILY_ORDER
}

# Untreated / vehicle labels only (not McFarland ``GENE+ctrl`` perturbation keys).
CONTROL_CONDITIONS = frozenset({'ctrl', 'control', 'vehicle', 'dmso'})


def _is_control_condition(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(CONTROL_CONDITIONS)


def drop_control_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Remove untreated control rows; keep treated / predicted profiles only."""
    if 'condition' not in df.columns:
        return df
    return df.loc[~_is_control_condition(df['condition'])].copy()


def _remap_gene_targets_to_labels(
    df: pd.DataFrame,
    *,
    mapping_path: str,
    target_col: str,
    label_col: str,
) -> pd.DataFrame:
    """Map GEARS/scFoundation ``condition`` gene targets onto observed drug labels."""
    if 'condition' not in df.columns:
        return df
    mapping = pd.read_csv(mapping_path)
    if target_col not in mapping.columns or label_col not in mapping.columns:
        # SciPlex map may store product_name with a redundant index column.
        mapping = pd.read_csv(mapping_path, index_col=0)
    if target_col not in mapping.columns or label_col not in mapping.columns:
        raise ValueError(f'Expected {target_col!r} and {label_col!r} in {mapping_path}')
    target_to_label = (
        mapping.dropna(subset=[target_col, label_col])
        .drop_duplicates(subset=target_col, keep='first')
        .set_index(target_col)[label_col]
    )
    out = df.copy()
    mapped = out['condition'].astype(str).map(target_to_label)
    # Keep already-labeled conditions; remap gene-target keys.
    out['condition'] = mapped.fillna(out['condition'].astype(str))
    return out


def _prepare_genetic_prm_profiles(df: pd.DataFrame, *, assay: str) -> pd.DataFrame:
    """Fold-average genetic PRMs; remap SciPlex gene targets onto product names."""
    out = _mean_over_folds(df)
    if assay == 'sciplex':
        # SciPlex observed pseudobulks use product names; GEARS/scf use gene targets.
        return _remap_gene_targets_to_labels(
            out,
            mapping_path=os.path.join(resources_dir, 'sciplex_drug_to_perturbation.csv'),
            target_col='target',
            label_col='product_name',
        )
    # McFarland observed + GEARS/scf already share gene-target condition labels.
    return out


def gene_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in PROFILE_META_COLS]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def _mean_over_folds(df: pd.DataFrame) -> pd.DataFrame:
    """Average predicted expression across CV folds for each (cell_line, condition)."""
    out = drop_control_conditions(df.copy())
    if 'cell_line' in out.columns:
        out['cell_line'] = out['cell_line'].astype(str).str.upper()
    keys = [c for c in ('cell_line', 'condition') if c in out.columns]
    out = out.drop(columns=[c for c in ('fold', 'split') if c in out.columns], errors='ignore')
    return out.groupby(keys, as_index=False, sort=False).mean(numeric_only=True)


def _align_to_reference(
    df: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    normalize_conditions: bool = False,
) -> pd.DataFrame:
    """Left-join ``df`` onto the reference (cell_line, condition) grid.

    When ``normalize_conditions`` is True (SciPlex), join on whitespace-stripped
    product names so Average-effect / PRM labels like ``Rigosertib (ON-01910)``
    match observed ``Rigosertib(ON-01910)``.
    """
    keys = ['cell_line', 'condition']
    left = reference[keys].copy()
    left['cell_line'] = left['cell_line'].astype(str).str.upper()
    right = df.copy()
    right['cell_line'] = right['cell_line'].astype(str).str.upper()

    if normalize_conditions:
        left = left.assign(_cond_key=_normalize_sciplex_product_name(left['condition']))
        right = right.assign(_cond_key=_normalize_sciplex_product_name(right['condition']))
        right = right.drop(columns=['condition'], errors='ignore')
        right = right.drop_duplicates(subset=['cell_line', '_cond_key'], keep='first')
        return left.merge(right, on=['cell_line', '_cond_key'], how='left').drop(
            columns=['_cond_key'], errors='ignore'
        )

    right = right.drop_duplicates(subset=keys, keep='first')
    return left.merge(right, on=keys, how='left')


def common_genes(all_predictions: dict, candidates: list[str] | None = None) -> list[str]:
    """Genes present in every profile table (optionally intersected with ``candidates``)."""
    shared = None
    for df in all_predictions.values():
        genes = set(gene_columns(df))
        shared = genes if shared is None else shared & genes
    if candidates is not None:
        shared = [g for g in candidates if g in shared]
    else:
        shared = sorted(shared)
    return shared


def get_predictions():
    """SciPlex post-treatment profiles for all PRMs + baselines, fold-averaged.

    Measured panel uses non-control observations only. All other panels use that
    model's predicted (or baseline) profiles with untreated controls removed.
    """
    observations = []
    for cell_line in ['mcf7', 'a549', 'k562']:
        print(f'Processing cell line: {cell_line}')
        obs = pd.read_csv(
            os.path.join(data_dir, 'observed_pseudobulk', f'sciplex_mean_post_{cell_line}.csv'),
            index_col=0,
        ).assign(cell_line=cell_line.upper())
        observations.append(obs)
    observations = drop_control_conditions(pd.concat(observations, ignore_index=True))
    observations['cell_line'] = observations['cell_line'].astype(str).str.upper()

    cpa_post, _ = get_CPA_predictions()
    chemcpa_post, _ = get_chemCPA_predictions()
    prnet_post, _ = get_PRnet_predictions()
    gears_post, _ = get_GEARS_predictions()
    scf_post, _ = get_scfoundation_predictions()
    average_post, _ = get_average_effect_predictions()
    no_effect_post = get_no_effect_predictions()

    tables = {
        'observations': observations,
        'CPA': _mean_over_folds(cpa_post),
        'chemCPA': _mean_over_folds(chemcpa_post),
        'PRnet': _mean_over_folds(prnet_post),
        'GEARS': _prepare_genetic_prm_profiles(gears_post, assay='sciplex'),
        'scFoundation': _prepare_genetic_prm_profiles(scf_post, assay='sciplex'),
        'average_effect': _mean_over_folds(average_post),
        'no_effect': _mean_over_folds(no_effect_post),
    }
    # Align predicted panels onto the measured (non-control) grid; never substitute
    # observed expression into predicted tables.
    return {
        name: _align_to_reference(
            drop_control_conditions(frame),
            observations,
            normalize_conditions=True,
        )
        for name, frame in tables.items()
    }


def get_mcfarland_predictions():
    """McFarland post-treatment profiles for all PRMs + baselines, fold-averaged.

    Measured panel uses non-control observations only. All other panels use that
    model's predicted (or baseline) profiles with untreated controls removed.
    """
    _, observations, _ = get_McFarland_mean_data()
    observations = drop_control_conditions(observations.copy())
    observations['cell_line'] = observations['cell_line'].astype(str).str.upper()

    cpa_post, _ = get_McFarland_CPA_predictions()
    chemcpa_post, _ = get_McFarland_chemCPA_predictions()
    prnet_post, _ = get_McFarland_PRnet_predictions()
    gears_post, _ = get_McFarland_GEARS_predictions()
    scf_post, _ = get_McFarland_scFoundation_predictions()
    average_post, _ = get_McFarland_average_effect_predictions()
    no_effect_post = get_McFarland_no_effect_predictions()

    tables = {
        'observations': observations,
        'CPA': _mean_over_folds(cpa_post),
        'chemCPA': _mean_over_folds(chemcpa_post),
        'PRnet': _mean_over_folds(prnet_post),
        'GEARS': _prepare_genetic_prm_profiles(gears_post, assay='mcfarland'),
        'scFoundation': _prepare_genetic_prm_profiles(scf_post, assay='mcfarland'),
        'average_effect': _mean_over_folds(average_post),
        'no_effect': _mean_over_folds(no_effect_post),
    }
    return {
        name: _align_to_reference(drop_control_conditions(frame), observations)
        for name, frame in tables.items()
    }


def get_mcfarland_post_feature_importance(
    results_prefix: str | None = None,
    *,
    smiles: bool = True,
) -> pd.DataFrame:
    """Mean ElasticNet gene weights from split-CV on measured McFarland post profiles.

    Defaults to the SMILES FI pickle and drops ECFP / fingerprint features.
    """
    from feature_importance_io import (
        DEFAULT_MCFARLAND_FI_PREFIX,
        filter_gene_features,
        load_split_cv_feature_importance_with_observed_fallback,
        _mean_coef_across_folds,
    )

    prefix = results_prefix or DEFAULT_MCFARLAND_FI_PREFIX
    fi_pickle = load_split_cv_feature_importance_with_observed_fallback(results_dir, prefix)
    candidates = (
        ['observed_post_treatment_smiles', 'observed_post_treatment']
        if smiles
        else ['observed_post_treatment']
    )
    key = next((k for k in candidates if k in fi_pickle), None)
    if key is None:
        raise KeyError(f'observed post key missing from {prefix}_feature_importance.pkl')
    importance = filter_gene_features(_mean_coef_across_folds(fi_pickle[key]))
    return (
        importance.rename('importance')
        .rename_axis('gene')
        .reset_index()
        .sort_values('gene')
    )


def _square_corr_from_reset(corr_df: pd.DataFrame) -> pd.DataFrame:
    """Square gene–gene matrix from ``reset_index()`` corr table (first col = gene)."""
    return corr_df.set_index(corr_df.iloc[:, 0]).drop(columns=corr_df.columns[0])


def _clustermap_gene_orders(corr: pd.DataFrame) -> tuple[list, list]:
    """Same gene ordering as the original seaborn clustermap panels."""
    g = sns.clustermap(
        corr,
        cmap='coolwarm', center=0, vmin=-1, vmax=1, figsize=(8, 8),
        row_cluster=True, col_cluster=True,
    )
    plt.close(g.fig)
    row_order = [corr.index[i] for i in g.dendrogram_row.reordered_ind]
    col_order = [corr.columns[i] for i in g.dendrogram_col.reordered_ind]
    return row_order, col_order


def _draw_corr_heatmap(ax, matrix: pd.DataFrame):
    """Seaborn heatmap styling with rasterized mesh (keeps large HVG PDFs openable)."""
    sns.heatmap(
        matrix,
        ax=ax,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=False,
        yticklabels=False,
        cbar=False,
    )
    for coll in ax.collections:
        coll.set_rasterized(True)
    return ax.collections[-1] if ax.collections else None


def plot_pooled_gene_gene_heatmaps(
    corrs: dict,
    outfile: str,
    ylabel: str = 'All cell lines',
    model_order: list[str] | None = None,
    title_map: dict | None = None,
    display_names: dict | None = None,
    figsize: tuple[float, float] = (21, 3.5),
):
    """Cluster measured gene-gene correlations and plot all models in that order."""
    model_order = model_order or ALL_FAMILY_ORDER
    title_map = title_map or ALL_FAMILY_TITLE_MAP
    display_names = display_names or ALL_FAMILY_DISPLAY_NAMES
    if figsize == (21, 3.5):
        figsize = (3.5 * len(model_order), 3.5)

    corr_observations_cl = _square_corr_from_reset(corrs['observations'].copy())
    row_order, col_order = _clustermap_gene_orders(corr_observations_cl)

    def upper_tri_vector(M):
        iu = np.triu_indices_from(M, k=1)
        return M.values[iu]

    def safe_pair(a, b):
        mask = np.isfinite(a) & np.isfinite(b)
        return a[mask], b[mask]

    obs_vec = upper_tri_vector(corr_observations_cl)
    fig, axes = plt.subplots(1, len(model_order), figsize=figsize, constrained_layout=True, dpi=300)
    cos_sims = {}
    last_mesh = None

    for j, m in enumerate(model_order):
        mod_corrs = _square_corr_from_reset(corrs[m].copy())
        modd_corrs = mod_corrs.loc[row_order, col_order]
        mod_vec = upper_tri_vector(mod_corrs)
        a, b = safe_pair(obs_vec, mod_vec)
        if a.size == 0:
            cos_sims[m] = np.nan
            continue
        cos_sims[m] = float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0])
        ax = axes[j]
        last_mesh = _draw_corr_heatmap(ax, modd_corrs)
        t = title_map.get(m, m)
        cos_val = cos_sims[m]
        if m == 'observations':
            title = f'{t}'
        else:
            title = f'{t}\ncos. sim.={cos_val:.2f}' if np.isfinite(cos_val) else f'{m}\ncos. sim.=NA'
        ax.set_title(title, fontsize=24)
        ax.set_ylabel(ylabel, fontsize=24) if j == 0 else ax.set_ylabel('')
        ax.set_xlabel('')

    if last_mesh is not None:
        cb_top = fig.colorbar(
            last_mesh,
            ax=axes[:].ravel().tolist(),
            fraction=0.03, pad=0.02,
        )
        cb_top.set_label('', rotation=270, labelpad=16, fontsize=20)

    print(f'Gene-gene cosine similarity vs measured ({ylabel}, upper triangle):')
    for m in model_order:
        if m == 'observations':
            continue
        print(f'  {display_names.get(m, m)}: {cos_sims[m]:.4f}')

    fig.savefig(outfile, bbox_inches='tight', dpi=300)
    return fig, cos_sims


def plot_gene_gene_heatmaps_by_cell_line(
    corrs: dict,
    outfile: str,
    cell_lines: tuple[str, ...] = ('MCF7', 'A549', 'K562'),
    model_order: list[str] | None = None,
    title_map: dict | None = None,
    display_names: dict | None = None,
    figsize: tuple[float, float] = (21, 10),
):
    """Gene-gene correlation panels per cell line (rows) × model (columns)."""
    model_order = model_order or ALL_FAMILY_ORDER
    title_map = title_map or ALL_FAMILY_TITLE_MAP
    display_names = display_names or ALL_FAMILY_DISPLAY_NAMES
    if figsize == (21, 10):
        figsize = (3.5 * len(model_order), 3.2 * max(len(cell_lines), 1))

    n_rows = len(cell_lines)
    fig, axes = plt.subplots(
        n_rows, len(model_order), figsize=figsize, constrained_layout=True, dpi=300
    )
    if n_rows == 1:
        axes = np.array([axes])

    corr_observations = corrs['observations'].copy()
    if 'level_0' not in corr_observations.columns:
        raise ValueError("Expected MultiIndex reset with 'level_0' column for per-cell-line plots.")

    print('Gene-gene cosine similarity vs measured (per cell line, upper triangle):')
    for row, cl in enumerate(cell_lines):
        corr_obs_cl = corr_observations[corr_observations['level_0'] == cl].drop(columns=['level_0'])
        corr_obs_cl = _square_corr_from_reset(corr_obs_cl)
        row_order, col_order = _clustermap_gene_orders(corr_obs_cl)
        obs_ord = corr_obs_cl.loc[row_order, col_order]
        obs_vec = obs_ord.values[np.triu_indices_from(obs_ord, k=1)]

        cos_sims: dict[str, float] = {}
        for j, m in enumerate(model_order):
            mod = corrs[m].copy()
            mod = mod[mod['level_0'] == cl].drop(columns=['level_0'])
            mod = _square_corr_from_reset(mod)
            mod_ord = mod.loc[row_order, col_order]
            mod_vec = mod_ord.values[np.triu_indices_from(mod_ord, k=1)]
            mask = np.isfinite(obs_vec) & np.isfinite(mod_vec)
            cos_sims[m] = (
                float(cosine_similarity(obs_vec[mask].reshape(1, -1), mod_vec[mask].reshape(1, -1))[0, 0])
                if mask.any()
                else np.nan
            )
            ax = axes[row, j]
            _draw_corr_heatmap(ax, mod_ord)
            t = title_map.get(m, m)
            cos_val = cos_sims[m]
            if row == 0:
                title = t if m == 'observations' else (
                    f'{t}\ncos. sim.={cos_val:.2f}' if np.isfinite(cos_val) else f'{m}\ncos. sim.=NA'
                )
            else:
                title = '' if m == 'observations' else (
                    f'cos. sim.={cos_val:.2f}' if np.isfinite(cos_val) else f'{m}\ncos. sim.=NA'
                )
            ax.set_title(title, fontsize=24)
            ax.set_ylabel(cl if j == 0 else '', fontsize=24)
            ax.set_xlabel('')

        print(f'  {cl}')
        for m in model_order:
            if m == 'observations':
                continue
            print(f'    {display_names.get(m, m)}: {cos_sims[m]:.4f}')

    for r in range(n_rows):
        fig.colorbar(
            axes[r, -1].collections[0],
            ax=axes[r, :].ravel().tolist(),
            fraction=0.03, pad=0.02,
        )
    fig.savefig(outfile, bbox_inches='tight', dpi=300)
    return fig


def _corr_matrix_for_cell_line(corr_df: pd.DataFrame, cell_line: str) -> pd.DataFrame:
    """Square gene–gene corr matrix for one cell line from a reset MultiIndex frame."""
    cl = corr_df[corr_df['level_0'] == cell_line].drop(columns=['level_0'])
    return cl.set_index(cl.iloc[:, 0]).drop(columns=cl.columns[0])


def gene_gene_cosine_sims_per_cell_line(
    corrs: dict,
    model_order: list[str] | None = None,
) -> pd.DataFrame:
    """Upper-triangle cosine similarity of each model's gene–gene corr vs Measured.

    Returns a long frame with columns ``cell_line``, ``model``, ``cosine_sim``.
    ``observations`` is omitted (always 1.0). Cell lines with no finite Measured
    upper-triangle pairs are skipped.
    """
    model_order = model_order or ALL_FAMILY_ORDER
    models = [m for m in model_order if m != 'observations' and m in corrs]
    corr_observations = corrs['observations']
    if 'level_0' not in corr_observations.columns:
        raise ValueError("Expected MultiIndex reset with 'level_0' column for per-cell-line sims.")

    cell_lines = sorted(corr_observations['level_0'].astype(str).unique())
    rows: list[dict] = []
    for cl in cell_lines:
        corr_obs_cl = _corr_matrix_for_cell_line(corr_observations, cl)
        if corr_obs_cl.empty or corr_obs_cl.shape[0] < 2:
            continue
        obs_vec = corr_obs_cl.values[np.triu_indices_from(corr_obs_cl, k=1)]
        if not np.isfinite(obs_vec).any():
            continue
        for m in models:
            mod = _corr_matrix_for_cell_line(corrs[m], cl)
            # Align to Measured gene order where possible
            shared = corr_obs_cl.index.intersection(mod.index)
            if len(shared) < 2:
                rows.append({'cell_line': cl, 'model': m, 'cosine_sim': np.nan})
                continue
            obs_aln = corr_obs_cl.loc[shared, shared]
            mod_aln = mod.loc[shared, shared]
            obs_v = obs_aln.values[np.triu_indices_from(obs_aln, k=1)]
            mod_v = mod_aln.values[np.triu_indices_from(mod_aln, k=1)]
            mask = np.isfinite(obs_v) & np.isfinite(mod_v)
            cos_val = (
                float(cosine_similarity(obs_v[mask].reshape(1, -1), mod_v[mask].reshape(1, -1))[0, 0])
                if mask.any()
                else np.nan
            )
            rows.append({'cell_line': cl, 'model': m, 'cosine_sim': cos_val})
    return pd.DataFrame(rows)


# Display-name colors matching step-03 reconstruction boxplots.
GENE_GENE_MODEL_COLORS = {
    'CPA': '#1f77b4',
    'chemCPA': '#d62728',
    'PRnet': '#17becf',
    'GEARS': '#2ca02c',
    'scFoundation': '#ff7f0e',
    'Average effect': '#9467bd',
    'No effect': '#7f7f7f',
}

# Match step-03 all-models order (Average effect before No effect) and x-group bars.
COSINE_BOXPLOT_MODEL_ORDER = [
    'CPA',
    'chemCPA',
    'PRnet',
    'GEARS',
    'scFoundation',
    'average_effect',
    'no_effect',
]
MODEL_X_GROUPS: tuple[tuple[str, set[str]], ...] = (
    ('chemical PRMs', {'CPA', 'chemCPA', 'PRnet'}),
    ('genetic PRMs', {'GEARS', 'scFoundation'}),
    ('baselines', {'Average effect', 'No effect'}),
)
BOX_WIDTH = 0.5


def _style_model_xticks(ax) -> None:
    ax.tick_params(axis='x', labelsize=20, rotation=45)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
        label.set_rotation_mode('anchor')


def _add_x_group_labels(
    ax,
    categories: list[str],
    groups: tuple[tuple[str, set[str]], ...] = MODEL_X_GROUPS,
    *,
    y_line: float = -0.54,
    y_text: float = -0.56,
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


def plot_gene_gene_cosine_sim_boxplot(
    sims: pd.DataFrame,
    outfile: str,
    model_order: list[str] | None = None,
    display_names: dict | None = None,
    figsize: tuple[float, float] = (12, 6),
    title: str | None = None,
) -> plt.Figure:
    """Boxplot of per–cell-line gene–gene cosine sim vs Measured (one box per method).

    Styling matches step-03 reconstruction boxplots (colors, ticks, x-group labels).
    """
    model_order = model_order or COSINE_BOXPLOT_MODEL_ORDER
    display_names = display_names or ALL_FAMILY_DISPLAY_NAMES
    models = [m for m in model_order if m in set(sims['model'])]
    if not models:
        raise ValueError('No models with cosine-sim values to plot')

    display_labels = [display_names.get(m, m) for m in models]
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    rng = np.random.default_rng(0)

    for x, m in enumerate(models):
        grp = sims.loc[sims['model'] == m, 'cosine_sim'].dropna()
        display = display_names.get(m, m)
        color = GENE_GENE_MODEL_COLORS.get(display, '#333333')
        ax.boxplot(
            [grp.values],
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
            np.full(len(grp), x) + jitter,
            grp.values,
            c=[color],
            s=20,
            alpha=1,
            linewidths=0,
            zorder=3,
        )

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(display_labels, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_xlim(-0.55, len(models) - 0.45)
    _style_model_xticks(ax)
    # _add_x_group_labels(ax, display_labels)
    ax.set_xlabel('')
    ax.set_ylabel('Cosine similarity to Measured', fontsize=14)
    if title:
        ax.set_title(title, fontsize=26, fontweight='bold', pad=10, loc='center')
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, which='major', axis='y', linestyle='--', color='#cccccc', zorder=0, alpha=0.7)
    ax.grid(False, axis='x')
    sns.despine(ax=ax, offset=8, trim=False)
    fig.subplots_adjust(bottom=0.28, top=0.88)
    fig.savefig(outfile, bbox_inches='tight', dpi=300)
    return fig


def build_gene_gene_corr_dict(
    all_predictions: dict[str, pd.DataFrame],
    important_features: list[str],
    *,
    per_cell_line: bool = True,
) -> dict[str, pd.DataFrame]:
    """Gene–gene correlations per model panel.

    ``observations`` → measured profiles only; every other key → that model's
    predicted (or baseline) profiles. Untreated controls are always dropped.
    """
    subset = important_features + ['cell_line', 'condition']
    corrs: dict[str, pd.DataFrame] = {}
    for name, frame in all_predictions.items():
        corrs[name] = compute_gene_gene_correlation(
            drop_control_conditions(frame),
            subset=subset,
            per_cell_line=per_cell_line,
        ).reset_index()
    return corrs


def get_sciplex_post_feature_importance(
    results_prefix: str | None = None,
    *,
    smiles: bool = True,
) -> pd.DataFrame:
    """Mean ElasticNet gene weights from SciPlex measured-post split-CV (SMILES by default)."""
    from feature_importance_io import (
        DEFAULT_SCIPLEX_FI_PREFIX,
        get_sciplex_post_feature_importance_gene_table,
    )

    prefix = results_prefix or DEFAULT_SCIPLEX_FI_PREFIX
    return get_sciplex_post_feature_importance_gene_table(
        results_dir, prefix, smiles=smiles,
    )


def get_observations():
    sciplex_observations = []
    for cl in ['mcf7', 'a549', 'k562']:
        observations = pd.read_csv(os.path.join(data_dir, 'observed_pseudobulk', f'sciplex_mean_post_{cl}.csv'), index_col=0)
        observations['cell_line'] = cl.upper()
        sciplex_observations.append(observations)

    sciplex_observations = drop_control_conditions(pd.concat(sciplex_observations, ignore_index=True))
    sciplex_observations['cell_line'] = sciplex_observations['cell_line'].astype(str).str.upper()

    sciplex_sensitivity = get_sciplex_AUCs()
    sciplex_observations = merge_sciplex_sensitivity(sciplex_observations, sciplex_sensitivity)

    return sciplex_observations


def compute_correlation_with_y(df, subset = None):
    sciplex_sensitivity = get_sciplex_AUCs().rename(columns={'y': 'sensitivity'})

    df = drop_control_conditions(df.copy())
    if subset is not None:
        keep = ['cell_line', 'condition'] + [c for c in subset if c not in PROFILE_META_COLS and c in df.columns]
        df = df[keep]
    df_with_sensitivity = merge_sciplex_sensitivity(
        df,
        sciplex_sensitivity.rename(columns={'sensitivity': 'y'}),
        value_col='y',
    ).rename(columns={'y': 'sensitivity'})
    df_with_sensitivity = df_with_sensitivity.drop(columns=['cell_line', 'condition'])

    return df_with_sensitivity.corrwith(df_with_sensitivity['sensitivity'], method='spearman').drop('sensitivity').fillna(0)


def compute_mse_per_gene(df, subset = None):
    sciplex_observations = get_observations()
    df = df.drop(columns=['cell_line', 'condition'])
    observations = sciplex_observations.drop(columns=['cell_line', 'condition'])

    if subset is not None:
        subset = list(set(subset).intersection(set(df.columns.tolist())))
        df = df[subset]
        observation_subset = observations[subset]

    else:
        subset = list(set(observations.columns.tolist()).intersection(set(df.columns.tolist())))
        df = df[subset]
        observation_subset = observations[subset]
        
    diff_df = df.subtract(observation_subset.values, axis=0)
    diff_df = diff_df**2

    return diff_df.mean()


def compute_mse_per_individual(df, subset = None):
    sciplex_observations = get_observations()
    df = df.set_index(['cell_line', 'condition'])
    observations = sciplex_observations.set_index(['cell_line', 'condition'])

    if subset is not None:
        subset = list(set(subset).intersection(set(df.columns.tolist())))
        df = df[subset]
        observation_subset = observations[subset]

    else:
        subset = list(set(observations.columns.tolist()).intersection(set(df.columns.tolist())))
        df = df[subset]
        observation_subset = observations[subset]

    df = df.T
    observation_subset = observation_subset.T

    diff_df = df.subtract(observation_subset.values, axis=0)

    diff_df = diff_df**2
    diff_df = diff_df.mean()
    # diff_df.index = df.index

    return diff_df


def compute_correlation_per_individual(df, subset=None):
    sciplex_observations = get_observations()
    df = df.set_index(['cell_line', 'condition'])
    observations = sciplex_observations.set_index(['cell_line', 'condition'])

    if subset is not None:
        subset = list(set(subset).intersection(set(df.columns.tolist())))
        df = df[subset]
        observation_subset = observations[subset]
    else:
        subset = list(set(observations.columns.tolist()).intersection(set(df.columns.tolist())))
        df = df[subset]
        observation_subset = observations[subset]

    correlations = df.corrwith(observation_subset, axis=1, method='pearson')
    return correlations.fillna(0)


def compute_gene_gene_correlation(df, subset = None, per_cell_line = True):
    df = drop_control_conditions(df)
    if subset is not None:
        subset = [c for c in subset if c not in PROFILE_META_COLS]
        subset = list(set(subset).intersection(set(df.columns.tolist())))
    else:
        subset = gene_columns(df)
    keep = subset + [c for c in ('cell_line',) if c in df.columns]
    df = df[keep]

    if per_cell_line:
        results = []
        for cell_line in df['cell_line'].unique():
            df_cl = df[df['cell_line'] == cell_line].drop(columns=['cell_line'])
            df_cl_corr = df_cl.corr()
            df_cl_corr.index = pd.MultiIndex.from_product([[cell_line], df_cl_corr.index])
            results.append(df_cl_corr)

        results_df = pd.concat(results, axis=0)
        results_df = results_df.fillna(0)
    else:
        results_df = df.drop(columns=['cell_line']).corr()
        
    return results_df
