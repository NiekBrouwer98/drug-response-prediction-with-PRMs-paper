"""Plot T1–T4 measured-profile drug-response performance (SciPlex | McFarland)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from config import config, setup_project
from task_cv import MODEL_LABELS, PLOT_MODEL_ORDER, _safe_pearson
from utils import ensure_directories_exist, setup_logging_for_script
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

setup_project()
logger = setup_logging_for_script(__file__)

results_dir = str(config.RESULTS_04_DIR)
figures_dir = str(config.FIGURES_04_DIR)
ensure_directories_exist(results_dir, figures_dir)

TASK_ORDER = ['T1', 'T2', 'T3', 'T4']
MODEL_ORDER = PLOT_MODEL_ORDER
# Gene-only + SMILES display labels (PLOT_MODEL_ORDER is SMILES-only by default).
ALL_MEASURED_MODEL_LABELS = tuple(dict.fromkeys([*MODEL_LABELS.values(), *MODEL_ORDER]))
DATASET_ORDER = ['SciPlex3', 'McFarland']
# Pool OOF predictions within these units before Pearson (stabilises T2–T4).
GROUP_COL_BY_DATASET = {
    'McFarland': 'tissue',
    'SciPlex3': 'cell_line',
}
# Canonical: T2 = unseen context, T3 = unseen drug (matches task_cv.py).


def _drop_no_effect_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop retired measured-task ``profile_source=no_effect`` rows if present."""
    if df is None or df.empty or 'profile_source' not in df.columns:
        return df
    return df[df['profile_source'].astype(str) != 'no_effect'].copy()


def task_axis_labels(context_noun: str) -> dict[str, str]:
    """Map T1–T4 codes to plot axis labels (T2=unseen context, T3=unseen drug)."""
    return {
        'T1': f'T1:\nSeen drug, \nseen {context_noun}',
        'T2': f'T2:\nSeen drug,\nunseen {context_noun}',
        'T3': f'T3:\nUnseen drug,\nseen {context_noun}',
        'T4': f'T4:\nUnseen drug,\nunseen {context_noun}',
    }
PALETTE = {
    'Pre+SMILES': '#A0A0A0',
    'Post+SMILES': '#FF7F7F',
    'LFC+SMILES': '#CD5C5C',
    # Opt-in expression-only modalities
    'Pre': '#808080',
    'Post': '#FF0000',
    'LFC': '#8B0000',
}
DODGE = {
    'Pre+SMILES': -0.20,
    'Post+SMILES': 0.0,
    'LFC+SMILES': 0.20,
    'Pre': -0.30,
    'Post': -0.06,
    'LFC': 0.18,
}
MARKERS = {
    'Pre+SMILES': 's',
    'Post+SMILES': 's',
    'LFC+SMILES': 's',
    'Pre': 'o',
    'Post': 'o',
    'LFC': 'o',
}


def _merge_modality_frames(base: pd.DataFrame | None, extra: pd.DataFrame | None) -> pd.DataFrame | None:
    """Merge prediction/score frames without dropping tasks present only in ``base``.

    Prefer ``extra`` for matching ``(dataset, task, model)`` keys (and ``head`` /
    ``threshold`` when present); keep all other ``base`` rows. This lets a
    SMILES T1-only exhaustive file replace T1 SMILES modalities without wiping
    T2–T4 for those models.
    """
    if extra is None or extra.empty:
        return base
    if base is None or base.empty:
        return extra

    extra = extra.copy()
    base = base.copy()
    key_cols = [c for c in ('dataset', 'task', 'model', 'head', 'threshold') if c in extra.columns]
    if 'task' not in key_cols or 'model' not in key_cols:
        # Fall back to model-level replace (legacy behaviour).
        models_in_extra = set(extra['model'].astype(str).unique())
        base_keep = base[~base['model'].astype(str).isin(models_in_extra)].copy()
        return pd.concat([base_keep, extra], ignore_index=True)

    keys = extra[key_cols].drop_duplicates()
    merged = base.merge(keys.assign(_from_extra=1), on=key_cols, how='left')
    base_keep = merged[merged['_from_extra'].isna()].drop(columns=['_from_extra'])
    return pd.concat([base_keep, extra], ignore_index=True)


def _prefer_t1_cv_scheme(
    df: pd.DataFrame,
    preferred: str,
) -> pd.DataFrame:
    """Keep T1 rows for ``preferred`` cv_scheme; fall back per model otherwise.

    ``preferred`` is ``'exhaustive'`` (leave-one-(drug, context)) or
    ``'predefined_fold'`` (the same 5 random splits as predicted-profile CV).
    Non-T1 tasks are unchanged (aside from dropping the non-preferred T1 scheme).
    """
    if preferred not in {'exhaustive', 'predefined_fold'}:
        raise ValueError("preferred must be 'exhaustive' or 'predefined_fold'")
    if df is None or df.empty or 'task' not in df.columns:
        return df
    out = df.copy()
    t1_mask = out['task'].astype(str).eq('T1')
    if not t1_mask.any():
        return out
    if 'cv_scheme' not in out.columns:
        if preferred == 'predefined_fold':
            logger.warning(
                'No cv_scheme column; treating available T1 as predefined-fold / legacy.'
            )
        else:
            n_groups = out.loc[t1_mask].groupby(
                [c for c in ('dataset', 'model') if c in out.columns], dropna=False
            ).size()
            if (n_groups <= 5).all():
                logger.warning(
                    'T1 predictions look like 5-fold CV (no cv_scheme). '
                    'Re-run measured-task CV with exhaustive T1 to refresh.'
                )
        return out

    t1 = out.loc[t1_mask].copy()
    other = out.loc[~t1_mask].copy()
    is_pref = t1['cv_scheme'].astype(str).eq(preferred)
    if not is_pref.any():
        logger.warning(
            'No T1 rows with cv_scheme=%s; using available T1.',
            preferred,
        )
        return out

    group_cols = [c for c in ('dataset', 'model') if c in t1.columns]
    models_with_pref = (
        t1.loc[is_pref, group_cols].drop_duplicates()
        if group_cols
        else t1.loc[is_pref, ['model']].drop_duplicates()
    )
    t1_pref = t1.loc[is_pref].copy()
    if group_cols:
        keyed = t1.merge(models_with_pref.assign(_has_pref=1), on=group_cols, how='left')
        t1_fallback = keyed[keyed['_has_pref'].isna()].drop(columns=['_has_pref'])
    else:
        models = set(t1_pref['model'].astype(str))
        t1_fallback = t1[~t1['model'].astype(str).isin(models)]
    preferred_frame = pd.concat([t1_pref, t1_fallback], ignore_index=True)
    logger.info(
        'T1 selection (preferred=%s): %d preferred rows, %d fallback rows',
        preferred,
        int(is_pref.sum()),
        int(len(t1_fallback)),
    )
    return pd.concat([other, preferred_frame], ignore_index=True)


def _load_fold_scores(t1_cv_scheme: str = 'predefined_fold') -> pd.DataFrame:
    combined = os.path.join(results_dir, 'combined_measured_tasks_fold_scores.csv')
    frames = []
    for prefix, dataset in (('sciplex', 'SciPlex3'), ('mcfarland', 'McFarland')):
        path = os.path.join(results_dir, f'{prefix}_measured_tasks_fold_scores.csv')
        smiles_path = os.path.join(results_dir, f'{prefix}_smiles_measured_tasks_fold_scores.csv')
        t1_5fold_path = os.path.join(results_dir, f'{prefix}_measured_tasks_t1_5fold_fold_scores.csv')
        t1_5fold_smiles_path = os.path.join(
            results_dir, f'{prefix}_smiles_measured_tasks_t1_5fold_fold_scores.csv'
        )
        base = None
        if os.path.exists(path):
            df = pd.read_csv(path)
            if 'dataset' not in df.columns:
                df['dataset'] = dataset
            base = df
        if os.path.exists(smiles_path):
            df = pd.read_csv(smiles_path)
            if 'dataset' not in df.columns:
                df['dataset'] = dataset
            base = _merge_modality_frames(base, df)
        if t1_cv_scheme == 'predefined_fold':
            for extra_path in (t1_5fold_path, t1_5fold_smiles_path):
                if os.path.exists(extra_path):
                    df = pd.read_csv(extra_path)
                    if 'dataset' not in df.columns:
                        df['dataset'] = dataset
                    base = _merge_modality_frames(base, df)
        if base is not None:
            frames.append(base)
    if frames:
        out = _drop_no_effect_rows(pd.concat(frames, ignore_index=True))
        out = _prefer_t1_cv_scheme(out, t1_cv_scheme)
        return out[out['model'].isin(MODEL_ORDER)].copy()
    if os.path.exists(combined):
        out = _drop_no_effect_rows(
            pd.read_csv(combined).query('model in @MODEL_ORDER').copy()
        )
        return _prefer_t1_cv_scheme(out, t1_cv_scheme)

    raise FileNotFoundError(
        'No T1–T4 fold-score CSVs found. Run task_cv / response_prediction first.'
    )


def _parse_threshold_tag(tag: str) -> float:
    """Convert output tag fragments like ``0p02`` / ``0p1`` to float thresholds."""
    return float(tag.replace('p', '.').replace('m', '-'))


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate column names (e.g. SciPlex ``cell_line`` / ``cell_line.1``)."""
    if not df.columns.duplicated().any():
        return df
    return df.loc[:, ~df.columns.duplicated()].copy()


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman ρ; 0 when ranking is undefined (constant true/pred or n < 2)."""
    if len(y_true) < 2:
        return 0.0
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    val = spearmanr(y_true, y_pred).correlation
    return float(val) if val is not None and not np.isnan(val) else 0.0


def score_predictions_by_group(
    predictions: pd.DataFrame,
    *,
    group_col_by_dataset: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Pearson / Spearman / RMSE for plotting T1–T4 measured-task results.

    For every task (T1–T4), pool OOF predictions within dataset-specific groups,
    then score. Default grouping: McFarland by ``tissue``; SciPlex3 by ``cell_line``.
    Pass ``group_col_by_dataset`` to override (e.g. McFarland ``condition`` for
    within-drug scores).
    """
    preds = _dedupe_columns(predictions).copy()
    if 'dataset' not in preds.columns:
        raise ValueError('predictions must include a dataset column')
    if 'model' not in preds.columns:
        raise ValueError('predictions must include a model column')

    if 'head' not in preds.columns:
        preds['head'] = 'continuous'
    if 'threshold' not in preds.columns:
        preds['threshold'] = np.nan

    group_map = dict(GROUP_COL_BY_DATASET)
    if group_col_by_dataset:
        group_map.update(group_col_by_dataset)

    rows: list[dict] = []
    for dataset, ds_pred in preds.groupby('dataset', dropna=False):
        group_col = group_map.get(str(dataset))
        if group_col is None:
            raise ValueError(f'No group column configured for dataset {dataset!r}')
        if group_col not in ds_pred.columns:
            raise ValueError(f'{dataset} predictions missing group column {group_col!r}')

        tasks = ds_pred[ds_pred['task'].isin(TASK_ORDER)]
        if tasks.empty:
            continue
        key_cols = ['task', 'model', 'head', 'threshold', group_col]
        for keys, grp in tasks.groupby(key_cols, dropna=False):
            task, model, head, threshold, group_value = keys
            y_true = grp['true'].to_numpy(dtype=float)
            y_pred = grp['pred'].to_numpy(dtype=float)
            rows.append({
                'dataset': dataset,
                'task': task,
                'model': model,
                'head': head,
                'threshold': threshold,
                'group_col': group_col,
                'group': group_value,
                'n_test': int(len(grp)),
                'pearson': _safe_pearson(y_true, y_pred),
                'spearman': _safe_spearman(y_true, y_pred),
                'rmse': float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Keep gene-only (Pre/Post/LFC) and +SMILES; do not drop nosmiles CV outputs.
    return out[out['model'].astype(str).isin(ALL_MEASURED_MODEL_LABELS)].copy()


def align_stratified_task_panel_groups(
    scores: pd.DataFrame,
    *,
    models: list[str] | None = None,
    tasks: list[str] | None = None,
) -> pd.DataFrame:
    """Keep only groups shared by every ``(task, model)`` panel in ``scores``.

    Ensures every boxplot in a stratified panel (e.g. SciPlex within cell line)
    uses the same set of cell lines / tissues / drugs.
    """
    if scores is None or scores.empty or 'group' not in scores.columns:
        return scores
    frame = scores.copy()
    if models is not None:
        frame = frame[frame['model'].astype(str).isin(models)].copy()
    use_tasks = list(tasks) if tasks is not None else list(TASK_ORDER)
    frame = frame[frame['task'].astype(str).isin(use_tasks)].copy()
    if frame.empty:
        return frame

    group_sets: list[set[str]] = []
    panel_keys: list[tuple[str, str]] = []
    for (task, model), sub in frame.groupby(['task', 'model'], dropna=False):
        groups = {str(g) for g in sub['group'].tolist() if pd.notna(g) and str(g) not in {'nan', 'none'}}
        if not groups:
            continue
        group_sets.append(groups)
        panel_keys.append((str(task), str(model)))

    if not group_sets:
        return frame

    common = set.intersection(*group_sets)
    logger.info(
        'align_stratified_task_panel_groups: %d common groups across %d (task, model) panels '
        '(example=%s)',
        len(common),
        len(group_sets),
        panel_keys[:6],
    )
    if not common:
        logger.warning(
            'align_stratified_task_panel_groups: empty intersection; leaving unaligned'
        )
        return frame
    return frame[frame['group'].astype(str).isin(common)].reset_index(drop=True).copy()


def _annotate_prediction_frame(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    out = _dedupe_columns(df)
    if 'dataset' not in out.columns:
        out['dataset'] = dataset
    if 'head' not in out.columns:
        out['head'] = 'continuous'
    if 'threshold' not in out.columns:
        out['threshold'] = np.nan
    return out


def _load_prediction_frames(
    *,
    include_two_stage: bool = False,
    t1_cv_scheme: str = 'predefined_fold',
    output_tag: str = '',
) -> pd.DataFrame:
    """Load continuous (and optionally two-stage) measured-task prediction CSVs.

    ``t1_cv_scheme`` selects which T1 rows to keep:
      - ``'predefined_fold'`` — perturbation-model 5-fold T1 (default)
      - ``'exhaustive'`` — leave-one-(drug, context) T1
    T2–T4 always use the exhaustive / leave-group holdouts from the main files.

    ``output_tag`` selects tagged outputs (e.g. ``nosmiles`` →
    ``sciplex_nosmiles_measured_tasks_predictions.csv``). Empty tag keeps the
    default loaders (untagged + legacy ``*_smiles_measured_tasks_*``).
    """
    if t1_cv_scheme not in {'exhaustive', 'predefined_fold'}:
        raise ValueError("t1_cv_scheme must be 'exhaustive' or 'predefined_fold'")

    frames: list[pd.DataFrame] = []
    for prefix, dataset in (('sciplex', 'SciPlex3'), ('mcfarland', 'McFarland')):
        file_prefix = f'{prefix}_{output_tag}' if output_tag else prefix
        cont_path = Path(results_dir) / f'{file_prefix}_measured_tasks_predictions.csv'
        t1_5fold_path = (
            Path(results_dir) / f'{file_prefix}_measured_tasks_t1_5fold_predictions.csv'
        )
        base = None
        if cont_path.exists():
            base = _annotate_prediction_frame(pd.read_csv(cont_path), dataset)
        if not output_tag:
            # Legacy SMILES-tagged runs (merged only when loading the default tag).
            smiles_path = (
                Path(results_dir) / f'{prefix}_smiles_measured_tasks_predictions.csv'
            )
            if smiles_path.exists():
                smiles = _annotate_prediction_frame(pd.read_csv(smiles_path), dataset)
                base = smiles if base is None else _merge_modality_frames(base, smiles)
        if t1_cv_scheme == 'predefined_fold':
            t1_paths = [t1_5fold_path]
            if not output_tag:
                t1_paths.append(
                    Path(results_dir)
                    / f'{prefix}_smiles_measured_tasks_t1_5fold_predictions.csv'
                )
            for path in t1_paths:
                if not path.exists():
                    continue
                five = _annotate_prediction_frame(pd.read_csv(path), dataset)
                base = five if base is None else _merge_modality_frames(base, five)
        if base is not None:
            frames.append(base)

        if not include_two_stage:
            continue
        two_stage_glob = (
            f'{file_prefix}_two_stage_t*_measured_tasks_predictions.csv'
            if output_tag
            else f'{prefix}_two_stage_t*_measured_tasks_predictions.csv'
        )
        two_stage_paths = sorted(Path(results_dir).glob(two_stage_glob))
        if not two_stage_paths:
            logger.warning(
                'include_two_stage=True but no files matched %s/%s',
                results_dir,
                two_stage_glob,
            )
        for path in two_stage_paths:
            stem = path.name
            tag = stem.split('_two_stage_t', 1)[1].split('_measured_tasks_', 1)[0]
            df = _annotate_prediction_frame(pd.read_csv(path), dataset)
            df['head'] = 'two_stage'
            df['threshold'] = _parse_threshold_tag(tag)
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f'No measured-task prediction CSVs found in {results_dir}'
            + (f' (output_tag={output_tag!r})' if output_tag else '')
        )
    out = _drop_no_effect_rows(pd.concat(frames, ignore_index=True))
    # Observations / measured-task figures use Measured profiles only.
    if 'profile_source' in out.columns:
        src = out['profile_source']
        keep = src.isna() | src.astype(str).isin({'measured', 'nan', ''})
        out = out.loc[keep].copy()
    # T2–T4 (and non-preferred T1) should not keep the alternate T1 scheme rows.
    if 'cv_scheme' in out.columns:
        if t1_cv_scheme == 'exhaustive':
            out = out[out['cv_scheme'].astype(str) != 'predefined_fold'].copy()
        else:
            # Keep predefined_fold T1; drop exhaustive T1 so scoring is not mixed.
            t1 = out['task'].astype(str).eq('T1')
            drop_ex = t1 & out['cv_scheme'].astype(str).eq('exhaustive')
            out = out.loc[~drop_ex].copy()
    out = _prefer_t1_cv_scheme(out, t1_cv_scheme)
    return out


def load_grouped_task_scores(
    *,
    include_two_stage: bool = False,
    t1_cv_scheme: str = 'predefined_fold',
) -> pd.DataFrame:
    """T1–T4 group-pooled OOF scores for continuous/two-stage heads.

    Default T1 = predefined 5-fold; T2–T4 = exhaustive leave-one-group.
    """
    preds = _load_prediction_frames(
        include_two_stage=include_two_stage,
        t1_cv_scheme=t1_cv_scheme,
    )
    return score_predictions_by_group(preds)


def load_stratified_task_panel_scores(
    *,
    include_two_stage: bool = False,
    models: list[str] | None = None,
    t1_cv_scheme: str = 'predefined_fold',
    align_common_groups: bool = True,
    output_tag: str = '',
) -> dict[str, pd.DataFrame]:
    """Score T1–T4 for the manuscript 3-panel stratification.

    Returns a dict with keys:
      ``sciplex_cell_line`` — SciPlex3 within cell line
      ``mcfarland_tissue`` — McFarland within tissue
      ``mcfarland_drug`` — McFarland within drug (``condition``)

    Default ``t1_cv_scheme='predefined_fold'`` (5-fold); T2–T4 remain exhaustive.
    When ``align_common_groups`` is True, each panel keeps only groups shared by
    every ``(task, model)`` combination so all boxplots show the same points.
    Pass ``output_tag='nosmiles'`` to load expression-only tagged CV outputs.
    """
    preds = _load_prediction_frames(
        include_two_stage=include_two_stage,
        t1_cv_scheme=t1_cv_scheme,
        output_tag=output_tag,
    )
    if models is not None:
        preds = preds[preds['model'].astype(str).isin(models)].copy()

    sciplex = preds[preds['dataset'].astype(str).eq('SciPlex3')].copy()
    mcf = preds[preds['dataset'].astype(str).eq('McFarland')].copy()

    panels = {
        'sciplex_cell_line': score_predictions_by_group(
            sciplex, group_col_by_dataset={'SciPlex3': 'cell_line'}
        ),
        'mcfarland_tissue': score_predictions_by_group(
            mcf, group_col_by_dataset={'McFarland': 'tissue'}
        ),
        'mcfarland_drug': score_predictions_by_group(
            mcf, group_col_by_dataset={'McFarland': 'condition'}
        ),
    }
    if align_common_groups:
        panels = {
            key: align_stratified_task_panel_groups(frame, models=models)
            for key, frame in panels.items()
        }
    for key, frame in panels.items():
        n_groups = (
            frame.groupby('task')['group'].nunique().to_dict() if len(frame) else {}
        )
        logger.info(
            '%s: %d scored rows (tasks=%s models=%s groups_per_task=%s)',
            key,
            len(frame),
            sorted(frame['task'].astype(str).unique()) if len(frame) else [],
            sorted(frame['model'].astype(str).unique()) if len(frame) else [],
            n_groups,
        )
    return panels


def _to_task_boxplot_frame(
    scores: pd.DataFrame,
    *,
    context_noun: str,
) -> pd.DataFrame:
    """Convert group scores to the schema expected by observation boxplots."""
    labels = task_axis_labels(context_noun)
    out = scores.copy()
    out['test'] = out['task'].map(labels)
    out['accuracy'] = pd.to_numeric(out['pearson'], errors='coerce').fillna(0.0)
    out['feature_selection'] = 'top1000 HVG'
    out['test_instance'] = out['group']
    return out


def _legend_display_label(model: str, model_order: list[str]) -> str:
    """Map Pre+SMILES → Pre when the base modality is not also plotted."""
    if model.endswith('+SMILES'):
        base = model[: -len('+SMILES')]
        bases = {m for m in model_order if not str(m).endswith('+SMILES')}
        if base not in bases:
            return base
    return model


def _add_task_boxplot_scatter(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    feature_order: list[str],
    model_order: list[str],
    colors: list[str],
    box_width: float = 0.6,
) -> None:
    """Overlay individual group scores on hue-dodged boxes (predicted-plot style)."""
    x_locs = {cat: i for i, cat in enumerate(feature_order)}
    n_models = max(len(model_order), 1)
    offset_lookup = {
        model: (idx - (n_models - 1) / 2) * (box_width / n_models)
        for idx, model in enumerate(model_order)
    }
    for test in feature_order:
        for j, model in enumerate(model_order):
            group = data[(data['test'] == test) & (data['model'] == model)]
            if group.empty:
                continue
            y = pd.to_numeric(group['accuracy'], errors='coerce').to_numpy(dtype=float)
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
                color=colors[j],
                edgecolor='black',
                linewidth=0.5,
                alpha=0.6,
                zorder=4,
                label=None,
            )


def _resolve_pre_lfc_pair(model_order: list[str]) -> tuple[str, str] | None:
    """Prefer Pre+SMILES/LFC+SMILES when present, else Pre/LFC."""
    order = [str(m) for m in model_order]
    if 'Pre+SMILES' in order and 'LFC+SMILES' in order:
        return 'Pre+SMILES', 'LFC+SMILES'
    if 'Pre' in order and 'LFC' in order:
        return 'Pre', 'LFC'
    return None


def _sig_label_obs(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return 'NS'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'NS'


def _format_mean_delta(mean_a: float, mean_b: float) -> str:
    """Mean difference (model_b − model_a); matches stratified/pooled summary Δ."""
    if np.isnan(mean_a) or np.isnan(mean_b):
        return ''
    return f'Δ{float(mean_b - mean_a):+.2f}'


def _pair_effect_obs(
    sub: pd.DataFrame,
    *,
    model_a: str,
    model_b: str,
    value_col: str = 'accuracy',
) -> tuple[float, float, float]:
    """Return (p, mean_a, mean_b) on paired groups (same pairing as summary Δ)."""
    from scipy.stats import ttest_ind, ttest_rel

    pair_col = None
    for cand in ('test_instance', 'group', 'tissue', 'cell_line', 'condition', 'fold'):
        if cand in sub.columns and sub[cand].notna().any():
            pair_col = cand
            break

    a_vals = b_vals = None
    p = float('nan')

    if pair_col is not None:
        tmp = sub.loc[sub['model'].isin([model_a, model_b]), [pair_col, 'model', value_col]].copy()
        tmp[value_col] = pd.to_numeric(tmp[value_col], errors='coerce')
        tmp = tmp.dropna(subset=[value_col])
        tmp = tmp.groupby([pair_col, 'model'], as_index=False)[value_col].mean()
        wide = tmp.pivot(index=pair_col, columns='model', values=value_col)
        if {model_a, model_b}.issubset(wide.columns):
            paired = wide[[model_a, model_b]].dropna()
            if len(paired) >= 2:
                p = float(ttest_rel(paired[model_b], paired[model_a]).pvalue)
                a_vals = paired[model_a]
                b_vals = paired[model_b]

    if a_vals is None or b_vals is None:
        a_vals = pd.to_numeric(sub.loc[sub['model'] == model_a, value_col], errors='coerce').dropna()
        b_vals = pd.to_numeric(sub.loc[sub['model'] == model_b, value_col], errors='coerce').dropna()
        if len(a_vals) >= 2 and len(b_vals) >= 2 and np.isnan(p):
            p = float(ttest_ind(b_vals, a_vals, equal_var=False).pvalue)

    if len(a_vals) == 0 or len(b_vals) == 0:
        return p, float('nan'), float('nan')
    return p, float(np.mean(a_vals)), float(np.mean(b_vals))


def _annotate_pre_vs_lfc_brackets(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    feature_order: list[str],
    model_order: list[str],
    value_col: str = 'accuracy',
    fontsize: int = 22,
) -> None:
    """Bracket + paired mean Δ (LFC − Pre), matching stratified summary tables."""
    pair = _resolve_pre_lfc_pair(model_order)
    if pair is None:
        return
    model_a, model_b = pair

    n_models = max(len(model_order), 1)
    box_width = 0.6
    offset_lookup = {
        model: (idx - (n_models - 1) / 2) * (box_width / n_models)
        for idx, model in enumerate(model_order)
    }
    x_locs = {cat: i for i, cat in enumerate(feature_order)}

    y0, y1 = ax.get_ylim()
    y_span = y1 - y0
    ax.set_ylim(y0, y1 + 0.20 * y_span)
    y0, y1 = ax.get_ylim()
    y_span = y1 - y0
    y_line = y1 - 0.11 * y_span
    y_text = y1 - 0.10 * y_span

    for test in feature_order:
        sub = data[data['test'] == test]
        if sub.empty:
            continue
        _p, mean_a, mean_b = _pair_effect_obs(
            sub, model_a=model_a, model_b=model_b, value_col=value_col
        )
        fold = _format_mean_delta(mean_a, mean_b)
        # Show paired mean Δ (sig computed but not drawn); matches summary Δ.
        label = fold if fold else _sig_label_obs(_p)
        if not label:
            continue
        x_pre = x_locs[test] + offset_lookup[model_a]
        x_lfc = x_locs[test] + offset_lookup[model_b]
        x_left, x_right = sorted((x_pre, x_lfc))
        ax.plot([x_left, x_right], [y_line, y_line], color='black', linewidth=0.9, clip_on=False)
        ax.plot(
            [x_left, x_left],
            [y_line - 0.015 * y_span, y_line],
            color='black',
            linewidth=0.9,
            clip_on=False,
        )
        ax.plot(
            [x_right, x_right],
            [y_line - 0.015 * y_span, y_line],
            color='black',
            linewidth=0.9,
            clip_on=False,
        )
        ax.text(
            0.5 * (x_left + x_right),
            y_text,
            label,
            ha='center',
            va='bottom',
            fontsize=fontsize,
            linespacing=1.05,
            clip_on=False,
        )


def plot_tasks_stratified_three_panel(
    panel_scores: dict[str, pd.DataFrame] | None = None,
    *,
    model_order: list[str] | None = None,
    palette: dict[str, str] | None = None,
    outfile: str | None = None,
    ylim: tuple[float, float] | None = None,  # Remove default (-1.0, 1.0)
    annotate_pre_lfc_delta: bool = True,
) -> plt.Figure:
    """Three-panel T1–T4 Pearson: SciPlex line | McFarland line | McFarland drug.

    Styling matches the predicted-profile stratified panels: hue-dodged boxes,
    individual point overlay, and light horizontal marker / grid lines.
    Pre vs LFC Δ brackets use paired mean Δ (same as stratified summary tables).
    """
    if panel_scores is None:
        panel_scores = load_stratified_task_panel_scores()
    if model_order is None:
        model_order = list(MODEL_ORDER)
    if palette is None:
        palette = PALETTE

    panel_defs = [
        (
            'sciplex_cell_line',
            'SciPlex3\n(within cell line)',
            'cell line',
            r'Within-line Pearson $r$  $\rightarrow$',
        ),
        (
            'mcfarland_tissue',
            'McFarland\n(within tissue)',
            'tissue',
            r'Within-tissue Pearson $r$  $\rightarrow$',
        ),
        (
            'mcfarland_drug',
            'McFarland\n(within drug)',
            'tissue',
            r'Within-drug Pearson $r$  $\rightarrow$',
        ),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(42, 12),
        dpi=300,
        sharey=False,
        gridspec_kw={'wspace': 0.18},
    )

    for ax, (key, title, context_noun, ylabel) in zip(axes, panel_defs):
        scores = panel_scores.get(key, pd.DataFrame())
        if scores is None or scores.empty:
            ax.set_title(f'{title}\n(no data yet)', fontsize=34, fontweight='bold', loc='center')
            ax.text(
                0.5,
                0.5,
                'Waiting for predefined-fold T1\nand/or SMILES task CV',
                ha='center',
                va='center',
                transform=ax.transAxes,
                fontsize=16,
            )
            if ylim is not None:
                ax.set_ylim(*ylim)
            continue

        plot_df = _to_task_boxplot_frame(scores, context_noun=context_noun)
        present = [m for m in model_order if m in set(plot_df['model'].astype(str))]
        feature_order = [task_axis_labels(context_noun)[t] for t in TASK_ORDER]
        feature_order = [t for t in feature_order if t in set(plot_df['test'].dropna())]
        colors = [palette.get(m, '#333333') for m in present]
        palette_map = {m: c for m, c in zip(present, colors)}

        # Resolve ylim (and marker lines) before drawing so grids sit behind boxes.
        if ylim is not None:
            y0, y1 = ylim
        else:
            vals = pd.to_numeric(plot_df['accuracy'], errors='coerce').dropna()
            row_min = float(vals.min()) if len(vals) else -0.25
            row_max = float(vals.max()) if len(vals) else 0.8
            y_pad = max(0.05, 0.08 * (row_max - row_min))
            y0, y1 = row_min - y_pad, row_max + y_pad
        grid_lines_y = np.linspace(y0, y1, num=6)
        for y in grid_lines_y:
            ax.axhline(y=y, color='lightgrey', linewidth=1, linestyle='-', alpha=0.7, zorder=0)
        ax.yaxis.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)

        sns.boxplot(
            data=plot_df,
            x='test',
            y='accuracy',
            hue='model',
            order=feature_order,
            hue_order=present,
            palette=palette_map,
            ax=ax,
            width=0.65,
            showfliers=False,
            linewidth=0.8,
        )
        _add_task_boxplot_scatter(
            ax,
            plot_df,
            feature_order=feature_order,
            model_order=present,
            colors=colors,
        )

        ax.set_ylim(y0, y1)
        if annotate_pre_lfc_delta:
            _annotate_pre_vs_lfc_brackets(
                ax,
                plot_df,
                feature_order=feature_order,
                model_order=present,
            )

        ax.set_xlabel('')
        ax.set_ylabel(ylabel, fontsize=28)
        ax.set_title(title, fontsize=34, fontweight='bold', loc='center')
        ax.tick_params(axis='x', labelsize=22)
        ax.tick_params(axis='y', labelsize=22)
        ax.margins(x=0.02)
        ax.set_xlim(-0.5, len(feature_order) - 0.5)
        sns.despine(ax=ax, offset=5, trim=False)

        handles, labels = ax.get_legend_handles_labels()
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        if handles:
            ax.legend(
                handles,
                [_legend_display_label(lab, present) for lab in labels],
                loc='lower left',
                fontsize=18,
                frameon=False,
            )

    fig.subplots_adjust(right=0.98, bottom=0.12)
    if outfile is None:
        outfile = os.path.join(
            figures_dir, 'measured_tasks_stratified_line_tissue_drug_pearson.pdf'
        )
    fig.savefig(outfile, bbox_inches='tight', dpi=300)
    logger.info('Saved %s', outfile)
    return fig


def load_head_comparison_fold_scores() -> pd.DataFrame:
    """Backward-compatible alias: group-pooled continuous + two-stage scores."""
    return load_grouped_task_scores(include_two_stage=True)


def _load_head_comparison_scores() -> pd.DataFrame | None:
    try:
        return load_grouped_task_scores(include_two_stage=True)
    except FileNotFoundError:
        return None


def _ci_bounds(values: np.ndarray, ci: float = 0.95) -> tuple[float, float, float]:
    vals = values[~np.isnan(values)]
    if len(vals) == 0:
        return float('nan'), float('nan'), float('nan')
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(vals, [alpha, 1.0 - alpha])
    return float(np.mean(vals)), float(lo), float(hi)


def plot_tasks_figure(
    fold_scores: pd.DataFrame,
    metric: str = 'pearson',
    outfile: str | None = None,
) -> plt.Figure:
    """One row per dataset; x = T1–T4; hue = Pre/Post/LFC; error bars = fold CI."""
    data = fold_scores.dropna(subset=[metric]).copy()
    data['task'] = pd.Categorical(data['task'], categories=TASK_ORDER, ordered=True)
    data['model'] = pd.Categorical(data['model'], categories=MODEL_ORDER, ordered=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 5), dpi=300, sharey=True)
    for ax, dataset in zip(axes, DATASET_ORDER):
        sub = data[data['dataset'] == dataset]
        if sub.empty:
            ax.set_visible(False)
            continue

        summary_rows = []
        for (task, model), grp in sub.groupby(['task', 'model'], observed=True):
            mean, lo, hi = _ci_bounds(grp[metric].to_numpy(dtype=float))
            summary_rows.append(
                {'task': task, 'model': model, 'mean': mean, 'lo': lo, 'hi': hi}
            )
        summary = pd.DataFrame(summary_rows)
        summary['task'] = pd.Categorical(summary['task'], categories=TASK_ORDER, ordered=True)
        summary['model'] = pd.Categorical(summary['model'], categories=MODEL_ORDER, ordered=True)

        # Dodged points with CI whiskers
        x_pos = {t: i for i, t in enumerate(TASK_ORDER)}
        for _, row in summary.iterrows():
            x = x_pos[row['task']] + DODGE[row['model']]
            color = PALETTE[row['model']]
            ax.errorbar(
                x,
                row['mean'],
                yerr=[[row['mean'] - row['lo']], [row['hi'] - row['mean']]],
                fmt=MARKERS[row['model']],
                color=color,
                ecolor=color,
                elinewidth=1.2,
                capsize=3,
                markersize=6,
                label=row['model'] if row['task'] == 'T1' else None,
            )

        # Overlay fold scatter (faint)
        rng = np.random.default_rng(0)
        for _, row in sub.iterrows():
            if row['model'] not in DODGE:
                continue
            x = x_pos[row['task']] + DODGE[row['model']] + rng.uniform(-0.03, 0.03)
            ax.scatter(
                x,
                row[metric],
                c=PALETTE[row['model']],
                s=12,
                alpha=0.25,
                linewidths=0,
                zorder=1,
            )

        context_noun = 'cell line' if dataset == 'SciPlex3' else 'tissue'
        axis_labels = task_axis_labels(context_noun)
        ax.set_xticks(range(len(TASK_ORDER)))
        ax.set_xticklabels([axis_labels[t] for t in TASK_ORDER], fontsize=11)
        ax.set_title(dataset, fontsize=16, fontweight='bold')
        ax.set_xlabel('')
        ax.axhline(0.0, color='#bbbbbb', linewidth=0.8, zorder=0)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        sns.despine(ax=ax)

    ylabel = r'Pearson $r$' if metric == 'pearson' else 'RMSE'
    axes[0].set_ylabel(ylabel, fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    # Deduplicate
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if l not in seen and not seen.add(l)]
    if uniq:
        fig.legend(
            [h for h, _ in uniq],
            [l for _, l in uniq],
            loc='lower center',
            ncol=len(MODEL_ORDER),
            frameon=False,
            fontsize=10,
            bbox_to_anchor=(0.5, -0.06),
        )
    fig.suptitle(
        'Measured-profile drug-response performance by generalization task',
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    if outfile is None:
        outfile = os.path.join(figures_dir, f'measured_tasks_{metric}.pdf')
    fig.savefig(outfile, bbox_inches='tight', dpi=300)
    logger.info('Saved %s', outfile)
    return fig


def plot_head_comparison(
    fold_scores: pd.DataFrame,
    metric: str = 'pearson',
    outfile: str | None = None,
) -> plt.Figure | None:
    """Threshold sweep plot: two-stage points with continuous ElasticNet reference lines."""
    required = {'dataset', 'task', 'model', 'head', 'threshold', metric}
    if not required.issubset(fold_scores.columns):
        missing = sorted(required.difference(fold_scores.columns))
        logger.warning('Skipping head comparison plot; missing columns: %s', missing)
        return None

    data = fold_scores.dropna(subset=[metric]).copy()
    two_stage = data[data['head'] == 'two_stage']
    continuous = data[data['head'] == 'continuous']
    if two_stage.empty:
        logger.warning(
            'Skipping head comparison plot: no two-stage rows '
            '(heads=%s). Run measured-task CV with TASK_CV_MODE=threshold-sweep '
            'to write *_two_stage_t*_measured_tasks_predictions.csv.',
            sorted(data['head'].dropna().astype(str).unique().tolist()),
        )
        return None

    summary = (
        two_stage.groupby(['dataset', 'task', 'model', 'threshold'], observed=True)[metric]
        .mean()
        .reset_index()
    )
    cont_summary = (
        continuous.groupby(['dataset', 'task', 'model'], observed=True)[metric]
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(
        len(DATASET_ORDER),
        len(TASK_ORDER),
        figsize=(18, 8),
        dpi=300,
        sharey=True,
        sharex=True,
    )
    for row_idx, dataset in enumerate(DATASET_ORDER):
        for col_idx, task in enumerate(TASK_ORDER):
            ax = axes[row_idx, col_idx]
            sub = summary[(summary['dataset'] == dataset) & (summary['task'] == task)]
            cont = cont_summary[(cont_summary['dataset'] == dataset) & (cont_summary['task'] == task)]
            if sub.empty and cont.empty:
                ax.set_visible(False)
                continue

            for model in MODEL_ORDER:
                model_sub = sub[sub['model'] == model].sort_values('threshold')
                color = PALETTE[model]
                if not model_sub.empty:
                    ax.plot(
                        model_sub['threshold'],
                        model_sub[metric],
                        marker=MARKERS[model],
                        color=color,
                        linewidth=1.2,
                        markersize=4,
                        label=model if row_idx == 0 and col_idx == 0 else None,
                    )
                cont_val = cont.loc[cont['model'] == model, metric]
                if len(cont_val):
                    ax.axhline(float(cont_val.iloc[0]), color=color, linestyle='--', linewidth=0.8, alpha=0.6)

            context_noun = 'cell line' if dataset == 'SciPlex3' else 'tissue'
            ax.set_title(
                f'{dataset}\n{task_axis_labels(context_noun)[task].replace(chr(10), " ")}',
                fontsize=9,
            )
            ax.grid(True, axis='y', linestyle='--', alpha=0.4)
            if row_idx == len(DATASET_ORDER) - 1:
                ax.set_xlabel('Two-stage threshold')
            if col_idx == 0:
                ax.set_ylabel(r'Pearson $r$' if metric == 'pearson' else 'RMSE')
            sns.despine(ax=ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=len(MODEL_ORDER), frameon=False)
    fig.suptitle(
        'Measured-profile continuous vs two-stage '
        '(T1–T4: within tissue / cell line)',
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    if outfile is None:
        outfile = os.path.join(figures_dir, f'measured_tasks_head_comparison_{metric}.pdf')
    fig.savefig(outfile, bbox_inches='tight', dpi=300)
    logger.info('Saved %s', outfile)
    return fig


def write_held_out_table(fold_scores: pd.DataFrame) -> str:
    """Supplementary-style table of held-out groups per fold."""
    cols = [
        'dataset', 'task', 'model', 'repeat', 'fold',
        'held_out_drugs', 'held_out_contexts', 'n_train', 'n_test', 'pearson', 'rmse',
    ]
    cols = [c for c in cols if c in fold_scores.columns]
    out = fold_scores[cols].sort_values(['dataset', 'task', 'model', 'repeat', 'fold'])
    path = os.path.join(results_dir, 'combined_measured_tasks_held_out_groups.csv')
    out.to_csv(path, index=False)
    logger.info('Wrote held-out group table: %s', path)
    return path


SMILES_MODEL_ORDER = ['Pre+SMILES', 'Post+SMILES', 'LFC+SMILES']
PANEL_SUMMARY_SPECS = (
    ('sciplex_cell_line', 'sciplex_within_line', 'SciPlex3', 'Within cell line'),
    ('mcfarland_tissue', 'mcfarland_within_tissue', 'McFarland', 'Within tissue'),
    ('mcfarland_drug', 'mcfarland_within_drug', 'McFarland', 'Within drug'),
)


def _display_modality(model: str) -> str:
    """Pre+SMILES → Pre for SMILES-only tables."""
    if str(model).endswith('+SMILES'):
        return str(model)[: -len('+SMILES')]
    return str(model)


def _latex_escape(text: object) -> str:
    s = '' if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)
    for old, new in (
        ('\\', r'\textbackslash{}'),
        ('&', r'\&'),
        ('%', r'\%'),
        ('$', r'\$'),
        ('#', r'\#'),
        ('_', r'\_'),
        ('{', r'\{'),
        ('}', r'\}'),
    ):
        s = s.replace(old, new)
    return s


def _fmt_mean_sd(mean: float, sd: float, digits: int = 3) -> str:
    if mean is None or (isinstance(mean, float) and not np.isfinite(mean)):
        return '--'
    if sd is None or (isinstance(sd, float) and not np.isfinite(sd)):
        return f'{float(mean):.{digits}f}'
    return f'{float(mean):.{digits}f} $\\pm$ {float(sd):.{digits}f}'


def _fmt_ci(mean: float, lo: float, hi: float, digits: int = 3) -> str:
    if mean is None or not np.isfinite(mean):
        return '--'
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return f'{float(mean):+.{digits}f}'
    return f'{float(mean):+.{digits}f} [{float(lo):.{digits}f}, {float(hi):.{digits}f}]'


def _fmt_sig_tex(sig: str) -> str:
    if sig == '**':
        return r'$^{**}$'
    if sig == '*':
        return r'$^{*}$'
    return 'NS'


def _fmt_pvalue(p: object, digits: int = 3) -> str:
    """Format a BH-FDR q-value (or raw p) for supplement tables."""
    if p is None or (isinstance(p, float) and (np.isnan(p) or not np.isfinite(p))):
        return '--'
    try:
        val = float(p)
    except (TypeError, ValueError):
        return '--'
    if val < 10 ** (-digits):
        return f'{val:.1e}'
    return f'{val:.{digits}g}'


def _significance_stars(p: float) -> str:
    if p is None or not np.isfinite(p):
        return 'NS'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'NS'


def _paired_delta_vs_ref(
    frame: pd.DataFrame,
    *,
    model: str,
    ref_model: str,
    metric: str,
    group_col: str = 'group',
) -> tuple[float, float, float, float, str, int]:
    """Paired mean Δ [95% CI] and sig for ``model - ref`` matched on ``group``."""
    from scipy import stats

    if metric not in frame.columns:
        return (np.nan, np.nan, np.nan, np.nan, 'NS', 0)
    a = (
        frame.loc[frame['model'].astype(str).eq(model), [group_col, metric]]
        .dropna()
        .drop_duplicates(group_col)
        .set_index(group_col)[metric]
    )
    b = (
        frame.loc[frame['model'].astype(str).eq(ref_model), [group_col, metric]]
        .dropna()
        .drop_duplicates(group_col)
        .set_index(group_col)[metric]
    )
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return (np.nan, np.nan, np.nan, np.nan, 'NS', 0)
    delta = a.loc[common].to_numpy(dtype=float) - b.loc[common].to_numpy(dtype=float)
    delta = delta[np.isfinite(delta)]
    n = int(len(delta))
    if n == 0:
        return (np.nan, np.nan, np.nan, np.nan, 'NS', 0)
    mean = float(np.mean(delta))
    if n < 2:
        return (mean, np.nan, np.nan, np.nan, 'NS', n)
    se = float(np.std(delta, ddof=1) / np.sqrt(n))
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    lo, hi = mean - tcrit * se, mean + tcrit * se
    _, p = stats.ttest_1samp(delta, 0.0)
    return (mean, float(lo), float(hi), float(p), _significance_stars(float(p)), n)


# Metrics with paired Δ vs Pre+SMILES in measured-task summary tables.
_MEASURED_DELTA_METRICS: tuple[str, ...] = ('pearson', 'spearman', 'rmse')


def _apply_bh_fdr_to_pre_smiles_pvalues(
    summary: pd.DataFrame,
    *,
    group_cols: list[str],
    metrics: tuple[str, ...] = _MEASURED_DELTA_METRICS,
) -> pd.DataFrame:
    """BH-FDR $q$ for all modality vs-Pre $p$-values within each group × metric.

    Correction family = all tasks × non-ref modalities in that panel/dataset for
    one metric. Raw ``*_p`` is retained; tables display ``*_p_fdr`` / stars from $q$.
    """
    if summary is None or summary.empty:
        return summary
    out = summary.copy()
    grouping = [c for c in group_cols if c in out.columns]
    for metric in metrics:
        p_col = f'{metric}_vs_pre_smiles_p'
        q_col = f'{metric}_vs_pre_smiles_p_fdr'
        sig_col = f'{metric}_vs_pre_smiles_sig'
        if p_col not in out.columns:
            continue
        out[q_col] = np.nan
        if not grouping:
            mask = out[p_col].notna()
            if mask.any():
                out.loc[mask, q_col] = multipletests(
                    out.loc[mask, p_col], alpha=0.05, method='fdr_bh'
                )[1]
        else:
            for _, idx in out.groupby(grouping, dropna=False).groups.items():
                idx = list(idx)
                vals = pd.to_numeric(out.loc[idx, p_col], errors='coerce')
                mask = vals.notna()
                if not mask.any():
                    continue
                q = multipletests(vals.loc[mask], alpha=0.05, method='fdr_bh')[1]
                out.loc[vals.index[mask], q_col] = q
        out[sig_col] = out[q_col].map(
            lambda v: _significance_stars(float(v)) if pd.notna(v) else 'NS'
        )
    return out


def _paired_delta_fields(
    frame: pd.DataFrame,
    *,
    model: str,
    ref_model: str,
    metrics: tuple[str, ...] = _MEASURED_DELTA_METRICS,
    group_col: str = 'group',
) -> dict[str, float | str | int]:
    """Paired Δ / CI / p / sig / n vs ``ref_model`` for each metric."""
    out: dict[str, float | str | int] = {}
    for metric in metrics:
        d_mean, d_lo, d_hi, d_p, d_sig, d_n = _paired_delta_vs_ref(
            frame,
            model=model,
            ref_model=ref_model,
            metric=metric,
            group_col=group_col,
        )
        prefix = f'{metric}_vs_pre_smiles'
        out[f'{prefix}_delta'] = d_mean
        out[f'{prefix}_ci_low'] = d_lo
        out[f'{prefix}_ci_high'] = d_hi
        out[f'{prefix}_p'] = d_p
        out[f'{prefix}_sig'] = d_sig
        out[f'{prefix}_n'] = d_n
    return out


def _fmt_delta_p_cells(row: pd.Series, metric: str, *, is_ref: bool) -> tuple[str, str]:
    """Return (Δ [95% CI], BH-FDR q with sig stars) for one metric vs Pre."""
    if is_ref:
        return '--', '--'
    digits = 2 if metric in ('pearson', 'spearman') else 3
    prefix = f'{metric}_vs_pre_smiles'
    delta_tex = _fmt_ci(
        row.get(f'{prefix}_delta', np.nan),
        row.get(f'{prefix}_ci_low', np.nan),
        row.get(f'{prefix}_ci_high', np.nan),
        digits=digits,
    )
    q = row.get(f'{prefix}_p_fdr', np.nan)
    if q is None or (isinstance(q, float) and np.isnan(q)):
        q = row.get(f'{prefix}_p', np.nan)
    p_tex = _fmt_pvalue(q)
    if p_tex == '--':
        return delta_tex, '--'
    sig_tex = _fmt_sig_tex(str(row.get(f'{prefix}_sig', 'NS')))
    if sig_tex != 'NS':
        p_tex = f'{p_tex}{sig_tex}'
    return delta_tex, p_tex


def build_measured_tasks_stratified_summary(
    panel_scores: dict[str, pd.DataFrame] | None = None,
    *,
    models: list[str] | None = None,
    ref_model: str = 'Pre+SMILES',
    t1_cv_scheme: str = 'predefined_fold',
) -> pd.DataFrame:
    """Summarise SMILES T1–T4 stratified scores (mean±sd + Δ vs Pre+SMILES).

    Matches the three figure panels: SciPlex within-line, McFarland within-tissue,
    McFarland within-drug.
    Default T1 = predefined 5-fold; T2–T4 = exhaustive.
    Paired Δ / CI / BH-FDR $q$ (within panel × metric across tasks×modalities)
    for Pearson, Spearman, and RMSE vs Pre+SMILES.
    """
    if models is None:
        models = list(SMILES_MODEL_ORDER)
    if panel_scores is None:
        panel_scores = load_stratified_task_panel_scores(
            include_two_stage=False,
            models=models,
            t1_cv_scheme=t1_cv_scheme,
        )

    rows: list[dict] = []
    for key, stratum, dataset, panel_label in PANEL_SUMMARY_SPECS:
        frame = panel_scores.get(key, pd.DataFrame())
        if frame is None or frame.empty:
            continue
        sub = frame[frame['model'].astype(str).isin(models)].copy()
        if sub.empty:
            continue
        for task in TASK_ORDER:
            task_frame = sub[sub['task'].astype(str).eq(task)]
            if task_frame.empty:
                continue
            for model in models:
                model_frame = task_frame[task_frame['model'].astype(str).eq(model)]
                if model_frame.empty:
                    continue
                pearson = pd.to_numeric(model_frame['pearson'], errors='coerce').dropna()
                spearman = (
                    pd.to_numeric(model_frame['spearman'], errors='coerce').dropna()
                    if 'spearman' in model_frame.columns
                    else pd.Series(dtype=float)
                )
                rmse = pd.to_numeric(model_frame['rmse'], errors='coerce').dropna()
                rows.append({
                    'panel_key': key,
                    'stratum': stratum,
                    'panel': panel_label,
                    'dataset': dataset,
                    'task': task,
                    'model': model,
                    'modality': _display_modality(model),
                    'n_groups': int(len(pearson)),
                    'pearson_mean': float(pearson.mean()) if len(pearson) else np.nan,
                    'pearson_sd': float(pearson.std(ddof=1)) if len(pearson) > 1 else np.nan,
                    'spearman_mean': float(spearman.mean()) if len(spearman) else np.nan,
                    'spearman_sd': float(spearman.std(ddof=1)) if len(spearman) > 1 else np.nan,
                    'rmse_mean': float(rmse.mean()) if len(rmse) else np.nan,
                    'rmse_sd': float(rmse.std(ddof=1)) if len(rmse) > 1 else np.nan,
                    **_paired_delta_fields(
                        task_frame, model=model, ref_model=ref_model
                    ),
                })
    return _apply_bh_fdr_to_pre_smiles_pvalues(
        pd.DataFrame(rows), group_cols=['panel_key']
    )


def write_measured_tasks_stratified_summary(
    summary: pd.DataFrame | None = None,
    *,
    models: list[str] | None = None,
    outdir: str | Path | None = None,
    wrap_landscape: bool = False,
) -> tuple[Path, list[Path]]:
    """Write CSV + LaTeX longtables for SMILES measured-task stratified results.

    By default tables are not wrapped in ``landscape`` so the supplement can place
    section headings on the same page as the first table.
    """
    if summary is None:
        summary = build_measured_tasks_stratified_summary(models=models)
    if summary.empty:
        raise ValueError('No measured-task stratified summary rows to write')

    outdir = Path(outdir) if outdir is not None else Path(figures_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(results_dir) / 'measured_tasks_smiles_stratified_summary.csv'
    summary.to_csv(csv_path, index=False)
    logger.info('Wrote %s (%d rows)', csv_path, len(summary))

    written: list[Path] = []
    # Portrait-friendly: one metric per longtable (6 columns).
    colspec = (
        r'>{\raggedright\arraybackslash}p{0.08\textwidth}'
        r'>{\raggedright\arraybackslash}p{0.10\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.06\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.16\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.28\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.12\textwidth}'
    )
    metric_specs = (
        ('pearson', 'Pearson $r$', 'pearson_mean', 'pearson_sd'),
        ('spearman', r'Spearman $\rho$', 'spearman_mean', 'spearman_sd'),
        ('rmse', 'RMSE', 'rmse_mean', 'rmse_sd'),
    )

    for key, stratum, dataset, panel_label in PANEL_SUMMARY_SPECS:
        sub = summary[summary['panel_key'] == key].copy()
        if sub.empty:
            continue
        model_cats = list(models) if models is not None else list(SMILES_MODEL_ORDER)
        sub['model'] = pd.Categorical(sub['model'], categories=model_cats, ordered=True)
        sub = sub.sort_values(['task', 'model']).reset_index(drop=True)

        lines: list[str] = [
            '% Auto-generated from measured_tasks_smiles_stratified_summary.csv',
            '% Portrait longtables: all vs-Pre p-values are BH-FDR q within panel × metric.',
            '',
        ]
        for i_metric, (metric, metric_label, mean_col, sd_col) in enumerate(metric_specs):
            digits = 2 if metric in ('pearson', 'spearman') else 3
            body: list[str] = []
            for _, row in sub.iterrows():
                is_ref = str(row['model']) == 'Pre+SMILES'
                delta_tex, p_tex = _fmt_delta_p_cells(row, metric, is_ref=is_ref)
                cells = [
                    _latex_escape(row['task']),
                    _latex_escape(row['modality']),
                    str(int(row['n_groups'])),
                    _fmt_mean_sd(
                        row.get(mean_col, np.nan),
                        row.get(sd_col, np.nan),
                        digits=digits,
                    ),
                    delta_tex,
                    p_tex,
                ]
                body.append(' & '.join(cells) + r' \\')

            caption = (
                f'{dataset} {panel_label.lower()}: {metric_label} '
                r'(mean $\pm$ s.d.; paired $\Delta$ [95\% CI], BH-FDR $q$ vs Pre+Morgan '
                r'within panel across tasks$\times$modalities; '
                r'$^{**}$ $q<0.01$, $^{*}$ $q<0.05$).'
            )
            header = (
                r'Task & Modality & $n$ & Mean $\pm$ s.d. & '
                r'$\Delta$ vs Pre+Morgan [95\% CI] & $q$ \\'
            )
            # No mid-stratum Needspace: that was splitting the three metrics across pages.
            block: list[str] = [
                r'\begingroup',
                r'\tiny',
                r'\setlength{\tabcolsep}{2pt}',
                r'\setlength{\abovecaptionskip}{2pt}',
                r'\setlength{\belowcaptionskip}{1pt}',
                r'\renewcommand{\arraystretch}{0.92}',
                r'\setlength{\LTleft}{\fill}',
                r'\setlength{\LTright}{\fill}',
                rf'\begin{{longtable}}{{{colspec}}}',
                rf'\caption{{{caption}}}',
                rf'\label{{tab:measured-tasks-smiles-{stratum}-{metric}}}\\',
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
                r'\multicolumn{6}{r}{\textit{Continued on next page}}\\',
                r'\endfoot',
                r'\bottomrule',
                r'\endlastfoot',
                *body,
                r'\end{longtable}',
                r'\endgroup',
            ]
            if i_metric < len(metric_specs) - 1:
                block.append(r'\vspace{0.35em}')
            block.append('')
            lines.extend(block)

        if wrap_landscape:
            lines = [r'\begin{landscape}', *lines, r'\end{landscape}', '']
        path = outdir / f'measured_tasks_smiles_stratified_summary_{stratum}.tex'
        path.write_text('\n'.join(lines), encoding='utf-8')
        written.append(path)
        logger.info('Wrote %s (%d rows × %d metrics)', path, len(sub), len(metric_specs))

    master = outdir / 'measured_tasks_smiles_stratified_summary_tables.tex'
    master_lines = [
        '% Auto-generated SMILES measured-task stratified summary tables.',
        '% Requires: booktabs, longtable (portrait).',
        '',
    ]
    for path in written:
        master_lines.append(rf'\input{{{path.name}}}')
        master_lines.append('')
    master_lines.append('')
    master.write_text('\n'.join(master_lines), encoding='utf-8')
    written.append(master)
    logger.info('Wrote master %s', master)
    return csv_path, written


def build_measured_tasks_pooled_summary(
    scores: pd.DataFrame | None = None,
    *,
    models: list[str] | None = None,
    ref_model: str = 'Pre+SMILES',
    t1_cv_scheme: str = 'predefined_fold',
) -> pd.DataFrame:
    """Summarise SMILES T1–T4 pooled group scores (SciPlex line / McFarland tissue).

    Uses the same grouping as ``load_grouped_task_scores`` / main boxplots.
    """
    if models is None:
        models = list(SMILES_MODEL_ORDER)
    if scores is None:
        scores = load_grouped_task_scores(
            include_two_stage=False, t1_cv_scheme=t1_cv_scheme
        )
    frame = scores[scores['model'].astype(str).isin(models)].copy()
    if frame.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for dataset in ('SciPlex3', 'McFarland'):
        ds = frame[frame['dataset'].astype(str).eq(dataset)]
        if ds.empty:
            continue
        for task in TASK_ORDER:
            task_frame = ds[ds['task'].astype(str).eq(task)]
            if task_frame.empty:
                continue
            for model in models:
                model_frame = task_frame[task_frame['model'].astype(str).eq(model)]
                if model_frame.empty:
                    continue
                pearson = pd.to_numeric(model_frame['pearson'], errors='coerce').dropna()
                spearman = (
                    pd.to_numeric(model_frame['spearman'], errors='coerce').dropna()
                    if 'spearman' in model_frame.columns
                    else pd.Series(dtype=float)
                )
                rmse = (
                    pd.to_numeric(model_frame['rmse'], errors='coerce').dropna()
                    if 'rmse' in model_frame.columns
                    else pd.Series(dtype=float)
                )
                rows.append({
                    'dataset': dataset,
                    'task': task,
                    'model': model,
                    'modality': _display_modality(model),
                    'n_groups': int(len(pearson)),
                    'pearson_mean': float(pearson.mean()) if len(pearson) else np.nan,
                    'pearson_sd': float(pearson.std(ddof=1)) if len(pearson) > 1 else np.nan,
                    'spearman_mean': float(spearman.mean()) if len(spearman) else np.nan,
                    'spearman_sd': float(spearman.std(ddof=1)) if len(spearman) > 1 else np.nan,
                    'rmse_mean': float(rmse.mean()) if len(rmse) else np.nan,
                    'rmse_sd': float(rmse.std(ddof=1)) if len(rmse) > 1 else np.nan,
                    **_paired_delta_fields(
                        task_frame, model=model, ref_model=ref_model
                    ),
                })
    return _apply_bh_fdr_to_pre_smiles_pvalues(
        pd.DataFrame(rows), group_cols=['dataset']
    )


def write_measured_tasks_pooled_summary(
    summary: pd.DataFrame | None = None,
    *,
    models: list[str] | None = None,
    outdir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write CSV + LaTeX longtable for pooled SMILES measured-task results."""
    if models is None:
        models = list(SMILES_MODEL_ORDER)
    if summary is None:
        summary = build_measured_tasks_pooled_summary(models=models)
    if summary.empty:
        raise ValueError('No measured-task pooled summary rows to write')

    outdir = Path(outdir) if outdir is not None else Path(figures_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(results_dir) / 'measured_tasks_smiles_pooled_summary.csv'
    summary.to_csv(csv_path, index=False)
    logger.info('Wrote %s (%d rows)', csv_path, len(summary))

    model_cats = list(models)
    body: list[str] = []
    for dataset in ('SciPlex3', 'McFarland'):
        sub = summary[summary['dataset'].astype(str).eq(dataset)].copy()
        if sub.empty:
            continue
        sub['model'] = pd.Categorical(sub['model'], categories=model_cats, ordered=True)
        sub = sub.sort_values(['task', 'model']).reset_index(drop=True)
        for _, row in sub.iterrows():
            is_ref = str(row['model']) == 'Pre+SMILES'
            pearson_d, pearson_p = _fmt_delta_p_cells(row, 'pearson', is_ref=is_ref)
            spearman_d, spearman_p = _fmt_delta_p_cells(row, 'spearman', is_ref=is_ref)
            rmse_d, rmse_p = _fmt_delta_p_cells(row, 'rmse', is_ref=is_ref)
            cells = [
                _latex_escape(row['dataset']),
                _latex_escape(row['task']),
                _latex_escape(row['modality']),
                str(int(row['n_groups'])),
                _fmt_mean_sd(row['pearson_mean'], row['pearson_sd'], digits=2),
                pearson_d,
                pearson_p,
                _fmt_mean_sd(
                    row.get('spearman_mean', np.nan),
                    row.get('spearman_sd', np.nan),
                    digits=2,
                ),
                spearman_d,
                spearman_p,
                _fmt_mean_sd(row['rmse_mean'], row['rmse_sd'], digits=3),
                rmse_d,
                rmse_p,
            ]
            body.append(' & '.join(cells) + r' \\')

    colspec = (
        r'>{\raggedright\arraybackslash}p{0.07\textwidth}'
        r'>{\raggedright\arraybackslash}p{0.04\textwidth}'
        r'>{\raggedright\arraybackslash}p{0.05\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.03\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.07\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.10\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.045\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.07\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.10\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.045\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.07\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.10\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.045\textwidth}'
    )
    caption = (
        r'Measured-profile ElasticNet: pooled SciPlex3 within cell line / McFarland within tissue '
        r'Pearson $r$ / Spearman $\rho$ / RMSE '
        r'(mean $\pm$ s.d.\ across groups; paired $\Delta$ [95\% CI] and BH-FDR $q$ '
        r'vs Pre+Morgan within dataset across tasks$\times$modalities for each metric; '
        r'sig: $^{**}$ $q<0.01$, $^{*}$ $q<0.05$).'
    )
    header = (
        r'Dataset & Task & Modality & $n$ & '
        r'Pearson $r$ & $\Delta r$ [95\% CI] & $q$ & '
        r'Spearman $\rho$ & $\Delta\rho$ [95\% CI] & $q$ & '
        r'RMSE & $\Delta$RMSE [95\% CI] & $q$ \\'
    )
    lines = [
        r'\begin{landscape}',
        '% Auto-generated from measured_tasks_smiles_pooled_summary.csv',
        r'\begingroup',
        r'\scriptsize',
        r'\setlength{\tabcolsep}{2pt}',
        r'\setlength{\LTleft}{\fill}',
        r'\setlength{\LTright}{\fill}',
        rf'\begin{{longtable}}{{{colspec}}}',
        rf'\caption{{{caption}}}',
        r'\label{tab:measured-tasks-smiles-pooled}\\',
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
        r'\multicolumn{13}{r}{\textit{Continued on next page}}\\',
        r'\endfoot',
        r'\bottomrule',
        r'\endlastfoot',
        *body,
        r'\end{longtable}',
        r'\endgroup',
        r'\end{landscape}',
        '',
    ]
    tex_path = outdir / 'measured_tasks_smiles_pooled_summary.tex'
    tex_path.write_text('\n'.join(lines), encoding='utf-8')
    logger.info('Wrote %s (%d rows)', tex_path, len(summary))
    return csv_path, tex_path


def build_measured_tasks_lfc_vs_pre_tests(
    scores: pd.DataFrame | None = None,
    *,
    lfc_model: str = 'LFC+SMILES',
    pre_model: str = 'Pre+SMILES',
    t1_cv_scheme: str = 'predefined_fold',
    include_stratified: bool = True,
) -> pd.DataFrame:
    """Paired LFC vs Pre Pearson tests for every dataset × task (and stratified panels).

    Default models are SMILES modalities. Pass ``LFC`` / ``Pre`` for gene-only.
    Pooled rows use SciPlex cell-line / McFarland tissue grouping (main boxplots).
    When ``include_stratified`` is True, also add SciPlex within-line, McFarland
    within-tissue, and McFarland within-drug panels.
    """
    models = [pre_model, lfc_model]
    rows: list[dict] = []

    def _append_rows(
        frame: pd.DataFrame,
        *,
        dataset: str,
        panel: str,
        group_col: str,
    ) -> None:
        if frame is None or frame.empty:
            return
        sub = frame[frame['model'].astype(str).isin(models)].copy()
        if sub.empty:
            return
        for task in TASK_ORDER:
            task_frame = sub[sub['task'].astype(str).eq(task)]
            if task_frame.empty:
                continue
            d_mean, d_lo, d_hi, d_p, d_sig, d_n = _paired_delta_vs_ref(
                task_frame,
                model=lfc_model,
                ref_model=pre_model,
                metric='pearson',
                group_col=group_col,
            )
            lfc_vals = pd.to_numeric(
                task_frame.loc[task_frame['model'].astype(str).eq(lfc_model), 'pearson'],
                errors='coerce',
            ).dropna()
            pre_vals = pd.to_numeric(
                task_frame.loc[task_frame['model'].astype(str).eq(pre_model), 'pearson'],
                errors='coerce',
            ).dropna()
            rows.append({
                'panel': panel,
                'dataset': dataset,
                'task': task,
                'lfc_model': lfc_model,
                'pre_model': pre_model,
                'n_groups': d_n,
                'lfc_pearson_mean': float(lfc_vals.mean()) if len(lfc_vals) else np.nan,
                'lfc_pearson_sd': float(lfc_vals.std(ddof=1)) if len(lfc_vals) > 1 else np.nan,
                'pre_pearson_mean': float(pre_vals.mean()) if len(pre_vals) else np.nan,
                'pre_pearson_sd': float(pre_vals.std(ddof=1)) if len(pre_vals) > 1 else np.nan,
                'delta_lfc_minus_pre': d_mean,
                'delta_ci_low': d_lo,
                'delta_ci_high': d_hi,
                'p_value': d_p,
                'significance': d_sig,
            })

    if scores is None:
        scores = load_grouped_task_scores(
            include_two_stage=False, t1_cv_scheme=t1_cv_scheme
        )
    for dataset in ('SciPlex3', 'McFarland'):
        ds = scores[scores['dataset'].astype(str).eq(dataset)]
        _append_rows(
            ds,
            dataset=dataset,
            panel='pooled',
            group_col='group',
        )

    if include_stratified:
        panels = load_stratified_task_panel_scores(
            include_two_stage=False,
            models=models,
            t1_cv_scheme=t1_cv_scheme,
            align_common_groups=True,
        )
        for key, stratum, dataset, panel_label in PANEL_SUMMARY_SPECS:
            _append_rows(
                panels.get(key, pd.DataFrame()),
                dataset=dataset,
                panel=panel_label,
                group_col='group',
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    panel_order = ['pooled', 'Within cell line', 'Within tissue', 'Within drug']
    out['panel'] = pd.Categorical(out['panel'], categories=panel_order, ordered=True)
    out['dataset'] = pd.Categorical(
        out['dataset'], categories=['SciPlex3', 'McFarland'], ordered=True
    )
    out['task'] = pd.Categorical(out['task'], categories=list(TASK_ORDER), ordered=True)
    out = out.sort_values(['panel', 'dataset', 'task']).reset_index(drop=True)
    # Primary family: BH-FDR within each panel × dataset across tasks.
    out['p_value_fdr'] = np.nan
    for _, idx in out.groupby(['panel', 'dataset'], dropna=False).groups.items():
        idx = list(idx)
        vals = pd.to_numeric(out.loc[idx, 'p_value'], errors='coerce')
        mask = vals.notna()
        if not mask.any():
            continue
        q = multipletests(vals.loc[mask], alpha=0.05, method='fdr_bh')[1]
        out.loc[vals.index[mask], 'p_value_fdr'] = q
    out['significance'] = out['p_value_fdr'].map(
        lambda v: _significance_stars(float(v)) if pd.notna(v) else 'NS'
    )
    return out


def write_measured_tasks_lfc_vs_pre_tests(
    summary: pd.DataFrame | None = None,
    *,
    lfc_model: str = 'LFC+SMILES',
    pre_model: str = 'Pre+SMILES',
    outdir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write CSV + LaTeX table for paired LFC vs Pre tests (dataset × task)."""
    if summary is None:
        summary = build_measured_tasks_lfc_vs_pre_tests(
            lfc_model=lfc_model, pre_model=pre_model
        )
    if summary.empty:
        raise ValueError('No LFC vs Pre test rows to write')

    outdir = Path(outdir) if outdir is not None else Path(figures_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(results_dir) / 'measured_tasks_smiles_lfc_vs_pre_tests.csv'
    summary.to_csv(csv_path, index=False)
    logger.info('Wrote %s (%d rows)', csv_path, len(summary))

    lfc_lab = _display_modality(lfc_model)
    pre_lab = _display_modality(pre_model)
    body: list[str] = []
    for _, row in summary.iterrows():
        cells = [
            _latex_escape(row['panel']),
            _latex_escape(row['dataset']),
            _latex_escape(row['task']),
            str(int(row['n_groups'])),
            _fmt_mean_sd(row['pre_pearson_mean'], row['pre_pearson_sd'], digits=2),
            _fmt_mean_sd(row['lfc_pearson_mean'], row['lfc_pearson_sd'], digits=2),
            _fmt_ci(
                row['delta_lfc_minus_pre'],
                row['delta_ci_low'],
                row['delta_ci_high'],
                digits=2,
            ),
            (
                f'{float(row["p_value_fdr"]):.3g}'
                if 'p_value_fdr' in row.index
                and pd.notna(row['p_value_fdr'])
                and np.isfinite(row['p_value_fdr'])
                else (
                    f'{float(row["p_value"]):.3g}'
                    if pd.notna(row['p_value']) and np.isfinite(row['p_value'])
                    else '--'
                )
            ),
            _fmt_sig_tex(str(row['significance'])),
        ]
        body.append(' & '.join(cells) + r' \\')

    colspec = (
        r'>{\raggedright\arraybackslash}p{0.12\textwidth}'
        r'>{\raggedright\arraybackslash}p{0.10\textwidth}'
        r'>{\raggedright\arraybackslash}p{0.06\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.04\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.12\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.12\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.16\textwidth}'
        r'>{\raggedleft\arraybackslash}p{0.08\textwidth}'
        r'>{\centering\arraybackslash}p{0.05\textwidth}'
    )
    caption = (
        rf'Measured-profile ElasticNet: primary paired {lfc_lab} vs {pre_lab} Pearson $r$ '
        r'(mean $\pm$ s.d.\ across groups; paired $\Delta$ [95\% CI]; BH-FDR $q$ within '
        r'each panel across tasks; sig: $^{**}$ $q<0.01$, $^{*}$ $q<0.05$).'
    )
    header = (
        rf'Panel & Dataset & Task & $n$ & {pre_lab} $r$ & {lfc_lab} $r$ & '
        rf'$\Delta$ ({lfc_lab}$-${pre_lab}) [95\% CI] & $q$ & Sig \\'
    )
    lines = [
        r'\begin{landscape}',
        '% Auto-generated from measured_tasks_smiles_lfc_vs_pre_tests.csv',
        r'\begingroup',
        r'\scriptsize',
        r'\setlength{\tabcolsep}{3pt}',
        r'\setlength{\LTleft}{\fill}',
        r'\setlength{\LTright}{\fill}',
        rf'\begin{{longtable}}{{{colspec}}}',
        rf'\caption{{{caption}}}',
        r'\label{tab:measured-tasks-smiles-lfc-vs-pre}\\',
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
        r'\multicolumn{9}{r}{\textit{Continued on next page}}\\',
        r'\endfoot',
        r'\bottomrule',
        r'\endlastfoot',
        *body,
        r'\end{longtable}',
        r'\endgroup',
        r'\end{landscape}',
        '',
    ]
    tex_path = outdir / 'measured_tasks_smiles_lfc_vs_pre_tests.tex'
    tex_path.write_text('\n'.join(lines), encoding='utf-8')
    logger.info('Wrote %s (%d rows)', tex_path, len(summary))
    return csv_path, tex_path


def plot_mcfarland_within_tissue_pred_vs_true(
    *,
    models: list[str] | None = None,
    tasks: list[str] | None = None,
    t1_cv_scheme: str = 'predefined_fold',
    outfile: str | None = None,
    n_cols: int = 5,
    point_size: float = 18,
) -> list[plt.Figure]:
    """Scatter predicted vs true sensitivity within each McFarland tissue.

    One figure per task; panels are tissues (sorted by Pre+SMILES Pearson when
    that model is present). Points are coloured by model. Matches the stratified
    within-tissue scoring (OOF predictions pooled within tissue).
    """
    from matplotlib.lines import Line2D

    if models is None:
        models = list(SMILES_MODEL_ORDER)
    if tasks is None:
        tasks = list(TASK_ORDER)

    preds = _load_prediction_frames(include_two_stage=False, t1_cv_scheme=t1_cv_scheme)
    mcf = preds[
        preds['dataset'].astype(str).eq('McFarland')
        & preds['model'].astype(str).isin(models)
        & preds['task'].astype(str).isin(tasks)
    ].copy()
    if mcf.empty:
        raise ValueError('No McFarland predictions for the requested models/tasks')
    if 'tissue' not in mcf.columns:
        raise ValueError('McFarland predictions missing tissue column')

    palette = {m: PALETTE.get(m, '#333333') for m in models}
    tissues_all = sorted(mcf['tissue'].dropna().astype(str).unique())
    figures: list[plt.Figure] = []

    for task in tasks:
        task_df = mcf[mcf['task'].astype(str).eq(task)].copy()
        if task_df.empty:
            continue

        # Sort tissues by Pre+SMILES (or first model) Pearson, ascending.
        sort_model = 'Pre+SMILES' if 'Pre+SMILES' in models else models[0]
        tissue_order: list[str] = []
        sort_scores: dict[str, float] = {}
        for tissue in tissues_all:
            g = task_df[
                task_df['tissue'].astype(str).eq(tissue)
                & task_df['model'].astype(str).eq(sort_model)
            ]
            if len(g) < 2:
                sort_scores[tissue] = np.nan
                continue
            sort_scores[tissue] = _safe_pearson(
                g['true'].to_numpy(dtype=float),
                g['pred'].to_numpy(dtype=float),
            )
        tissue_order = sorted(
            tissues_all,
            key=lambda t: (
                np.isnan(sort_scores.get(t, np.nan)),
                sort_scores.get(t, np.nan),
            ),
        )

        n = len(tissue_order)
        n_cols_use = max(1, min(n_cols, n))
        n_rows = int(np.ceil(n / n_cols_use))
        fig, axes = plt.subplots(
            n_rows,
            n_cols_use,
            figsize=(3.2 * n_cols_use, 3.0 * n_rows),
            dpi=200,
            squeeze=False,
        )
        axes_flat = axes.ravel()

        for ax_i, tissue in enumerate(tissue_order):
            ax = axes_flat[ax_i]
            sub = task_df[task_df['tissue'].astype(str).eq(tissue)]
            lo_hi = []
            r_lines: list[str] = []
            for model in models:
                g = sub[sub['model'].astype(str).eq(model)]
                if g.empty:
                    continue
                y_true = g['true'].to_numpy(dtype=float)
                y_pred = g['pred'].to_numpy(dtype=float)
                mask = np.isfinite(y_true) & np.isfinite(y_pred)
                y_true, y_pred = y_true[mask], y_pred[mask]
                if len(y_true) == 0:
                    continue
                r = _safe_pearson(y_true, y_pred)
                ax.scatter(
                    y_true,
                    y_pred,
                    s=point_size,
                    alpha=0.75,
                    color=palette[model],
                    edgecolors='none',
                )
                if np.isfinite(r):
                    r_lines.append(f'{model}: r={r:.2f} (n={len(y_true)})')
                lo_hi.extend([float(y_true.min()), float(y_true.max()),
                              float(y_pred.min()), float(y_pred.max())])
            if r_lines:
                ax.text(
                    0.02,
                    0.98,
                    '\n'.join(r_lines),
                    transform=ax.transAxes,
                    va='top',
                    ha='left',
                    fontsize=7,
                    family='monospace',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'),
                )
            if lo_hi:
                lo, hi = min(lo_hi), max(lo_hi)
                pad = 0.05 * (hi - lo + 1e-6)
                lims = (lo - pad, hi + pad)
                ax.plot(lims, lims, color='0.5', lw=0.8, ls='--', zorder=0)
                ax.set_xlim(lims)
                ax.set_ylim(lims)
            ax.set_title(str(tissue), fontsize=10, fontweight='bold')
            ax.set_aspect('equal', adjustable='box')
            ax.tick_params(labelsize=8)
            if ax_i % n_cols_use == 0:
                ax.set_ylabel('Predicted sensitivity', fontsize=9)
            if ax_i >= (n_rows - 1) * n_cols_use:
                ax.set_xlabel('Observed sensitivity', fontsize=9)

        for ax in axes_flat[n:]:
            ax.axis('off')

        handles = [
            Line2D(
                [0], [0],
                marker='o',
                color='none',
                markerfacecolor=palette[m],
                markersize=8,
                label=m,
            )
            for m in models
        ]
        fig.legend(
            handles,
            models,
            loc='upper center',
            ncol=len(models),
            frameon=False,
            fontsize=11,
            bbox_to_anchor=(0.5, 1.02),
        )
        fig.suptitle(
            f'McFarland within-tissue predicted vs observed ({task}, '
            f'T1 cv={t1_cv_scheme})',
            fontsize=13,
            fontweight='bold',
            y=1.06,
        )
        fig.tight_layout()
        figures.append(fig)

        if outfile is None:
            stem = (
                f'measured_tasks_mcfarland_within_tissue_pred_vs_true_{task.lower()}'
            )
            out_path = os.path.join(figures_dir, f'{stem}.pdf')
        else:
            # If a single path is given for multiple tasks, insert task before .pdf
            if len(tasks) > 1 and str(outfile).endswith('.pdf'):
                out_path = str(outfile).replace('.pdf', f'_{task.lower()}.pdf')
            else:
                out_path = outfile
        fig.savefig(out_path, bbox_inches='tight')
        logger.info('Saved %s', out_path)

    return figures


# Backwards-compatible alias after brief within-line experiment.
plot_mcfarland_within_cell_line_pred_vs_true = plot_mcfarland_within_tissue_pred_vs_true


def plot_task_pred_vs_true_by_modality(
    *,
    dataset: str = 'SciPlex3',
    task: str = 'T2',
    models: list[str] | None = None,
    color_by: str = 'held_out_contexts',
    t1_cv_scheme: str = 'predefined_fold',
    outfile: str | Path | None = None,
    point_size: float = 22,
    title: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """True vs predicted sensitivity for one dataset/task, one panel per modality.

    Default use case: SciPlex T2 (leave-one-cell-line-out) with Pre/Post/LFC+SMILES.
    Points are coloured by ``color_by`` (default: held-out context).
    """
    if models is None:
        models = list(SMILES_MODEL_ORDER)

    preds = _load_prediction_frames(include_two_stage=False, t1_cv_scheme=t1_cv_scheme)
    df = preds[
        preds['dataset'].astype(str).eq(dataset)
        & preds['task'].astype(str).eq(task)
        & preds['model'].astype(str).isin(models)
    ].copy()
    if df.empty:
        raise ValueError(f'No predictions for dataset={dataset!r} task={task!r}')
    if color_by not in df.columns or df[color_by].isna().all():
        fallback = GROUP_COL_BY_DATASET.get(dataset, 'cell_line')
        if fallback not in df.columns:
            raise ValueError(f'Missing colour column {color_by!r} and fallback {fallback!r}')
        color_by = fallback

    df = df.dropna(subset=['true', 'pred'])
    groups = sorted(df[color_by].dropna().astype(str).unique())
    palette = dict(zip(groups, sns.color_palette('colorblind', n_colors=max(len(groups), 1))))

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.5), dpi=300, squeeze=False)
    axes = axes.ravel()

    for ax, model in zip(axes, models):
        sub = df.loc[df['model'].astype(str).eq(model)].copy()
        for group in groups:
            pts = sub.loc[sub[color_by].astype(str).eq(group)]
            if pts.empty:
                continue
            ax.scatter(
                pts['true'],
                pts['pred'],
                s=point_size,
                alpha=0.75,
                color=palette[group],
                edgecolors='none',
                label=group,
                zorder=2,
            )

        lo = float(min(sub['true'].min(), sub['pred'].min(), 0.0))
        hi = float(max(sub['true'].max(), sub['pred'].max(), 1.0))
        pad = 0.04 * (hi - lo if hi > lo else 1.0)
        lim = (lo - pad, hi + pad)
        ax.plot(lim, lim, color='0.4', lw=1.0, ls='--', zorder=1)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect('equal', adjustable='box')

        r_lines = [
            f'all: r={_safe_pearson(sub["true"].to_numpy(), sub["pred"].to_numpy()):.2f}'
        ]
        for group in groups:
            pts = sub.loc[sub[color_by].astype(str).eq(group)]
            if len(pts) < 2:
                continue
            r = _safe_pearson(pts['true'].to_numpy(), pts['pred'].to_numpy())
            r_lines.append(f'{group}: r={r:.2f}')
        ax.text(
            0.03,
            0.97,
            '\n'.join(r_lines),
            transform=ax.transAxes,
            va='top',
            ha='left',
            fontsize=9,
            family='monospace',
            bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'none', 'pad': 2},
        )
        ax.set_xlabel('True sensitivity')
        ax.set_ylabel('Predicted sensitivity' if ax is axes[0] else '')
        ax.set_title(model, fontsize=13, fontweight='bold')
        sns.despine(ax=ax)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title='Held-out context' if color_by == 'held_out_contexts' else color_by,
            loc='center left',
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            fontsize=9,
            title_fontsize=10,
            markerscale=1.3,
        )
    if title is None:
        title = f'{dataset} {task}: predicted vs observed'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    if outfile is None:
        stem = f'measured_tasks_{dataset.lower()}_{task.lower()}_pred_vs_true'
        outfile = Path(figures_dir) / f'{stem}.pdf'
    else:
        outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, bbox_inches='tight')
    logger.info('Saved %s', outfile)
    return fig, axes


def load_measured_task_predictions(
    *,
    dataset: str = 'SciPlex3',
    output_tag: str = '',
    tasks: list[str] | None = None,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Load measured-task OOF predictions for one dataset / optional output tag.

    ``output_tag='nosmiles'`` → ``sciplex_nosmiles_measured_tasks_predictions.csv``.
    Empty tag → ``sciplex_measured_tasks_predictions.csv`` (plus T1 5-fold file).
    """
    prefix = 'sciplex' if 'sciplex' in dataset.lower() else 'mcfarland'
    if output_tag:
        prefix = f'{prefix}_{output_tag}'
    paths = [
        Path(results_dir) / f'{prefix}_measured_tasks_predictions.csv',
        Path(results_dir) / f'{prefix}_measured_tasks_t1_5fold_predictions.csv',
    ]
    frames = []
    for path in paths:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(
            f'No measured-task predictions for prefix={prefix!r} under {results_dir}'
        )
    df = pd.concat(frames, ignore_index=True)
    if 'dataset' not in df.columns:
        df['dataset'] = dataset
    if tasks is not None:
        df = df[df['task'].astype(str).isin(tasks)].copy()
    if models is not None:
        df = df[df['model'].astype(str).isin(models)].copy()
    return df


def plot_smiles_vs_nosmiles_pred_vs_true(
    *,
    dataset: str = 'SciPlex3',
    task: str = 'T2',
    smiles_tag: str = '',
    nosmiles_tag: str = 'nosmiles',
    modalities: tuple[str, ...] = ('Pre', 'Post', 'LFC'),
    color_by: str = 'held_out_contexts',
    outfile: str | Path | None = None,
    point_size: float = 18,
) -> tuple[plt.Figure, np.ndarray]:
    """2×3 scatter: rows = −SMILES / +SMILES, columns = Pre / Post / LFC."""
    smiles_models = [f'{m}+SMILES' for m in modalities]
    nosmiles_models = list(modalities)

    smiles = load_measured_task_predictions(
        dataset=dataset, output_tag=smiles_tag, tasks=[task], models=smiles_models
    )
    nosmiles = load_measured_task_predictions(
        dataset=dataset, output_tag=nosmiles_tag, tasks=[task], models=nosmiles_models
    )

    row_specs = (
        ('−SMILES (expression only)', nosmiles, nosmiles_models),
        ('+SMILES', smiles, smiles_models),
    )
    groups = sorted(
        set(nosmiles[color_by].dropna().astype(str))
        | set(smiles[color_by].dropna().astype(str))
    )
    palette = dict(zip(groups, sns.color_palette('colorblind', n_colors=max(len(groups), 1))))

    fig, axes = plt.subplots(
        2,
        len(modalities),
        figsize=(4.4 * len(modalities), 8.4),
        dpi=300,
        squeeze=False,
    )
    for row_i, (row_title, frame, models) in enumerate(row_specs):
        for col_i, (modality, model) in enumerate(zip(modalities, models)):
            ax = axes[row_i, col_i]
            sub = frame.loc[frame['model'].astype(str).eq(model)].dropna(subset=['true', 'pred'])
            if sub.empty:
                ax.set_title(f'{modality}\n(no data)', fontsize=12)
                ax.axis('off')
                continue
            for group in groups:
                pts = sub.loc[sub[color_by].astype(str).eq(group)]
                if pts.empty:
                    continue
                ax.scatter(
                    pts['true'],
                    pts['pred'],
                    s=point_size,
                    alpha=0.7,
                    color=palette[group],
                    edgecolors='none',
                    label=group,
                    zorder=2,
                )
            lo = float(min(sub['true'].min(), sub['pred'].min(), 0.0))
            hi = float(max(sub['true'].max(), sub['pred'].max(), 1.0))
            pad = 0.04 * (hi - lo if hi > lo else 1.0)
            lim = (lo - pad, hi + pad)
            ax.plot(lim, lim, color='0.4', lw=1.0, ls='--', zorder=1)
            ax.set_xlim(*lim)
            ax.set_ylim(*lim)
            ax.set_aspect('equal', adjustable='box')
            r_all = _safe_pearson(sub['true'].to_numpy(), sub['pred'].to_numpy())
            ax.text(
                0.03,
                0.97,
                f'r={r_all:.2f}\nn={len(sub)}',
                transform=ax.transAxes,
                va='top',
                ha='left',
                fontsize=9,
                family='monospace',
                bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'none', 'pad': 2},
            )
            if row_i == 0:
                ax.set_title(modality, fontsize=13, fontweight='bold')
            if col_i == 0:
                ax.set_ylabel(f'{row_title}\nPredicted', fontsize=11)
            else:
                ax.set_ylabel('')
            ax.set_xlabel('True sensitivity' if row_i == 1 else '')
            sns.despine(ax=ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title='Held-out context',
            loc='center left',
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            fontsize=9,
            title_fontsize=10,
        )
    fig.suptitle(
        f'{dataset} {task}: expression-only vs +SMILES',
        fontsize=14,
        fontweight='bold',
        y=1.01,
    )
    fig.tight_layout()
    if outfile is None:
        outfile = (
            Path(figures_dir)
            / f'measured_tasks_{dataset.lower()}_{task.lower()}_smiles_vs_nosmiles.pdf'
        )
    else:
        outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, bbox_inches='tight')
    logger.info('Saved %s', outfile)
    return fig, axes


GENE_ONLY_MODEL_ORDER = ['Pre', 'Post', 'LFC']


def main() -> None:
    fold_scores = _load_fold_scores(t1_cv_scheme='predefined_fold')
    write_held_out_table(fold_scores)
    grouped_scores = load_grouped_task_scores(
        include_two_stage=False, t1_cv_scheme='predefined_fold'
    )
    smiles_scores = grouped_scores[
        grouped_scores['model'].astype(str).isin(SMILES_MODEL_ORDER)
    ].copy()
    plot_tasks_figure(smiles_scores, metric='pearson')
    plot_tasks_figure(smiles_scores, metric='rmse')
    comparison_scores = _load_head_comparison_scores()
    if comparison_scores is not None:
        plot_head_comparison(comparison_scores, metric='pearson')
        plot_head_comparison(comparison_scores, metric='rmse')
    # SciPlex within-line | McFarland within-tissue | McFarland within-drug
    plot_tasks_stratified_three_panel(model_order=SMILES_MODEL_ORDER)
    write_measured_tasks_stratified_summary(
        models=SMILES_MODEL_ORDER, wrap_landscape=False
    )
    write_measured_tasks_pooled_summary(
        build_measured_tasks_pooled_summary(
            smiles_scores, models=SMILES_MODEL_ORDER, t1_cv_scheme='predefined_fold'
        ),
        models=SMILES_MODEL_ORDER,
    )
    write_measured_tasks_lfc_vs_pre_tests(
        build_measured_tasks_lfc_vs_pre_tests(
            smiles_scores,
            lfc_model='LFC+SMILES',
            pre_model='Pre+SMILES',
            t1_cv_scheme='predefined_fold',
            include_stratified=True,
        )
    )
    # Expression-only stratified panel when nosmiles task-CV outputs exist
    nosmiles_present = any(
        (Path(results_dir) / f'{ds}_nosmiles_measured_tasks_predictions.csv').exists()
        for ds in ('sciplex', 'mcfarland')
    )
    if nosmiles_present:
        plot_tasks_stratified_three_panel(
            load_stratified_task_panel_scores(
                include_two_stage=False,
                models=GENE_ONLY_MODEL_ORDER,
                t1_cv_scheme='predefined_fold',
                align_common_groups=True,
                output_tag='nosmiles',
            ),
            model_order=GENE_ONLY_MODEL_ORDER,
            outfile=os.path.join(
                figures_dir, 'measured_tasks_stratified_line_drug_pearson_nosmiles.pdf'
            ),
            ylim=(-1.0, 1.0),
        )
    else:
        logger.info(
            'Skipping nosmiles stratified panel; no *_nosmiles_measured_tasks_predictions.csv'
        )


if __name__ == '__main__':
    main()
