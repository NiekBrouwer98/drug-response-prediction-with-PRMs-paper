"""Group-aware T1–T4 cross-validation for measured-profile drug-response claims.

Tasks
-----
T1  seen drug, seen tissue/line   — exhaustive leave-one-(drug, context) with
                                    train = complement (interpolation); also saves
                                    predefined 5-fold CV for predicted-profile plots
T2  seen drug, unseen tissue/line — exhaustive leave-one-context (tissue / cell_line)
T3  unseen drug, seen tissue/line — exhaustive leave-one-drug
T4  unseen drug and tissue/line   — exhaustive combinatorial leave-one-(drug, context)

McFarland uses ``tissue`` for T2/T4; SciPlex uses ``cell_line`` (only three lines).

Canonical labels (predictions and plots)
----------------------------------------
T2 = unseen context (tissue / cell line); T3 = unseen drug. Do not remap.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

TASK_DEFINITIONS: dict[str, dict[str, str]] = {
    'T1': {
        'name': 'seen_seen',
        'label': 'T1: seen drug, seen tissue/line',
        'claim': 'interpolation',
    },
    'T2': {
        'name': 'unseen_context',
        'label': 'T2: unseen tissue/line',
        'claim': 'context transfer',
    },
    'T3': {
        'name': 'unseen_drug',
        'label': 'T3: unseen drug',
        'claim': 'drug transfer',
    },
    'T4': {
        'name': 'unseen_drug_and_context',
        'label': 'T4: unseen drug and tissue/line',
        'claim': 'drug × context transfer',
    },
}

MODEL_LABELS = {
    'pre_treatment': 'Pre',
    'post_treatment': 'Post',
    'LFC': 'LFC',
    'pre_treatment_smiles': 'Pre+SMILES',
    'post_treatment_smiles': 'Post+SMILES',
    'LFC_smiles': 'LFC+SMILES',
}

# Default measured-profile CV: Pre/Post/LFC + ECFP (SMILES) fingerprints.
MEASURED_MODEL_KEYS: tuple[str, ...] = (
    'pre_treatment_smiles',
    'post_treatment_smiles',
    'LFC_smiles',
)
PLOT_MODEL_ORDER: list[str] = [MODEL_LABELS[k] for k in MEASURED_MODEL_KEYS]


def resolve_measured_model_keys(models: Iterable[str] | None = None) -> tuple[str, ...]:
    """Map CLI tokens (keys or display labels) to measured-model keys."""
    if models is None:
        return MEASURED_MODEL_KEYS
    requested = [str(m).strip() for m in models if str(m).strip()]
    if not requested:
        return MEASURED_MODEL_KEYS

    label_to_key = {label: key for key, label in MODEL_LABELS.items()}
    aliases = {
        'pre': 'pre_treatment',
        'post': 'post_treatment',
        'lfc': 'LFC',
        'pre+smiles': 'pre_treatment_smiles',
        'post+smiles': 'post_treatment_smiles',
        'lfc+smiles': 'LFC_smiles',
        'pre_smiles': 'pre_treatment_smiles',
        'post_smiles': 'post_treatment_smiles',
        'lfc_smiles': 'LFC_smiles',
    }
    resolved: list[str] = []
    for token in requested:
        key = (
            token
            if token in MODEL_LABELS
            else label_to_key.get(token)
            or aliases.get(token.lower())
        )
        if key is None or key not in MODEL_LABELS:
            raise ValueError(
                f'Unknown measured-model token {token!r}. '
                f'Use keys {list(MODEL_LABELS)} or labels {list(MODEL_LABELS.values())}.'
            )
        if key not in resolved:
            resolved.append(key)
    return tuple(resolved)

_PARAM_GRID = {
    'elasticnet__alpha': [0.1, 1.0, 10.0],
    'elasticnet__l1_ratio': [0.0, 0.1, 0.5, 1.0],
}
DEFAULT_TWO_STAGE_THRESHOLDS: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)


def _safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson r; 0 when ranking is undefined (constant true/pred or n < 2)."""
    if len(y_true) < 2:
        return 0.0
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        # Common for Pre on leave-one-cell-line folds: basal expression is
        # identical across drugs within a line, so predictions are constant.
        return 0.0
    val = pearsonr(y_true, y_pred)[0]
    return float(val) if not np.isnan(val) else 0.0


def _t1_fold_ids(df: pd.DataFrame, context_col: str, n_folds: int, rng: np.random.Generator) -> np.ndarray:
    """Assign folds within each context so every fold sees all contexts when possible."""
    fold_id = np.full(len(df), -1, dtype=int)
    for context, idx in df.groupby(context_col, dropna=False).groups.items():
        order = np.array(list(idx))
        if len(order) > n_folds:
            rng.shuffle(order)
        fold_id[order] = np.arange(len(order)) % n_folds
    unassigned = fold_id < 0
    if unassigned.any():
        order = np.where(unassigned)[0]
        rng.shuffle(order)
        fold_id[order] = np.arange(len(order)) % n_folds
    return fold_id


def _attach_predefined_folds(
    profiles: pd.DataFrame,
    fold_reference: pd.DataFrame,
) -> pd.DataFrame:
    """Attach perturbation-model fold labels by measured profile key."""
    key_cols = ['cell_line', 'condition']
    missing = [c for c in key_cols + ['fold'] if c not in fold_reference.columns]
    if missing:
        raise ValueError(f'fold_reference missing required columns: {missing}')
    if any(c not in profiles.columns for c in key_cols):
        missing_profiles = [c for c in key_cols if c not in profiles.columns]
        raise ValueError(f'profiles missing required columns for fold merge: {missing_profiles}')

    fold_keys = fold_reference[key_cols + ['fold']].dropna().drop_duplicates()
    out = profiles.drop(columns=['fold'], errors='ignore').merge(fold_keys, on=key_cols, how='left')
    n_with_fold = int(out['fold'].notna().sum())
    logger.info(
        '_attach_predefined_folds: matched fold labels for %d / %d rows',
        n_with_fold,
        len(out),
    )
    if n_with_fold == 0 and len(out) > 0:
        raise ValueError('No measured profiles matched perturbation-model fold labels')
    return out


def _prepare_fold_matrices(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    selected_features: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    meta = {'tissue', 'cell_line', 'condition', 'y', 'target', 'sens', 'sens_label', 'fold', 'drug', 'product_name'}
    features = [
        f
        for f in selected_features
        if f in X_train.columns and f in X_test.columns and f not in meta
    ]
    X_tr = X_train[features].astype(float).fillna(0.0)
    X_te = X_test[features].astype(float).fillna(0.0)
    y_tr = y_train.astype(float)

    if X_tr.shape[0] < 2 or X_te.shape[0] < 1:
        raise ValueError(f'train={X_tr.shape[0]} test={X_te.shape[0]}')
    if X_tr.shape[1] == 0:
        raise ValueError('No selected feature columns available')
    return X_tr, y_tr, X_te


def _fit_continuous_elasticnet(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_te: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    cv_folds = min(5, max(2, X_tr.shape[0]))

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('elasticnet', ElasticNet(max_iter=2000)),
    ])
    grid = GridSearchCV(pipeline, _PARAM_GRID, cv=min(cv_folds, X_tr.shape[0]))
    grid.fit(X_tr, y_tr)
    y_pred = grid.predict(X_te)
    y_train_pred = grid.predict(X_tr)
    return y_pred, y_train_pred


def _fit_two_stage(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_te: pd.DataFrame,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    y_bin = (y_tr > threshold).astype(int)

    if int(y_bin.nunique()) < 2:
        if int(y_bin.iloc[0]) == 0:
            return np.zeros(X_te.shape[0]), np.zeros(X_tr.shape[0])
        return _fit_continuous_elasticnet(X_tr, y_tr, X_te)

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000)),
    ])
    clf.fit(X_tr, y_bin)
    test_bin_pred = clf.predict(X_te)
    train_bin_pred = clf.predict(X_tr)

    positive_mask = y_tr > threshold
    n_positive = int(positive_mask.sum())
    if n_positive < 2:
        positive_mean = float(y_tr.loc[positive_mask].mean()) if n_positive else 0.0
        test_reg_pred = np.full(X_te.shape[0], positive_mean)
        train_reg_pred = np.full(X_tr.shape[0], positive_mean)
    else:
        test_reg_pred, train_reg_pred_positive = _fit_continuous_elasticnet(
            X_tr.loc[positive_mask],
            y_tr.loc[positive_mask],
            X_te,
        )
        train_reg_pred = np.full(X_tr.shape[0], float(y_tr.loc[positive_mask].mean()))
        train_reg_pred[positive_mask.to_numpy()] = train_reg_pred_positive

    test_pred = np.where(test_bin_pred == 1, test_reg_pred, 0.0)
    train_pred = np.where(train_bin_pred == 1, train_reg_pred, 0.0)
    return test_pred, train_pred


def _fit_predict_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    selected_features: list[str],
    head: str = 'continuous',
    threshold: float = 0.02,
) -> tuple[np.ndarray, float]:
    X_tr, y_tr, X_te = _prepare_fold_matrices(X_train, y_train, X_test, selected_features)

    if head == 'continuous':
        y_pred, y_train_pred = _fit_continuous_elasticnet(X_tr, y_tr, X_te)
    elif head == 'two_stage':
        y_pred, y_train_pred = _fit_two_stage(X_tr, y_tr, X_te, threshold=threshold)
    else:
        raise ValueError(f"Unknown prediction head {head!r}; expected 'continuous' or 'two_stage'")

    return y_pred, float(np.sqrt(mean_squared_error(y_tr, y_train_pred)))


def run_task_cv(
    df_with_y: pd.DataFrame,
    selected_features: list[str],
    task: str,
    context_col: str = 'tissue',
    drug_col: str = 'condition',
    n_folds: int | None = None,
    n_repeats: int = 5,
    random_state: int = 1,
    head: str = 'continuous',
    threshold: float = 0.02,
    t1_scheme: str = 'exhaustive',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run one T1–T4 task.

    T1 ``exhaustive``: leave-one-(drug, context) with ``train = ~test`` (both factors
    remain seen via other pairs). T1 ``predefined_fold``: perturbation-model 5-fold
    labels (or stratified random folds) for predicted-profile figure injection.
    T2/T3/T4 are exhaustive group holdouts. For T4, the training set excludes both
    the held-out drug and the held-out context; it is not merely the complement of
    the held-out drug-context pair.

    Returns
    -------
    scores : fold-level metrics (pearson, rmse) with held-out group IDs
    predictions : per-sample predictions
    """
    if task not in TASK_DEFINITIONS:
        raise ValueError(f'Unknown task {task!r}; expected one of {list(TASK_DEFINITIONS)}')
    if head not in {'continuous', 'two_stage'}:
        raise ValueError("head must be 'continuous' or 'two_stage'")
    if t1_scheme not in {'exhaustive', 'predefined_fold'}:
        raise ValueError("t1_scheme must be 'exhaustive' or 'predefined_fold'")

    df = df_with_y.reset_index(drop=True).copy()
    meta_cols = [c for c in ('y', context_col, 'cell_line', drug_col) if c in df.columns]
    df = df.dropna(subset=meta_cols)
    if task == 'T1' and t1_scheme == 'predefined_fold' and 'fold' in df.columns:
        df = df.dropna(subset=['fold'])
    if context_col not in df.columns:
        raise ValueError(f'context_col {context_col!r} missing from frame')
    if drug_col not in df.columns:
        raise ValueError(f'drug_col {drug_col!r} missing from frame')

    drugs = sorted(df[drug_col].dropna().astype(str).unique().tolist())
    contexts = sorted(df[context_col].dropna().astype(str).unique().tolist())

    selected_features = list(set(selected_features).intersection(df.columns))
    score_rows: list[dict] = []
    pred_frames: list[pd.DataFrame] = []

    split_specs: list[tuple[int, int, pd.Series, pd.Series, dict[str, str]]] = []
    drug_values = df[drug_col].astype(str)
    context_values = df[context_col].astype(str)
    cv_scheme = 'exhaustive'

    if task == 'T1' and t1_scheme == 'predefined_fold':
        cv_scheme = 'predefined_fold'
        if 'fold' in df.columns and df['fold'].notna().any():
            fold_values = sorted(df['fold'].dropna().unique().tolist())
            for fold_idx, fold_value in enumerate(fold_values):
                test_mask = df['fold'] == fold_value
                train_mask = ~test_mask
                split_specs.append((
                    0,
                    fold_idx,
                    train_mask,
                    test_mask,
                    {'held_out_drugs': '', 'held_out_contexts': ''},
                ))
        else:
            n_t1_folds = int(n_folds) if n_folds is not None else 5
            for repeat in range(n_repeats):
                rng = np.random.default_rng(random_state + repeat)
                t1_fold_id = _t1_fold_ids(df, context_col, n_t1_folds, rng)
                for fold_idx in range(n_t1_folds):
                    test_mask = pd.Series(t1_fold_id == fold_idx, index=df.index)
                    train_mask = ~test_mask
                    split_specs.append((
                        repeat,
                        fold_idx,
                        train_mask,
                        test_mask,
                        {'held_out_drugs': '', 'held_out_contexts': ''},
                    ))
    elif task == 'T1':
        # Interpolation: hold out each observed (drug, context); train on the rest
        # so both the drug and the context remain seen via other pairs.
        fold_idx = 0
        for drug in drugs:
            for context in contexts:
                test_mask = (drug_values == drug) & (context_values == context)
                if not bool(test_mask.any()):
                    continue
                train_mask = ~test_mask
                split_specs.append((
                    0,
                    fold_idx,
                    train_mask,
                    test_mask,
                    {'held_out_drugs': drug, 'held_out_contexts': context},
                ))
                fold_idx += 1
    elif task == 'T2':
        # Unseen context (tissue / cell line), seen drugs.
        for fold_idx, context in enumerate(contexts):
            test_mask = context_values == context
            train_mask = ~test_mask
            split_specs.append((
                0,
                fold_idx,
                train_mask,
                test_mask,
                {'held_out_drugs': '', 'held_out_contexts': context},
            ))
    elif task == 'T3':
        # Unseen drug, seen contexts.
        for fold_idx, drug in enumerate(drugs):
            test_mask = drug_values == drug
            train_mask = ~test_mask
            split_specs.append((
                0,
                fold_idx,
                train_mask,
                test_mask,
                {'held_out_drugs': drug, 'held_out_contexts': ''},
            ))
    else:  # T4
        fold_idx = 0
        for drug in drugs:
            for context in contexts:
                test_mask = (drug_values == drug) & (context_values == context)
                train_mask = (drug_values != drug) & (context_values != context)
                split_specs.append((
                    0,
                    fold_idx,
                    train_mask,
                    test_mask,
                    {'held_out_drugs': drug, 'held_out_contexts': context},
                ))
                fold_idx += 1

    for repeat, fold_idx, train_mask, test_mask, held_out in split_specs:
        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())
        if n_test < 1 or n_train < 2:
            logger.info(
                'task=%s repeat=%d fold=%d skipped (n_train=%d n_test=%d)',
                task, repeat, fold_idx, n_train, n_test,
            )
            continue

        X_train = df.loc[train_mask]
        X_test = df.loc[test_mask]
        y_train = X_train['y']
        y_test = X_test['y'].astype(float).to_numpy()
        # Deduplicate: SciPlex context_col is cell_line, so do not select it twice.
        id_cols = list(dict.fromkeys(
            [c for c in ('cell_line', drug_col, context_col) if c in X_test.columns]
        ))
        identifiers = X_test[id_cols].reset_index(drop=True)
        if 'tissue' not in identifiers.columns and 'tissue' in X_test.columns:
            identifiers = identifiers.assign(tissue=X_test['tissue'].to_numpy())

        try:
            y_pred, _ = _fit_predict_fold(
                X_train,
                y_train,
                X_test,
                selected_features,
                head=head,
                threshold=threshold,
            )
        except Exception as exc:
            logger.warning(
                'task=%s repeat=%d fold=%d fit failed: %s', task, repeat, fold_idx, exc
            )
            continue

        pearson = _safe_pearson(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        score_rows.append({
            'task': task,
            'task_label': TASK_DEFINITIONS[task]['label'],
            'claim': TASK_DEFINITIONS[task]['claim'],
            'repeat': repeat,
            'fold': fold_idx,
            'n_train': n_train,
            'n_test': n_test,
            'pearson': pearson,
            'rmse': rmse,
            'head': head,
            'threshold': threshold if head == 'two_stage' else np.nan,
            'cv_scheme': cv_scheme,
            **held_out,
        })

        pred_frames.append(
            pd.concat(
                [
                    identifiers,
                    pd.DataFrame({'true': y_test, 'pred': y_pred}),
                ],
                axis=1,
            ).assign(
                task=task,
                repeat=repeat,
                fold=fold_idx,
                head=head,
                threshold=threshold if head == 'two_stage' else np.nan,
                cv_scheme=cv_scheme,
                **held_out,
            )
        )

        logger.info(
            'task=%s repeat=%d fold=%d head=%s threshold=%s pearson=%.3f rmse=%.3f n_train=%d n_test=%d held_drugs=%s held_ctx=%s',
            task,
            repeat,
            fold_idx,
            head,
            threshold if head == 'two_stage' else 'NA',
            pearson if not np.isnan(pearson) else -9,
            rmse,
            n_train,
            n_test,
            held_out['held_out_drugs'][:80],
            held_out['held_out_contexts'][:80],
        )

    scores = pd.DataFrame(score_rows)
    predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    return scores, predictions


def summarize_task_scores(scores: pd.DataFrame, ci: float = 0.95) -> pd.DataFrame:
    """Mean ± CI over fold-level scores (grouped by dataset/task/model/metric)."""
    if scores.empty:
        return scores

    group_cols = [
        c
        for c in ('dataset', 'task', 'task_label', 'claim', 'model', 'head', 'threshold')
        if c in scores.columns
    ]
    rows = []
    alpha = (1.0 - ci) / 2.0
    for keys, grp in scores.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_dict = dict(zip(group_cols, keys))
        for metric in ('pearson', 'rmse'):
            vals = grp[metric].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            lo, hi = np.quantile(vals, [alpha, 1.0 - alpha])
            rows.append({
                **key_dict,
                'metric': metric,
                'n_folds': len(vals),
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                'ci_low': float(lo),
                'ci_high': float(hi),
            })
    return pd.DataFrame(rows)


def _prepare_profile_frames(
    pre: pd.DataFrame,
    post: pd.DataFrame,
    lfc: pd.DataFrame,
    *,
    feature_selection_fn,
    n_features: int = 1000,
    include_smiles: bool = True,
    n_fp_bits: int = 2048,
    apply_cov: bool = False,
) -> tuple[dict[str, pd.DataFrame], list[str], list[str]]:
    from drug_fingerprints import attach_ecfp_features, gene_and_fingerprint_features
    from prediction_utils import add_y_and_normalize, filter_on_coefficient_of_variation

    keep = ['condition', 'tissue', 'cell_line', 'drug', 'fold']
    frames: dict[str, pd.DataFrame] = {}
    for label, raw in (
        ('pre_treatment', pre),
        ('post_treatment', post),
        ('LFC', lfc),
    ):
        df = (
            filter_on_coefficient_of_variation(raw, groupby=['condition'])
            if apply_cov
            else raw
        )
        frames[label] = add_y_and_normalize(df, 'sens', normalize=False, keep=keep)

    gene_features = feature_selection_fn(
        frames['pre_treatment'].drop(columns=['y'], errors='ignore'), n_features
    )
    fp_features = gene_and_fingerprint_features(gene_features, n_bits=n_fp_bits)

    if include_smiles:
        for base_key, smiles_key in (
            ('pre_treatment', 'pre_treatment_smiles'),
            ('post_treatment', 'post_treatment_smiles'),
            ('LFC', 'LFC_smiles'),
        ):
            frames[smiles_key] = attach_ecfp_features(
                frames[base_key], drug_col='drug', n_bits=n_fp_bits
            )

    return frames, gene_features, fp_features


def run_dataset_tasks(
    dataset: str,
    frames: dict[str, pd.DataFrame],
    gene_features: list[str],
    fp_features: list[str],
    context_col: str,
    tasks: Iterable[str] = ('T1', 'T2', 'T3', 'T4'),
    n_repeats: int = 5,
    random_state: int = 1,
    head: str = 'continuous',
    threshold: float = 0.02,
    model_keys: Iterable[str] | None = None,
    t1_schemes: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run T1–T4 for measured modalities (expression ± SMILES); return scores, preds, summary.

    T1 schemes default to ``predefined_fold`` (5-fold matching PRM splits). Pass
    ``t1_schemes=('exhaustive',)`` or ``('exhaustive', 'predefined_fold')`` to add
    leave-one-(drug, context) T1. T2–T4 always use exhaustive leave-one-group holdouts.
    """
    selected_keys = resolve_measured_model_keys(model_keys)
    if t1_schemes is None:
        t1_scheme_list: tuple[str, ...] = ('predefined_fold',)
    else:
        t1_scheme_list = tuple(t1_schemes)
        bad = [s for s in t1_scheme_list if s not in {'exhaustive', 'predefined_fold'}]
        if bad:
            raise ValueError(f'Unknown t1_schemes {bad}')
        if not t1_scheme_list:
            raise ValueError('t1_schemes must be non-empty')
    all_scores = []
    all_preds = []
    for task in tasks:
        schemes = t1_scheme_list if task == 'T1' else ('exhaustive',)
        for model_key in selected_keys:
            df = frames.get(model_key)
            if df is None:
                continue
            features = fp_features if model_key.endswith('_smiles') else gene_features
            for t1_scheme in schemes:
                scores, preds = run_task_cv(
                    df,
                    features,
                    task=task,
                    context_col=context_col,
                    n_repeats=n_repeats,
                    random_state=random_state,
                    head=head,
                    threshold=threshold,
                    t1_scheme=t1_scheme if task == 'T1' else 'exhaustive',
                )
                if scores.empty:
                    logger.warning(
                        '%s %s %s (%s): no folds completed',
                        dataset,
                        task,
                        model_key,
                        t1_scheme if task == 'T1' else 'exhaustive',
                    )
                    continue
                scores = scores.assign(
                    dataset=dataset,
                    model=MODEL_LABELS.get(model_key, model_key),
                    profile_source='measured',
                )
                preds = preds.assign(
                    dataset=dataset,
                    model=MODEL_LABELS.get(model_key, model_key),
                    profile_source='measured',
                )
                all_scores.append(scores)
                all_preds.append(preds)

    scores_df = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    preds_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    summary_df = summarize_task_scores(_scores_for_summary(scores_df))
    return scores_df, preds_df, summary_df


def _scores_for_summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Fold scores used for mean±CI summaries.

    Prefer exhaustive T1 when present so 5-fold does not dilute those means; if
    only predefined-fold T1 was run, keep it. Always drop retired no-effect rows.
    """
    if scores is None or scores.empty:
        return scores if scores is not None else pd.DataFrame()
    out = scores.copy()
    if 'profile_source' in out.columns:
        out = out[out['profile_source'].astype(str) != 'no_effect'].copy()
    if 'cv_scheme' not in out.columns or 'task' not in out.columns:
        return out
    is_t1 = out['task'].astype(str).eq('T1')
    is_5fold = out['cv_scheme'].astype(str).eq('predefined_fold')
    if bool((is_t1 & ~is_5fold).any()):
        return out.loc[~(is_t1 & is_5fold)].copy()
    return out


def _split_primary_and_t1_5fold(
    scores: pd.DataFrame,
    preds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separate predefined-fold T1 rows from primary T1–T4 exhaustive outputs."""
    if scores.empty or 'cv_scheme' not in scores.columns:
        return scores, preds, pd.DataFrame(), pd.DataFrame()

    is_5fold = scores['cv_scheme'].astype(str).eq('predefined_fold')
    primary_scores = scores.loc[~is_5fold].copy()
    t1_5fold_scores = scores.loc[is_5fold].copy()

    if preds.empty or 'cv_scheme' not in preds.columns:
        return primary_scores, preds, t1_5fold_scores, pd.DataFrame()

    is_pred_5fold = preds['cv_scheme'].astype(str).eq('predefined_fold')
    primary_preds = preds.loc[~is_pred_5fold].copy()
    t1_5fold_preds = preds.loc[is_pred_5fold].copy()
    return primary_scores, primary_preds, t1_5fold_scores, t1_5fold_preds


def _dedupe_frame_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate-named columns and pandas ``col.1`` renames of an existing ``col``."""
    if df is None or df.empty:
        return df
    out = df.loc[:, ~df.columns.duplicated()].copy() if df.columns.duplicated().any() else df.copy()
    drop: list[str] = []
    present = set(map(str, out.columns))
    for col in list(out.columns):
        name = str(col)
        if '.' in name and name.rsplit('.', 1)[-1].isdigit():
            base = name.rsplit('.', 1)[0]
            if base in present:
                drop.append(col)
    if drop:
        out = out.drop(columns=drop)
    return out


def _merge_task_frames(existing_path: str, new_df: pd.DataFrame) -> pd.DataFrame:
    """Replace rows for matching ``(task, model)`` keys; keep other rows from disk.

    Falls back to task-only replace when ``model`` is missing. Always drops
    legacy ``profile_source=no_effect`` rows from both sides.
    """
    if new_df is None or new_df.empty:
        if os.path.exists(existing_path):
            old = _dedupe_frame_columns(pd.read_csv(existing_path))
            if 'profile_source' in old.columns:
                old = old[old['profile_source'].astype(str) != 'no_effect'].copy()
            return old
        return pd.DataFrame() if new_df is None else new_df
    new_df = _dedupe_frame_columns(new_df)
    if 'profile_source' in new_df.columns:
        new_df = new_df[new_df['profile_source'].astype(str) != 'no_effect'].copy()
    if 'task' not in new_df.columns or not os.path.exists(existing_path):
        return new_df
    old = _dedupe_frame_columns(pd.read_csv(existing_path))
    if 'profile_source' in old.columns:
        old = old[old['profile_source'].astype(str) != 'no_effect'].copy()
    if old.empty or 'task' not in old.columns:
        return new_df

    key_cols = [c for c in ('task', 'model') if c in new_df.columns and c in old.columns]
    if not key_cols:
        return new_df
    keys = new_df[key_cols].drop_duplicates()
    merged = old.merge(keys.assign(_replace=1), on=key_cols, how='left')
    keep = merged[merged['_replace'].isna()].drop(columns=['_replace'])
    if keep.empty:
        return new_df
    cols = list(dict.fromkeys([*keep.columns.tolist(), *new_df.columns.tolist()]))
    return pd.concat(
        [keep.reindex(columns=cols), new_df.reindex(columns=cols)],
        ignore_index=True,
    )


def _write_outputs(
    results_dir: str,
    prefix: str,
    scores: pd.DataFrame,
    preds: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    output_shard: str = '',
) -> None:
    """Write fold scores / predictions.

    When ``output_shard`` is set (e.g. ``T1``), write only that task's shard files
    (``…_fold_scores__T1.csv``) so parallel Slurm jobs do not race on one CSV.
    """
    os.makedirs(results_dir, exist_ok=True)
    primary_scores, primary_preds, t1_5fold_scores, t1_5fold_preds = _split_primary_and_t1_5fold(
        scores, preds
    )
    primary_scores = _dedupe_frame_columns(primary_scores)
    primary_preds = _dedupe_frame_columns(primary_preds)
    t1_5fold_scores = _dedupe_frame_columns(t1_5fold_scores)
    t1_5fold_preds = _dedupe_frame_columns(t1_5fold_preds)

    shard = str(output_shard).strip().upper()
    shard_suffix = f'__{shard}' if shard else ''

    scores_path = os.path.join(
        results_dir, f'{prefix}_measured_tasks_fold_scores{shard_suffix}.csv'
    )
    preds_path = os.path.join(
        results_dir, f'{prefix}_measured_tasks_predictions{shard_suffix}.csv'
    )
    summary_path = os.path.join(results_dir, f'{prefix}_measured_tasks_summary.csv')

    if shard:
        primary_scores.to_csv(scores_path, index=False)
        primary_preds.to_csv(preds_path, index=False)
        logger.info('Wrote shard %s (%d fold scores)', scores_path, len(primary_scores))
        logger.info('Wrote shard %s (%d predictions)', preds_path, len(primary_preds))
        if not t1_5fold_scores.empty or not t1_5fold_preds.empty:
            t1_scores_path = os.path.join(
                results_dir,
                f'{prefix}_measured_tasks_t1_5fold_fold_scores{shard_suffix}.csv',
            )
            t1_preds_path = os.path.join(
                results_dir,
                f'{prefix}_measured_tasks_t1_5fold_predictions{shard_suffix}.csv',
            )
            t1_5fold_scores.to_csv(t1_scores_path, index=False)
            t1_5fold_preds.to_csv(t1_preds_path, index=False)
            logger.info('Wrote shard %s (%d T1 5-fold scores)', t1_scores_path, len(t1_5fold_scores))
            logger.info('Wrote shard %s (%d T1 5-fold predictions)', t1_preds_path, len(t1_5fold_preds))
        return

    primary_scores = _merge_task_frames(scores_path, primary_scores)
    primary_preds = _merge_task_frames(preds_path, primary_preds)
    summary_parts = [df for df in (primary_scores, t1_5fold_scores) if df is not None and not df.empty]
    if summary_parts:
        summary = summarize_task_scores(_scores_for_summary(pd.concat(summary_parts, ignore_index=True)))

    primary_scores.to_csv(scores_path, index=False)
    primary_preds.to_csv(preds_path, index=False)
    summary.to_csv(summary_path, index=False)
    logger.info('Wrote %s (%d fold scores)', scores_path, len(primary_scores))
    logger.info('Wrote %s (%d predictions)', preds_path, len(primary_preds))
    logger.info('Wrote %s (%d summary rows)', summary_path, len(summary))

    if not t1_5fold_scores.empty or not t1_5fold_preds.empty:
        t1_scores_path = os.path.join(results_dir, f'{prefix}_measured_tasks_t1_5fold_fold_scores.csv')
        t1_preds_path = os.path.join(results_dir, f'{prefix}_measured_tasks_t1_5fold_predictions.csv')
        t1_5fold_scores.to_csv(t1_scores_path, index=False)
        t1_5fold_preds.to_csv(t1_preds_path, index=False)
        logger.info('Wrote %s (%d T1 5-fold scores)', t1_scores_path, len(t1_5fold_scores))
        logger.info('Wrote %s (%d T1 5-fold predictions)', t1_preds_path, len(t1_5fold_preds))


def run_mcfarland_measured_tasks(
    results_dir: str,
    n_features: int = 1000,
    n_repeats: int = 5,
    random_state: int = 1,
    pseudobulk: str = 'mean',
    head: str = 'continuous',
    threshold: float = 0.02,
    output_tag: str = '',
    model_keys: Iterable[str] | None = None,
    tasks: Iterable[str] = ('T1', 'T2', 'T3', 'T4'),
    t1_schemes: Iterable[str] | None = None,
    output_shard: str = '',
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from prediction_utils import (
        feature_selection,
        get_McFarland_CPA_predictions,
        get_McFarland_count_data,
        get_McFarland_mean_data,
        get_McFarland_sensitivityinfo_for_profile_merge,
    )

    from drug_fingerprints import merge_drug_names_mcfarland

    getter = get_McFarland_count_data if pseudobulk == 'count' else get_McFarland_mean_data
    pre, post, lfc = getter()
    sens = get_McFarland_sensitivityinfo_for_profile_merge()
    pre = pd.merge(pre, sens, left_on=['cell_line', 'condition'], right_on=['cell_line', 'target'], how='left')
    post = pd.merge(post, sens, left_on=['cell_line', 'condition'], right_on=['cell_line', 'target'], how='left')
    lfc = pd.merge(lfc, sens, left_on=['cell_line', 'condition'], right_on=['cell_line', 'target'], how='left')
    pre = merge_drug_names_mcfarland(pre)
    post = merge_drug_names_mcfarland(post)
    lfc = merge_drug_names_mcfarland(lfc)
    fold_reference, _ = get_McFarland_CPA_predictions()
    pre = _attach_predefined_folds(pre, fold_reference)
    post = _attach_predefined_folds(post, fold_reference)
    lfc = _attach_predefined_folds(lfc, fold_reference)

    frames, gene_features, fp_features = _prepare_profile_frames(
        pre,
        post,
        lfc,
        feature_selection_fn=feature_selection,
        n_features=n_features,
        apply_cov=False,
    )
    scores, preds, summary = run_dataset_tasks(
        'McFarland',
        frames,
        gene_features,
        fp_features,
        context_col='tissue',
        tasks=tasks,
        n_repeats=n_repeats,
        random_state=random_state,
        head=head,
        threshold=threshold,
        model_keys=model_keys,
        t1_schemes=t1_schemes,
    )
    prefix = 'mcfarland_count' if pseudobulk == 'count' else 'mcfarland'
    if output_tag:
        prefix = f'{prefix}_{output_tag}'
    _write_outputs(results_dir, prefix, scores, preds, summary, output_shard=output_shard)
    return scores, preds, summary


def run_sciplex_measured_tasks(
    results_dir: str,
    n_features: int = 1000,
    n_repeats: int = 5,
    random_state: int = 1,
    pseudobulk: str = 'mean',
    head: str = 'continuous',
    threshold: float = 0.02,
    output_tag: str = '',
    model_keys: Iterable[str] | None = None,
    tasks: Iterable[str] = ('T1', 'T2', 'T3', 'T4'),
    t1_schemes: Iterable[str] | None = None,
    output_shard: str = '',
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from prediction_utils import (
        feature_selection,
        get_CPA_predictions,
        get_sciplex_AUCs,
        get_sciplex_count_data,
        get_sciplex_mean_data,
        get_sciplex_pair_fold_keys,
        merge_sciplex_sensitivity,
    )

    from drug_fingerprints import merge_drug_names_sciplex

    getter = get_sciplex_count_data if pseudobulk == 'count' else get_sciplex_mean_data
    pre, post, lfc = getter()
    sens = get_sciplex_AUCs().rename(columns={'y': 'sens'})
    pre = merge_sciplex_sensitivity(pre, sens, value_col='sens')
    post = merge_sciplex_sensitivity(post, sens, value_col='sens')
    lfc = merge_sciplex_sensitivity(lfc, sens, value_col='sens')
    pre = merge_drug_names_sciplex(pre)
    post = merge_drug_names_sciplex(post)
    lfc = merge_drug_names_sciplex(lfc)
    for df in (pre, post, lfc):
        if 'target' not in df.columns:
            df['target'] = df['condition']
        df['sens_label'] = (df['sens'] >= 0.2).astype(int)
        # SciPlex has no tissue ontology; T2/T4 hold out cell lines.
        if 'tissue' not in df.columns:
            df['tissue'] = df['cell_line']
    fold_reference = get_sciplex_pair_fold_keys()
    pre = _attach_predefined_folds(pre, fold_reference)
    post = _attach_predefined_folds(post, fold_reference)
    lfc = _attach_predefined_folds(lfc, fold_reference)

    n_sens = int(pre['sens'].notna().sum()) if 'sens' in pre.columns else 0
    if len(pre) == 0 or n_sens == 0:
        raise ValueError(
            f'SciPlex measured tasks: no sensitivity labels after merge '
            f'(profile rows={len(pre)}, sens matches={n_sens}). '
            'Check observed_pseudobulk condition keys vs resources/sciplex_sensitivity_info.csv.'
        )

    frames, gene_features, fp_features = _prepare_profile_frames(
        pre,
        post,
        lfc,
        feature_selection_fn=feature_selection,
        n_features=n_features,
        apply_cov=False,
    )
    scores, preds, summary = run_dataset_tasks(
        'SciPlex3',
        frames,
        gene_features,
        fp_features,
        context_col='cell_line',
        tasks=tasks,
        n_repeats=n_repeats,
        random_state=random_state,
        head=head,
        threshold=threshold,
        model_keys=model_keys,
        t1_schemes=t1_schemes,
    )
    prefix = 'sciplex_count' if pseudobulk == 'count' else 'sciplex'
    if output_tag:
        prefix = f'{prefix}_{output_tag}'
    _write_outputs(results_dir, prefix, scores, preds, summary, output_shard=output_shard)
    return scores, preds, summary


def _dataset_prefixes(
    dataset: str,
    *,
    pseudobulk: str = 'mean',
    output_tag: str = '',
) -> list[str]:
    dataset = dataset.lower()
    prefixes: list[str] = []
    if dataset in {'both', 'sciplex'}:
        prefixes.append('sciplex_count' if pseudobulk == 'count' else 'sciplex')
    if dataset in {'both', 'mcfarland'}:
        prefixes.append('mcfarland_count' if pseudobulk == 'count' else 'mcfarland')
    if output_tag:
        prefixes = [f'{p}_{output_tag}' for p in prefixes]
    return prefixes


def _concat_shard_files(paths: list[str]) -> pd.DataFrame:
    frames = [_dedupe_frame_columns(pd.read_csv(p)) for p in paths if os.path.exists(p)]
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    return _dedupe_frame_columns(pd.concat(frames, ignore_index=True))


def assemble_measured_task_shards(
    results_dir: str,
    *,
    pseudobulk: str = 'mean',
    output_tag: str = '',
    dataset: str = 'both',
    remove_shards: bool = True,
) -> None:
    """Merge per-task shard CSVs (``…__T1.csv`` …) into canonical dataset files."""
    kinds = (
        'fold_scores',
        'predictions',
        't1_5fold_fold_scores',
        't1_5fold_predictions',
    )
    for prefix in _dataset_prefixes(dataset, pseudobulk=pseudobulk, output_tag=output_tag):
        for kind in kinds:
            pattern = os.path.join(results_dir, f'{prefix}_measured_tasks_{kind}__T*.csv')
            shard_paths = sorted(glob.glob(pattern))
            if not shard_paths:
                continue
            assembled = _concat_shard_files(shard_paths)
            out_path = os.path.join(results_dir, f'{prefix}_measured_tasks_{kind}.csv')
            assembled.to_csv(out_path, index=False)
            logger.info(
                'Assembled %s from %d shards (%d rows)',
                out_path,
                len(shard_paths),
                len(assembled),
            )
            if remove_shards:
                for shard_path in shard_paths:
                    os.remove(shard_path)

        scores_path = os.path.join(results_dir, f'{prefix}_measured_tasks_fold_scores.csv')
        t1_path = os.path.join(results_dir, f'{prefix}_measured_tasks_t1_5fold_fold_scores.csv')
        parts = []
        if os.path.exists(scores_path):
            parts.append(pd.read_csv(scores_path))
        if os.path.exists(t1_path):
            parts.append(pd.read_csv(t1_path))
        if parts:
            summary = summarize_task_scores(_scores_for_summary(pd.concat(parts, ignore_index=True)))
            summary_path = os.path.join(results_dir, f'{prefix}_measured_tasks_summary.csv')
            summary.to_csv(summary_path, index=False)
            logger.info('Wrote %s (%d summary rows)', summary_path, len(summary))


def combine_measured_task_outputs(
    results_dir: str,
    *,
    pseudobulk: str = 'mean',
    output_tag: str = '',
    dataset: str = 'both',
    assemble_shards: bool = True,
) -> pd.DataFrame:
    """Merge on-disk SciPlex/McFarland fold scores into combined summary CSVs."""
    dataset = dataset.lower()
    if dataset not in {'both', 'sciplex', 'mcfarland'}:
        raise ValueError(f"dataset must be 'both', 'sciplex', or 'mcfarland', got {dataset}")

    if assemble_shards:
        assemble_measured_task_shards(
            results_dir,
            pseudobulk=pseudobulk,
            output_tag=output_tag,
            dataset=dataset,
        )

    tag = 'count_' if pseudobulk == 'count' else ''
    if output_tag:
        tag = f'{tag}{output_tag}_'

    disk_frames: list[pd.DataFrame] = []
    for prefix in _dataset_prefixes(dataset, pseudobulk=pseudobulk, output_tag=output_tag):
        for path in (
            os.path.join(results_dir, f'{prefix}_measured_tasks_fold_scores.csv'),
            os.path.join(results_dir, f'{prefix}_measured_tasks_t1_5fold_fold_scores.csv'),
        ):
            if os.path.exists(path):
                disk_frames.append(pd.read_csv(path))

    combined_scores = (
        pd.concat(disk_frames, ignore_index=True) if disk_frames else pd.DataFrame()
    )
    combined_scores = _scores_for_summary(combined_scores)
    combined_summary = summarize_task_scores(combined_scores)
    combined_scores.to_csv(
        os.path.join(results_dir, f'combined_{tag}measured_tasks_fold_scores.csv'), index=False
    )
    combined_summary.to_csv(
        os.path.join(results_dir, f'combined_{tag}measured_tasks_summary.csv'), index=False
    )
    logger.info(
        'combine_measured_task_outputs: wrote combined_%smeasured_tasks_* (%d fold rows, %d summary rows)',
        tag,
        len(combined_scores),
        len(combined_summary),
    )
    return combined_summary


def run_all_measured_tasks(
    results_dir: str,
    n_features: int = 1000,
    n_repeats: int = 5,
    random_state: int = 1,
    pseudobulk: str = 'mean',
    head: str = 'continuous',
    threshold: float = 0.02,
    output_tag: str = '',
    model_keys: Iterable[str] | None = None,
    tasks: Iterable[str] = ('T1', 'T2', 'T3', 'T4'),
    dataset: str = 'both',
    t1_schemes: Iterable[str] | None = None,
    output_shard: str = '',
    skip_combine: bool = False,
) -> pd.DataFrame:
    """Run SciPlex and/or McFarland T1–T4 and write a combined summary."""
    dataset = dataset.lower()
    if dataset not in {'both', 'sciplex', 'mcfarland'}:
        raise ValueError(f"dataset must be 'both', 'sciplex', or 'mcfarland', got {dataset}")

    if dataset in {'both', 'sciplex'}:
        run_sciplex_measured_tasks(
            results_dir,
            n_features=n_features,
            n_repeats=n_repeats,
            random_state=random_state,
            pseudobulk=pseudobulk,
            head=head,
            threshold=threshold,
            output_tag=output_tag,
            model_keys=model_keys,
            tasks=tasks,
            t1_schemes=t1_schemes,
            output_shard=output_shard,
        )
    if dataset in {'both', 'mcfarland'}:
        run_mcfarland_measured_tasks(
            results_dir,
            n_features=n_features,
            n_repeats=n_repeats,
            random_state=random_state,
            pseudobulk=pseudobulk,
            head=head,
            threshold=threshold,
            output_tag=output_tag,
            model_keys=model_keys,
            tasks=tasks,
            t1_schemes=t1_schemes,
            output_shard=output_shard,
        )

    if skip_combine or output_shard:
        return pd.DataFrame()

    return combine_measured_task_outputs(
        results_dir,
        pseudobulk=pseudobulk,
        output_tag=output_tag,
        dataset=dataset,
    )


def _threshold_tag(threshold: float) -> str:
    return str(threshold).replace('-', 'm').replace('.', 'p')


def run_all_measured_task_head_sweep(
    results_dir: str,
    thresholds: Iterable[float] = DEFAULT_TWO_STAGE_THRESHOLDS,
    n_features: int = 1000,
    n_repeats: int = 5,
    random_state: int = 1,
    pseudobulk: str = 'mean',
    model_keys: Iterable[str] | None = None,
    dataset: str = 'both',
) -> pd.DataFrame:
    """Run continuous ElasticNet and two-stage heads across thresholds."""
    dataset = dataset.lower()
    if dataset not in {'both', 'sciplex', 'mcfarland'}:
        raise ValueError(f"dataset must be 'both', 'sciplex', or 'mcfarland', got {dataset}")

    all_scores: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []

    continuous_frames: list[pd.DataFrame] = []
    continuous_summaries: list[pd.DataFrame] = []
    if dataset in {'both', 'sciplex'}:
        sci_scores, _, sci_summary = run_sciplex_measured_tasks(
            results_dir,
            n_features=n_features,
            n_repeats=n_repeats,
            random_state=random_state,
            pseudobulk=pseudobulk,
            head='continuous',
            model_keys=model_keys,
        )
        continuous_frames.append(sci_scores)
        continuous_summaries.append(sci_summary)
    if dataset in {'both', 'mcfarland'}:
        mcf_scores, _, mcf_summary = run_mcfarland_measured_tasks(
            results_dir,
            n_features=n_features,
            n_repeats=n_repeats,
            random_state=random_state,
            pseudobulk=pseudobulk,
            head='continuous',
            model_keys=model_keys,
        )
        continuous_frames.append(mcf_scores)
        continuous_summaries.append(mcf_summary)
    continuous_scores = pd.concat(continuous_frames, ignore_index=True)
    continuous_summary = pd.concat(continuous_summaries, ignore_index=True)
    tag = 'count_' if pseudobulk == 'count' else ''
    continuous_scores.to_csv(
        os.path.join(results_dir, f'combined_{tag}measured_tasks_fold_scores.csv'),
        index=False,
    )
    continuous_summary.to_csv(
        os.path.join(results_dir, f'combined_{tag}measured_tasks_summary.csv'),
        index=False,
    )
    all_scores.append(continuous_scores)
    all_summaries.append(continuous_summary)

    for threshold in thresholds:
        output_tag = f'two_stage_t{_threshold_tag(float(threshold))}'
        stage_frames: list[pd.DataFrame] = []
        stage_summaries: list[pd.DataFrame] = []
        if dataset in {'both', 'sciplex'}:
            sci_scores, _, sci_summary = run_sciplex_measured_tasks(
                results_dir,
                n_features=n_features,
                n_repeats=n_repeats,
                random_state=random_state,
                pseudobulk=pseudobulk,
                head='two_stage',
                threshold=float(threshold),
                output_tag=output_tag,
                model_keys=model_keys,
            )
            stage_frames.append(sci_scores)
            stage_summaries.append(sci_summary)
        if dataset in {'both', 'mcfarland'}:
            mcf_scores, _, mcf_summary = run_mcfarland_measured_tasks(
                results_dir,
                n_features=n_features,
                n_repeats=n_repeats,
                random_state=random_state,
                pseudobulk=pseudobulk,
                head='two_stage',
                threshold=float(threshold),
                output_tag=output_tag,
                model_keys=model_keys,
            )
            stage_frames.append(mcf_scores)
            stage_summaries.append(mcf_summary)
        all_scores.append(pd.concat(stage_frames, ignore_index=True))
        all_summaries.append(pd.concat(stage_summaries, ignore_index=True))

    comparison_scores = pd.concat(all_scores, ignore_index=True)
    comparison_summary = pd.concat(all_summaries, ignore_index=True)
    comparison_scores.to_csv(
        os.path.join(results_dir, f'combined_{tag}measured_tasks_head_comparison_fold_scores.csv'),
        index=False,
    )
    comparison_summary.to_csv(
        os.path.join(results_dir, f'combined_{tag}measured_tasks_head_comparison_summary.csv'),
        index=False,
    )
    return comparison_summary
