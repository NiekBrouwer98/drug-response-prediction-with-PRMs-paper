import numpy as np
import pandas as pd
import os
import pickle
import sys
from pathlib import Path

from sklearn.linear_model import ElasticNet, LogisticRegression, LogisticRegressionCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

# Add project root to path for imports (must be before importing prediction_utils)
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config, setup_project
from prediction_utils import *
from prediction_utils import _normalize_mcfarland_profile_columns
from utils import setup_logging_for_script, log_script_start, log_script_end, ensure_directories_exist

# Setup project and logging
setup_project()
logger = setup_logging_for_script(__file__)

# Set project directories using configuration
path = os.getcwd()
home_dir = str(config.PROJECT_ROOT)
data_dir = str(config.DATA_DIR)
resources_dir = str(config.RESOURCES_DIR)
results_dir = str(config.RESULTS_04_DIR)
figures_dir = str(config.FIGURES_04_DIR)

# Ensure directories exist
ensure_directories_exist(results_dir, figures_dir, data_dir, resources_dir)

TWOPART_DEFAULT_THRESHOLD = 0.02
TWOPART_SWEEP_THRESHOLDS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)
ELASTICNET_PARAM_GRID = {
    'elasticnet__alpha': [0.1, 1.0, 10.0],
    'elasticnet__l1_ratio': [0.0, 0.1, 0.5, 1.0],
}


def _fit_continuous_elasticnet(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    cv_folds: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Grid-search ElasticNet on all training samples. Returns (pred, coef)."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('elasticnet', ElasticNet(max_iter=1000)),
    ])
    grid_search = GridSearchCV(pipeline, ELASTICNET_PARAM_GRID, cv=cv_folds, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    coef = grid_search.best_estimator_.named_steps['elasticnet'].coef_
    return grid_search.predict(X_test), np.asarray(coef)


def _fit_two_part(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    cv_folds: int,
    threshold: float = TWOPART_DEFAULT_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hurdle model: classify y > threshold, then ElasticNet on the positive class.

    Returns (combined_pred, sensitive_probability, regression_coef).
    """
    y_bin = (y_train > threshold).astype(int)
    if int(y_bin.nunique()) < 2:
        pred, coef = _fit_continuous_elasticnet(X_train, y_train, X_test, cv_folds)
        proba = np.full(len(X_test), float(y_bin.iloc[0]) if len(y_bin) else 0.0)
        return pred, proba, coef

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000)),
    ])
    try:
        clf.fit(X_train, y_bin)
        proba = clf.predict_proba(X_test)[:, 1]
        y_bin_pred = clf.predict(X_test)
    except Exception:
        pred, coef = _fit_continuous_elasticnet(X_train, y_train, X_test, cv_folds)
        proba = np.full(len(X_test), float(y_bin.mean()))
        return pred, proba, coef

    mask = y_train > threshold
    n_pos = int(mask.sum())
    inner_cv = min(cv_folds, n_pos)
    if inner_cv < 2:
        pred, coef = _fit_continuous_elasticnet(X_train, y_train, X_test, cv_folds)
        return np.where(y_bin_pred == 1, pred, 0.0), proba, coef

    try:
        pred, coef = _fit_continuous_elasticnet(
            X_train.loc[mask], y_train.loc[mask], X_test, inner_cv
        )
    except Exception:
        pred, coef = _fit_continuous_elasticnet(X_train, y_train, X_test, cv_folds)
    combined = np.where(y_bin_pred == 1, pred, 0.0)
    return combined, proba, coef


def _mcfarland_predefined_fold_cv(
    df_with_y: pd.DataFrame,
    selected_features: list[str],
    dedupe_train: bool = False,
    test_profiles_df: pd.DataFrame | None = None,
    two_part: bool = False,
    threshold: float = TWOPART_DEFAULT_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    ElasticNet regression with train/test splits defined by the ``fold`` column.

    For each test fold, rows with ``fold == test_fold`` are held out; all other folds
    are used for training. When ``dedupe_train`` is True, training rows are deduplicated
    on (cell_line, condition) so identical observed profiles are not overweighted.

    If ``two_part`` is True, use the hurdle model (classify y > ``threshold``, then
    regress on the positive class). Otherwise fit a single continuous ElasticNet.

    If ``test_profiles_df`` is set, train on ``df_with_y`` and evaluate on
    ``test_profiles_df`` for the held-out fold (optional; default is train and test on
    the same profile table).
    """
    if 'fold' not in df_with_y.columns:
        raise ValueError("df_with_y must contain a 'fold' column for predefined-fold CV")
    if test_profiles_df is not None and 'fold' not in test_profiles_df.columns:
        raise ValueError("test_profiles_df must contain a 'fold' column")

    _meta_cols = [c for c in ('y', 'tissue', 'cell_line', 'condition', 'fold') if c in df_with_y.columns]
    df_with_y = df_with_y.reset_index(drop=True).copy()
    if _meta_cols:
        df_with_y = df_with_y.dropna(axis=0, subset=_meta_cols)

    if test_profiles_df is not None:
        test_profiles_df = test_profiles_df.reset_index(drop=True).copy()
        test_meta = [c for c in _meta_cols if c in test_profiles_df.columns]
        if test_meta:
            test_profiles_df = test_profiles_df.dropna(axis=0, subset=test_meta)

    selected_features = list(set(selected_features).intersection(set(df_with_y.columns)))
    if test_profiles_df is not None:
        selected_features = list(set(selected_features).intersection(set(test_profiles_df.columns)))
    if len(selected_features) == 0:
        logger.error("_mcfarland_predefined_fold_cv: no selected features intersect dataframe columns")
        return (
            pd.DataFrame(columns=['accuracy', 'split']),
            pd.DataFrame(columns=['tissue', 'cell_line', 'condition', 'pred', 'true', 'split']),
            {},
        )

    test_folds = sorted(df_with_y['fold'].dropna().unique())

    outer_scores = []
    outcomes = []
    feature_importance_dict = {}

    fold_key_cols = ['cell_line', 'condition', 'fold']

    for test_fold in test_folds:
        train_df = df_with_y.loc[df_with_y['fold'] != test_fold].copy()
        if test_profiles_df is None:
            test_df = df_with_y.loc[df_with_y['fold'] == test_fold].copy()
        else:
            key_cols = [c for c in fold_key_cols if c in df_with_y.columns and c in test_profiles_df.columns]
            observed_test_keys = df_with_y.loc[df_with_y['fold'] == test_fold, key_cols].drop_duplicates()
            test_df = test_profiles_df.loc[test_profiles_df['fold'] == test_fold].copy()
            if key_cols:
                test_df = test_df.merge(observed_test_keys, on=key_cols, how='inner')

        if dedupe_train:
            train_df = train_df.drop_duplicates(subset=['cell_line', 'condition'], keep='first')

        if len(train_df) < 2 or len(test_df) < 1:
            logger.info(
                "_mcfarland_predefined_fold_cv skip fold=%s: n_train=%d n_test=%d",
                test_fold,
                len(train_df),
                len(test_df),
            )
            continue

        y_train = train_df['y'].astype(float)
        y_test = test_df['y'].astype(float).reset_index(drop=True)
        X_train = train_df.drop(columns=['y'])
        X_test = test_df.drop(columns=['y'])

        identifiers = X_test[['tissue', 'cell_line', 'condition']].reset_index(drop=True)
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        X_train = X_train.drop(columns=['tissue', 'cell_line', 'condition', 'fold'], errors='ignore').astype(float)
        X_test = X_test.drop(columns=['tissue', 'cell_line', 'condition', 'fold'], errors='ignore').astype(float)
        X_train = X_train.fillna(0.0)
        X_test = X_test.fillna(0.0)

        cv_folds = min(5, len(X_train))
        if cv_folds < 2:
            logger.info("_mcfarland_predefined_fold_cv skip fold=%s: inner cv_folds=%d", test_fold, cv_folds)
            continue

        try:
            if two_part:
                y_pred, y_score, coef = _fit_two_part(
                    X_train, y_train, X_test, cv_folds, threshold=threshold
                )
            else:
                y_pred, coef = _fit_continuous_elasticnet(X_train, y_train, X_test, cv_folds)
                y_score = y_pred
        except Exception as exc:
            logger.exception(
                "_mcfarland_predefined_fold_cv: fit failed test_fold=%s two_part=%s: %s",
                test_fold,
                two_part,
                exc,
            )
            continue

        outer_acc = np.sqrt(mean_squared_error(y_test, y_pred))

        fold_name = f'fold={test_fold}'
        feature_importance_dict[fold_name] = pd.Series(
            index=X_train.columns, data=coef, name=fold_name
        )

        split_outcomes = pd.concat(
            [identifiers, pd.DataFrame({'pred': y_pred, 'score': y_score, 'true': y_test})],
            axis=1,
        )
        outcomes.append(split_outcomes.assign(split=test_fold))
        outer_scores.append(pd.DataFrame({'accuracy': [outer_acc], 'split': [test_fold]}))
        logger.info(
            "_mcfarland_predefined_fold_cv ok: test_fold=%s two_part=%s threshold=%s RMSE=%.4f n_train=%d n_test=%d",
            test_fold,
            two_part,
            threshold if two_part else 'NA',
            outer_acc,
            len(X_train),
            len(X_test),
        )

    if outer_scores:
        outer_scores = pd.concat(outer_scores, ignore_index=True)
    else:
        outer_scores = pd.DataFrame(columns=['accuracy', 'split'])

    if outcomes:
        outcomes = pd.concat(outcomes, ignore_index=True)
    else:
        outcomes = pd.DataFrame(columns=['tissue', 'cell_line', 'condition', 'pred', 'score', 'true', 'split'])

    return outer_scores, outcomes, feature_importance_dict


NEW_PREDICTED_MODELS = ('CPA', 'chemCPA', 'PRnet')
# McFarland genetic PRMs keep native fold labels. SciPlex genetic PRMs have no
# reliable native folds in the mean tables — they use chemical (CPA / SciPlex) keys.
NATIVE_FOLD_PREDICTED_MODELS = frozenset({'GEARS', 'scFoundation'})

_MCFARLAND_PREDICTED_LOADERS = {
    'CPA': get_McFarland_CPA_predictions,
    'chemCPA': get_McFarland_chemCPA_predictions,
    'PRnet': get_McFarland_PRnet_predictions,
    'GEARS': get_McFarland_GEARS_predictions,
    'scFoundation': get_McFarland_scFoundation_predictions,
}
_SCIPLEX_PREDICTED_LOADERS = {
    'CPA': get_CPA_predictions,
    'chemCPA': get_chemCPA_predictions,
    'PRnet': get_PRnet_predictions,
    'GEARS': get_GEARS_predictions,
    'scFoundation': get_scfoundation_predictions,
}


def _zero_lfc_from_post(post_df: pd.DataFrame) -> pd.DataFrame:
    lfc = post_df.copy()
    gene_cols = [c for c in lfc.columns if c not in PROFILE_META_COLS]
    lfc[gene_cols] = 0.0
    return lfc


def _prepare_fold_keys(df: pd.DataFrame) -> pd.DataFrame:
    if 'fold' not in df.columns:
        raise ValueError("Predicted profiles must contain a fold/split column")
    return df[['cell_line', 'condition', 'fold']].drop_duplicates()


def predict_with_split_profiles_CV(
    dataset: str,
    n_features: int = 1000,
    results_prefix: str | None = None,
    include_models: tuple[str, ...] = NEW_PREDICTED_MODELS,
    two_part: bool = False,
    threshold: float = TWOPART_DEFAULT_THRESHOLD,
    return_configs: bool = False,
    include_predicted_smiles: bool = True,
    include_observed: bool = True,
    smiles_only: bool = False,
    train_on_measured_test_on_predicted: bool = False,
) -> None | tuple:
    """
    Drug-response CV using perturbation-model ``split``/``fold`` labels.

    Train/test folds follow the predicted-profile split column. Observed, average-effect,
    and no-effect profiles are expanded to the same (cell_line, condition, fold) keys
    as the first available predicted model (CPA when included). By default each profile
    source is trained and tested on itself only.

    When ``train_on_measured_test_on_predicted`` is True, ElasticNet is trained on
    Measured Post/LFC (±SMILES) for non-held-out folds and evaluated on predicted
    (PRM / baseline) profiles for the held-out fold. Measured→Measured heads are
    also written (same as ``include_observed``) so plots can use in-file Measured
    without attaching a separate split-CV run. For McFarland GEARS / scFoundation,
    Measured train rows are expanded onto **that model's native fold keys**;
    SciPlex genetic transfer uses the chemical Measured folds.

    Chemical PRMs (CPA / chemCPA / PRnet) share CPA / SciPlex official fold keys.
    McFarland GEARS / scFoundation keep their own predefined ``fold`` labels.
    SciPlex GEARS / scFoundation (no native folds in the mean tables) are expanded
    onto the chemical SciPlex fold keys. In transfer mode, Measured train for
    McFarland genetic PRMs is re-expanded onto that model's native folds; SciPlex
    genetic transfer uses the chemical Measured folds.

    When ``include_predicted_smiles`` is True, each predicted Post/LFC profile — and the
    average-effect / no-effect baselines — is also evaluated with Morgan (ECFP) drug
    fingerprints concatenated to the gene features.

    ``include_observed=False`` skips Measured Pre/Post/LFC (train/test on predicted
    profiles and baselines only). Forced True when ``train_on_measured_test_on_predicted``
    is True (Measured train features are required). ``smiles_only=True`` keeps only
    +SMILES heads.
    """
    dataset = dataset.lower()
    if dataset not in {'mcfarland', 'sciplex'}:
        raise ValueError(f"dataset must be 'mcfarland' or 'sciplex', got {dataset}")
    include_models = tuple(include_models)
    if train_on_measured_test_on_predicted:
        include_observed = True
    if results_prefix is None:
        if train_on_measured_test_on_predicted:
            results_prefix = f'{dataset}_split_cv_train_measured_test_predicted'
        else:
            results_prefix = f'{dataset}_split_cv'

    if dataset == 'mcfarland':
        mean_observed_pre, mean_observed_post, mean_observed_lfc = get_McFarland_mean_data()
        mean_avg_post, mean_avg_lfc = get_McFarland_average_effect_predictions()
        sensitivity_info = get_McFarland_sensitivityinfo_for_profile_merge()

        def _merge_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
            return pd.merge(
                df,
                sensitivity_info,
                left_on=['cell_line', 'condition'],
                right_on=['cell_line', 'target'],
                how='left',
            )

        loaders = _MCFARLAND_PREDICTED_LOADERS
    else:
        mean_observed_pre, mean_observed_post, mean_observed_lfc = get_sciplex_mean_data()
        mean_avg_post, mean_avg_lfc = get_average_effect_predictions()
        sensitivity_info = get_sciplex_AUCs().rename(columns={'y': 'sens'})

        def _merge_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
            out = merge_sciplex_sensitivity(df, sensitivity_info, value_col='sens')
            out['target'] = out['condition']
            out['sens_label'] = (out['sens'] >= 0.2).astype(int)
            if 'tissue' not in out.columns:
                out['tissue'] = out['cell_line']
            return out

        loaders = _SCIPLEX_PREDICTED_LOADERS
        for _df in (mean_observed_pre, mean_observed_post, mean_observed_lfc, mean_avg_post, mean_avg_lfc):
            if 'tissue' not in _df.columns:
                _df['tissue'] = _df['cell_line']

    # Keep all drugs/conditions with sensitivity labels (no CoV filtering).
    apply_cov = False

    from drug_fingerprints import (
        attach_ecfp_features,
        gene_and_fingerprint_features,
        merge_drug_names_mcfarland,
        merge_drug_names_sciplex,
    )

    merge_drug = merge_drug_names_mcfarland if dataset == 'mcfarland' else merge_drug_names_sciplex
    mean_observed_pre = merge_drug(mean_observed_pre)
    mean_observed_post = merge_drug(mean_observed_post)
    mean_observed_lfc = merge_drug(mean_observed_lfc)

    predicted_tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for model_name in include_models:
        if model_name not in loaders:
            raise KeyError(f"Unknown predicted model '{model_name}' for {dataset}")
        post_df, lfc_df = loaders[model_name]()
        predicted_tables[model_name] = (post_df.copy(), lfc_df.copy())

    if not predicted_tables:
        raise ValueError("include_models must contain at least one predicted-profile source")

    frames_to_merge = {
        'observed_pre': mean_observed_pre,
        'observed_post': mean_observed_post,
        'observed_lfc': mean_observed_lfc,
        'avg_post': mean_avg_post,
        'avg_lfc': mean_avg_lfc,
    }
    for model_name, (post_df, lfc_df) in predicted_tables.items():
        frames_to_merge[f'{model_name}_post'] = post_df
        frames_to_merge[f'{model_name}_lfc'] = lfc_df

    for key, frame in list(frames_to_merge.items()):
        frame = _merge_sensitivity(frame)
        if apply_cov:
            frame = filter_on_coefficient_of_variation(frame, groupby=['condition'])
        frames_to_merge[key] = frame

    fold_reference_name = next(
        (
            m
            for m in ('CPA',) + include_models
            if f'{m}_post' in frames_to_merge and 'fold' in frames_to_merge[f'{m}_post'].columns
        ),
        None,
    )
    if fold_reference_name is None:
        # SciPlex GEARS/scFoundation-only runs still need CPA fold keys for expansion.
        cpa_loader = loaders.get('CPA')
        if cpa_loader is None:
            raise ValueError(
                'No fold-labeled predicted profiles available to define CV splits '
                f'(models={list(include_models)})'
            )
        cpa_post, _ = cpa_loader()
        cpa_post = _merge_sensitivity(cpa_post)
        if apply_cov:
            cpa_post = filter_on_coefficient_of_variation(cpa_post, groupby=['condition'])
        frames_to_merge['CPA_post'] = cpa_post
        fold_reference_name = 'CPA'
        logger.info(
            'predict_with_split_profiles_CV [%s]: using CPA folds as reference '
            '(not included in --models)',
            dataset,
        )

    fold_reference_post = frames_to_merge[f'{fold_reference_name}_post']
    # Measured + baselines: SciPlex uses official split IDs (~188×3); McFarland keeps CPA folds.
    observed_fold_reference = (
        get_sciplex_pair_fold_keys() if dataset == 'sciplex' else fold_reference_post
    )
    mean_observed_pre = expand_mcfarland_profiles_with_folds(
        frames_to_merge['observed_pre'], observed_fold_reference
    )
    mean_observed_post = expand_mcfarland_profiles_with_folds(
        frames_to_merge['observed_post'], observed_fold_reference
    )
    mean_observed_lfc = expand_mcfarland_profiles_with_folds(
        frames_to_merge['observed_lfc'], observed_fold_reference
    )
    # No effect Post := Measured Pre (same basal features, folds, tissues, y).
    mean_no_effect_post = mean_observed_pre.copy()
    logger.info(
        'predict_with_split_profiles_CV [%s]: No effect Post uses Measured Pre '
        '(%d rows) by construction',
        dataset,
        len(mean_no_effect_post),
    )
    mean_avg_post = expand_mcfarland_profiles_with_folds(
        frames_to_merge['avg_post'], observed_fold_reference
    )
    mean_avg_lfc = expand_mcfarland_profiles_with_folds(
        frames_to_merge['avg_lfc'], observed_fold_reference
    )

    # Shared held-out fold keys for chemical PRMs (+ SciPlex genetic). Do NOT use
    # chemCPA self-keys: chemCPA exports every (cell_line, condition) under all 5
    # split IDs, so self-keys would keep train identities in every CV fold.
    # McFarland GEARS / scFoundation keep native folds instead.
    shared_fold_keys = (
        observed_fold_reference
        if dataset == 'sciplex'
        else _prepare_fold_keys(fold_reference_post)
    )

    for model_name in include_models:
        post_df = frames_to_merge[f'{model_name}_post']
        lfc_df = frames_to_merge[f'{model_name}_lfc']
        use_native_folds = (
            dataset == 'mcfarland' and model_name in NATIVE_FOLD_PREDICTED_MODELS
        )
        if use_native_folds:
            if 'fold' not in post_df.columns or 'fold' not in lfc_df.columns:
                raise ValueError(
                    f'McFarland {model_name} predictions must include a native '
                    f"'fold' column (got post/LFC without fold)."
                )
            post_df = _normalize_mcfarland_profile_columns(post_df)
            lfc_df = _normalize_mcfarland_profile_columns(lfc_df)
            logger.info(
                'predict_with_split_profiles_CV [%s]: %s using native folds=%s '
                '(n_post=%d n_lfc=%d)',
                dataset,
                model_name,
                sorted(post_df['fold'].dropna().unique().tolist()),
                len(post_df),
                len(lfc_df),
            )
            predicted_tables[model_name] = (post_df, lfc_df)
        else:
            # Chemical PRMs, and SciPlex genetic: expand/align to chemical fold keys.
            if 'fold' not in post_df.columns:
                logger.info(
                    'predict_with_split_profiles_CV [%s]: expanding %s onto fold '
                    'reference (%s)%s',
                    dataset,
                    model_name,
                    fold_reference_name,
                    (
                        ' [SciPlex genetic → chemical splits]'
                        if model_name in NATIVE_FOLD_PREDICTED_MODELS
                        else ''
                    ),
                )
                post_df = expand_mcfarland_profiles_with_folds(post_df, fold_reference_post)
                lfc_df = expand_mcfarland_profiles_with_folds(lfc_df, fold_reference_post)
            predicted_tables[model_name] = (
                align_mcfarland_profiles_to_fold_keys(post_df, shared_fold_keys),
                align_mcfarland_profiles_to_fold_keys(lfc_df, shared_fold_keys),
            )
        log_mcfarland_profile_expression_difference(
            mean_observed_post, predicted_tables[model_name][0], f'{model_name} post-treatment'
        )
        log_mcfarland_profile_expression_difference(
            mean_observed_lfc, predicted_tables[model_name][1], f'{model_name} LFC'
        )

    mean_no_effect_lfc = _zero_lfc_from_post(mean_no_effect_post)

    feature_to_keep = ['condition', 'tissue', 'cell_line', 'fold']

    def _with_y(df: pd.DataFrame) -> pd.DataFrame:
        return add_y_and_normalize(df, 'sens', normalize=False, keep=feature_to_keep)

    observed_pre_y = _with_y(mean_observed_pre)
    observed_post_y = _with_y(mean_observed_post)
    observed_lfc_y = _with_y(mean_observed_lfc)
    # Same frame object path as Measured Pre so ElasticNet sees identical X/y/folds.
    no_effect_post_y = observed_pre_y
    no_effect_lfc_y = _with_y(mean_no_effect_lfc)
    avg_post_y = _with_y(mean_avg_post)
    avg_lfc_y = _with_y(mean_avg_lfc)

    features = feature_selection(
        observed_pre_y.drop(columns=['y', 'fold'], errors='ignore'),
        n_features=n_features,
    )
    fp_features = gene_and_fingerprint_features(features)
    observed_tables = {
        'pre_treatment': mean_observed_pre,
        'post_treatment': mean_observed_post,
        'LFC': mean_observed_lfc,
    }
    observed_smiles_y: dict[str, pd.DataFrame] = {}
    for base_key, smiles_key in (
        ('pre_treatment', 'pre_treatment_smiles'),
        ('post_treatment', 'post_treatment_smiles'),
        ('LFC', 'LFC_smiles'),
    ):
        frame = observed_tables[base_key].copy()
        if 'drug' not in frame.columns:
            frame = merge_drug(frame)
        observed_smiles_y[smiles_key] = _with_y(attach_ecfp_features(frame, drug_col='drug'))
    logger.info(
        "predict_with_split_profiles_CV [%s]: using %d gene features, %d fp features",
        dataset,
        len(features),
        len(fp_features),
    )

    if smiles_only and not include_predicted_smiles:
        raise ValueError('smiles_only=True requires include_predicted_smiles=True')

    use_gene = not smiles_only
    use_smiles = include_predicted_smiles or smiles_only

    # train_df, optional test_df (None => train and test on train_df).
    profile_configs: list[
        tuple[str, str, pd.DataFrame, bool, list[str], pd.DataFrame | None]
    ] = []

    def _append_config(
        source: str,
        label: str,
        train_df: pd.DataFrame,
        dedupe: bool,
        feats: list[str],
        test_df: pd.DataFrame | None = None,
    ) -> None:
        profile_configs.append((source, label, train_df, dedupe, feats, test_df))

    if train_on_measured_test_on_predicted:
        logger.info(
            'predict_with_split_profiles_CV [%s]: transfer mode — '
            'train Measured Post/LFC (±SMILES), test on predicted profiles; '
            'also write Measured→Measured heads',
            dataset,
        )
        # Measured train/test on itself (for plot baseline panels).
        if use_gene:
            _append_config('observed', 'pre_treatment', observed_pre_y, True, features)
            _append_config('observed', 'post_treatment', observed_post_y, True, features)
            _append_config('observed', 'LFC', observed_lfc_y, True, features)
        if use_smiles:
            _append_config(
                'observed',
                'pre_treatment_smiles',
                observed_smiles_y['pre_treatment_smiles'],
                True,
                fp_features,
            )
            _append_config(
                'observed',
                'post_treatment_smiles',
                observed_smiles_y['post_treatment_smiles'],
                True,
                fp_features,
            )
            _append_config(
                'observed',
                'LFC_smiles',
                observed_smiles_y['LFC_smiles'],
                True,
                fp_features,
            )
        measured_train = {
            'post_treatment': observed_post_y,
            'LFC': observed_lfc_y,
            'post_treatment_smiles': observed_smiles_y['post_treatment_smiles'],
            'LFC_smiles': observed_smiles_y['LFC_smiles'],
        }

        def _measured_train_for_fold_reference(
            fold_reference: pd.DataFrame,
        ) -> dict[str, pd.DataFrame]:
            """Expand Measured Post/LFC (±SMILES) onto ``fold_reference`` keys.

            Used so train-Measured → test-genetic uses that PRM's native folds,
            not the chemical CPA / SciPlex keys.
            """
            post = expand_mcfarland_profiles_with_folds(
                frames_to_merge['observed_post'], fold_reference
            )
            lfc = expand_mcfarland_profiles_with_folds(
                frames_to_merge['observed_lfc'], fold_reference
            )
            post_y = _with_y(post)
            lfc_y = _with_y(lfc)
            out = {'post_treatment': post_y, 'LFC': lfc_y}
            if use_smiles:
                post_frame = post.copy()
                lfc_frame = lfc.copy()
                if 'drug' not in post_frame.columns:
                    post_frame = merge_drug(post_frame)
                if 'drug' not in lfc_frame.columns:
                    lfc_frame = merge_drug(lfc_frame)
                out['post_treatment_smiles'] = _with_y(
                    attach_ecfp_features(post_frame, drug_col='drug')
                )
                out['LFC_smiles'] = _with_y(
                    attach_ecfp_features(lfc_frame, drug_col='drug')
                )
            return out

        for model_name in include_models:
            post_df = predicted_tables[model_name][0].copy()
            lfc_df = predicted_tables[model_name][1].copy()
            if 'drug' not in post_df.columns:
                post_df = merge_drug(post_df)
            if 'drug' not in lfc_df.columns:
                lfc_df = merge_drug(lfc_df)
            post_y = _with_y(post_df)
            lfc_y = _with_y(lfc_df)
            source = f'{model_name}_predicted'
            if (
                dataset == 'mcfarland'
                and model_name in NATIVE_FOLD_PREDICTED_MODELS
            ):
                train_pack = _measured_train_for_fold_reference(post_df)
                logger.info(
                    'predict_with_split_profiles_CV [%s]: Measured train for %s '
                    'uses native folds (n_post_train=%d)',
                    dataset,
                    model_name,
                    len(train_pack['post_treatment']),
                )
            else:
                train_pack = measured_train
            if use_gene:
                _append_config(
                    source,
                    'post_treatment',
                    train_pack['post_treatment'],
                    True,
                    features,
                    post_y,
                )
                _append_config(
                    source, 'LFC', train_pack['LFC'], True, features, lfc_y
                )
            if use_smiles:
                post_smiles_y = _with_y(attach_ecfp_features(post_df, drug_col='drug'))
                lfc_smiles_y = _with_y(attach_ecfp_features(lfc_df, drug_col='drug'))
                _append_config(
                    source,
                    'post_treatment_smiles',
                    train_pack['post_treatment_smiles'],
                    True,
                    fp_features,
                    post_smiles_y,
                )
                _append_config(
                    source,
                    'LFC_smiles',
                    train_pack['LFC_smiles'],
                    True,
                    fp_features,
                    lfc_smiles_y,
                )
        if use_gene:
            # No-effect Post = Measured Pre (train+test on basal features), matching
            # the self-trained baseline and the Measured Pre panel. Do NOT train on
            # Measured Post here — that made No-effect ≠ Pre in transfer plots.
            _append_config(
                'no_effect', 'post_treatment', observed_pre_y, True, features
            )
            _append_config(
                'no_effect', 'LFC', no_effect_lfc_y, True, features
            )
            _append_config(
                'average_effect',
                'post_treatment',
                measured_train['post_treatment'],
                True,
                features,
                avg_post_y,
            )
            _append_config(
                'average_effect', 'LFC', measured_train['LFC'], True, features, avg_lfc_y
            )
        if use_smiles:
            avg_post_frame = mean_avg_post.copy()
            avg_lfc_frame = mean_avg_lfc.copy()
            no_effect_lfc_frame = mean_no_effect_lfc.copy()
            if 'drug' not in avg_post_frame.columns:
                avg_post_frame = merge_drug(avg_post_frame)
            if 'drug' not in avg_lfc_frame.columns:
                avg_lfc_frame = merge_drug(avg_lfc_frame)
            if 'drug' not in no_effect_lfc_frame.columns:
                no_effect_lfc_frame = merge_drug(no_effect_lfc_frame)
            _append_config(
                'no_effect',
                'post_treatment_smiles',
                observed_smiles_y['pre_treatment_smiles'],
                True,
                fp_features,
            )
            _append_config(
                'no_effect',
                'LFC_smiles',
                _with_y(attach_ecfp_features(no_effect_lfc_frame, drug_col='drug')),
                True,
                fp_features,
            )
            _append_config(
                'average_effect',
                'post_treatment_smiles',
                measured_train['post_treatment_smiles'],
                True,
                fp_features,
                _with_y(attach_ecfp_features(avg_post_frame, drug_col='drug')),
            )
            _append_config(
                'average_effect',
                'LFC_smiles',
                measured_train['LFC_smiles'],
                True,
                fp_features,
                _with_y(attach_ecfp_features(avg_lfc_frame, drug_col='drug')),
            )
    else:
        if include_observed:
            if use_gene:
                _append_config('observed', 'pre_treatment', observed_pre_y, True, features)
                _append_config('observed', 'post_treatment', observed_post_y, True, features)
                _append_config('observed', 'LFC', observed_lfc_y, True, features)
            if use_smiles:
                _append_config(
                    'observed',
                    'pre_treatment_smiles',
                    observed_smiles_y['pre_treatment_smiles'],
                    True,
                    fp_features,
                )
                _append_config(
                    'observed',
                    'post_treatment_smiles',
                    observed_smiles_y['post_treatment_smiles'],
                    True,
                    fp_features,
                )
                _append_config(
                    'observed',
                    'LFC_smiles',
                    observed_smiles_y['LFC_smiles'],
                    True,
                    fp_features,
                )
        else:
            logger.info(
                'predict_with_split_profiles_CV [%s]: skipping Measured (observed) profiles',
                dataset,
            )

        for model_name in include_models:
            post_df = predicted_tables[model_name][0].copy()
            lfc_df = predicted_tables[model_name][1].copy()
            if 'drug' not in post_df.columns:
                post_df = merge_drug(post_df)
            if 'drug' not in lfc_df.columns:
                lfc_df = merge_drug(lfc_df)
            post_y = _with_y(post_df)
            lfc_y = _with_y(lfc_df)
            source = f'{model_name}_predicted'
            if use_gene:
                _append_config(source, 'post_treatment', post_y, False, features)
                _append_config(source, 'LFC', lfc_y, False, features)
            if use_smiles:
                post_smiles_y = _with_y(attach_ecfp_features(post_df, drug_col='drug'))
                lfc_smiles_y = _with_y(attach_ecfp_features(lfc_df, drug_col='drug'))
                _append_config(
                    source, 'post_treatment_smiles', post_smiles_y, False, fp_features
                )
                _append_config(source, 'LFC_smiles', lfc_smiles_y, False, fp_features)
        if use_gene:
            _append_config('no_effect', 'post_treatment', no_effect_post_y, True, features)
            _append_config('no_effect', 'LFC', no_effect_lfc_y, True, features)
            _append_config('average_effect', 'post_treatment', avg_post_y, True, features)
            _append_config('average_effect', 'LFC', avg_lfc_y, True, features)
        if use_smiles:
            baseline_tables = {
                'no_effect_post_smiles': observed_smiles_y['pre_treatment_smiles'],
                'no_effect_lfc': mean_no_effect_lfc,
                'average_effect_post': mean_avg_post,
                'average_effect_lfc': mean_avg_lfc,
            }
            for key, frame in list(baseline_tables.items()):
                if key == 'no_effect_post_smiles':
                    continue
                frame = frame.copy()
                if 'drug' not in frame.columns:
                    frame = merge_drug(frame)
                baseline_tables[key] = frame
            _append_config(
                'no_effect',
                'post_treatment_smiles',
                baseline_tables['no_effect_post_smiles'],
                True,
                fp_features,
            )
            _append_config(
                'no_effect',
                'LFC_smiles',
                _with_y(attach_ecfp_features(baseline_tables['no_effect_lfc'], drug_col='drug')),
                True,
                fp_features,
            )
            _append_config(
                'average_effect',
                'post_treatment_smiles',
                _with_y(
                    attach_ecfp_features(baseline_tables['average_effect_post'], drug_col='drug')
                ),
                True,
                fp_features,
            )
            _append_config(
                'average_effect',
                'LFC_smiles',
                _with_y(
                    attach_ecfp_features(baseline_tables['average_effect_lfc'], drug_col='drug')
                ),
                True,
                fp_features,
            )

    if return_configs:
        return profile_configs

    all_scores = []
    all_predictions = []
    feature_importance: dict[str, dict] = {}

    for (
        profile_source,
        model_label,
        df_with_y,
        dedupe_train,
        selected_features,
        test_profiles_df,
    ) in profile_configs:
        train_set = (
            'observed'
            if test_profiles_df is not None or profile_source == 'observed'
            else 'predicted'
        )
        n_test = len(test_profiles_df) if test_profiles_df is not None else len(df_with_y)
        logger.info(
            "predict_with_split_profiles_CV [%s]: %s / %s "
            "(train_rows=%d test_rows=%d train_set=%s)",
            dataset,
            profile_source,
            model_label,
            len(df_with_y),
            n_test,
            train_set,
        )
        scores, predictions, fi = _mcfarland_predefined_fold_cv(
            df_with_y,
            selected_features,
            dedupe_train=dedupe_train,
            test_profiles_df=test_profiles_df,
            two_part=two_part,
            threshold=threshold,
        )
        scores = (
            scores.assign(profile_source=profile_source)
            .assign(model=model_label)
            .assign(feature_selection=f'top{n_features} HVG')
            .assign(dataset=dataset)
            .assign(train_set=train_set)
        )
        predictions = (
            predictions.assign(profile_source=profile_source)
            .assign(model=model_label)
            .assign(feature_selection=f'top{n_features} HVG')
            .assign(dataset=dataset)
            .assign(head='twopart' if two_part else 'continuous')
            .assign(threshold=threshold if two_part else np.nan)
            .assign(train_set=train_set)
        )
        all_scores.append(scores)
        all_predictions.append(predictions)
        feature_importance[f'{profile_source}_{model_label}'] = fi

    results = pd.concat(all_scores, ignore_index=True)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    results.to_csv(os.path.join(results_dir, f'{results_prefix}_results.csv'))
    predictions_df.to_csv(os.path.join(results_dir, f'{results_prefix}_predictions.csv'))
    with open(os.path.join(results_dir, f'{results_prefix}_feature_importance.pkl'), 'wb') as f:
        pickle.dump(feature_importance, f)

    logger.info(
        "predict_with_split_profiles_CV [%s]: wrote %s (%d score rows, %d prediction rows)",
        dataset,
        results_prefix,
        len(results),
        len(predictions_df),
    )


def predict_McFarland_with_CPA_profiles_CV(
    n_features: int = 1000,
    results_prefix: str = 'mcfarland_split_cv',
    include_models: tuple[str, ...] = NEW_PREDICTED_MODELS,
    include_predicted_smiles: bool = True,
) -> None:
    """McFarland drug-response CV on predicted-profile split labels (CPA / chemCPA / PRnet by default)."""
    predict_with_split_profiles_CV(
        'mcfarland',
        n_features=n_features,
        results_prefix=results_prefix,
        include_models=include_models,
        include_predicted_smiles=include_predicted_smiles,
    )


def predict_sciplex_with_split_profiles_CV(
    n_features: int = 1000,
    results_prefix: str = 'sciplex_split_cv',
    include_models: tuple[str, ...] = NEW_PREDICTED_MODELS,
    include_predicted_smiles: bool = True,
) -> None:
    """SciPlex drug-response CV on predicted-profile split labels (CPA / chemCPA / PRnet by default)."""
    predict_with_split_profiles_CV(
        'sciplex',
        n_features=n_features,
        results_prefix=results_prefix,
        include_models=include_models,
        include_predicted_smiles=include_predicted_smiles,
    )


def _run_configs_fold_cv(
    profile_configs: list[tuple],
    *,
    dataset: str,
    n_features: int,
    two_part: bool,
    threshold: float,
) -> pd.DataFrame:
    frames = []
    for config in profile_configs:
        if len(config) == 5:
            profile_source, model_label, df_with_y, dedupe_train, selected_features = config
            test_profiles_df = None
        else:
            (
                profile_source,
                model_label,
                df_with_y,
                dedupe_train,
                selected_features,
                test_profiles_df,
            ) = config
        logger.info(
            "_run_configs_fold_cv [%s]: %s / %s two_part=%s threshold=%s rows=%d",
            dataset,
            profile_source,
            model_label,
            two_part,
            threshold if two_part else 'NA',
            len(df_with_y),
        )
        _, predictions, _ = _mcfarland_predefined_fold_cv(
            df_with_y,
            selected_features,
            dedupe_train=dedupe_train,
            test_profiles_df=test_profiles_df,
            two_part=two_part,
            threshold=threshold,
        )
        frames.append(
            predictions.assign(
                profile_source=profile_source,
                model=model_label,
                feature_selection=f'top{n_features} HVG',
                dataset=dataset,
                head='twopart' if two_part else 'continuous',
                threshold=threshold if two_part else np.nan,
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def run_two_part_threshold_sweep(
    dataset: str,
    thresholds: tuple[float, ...] = TWOPART_SWEEP_THRESHOLDS,
    n_features: int = 1000,
    include_models: tuple[str, ...] = NEW_PREDICTED_MODELS,
    results_prefix: str | None = None,
    include_predicted_smiles: bool = True,
    include_observed: bool = True,
    smiles_only: bool = False,
    train_on_measured_test_on_predicted: bool = False,
) -> pd.DataFrame:
    """Train the two-stage hurdle model at each sensitivity threshold and save predictions.

    Writes only ``{results_prefix}_predictions.csv`` (all thresholds) so continuous
    ``*_split_cv_*`` outputs are never overwritten. Default prefix is
    ``{dataset}_twopart_threshold_sweep``.
    """
    dataset = dataset.lower()
    if results_prefix is None:
        results_prefix = f'{dataset}_twopart_threshold_sweep'
    logger.info(
        "run_two_part_threshold_sweep [%s]: thresholds=%s prefix=%s "
        "include_observed=%s smiles_only=%s transfer=%s",
        dataset,
        thresholds,
        results_prefix,
        include_observed,
        smiles_only,
        train_on_measured_test_on_predicted,
    )
    profile_configs = predict_with_split_profiles_CV(
        dataset,
        n_features=n_features,
        include_models=include_models,
        return_configs=True,
        include_predicted_smiles=include_predicted_smiles,
        include_observed=include_observed,
        smiles_only=smiles_only,
        train_on_measured_test_on_predicted=train_on_measured_test_on_predicted,
    )
    profile_configs = [
        c for c in profile_configs
        if c[1] not in ('pre_treatment', 'pre_treatment_smiles')
    ]
    frames = []
    for threshold in thresholds:
        logger.info("two-part sweep [%s]: threshold=%.3f", dataset, threshold)
        preds = _run_configs_fold_cv(
            profile_configs,
            dataset=dataset,
            n_features=n_features,
            two_part=True,
            threshold=float(threshold),
        )
        frames.append(preds)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_path = os.path.join(results_dir, f'{results_prefix}_predictions.csv')
    out.to_csv(out_path, index=False)
    logger.info(
        "run_two_part_threshold_sweep [%s]: wrote %s (%d rows)",
        dataset,
        out_path,
        len(out),
    )
    return out

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Drug-response prediction from predicted profiles')
    parser.add_argument(
        '--dataset',
        choices=['sciplex', 'mcfarland', 'both'],
        default='both',
        help='Which split-CV run to execute',
    )
    parser.add_argument(
        '--mode',
        choices=['continuous', 'twopart', 'threshold-sweep'],
        default='continuous',
        help='continuous ElasticNet (default), two-stage hurdle at --threshold, or a threshold sweep',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=TWOPART_DEFAULT_THRESHOLD,
        help='Hurdle threshold for --mode twopart (default 0.02)',
    )
    parser.add_argument(
        '--thresholds',
        default=','.join(str(t) for t in TWOPART_SWEEP_THRESHOLDS),
        help=(
            'Comma-separated thresholds for --mode threshold-sweep '
            f'(default: {",".join(str(t) for t in TWOPART_SWEEP_THRESHOLDS)})'
        ),
    )
    parser.add_argument(
        '--models',
        default=','.join(NEW_PREDICTED_MODELS),
        help=(
            'Comma-separated predicted-profile sources '
            '(default: CPA,chemCPA,PRnet; also supports GEARS,scFoundation).'
        ),
    )
    parser.add_argument(
        '--include-predicted-smiles',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Also train Post+SMILES / LFC+SMILES heads on predicted profiles and '
            'average/no-effect baselines (default: on).'
        ),
    )
    parser.add_argument(
        '--include-observed',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Include Measured Pre/Post/LFC (±SMILES) in split-CV. '
            'Use --no-include-observed to train only on predicted profiles + baselines.'
        ),
    )
    parser.add_argument(
        '--smiles-only',
        action='store_true',
        default=False,
        help='Train only +SMILES heads (skip gene-only Post/LFC/Pre).',
    )
    parser.add_argument(
        '--train-on-measured-test-on-predicted',
        action='store_true',
        default=False,
        help=(
            'Train ElasticNet on Measured Post/LFC (±SMILES) and evaluate on '
            'predicted PRM / baseline profiles for the held-out fold.'
        ),
    )
    parser.add_argument(
        '--results-suffix',
        default='',
        help='Optional suffix for output prefixes, e.g. "_smiles" -> mcfarland_split_cv_smiles.',
    )
    args = parser.parse_args()

    include_models = tuple(m.strip() for m in args.models.split(',') if m.strip())
    if not include_models:
        raise ValueError('--models must list at least one predicted-profile source')

    log_script_start(__file__, logger)
    
    try:
        logger.info(f"Project root: {home_dir}")
        logger.info(f"Results directory: {results_dir}")
        logger.info(f"Resources directory: {resources_dir}")
        logger.info("Predicted models: %s", list(include_models))
        logger.info("Include predicted SMILES: %s", args.include_predicted_smiles)
        logger.info("Include observed (Measured): %s", args.include_observed)
        logger.info("SMILES-only heads: %s", args.smiles_only)
        logger.info(
            "Train measured → test predicted: %s",
            args.train_on_measured_test_on_predicted,
        )

        datasets = ['sciplex', 'mcfarland'] if args.dataset == 'both' else [args.dataset]
        for dataset in datasets:
            suffix = args.results_suffix
            if suffix and not suffix.startswith('_'):
                suffix = f'_{suffix}'
            if args.train_on_measured_test_on_predicted and not suffix:
                suffix = '_train_measured_test_predicted'
            common_kwargs = dict(
                include_models=include_models,
                include_predicted_smiles=args.include_predicted_smiles,
                include_observed=args.include_observed,
                smiles_only=args.smiles_only,
                train_on_measured_test_on_predicted=args.train_on_measured_test_on_predicted,
            )
            if args.mode == 'continuous':
                logger.info("Continuous ElasticNet split-CV [%s]...", dataset)
                predict_with_split_profiles_CV(
                    dataset,
                    two_part=False,
                    results_prefix=f'{dataset}_split_cv{suffix}' if suffix else None,
                    **common_kwargs,
                )
            elif args.mode == 'twopart':
                logger.info("Two-stage hurdle split-CV [%s] threshold=%.3f...", dataset, args.threshold)
                predict_with_split_profiles_CV(
                    dataset,
                    two_part=True,
                    threshold=args.threshold,
                    results_prefix=f'{dataset}_twopart{suffix}',
                    **common_kwargs,
                )
            else:
                sweep_thresholds = tuple(
                    float(t.strip()) for t in args.thresholds.split(',') if t.strip()
                )
                if not sweep_thresholds:
                    raise ValueError('--thresholds must list at least one value')
                logger.info(
                    "Two-stage threshold sweep [%s] thresholds=%s...",
                    dataset,
                    sweep_thresholds,
                )
                run_two_part_threshold_sweep(
                    dataset,
                    thresholds=sweep_thresholds,
                    results_prefix=f'{dataset}_twopart_threshold_sweep{suffix}',
                    **common_kwargs,
                )

        logger.info("Drug sensitivity prediction pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in drug sensitivity prediction: {str(e)}")
        raise
    
    finally:
        log_script_end(__file__, logger)