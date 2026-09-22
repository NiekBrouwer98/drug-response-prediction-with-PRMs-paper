"""Load ElasticNet feature importances from split-CV drug-response runs."""

from __future__ import annotations

import os
import pickle
from typing import Iterable

import pandas as pd

CPA_FAMILY_PROFILES: tuple[tuple[str, str], ...] = (
    ('observed', 'Measured'),
    ('CPA_predicted', 'CPA'),
    ('chemCPA_predicted', 'chemCPA'),
    ('PRnet_predicted', 'PRnet'),
    ('GEARS_predicted', 'GEARS'),
    ('scFoundation_predicted', 'scFoundation'),
    ('average_effect', 'Average effect'),
    ('no_effect', 'No effect'),
)
SCIPLEX_CPA_FAMILY_PROFILES = CPA_FAMILY_PROFILES

# Default split-CV runs that include drug-descriptor (±SMILES) ElasticNets.
DEFAULT_SCIPLEX_FI_PREFIX = 'sciplex_split_cv_smiles'
DEFAULT_MCFARLAND_FI_PREFIX = 'mcfarland_split_cv_smiles'

GSEA_MODEL_ORDER_POST: tuple[str, ...] = (
    'Measured',
    'CPA',
    'chemCPA',
    'PRnet',
    'GEARS',
    'scFoundation',
    'Average effect',
    'No effect',
)
GSEA_MODEL_ORDER_LFC: tuple[str, ...] = (
    'Measured',
    'CPA',
    'chemCPA',
    'PRnet',
    'GEARS',
    'scFoundation',
    'Average effect',
)
SCIPLEX_GSEA_MODEL_ORDER_POST = GSEA_MODEL_ORDER_POST
SCIPLEX_GSEA_MODEL_ORDER_LFC = GSEA_MODEL_ORDER_LFC


def _is_gene_feature(name: object) -> bool:
    """True for gene symbols; False for ECFP / other drug-descriptor columns."""
    text = str(name)
    return not (
        text.startswith('ecfp_')
        or text.startswith('fp_')
        or text.startswith('morgan_')
    )


def filter_gene_features(importance: pd.Series) -> pd.Series:
    """Keep gene coefficients only (drop ECFP / fingerprint features)."""
    if importance.empty:
        return importance
    keep = [_is_gene_feature(idx) for idx in importance.index]
    return importance.loc[keep]


def _mean_coef_across_folds(fi_by_fold: dict) -> pd.Series:
    if not fi_by_fold:
        return pd.Series(dtype=float)
    matrix = pd.concat(fi_by_fold, axis=1)
    return matrix.mean(axis=1)


def _single_row_importance(
    importance: pd.Series,
    *,
    condition: str = 'Full Sciplex dataset',
    model: str = 'Post',
) -> pd.DataFrame:
    row = importance.to_frame().T
    row.insert(0, 'condition', condition)
    row.insert(1, 'model', model)
    return row.reset_index(drop=True)


def load_split_cv_feature_importance_pickle(
    results_dir: str,
    prefix: str,
) -> dict[str, dict[str, pd.Series]]:
    path = os.path.join(results_dir, f'{prefix}_feature_importance.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Split-CV feature importance not found: {path}')
    with open(path, 'rb') as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f'Expected dict in {path}, got {type(raw)!r}')
    return raw


def load_split_cv_feature_importance_with_observed_fallback(
    results_dir: str,
    prefix: str,
) -> dict[str, dict[str, pd.Series]]:
    """Load ``prefix`` FI; fill missing keys from the non-``_smiles`` sibling pickle.

    Retrained SMILES-only runs often omit ``observed_*`` entries. The gene-only
    ``*_split_cv_feature_importance.pkl`` still carries measured Pre/Post/LFC.
    """
    primary = load_split_cv_feature_importance_pickle(results_dir, prefix)
    if not prefix.endswith('_smiles'):
        return primary
    base_prefix = prefix[: -len('_smiles')]
    base_path = os.path.join(results_dir, f'{base_prefix}_feature_importance.pkl')
    if not os.path.exists(base_path):
        return primary
    base = load_split_cv_feature_importance_pickle(results_dir, base_prefix)
    merged = dict(primary)
    for key, value in base.items():
        if key not in merged:
            merged[key] = value
    return merged


def build_cpa_family_gsea_feature_tables(
    fi_pickle: dict[str, dict[str, pd.Series]],
    *,
    condition: str,
    smiles: bool = True,
    gene_features_only: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Build single-row gene-importance tables for prerank GSEA.

    Keys: ``post``, ``lfc``, ``post_CPA``, ``lfc_chemCPA``, ``post_Average effect``, ...

    When ``smiles`` is True, prefer ``*_smiles`` ElasticNet keys (gene+ECFP models).
    When ``gene_features_only`` is True, drop ECFP / fingerprint coefficients before GSEA.
    """
    tables: dict[str, pd.DataFrame] = {}
    modality_pairs = (
        (('post_treatment_smiles', 'post'), ('LFC_smiles', 'lfc'))
        if smiles
        else (('post_treatment', 'post'), ('LFC', 'lfc'))
    )

    for profile_key, display_name in CPA_FAMILY_PROFILES:
        for model_label, gsea_prefix in modality_pairs:
            pickle_key = f'{profile_key}_{model_label}'
            if pickle_key not in fi_pickle and smiles:
                # Fall back to gene-only modality if SMILES key is absent.
                fallback = (
                    'post_treatment' if gsea_prefix == 'post' else 'LFC'
                )
                pickle_key = f'{profile_key}_{fallback}'
            if pickle_key not in fi_pickle:
                continue
            importance = _mean_coef_across_folds(fi_pickle[pickle_key])
            if gene_features_only:
                importance = filter_gene_features(importance)
            if importance.empty:
                continue
            if profile_key == 'observed':
                gsea_key = gsea_prefix
            else:
                gsea_key = f'{gsea_prefix}_{display_name}'
            model_name = 'Post' if gsea_prefix == 'post' else 'Log(fold change)'
            tables[gsea_key] = _single_row_importance(
                importance, condition=condition, model=model_name,
            )

    return tables


def build_sciplex_gsea_feature_tables(
    fi_pickle: dict[str, dict[str, pd.Series]],
    *,
    smiles: bool = True,
    gene_features_only: bool = True,
) -> dict[str, pd.DataFrame]:
    return build_cpa_family_gsea_feature_tables(
        fi_pickle,
        condition='Full Sciplex dataset',
        smiles=smiles,
        gene_features_only=gene_features_only,
    )


def get_sciplex_post_feature_importance_gene_table(
    results_dir: str,
    prefix: str = DEFAULT_SCIPLEX_FI_PREFIX,
    *,
    smiles: bool = True,
) -> pd.DataFrame:
    """Mean observed-post ElasticNet gene weights as ``gene`` / ``importance``."""
    fi_pickle = load_split_cv_feature_importance_with_observed_fallback(
        results_dir, prefix,
    )
    candidates = (
        ['observed_post_treatment_smiles', 'observed_post_treatment']
        if smiles
        else ['observed_post_treatment']
    )
    key = next((k for k in candidates if k in fi_pickle), None)
    if key is None:
        raise KeyError(
            f'observed post key missing from {prefix}_feature_importance.pkl '
            f'(and non-smiles fallback); keys={sorted(fi_pickle)}'
        )
    importance = filter_gene_features(_mean_coef_across_folds(fi_pickle[key]))
    return (
        importance.rename('importance')
        .rename_axis('gene')
        .reset_index()
        .sort_values('gene')
    )


def gsea_model_column_name(raw_key: str) -> str:
    """Map GSEA result key (``post_CPA``) to dotplot model label."""
    if raw_key in {'post', 'lfc'}:
        return f'{raw_key} Measured'
    if raw_key.startswith('post_'):
        return f"post {raw_key.removeprefix('post_')}"
    if raw_key.startswith('lfc_'):
        return f"lfc {raw_key.removeprefix('lfc_')}"
    return raw_key.replace('_', ' ')


def sciplex_dotplot_model_label(model: str) -> str:
    """Normalize dotplot model strings to display names."""
    label = model.replace('post ', '').replace('lfc ', '').strip()
    if label in {'post', 'lfc', 'Measured', 'Observed'}:
        return 'Measured'
    return label


def filter_sciplex_gsea_dotplot_models(
    dotplot_data: pd.DataFrame,
    *,
    profile_kind: str,
    model_order: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Keep CPA-family models for post or LFC GSEA dotplots."""
    prefix = 'post ' if profile_kind == 'post' else 'lfc '
    out = dotplot_data[dotplot_data['model'].str.startswith(prefix)].copy()
    out['model'] = out['model'].str.replace(prefix, '', regex=False)
    out['model'] = out['model'].map(sciplex_dotplot_model_label)
    order = list(model_order or GSEA_MODEL_ORDER_POST)
    if profile_kind == 'lfc':
        order = list(model_order or GSEA_MODEL_ORDER_LFC)
    out = out[out['model'].isin(order)].copy()
    out['model'] = pd.Categorical(out['model'], categories=order, ordered=True)
    return out.sort_values('model')
