"""Systema reference vectors for profile evaluation."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

SPLIT_COLUMN_ALIASES: tuple[str, ...] = ('fold', 'split')
DEFAULT_CONTROL_CONDITIONS: frozenset[str] = frozenset({'ctrl', 'control', 'vehicle', 'dmso'})


def resolve_split_column(df: pd.DataFrame) -> str | None:
    for col in SPLIT_COLUMN_ALIASES:
        if col in df.columns:
            return col
    return None


def normalize_split_values(df: pd.DataFrame, split_col: str | None = None) -> pd.DataFrame:
    """Coerce fold/split labels to int (handles ``split_0`` from PRnet)."""
    split_col = split_col or resolve_split_column(df)
    if split_col is None:
        return df
    out = df.copy()
    series = out[split_col]
    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        extracted = series.astype(str).str.extract(r'(\d+)', expand=False)
        out[split_col] = extracted.astype(int)
    else:
        out[split_col] = series.astype(int)
    return out


def _normalize_profile_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Align cell_line / condition naming (handles cell_type from pseudobulk CSVs)."""
    out = df.copy()
    if 'cell_type' in out.columns and 'cell_line' not in out.columns:
        out = out.rename(columns={'cell_type': 'cell_line'})
    if 'cell_line' in out.columns:
        out['cell_line'] = (
            out['cell_line'].astype(str).str.strip().str.split('_').str[0].str.strip().str.upper()
        )
    if 'condition' in out.columns:
        out['condition'] = out['condition'].astype(str).str.strip()
    return normalize_split_values(out)


def resolve_cell_line_column(
    df: pd.DataFrame,
    preferred: tuple[str, ...] = ('cell_line', 'cell_type'),
) -> str:
    for col in preferred:
        if col in df.columns:
            return col
    raise ValueError(f'No cell line column found (tried {preferred}).')


def _cell_line_col_after_normalize(
    df: pd.DataFrame, cell_line_col: str | None = None
) -> str:
    """Resolve cell-line column on a frame already passed through ``_normalize_profile_keys``."""
    if cell_line_col is not None and cell_line_col in df.columns:
        return cell_line_col
    return resolve_cell_line_column(df)


def _is_control_condition(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(DEFAULT_CONTROL_CONDITIONS)


def reference_vector_all_non_control(
    observations: pd.DataFrame,
    gene_columns: list[str],
    cell_line_col: str | None = None,
) -> pd.Series:
    """Single mean gene vector over all non-control (cell_line, condition) pairs."""
    observations = _normalize_profile_keys(observations)
    cell_line_col = _cell_line_col_after_normalize(observations, cell_line_col)
    obs = observations.loc[~_is_control_condition(observations['condition'])].copy()
    obs = obs.drop_duplicates(subset=[cell_line_col, 'condition'], keep='first')
    if obs.empty:
        raise ValueError('No non-control observations available to compute Systema reference vector.')
    return obs[gene_columns].mean(axis=0)


def reference_vectors_per_cell_line_non_control(
    observations: pd.DataFrame,
    gene_columns: list[str],
    cell_line_col: str | None = None,
) -> dict[str, pd.Series]:
    """Mean observed profile per cell line (non-control), one vector per cell line."""
    observations = _normalize_profile_keys(observations)
    cell_line_col = _cell_line_col_after_normalize(observations, cell_line_col)
    obs = observations.loc[~_is_control_condition(observations['condition'])].copy()
    obs = obs.drop_duplicates(subset=[cell_line_col, 'condition'], keep='first')
    refs: dict[str, pd.Series] = {}
    for cell_line in obs[cell_line_col].unique():
        subset = obs[obs[cell_line_col] == cell_line]
        if not subset.empty:
            refs[str(cell_line)] = subset[gene_columns].mean(axis=0)
    return refs


def leave_one_split_reference_vectors(
    profiles: pd.DataFrame,
    gene_columns: list[str],
    split_col: str,
) -> dict[int, pd.Series]:
    """
    For each split value s, return the mean gene vector over all rows with split != s.
    """
    profiles = profiles.copy()
    profiles[split_col] = profiles[split_col].astype(int)
    splits = sorted(profiles[split_col].unique())
    refs: dict[int, pd.Series] = {}
    for split_value in splits:
        other = profiles[profiles[split_col] != split_value]
        if other.empty:
            raise ValueError(f'No profiles outside split {split_value} to build Systema reference.')
        refs[split_value] = other[gene_columns].mean(axis=0)
    return refs


def leave_one_split_reference_vectors_per_cell_line(
    profiles: pd.DataFrame,
    gene_columns: list[str],
    split_col: str,
    cell_line_col: str | None = None,
) -> dict[tuple[int, str], pd.Series]:
    """
    For each (split s, cell line), return the mean gene vector over training rows
    (split != s) in that same cell line.
    """
    profiles = _normalize_profile_keys(profiles.copy())
    profiles[split_col] = profiles[split_col].astype(int)
    cell_line_col = _cell_line_col_after_normalize(profiles, cell_line_col)

    refs: dict[tuple[int, str], pd.Series] = {}
    for split_value in sorted(profiles[split_col].unique()):
        for cell_line in profiles[cell_line_col].unique():
            other = profiles[
                (profiles[split_col] != split_value)
                & (profiles[cell_line_col] == cell_line)
            ]
            if other.empty:
                continue
            refs[(int(split_value), str(cell_line))] = other[gene_columns].mean(axis=0)
    return refs


def attach_split_from_template(
    predictions: pd.DataFrame,
    template: pd.DataFrame,
    key_cols: Iterable[str] = ('cell_line', 'condition'),
) -> pd.DataFrame:
    """Replicate split labels from ``template`` when predictions lack them (e.g. baselines)."""
    predictions = _normalize_profile_keys(predictions)
    template = _normalize_profile_keys(template)
    split_col = resolve_split_column(template)
    if split_col is None:
        return predictions
    if resolve_split_column(predictions) is not None:
        return predictions
    merge_cols = list(key_cols)
    template_keys = template[merge_cols + [split_col]].drop_duplicates()
    return predictions.merge(template_keys, on=merge_cols, how='inner')


def observation_profiles_with_split(
    observations: pd.DataFrame,
    split_assignments: pd.DataFrame,
    key_cols: Iterable[str] = ('cell_line', 'condition'),
) -> pd.DataFrame:
    """Observed expression with one row per (keys..., split) from ``split_assignments``."""
    split_col = resolve_split_column(split_assignments)
    if split_col is None:
        raise ValueError('split_assignments must contain a fold or split column.')
    merge_cols = list(key_cols)
    observations = _normalize_profile_keys(observations)
    split_assignments = _normalize_profile_keys(split_assignments)
    obs = observations.drop_duplicates(subset=merge_cols, keep='first')
    keys = split_assignments[merge_cols + [split_col]].drop_duplicates()
    return keys.merge(obs, on=merge_cols, how='inner')
