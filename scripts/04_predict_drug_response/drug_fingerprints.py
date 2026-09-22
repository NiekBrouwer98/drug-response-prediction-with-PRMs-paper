"""Drug SMILES lookup and Morgan (ECFP) fingerprint features for response prediction."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FP_PREFIX = 'ecfp_'
DEFAULT_N_BITS = 2048
DEFAULT_RADIUS = 2


def fingerprint_column_names(n_bits: int = DEFAULT_N_BITS) -> list[str]:
    return [f'{FP_PREFIX}{i}' for i in range(n_bits)]


def _resources_dir() -> Path:
    from config import config

    return Path(config.RESOURCES_DIR)


def ensure_drug_smiles_table(
    path: Path | None = None,
    *,
    build_if_missing: bool = True,
) -> Path:
    """Return path to ``drug_smiles.csv``, building from PubChem if absent."""
    csv_path = path or (_resources_dir() / 'drug_smiles.csv')
    if csv_path.exists():
        return csv_path
    if not build_if_missing:
        raise FileNotFoundError(
            f'Drug SMILES table not found at {csv_path}. '
            'Run scripts/04_predict_drug_response/build_drug_smiles.py first.'
        )
    from build_drug_smiles import write_drug_smiles_table

    logger.info('drug_smiles.csv missing; building from PubChem at %s', csv_path)
    return write_drug_smiles_table(csv_path)


@lru_cache(maxsize=1)
def load_drug_smiles_table(path: Path | None = None) -> pd.DataFrame:
    """Load ``resources/drug_smiles.csv`` (columns: drug_key, smiles)."""
    csv_path = ensure_drug_smiles_table(path)
    df = pd.read_csv(csv_path)
    required = {'drug_key', 'smiles'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{csv_path} missing columns: {sorted(missing)}')
    df = df.copy()
    df['drug_key'] = df['drug_key'].astype(str).str.strip()
    df['smiles'] = df['smiles'].fillna('').astype(str).str.strip()
    return df.drop_duplicates(subset='drug_key', keep='first')


def normalize_drug_key(names: pd.Series) -> pd.Series:
    """Canonical key for SMILES lookup (strip spaces; keep punctuation)."""
    return names.astype(str).str.strip()


def ecfp_cache_path(n_bits: int = DEFAULT_N_BITS) -> Path:
    return _resources_dir() / f'drug_ecfp_{n_bits}.npz'


def write_ecfp_cache(
    n_bits: int = DEFAULT_N_BITS,
    radius: int = DEFAULT_RADIUS,
    smiles_table: pd.DataFrame | None = None,
    out_path: Path | None = None,
) -> Path:
    """Precompute Morgan fingerprints for all drugs in ``drug_smiles.csv`` (requires RDKit)."""
    table = smiles_table if smiles_table is not None else load_drug_smiles_table()
    target = out_path or ecfp_cache_path(n_bits)
    target.parent.mkdir(parents=True, exist_ok=True)

    keys: list[str] = []
    rows: list[np.ndarray] = []
    for _, row in table.iterrows():
        key = str(row['drug_key']).strip()
        smi = str(row['smiles']).strip()
        keys.append(key)
        rows.append(_morgan_bits(smi, n_bits, radius) if smi else np.zeros(n_bits, dtype=np.float32))

    fps = np.stack(rows, axis=0).astype(np.float32)
    np.savez_compressed(
        target,
        drug_keys=np.array(keys, dtype=object),
        fps=fps,
        n_bits=np.array(n_bits),
        radius=np.array(radius),
    )
    logger.info('Wrote precomputed ECFP cache %s (%d drugs × %d bits)', target, len(keys), n_bits)
    return target


@lru_cache(maxsize=4)
def _load_ecfp_cache_dict(n_bits: int = DEFAULT_N_BITS) -> dict[str, np.ndarray]:
    path = ecfp_cache_path(n_bits)
    if not path.exists():
        raise FileNotFoundError(
            f'Precomputed ECFP cache not found at {path}. '
            'Run build_drug_smiles.py locally (with RDKit) to generate drug_ecfp_2048.npz.'
        )
    data = np.load(path, allow_pickle=True)
    if int(data['n_bits']) != n_bits:
        raise ValueError(f'ECFP cache {path} has n_bits={int(data["n_bits"])}, expected {n_bits}')
    keys = [str(k) for k in data['drug_keys'].tolist()]
    fps = data['fps'].astype(np.float32)
    return dict(zip(keys, fps))


def _ecfp_lookup(
    unique_keys: list[str],
    n_bits: int,
    radius: int,
    smiles_by_key: dict[str, str],
) -> dict[str, np.ndarray]:
    """Resolve fingerprints from precomputed cache, falling back to RDKit if needed."""
    try:
        cached = _load_ecfp_cache_dict(n_bits)
        return {key: cached.get(key, np.zeros(n_bits, dtype=np.float32)) for key in unique_keys}
    except FileNotFoundError as exc:
        logger.info('ECFP cache unavailable (%s); computing with RDKit', exc)

    cache: dict[str, np.ndarray] = {}
    for key in unique_keys:
        smi = smiles_by_key.get(key, '')
        cache[key] = _morgan_bits(smi, n_bits, radius) if smi else np.zeros(n_bits, dtype=np.float32)
    return cache


def _morgan_bits(smiles: str, n_bits: int, radius: int) -> np.ndarray:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def attach_ecfp_features(
    df: pd.DataFrame,
    drug_col: str = 'drug',
    *,
    n_bits: int = DEFAULT_N_BITS,
    radius: int = DEFAULT_RADIUS,
    smiles_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add Morgan fingerprint columns ``ecfp_*`` keyed on ``drug_col``."""
    if drug_col not in df.columns:
        raise KeyError(f'drug_col {drug_col!r} not in dataframe')

    out = df.copy()
    if len(out) == 0:
        fp_cols = fingerprint_column_names(n_bits)
        return pd.concat([out, pd.DataFrame(columns=fp_cols)], axis=1)

    table = smiles_table if smiles_table is not None else load_drug_smiles_table()
    smiles_by_key = table.set_index('drug_key')['smiles'].to_dict()

    fp_cols = fingerprint_column_names(n_bits)
    keys = normalize_drug_key(out[drug_col])
    unique_keys = keys.dropna().unique().tolist()

    cache = _ecfp_lookup(unique_keys, n_bits, radius, smiles_by_key)
    missing = [k for k in unique_keys if not smiles_by_key.get(k, '')]

    if missing:
        logger.warning(
            'attach_ecfp_features: no SMILES for %d / %d drugs (zero vector); sample: %s',
            len(missing),
            len(unique_keys),
            missing[:8],
        )

    fp_matrix = np.vstack([cache[k] for k in keys.to_numpy()])
    fp_df = pd.DataFrame(fp_matrix, columns=fp_cols, index=out.index)
    return pd.concat([out, fp_df], axis=1)


def merge_drug_names_mcfarland(df: pd.DataFrame, resources_dir: Path | None = None) -> pd.DataFrame:
    """Attach ``drug`` column from ``mcfarland_drug_to_perturbation.csv`` via ``condition``/target."""
    res = resources_dir or _resources_dir()
    drug_map = pd.read_csv(res / 'mcfarland_drug_to_perturbation.csv')
    drug_map = drug_map[['drug', 'target']].drop_duplicates(subset='target', keep='first')
    out = df.merge(drug_map, left_on='condition', right_on='target', how='left')
    out['drug'] = out['drug'].fillna(out['condition'])
    return out


def merge_drug_names_sciplex(df: pd.DataFrame, resources_dir: Path | None = None) -> pd.DataFrame:
    """Attach ``drug`` and ``target`` from SciPlex mapping via normalized product name."""
    res = resources_dir or _resources_dir()
    drug_map = pd.read_csv(res / 'sciplex_drug_to_perturbation.csv', index_col=0)
    drug_map['product_name_key'] = normalize_drug_key(drug_map['product_name']).str.replace(
        r'\s+', '', regex=True
    )
    drug_map = drug_map[['product_name', 'target', 'product_name_key']].drop_duplicates(
        subset='product_name_key', keep='first'
    )
    out = df.copy()
    out['_drug_key'] = normalize_drug_key(out['condition']).str.replace(r'\s+', '', regex=True)
    out = out.merge(drug_map, left_on='_drug_key', right_on='product_name_key', how='left')
    out['drug'] = out['product_name'].fillna(out['condition'])
    if 'target' not in out.columns or out['target'].isna().all():
        out['target'] = out['condition']
    else:
        out['target'] = out['target'].fillna(out['condition'])
    return out.drop(columns=['_drug_key', 'product_name_key'], errors='ignore')


def gene_and_fingerprint_features(
    gene_features: Iterable[str],
    n_bits: int = DEFAULT_N_BITS,
) -> list[str]:
    """Gene subset plus all ECFP columns (for profile + SMILES models)."""
    fp_cols = fingerprint_column_names(n_bits)
    meta = {'tissue', 'cell_line', 'condition', 'y', 'target', 'sens', 'sens_label', 'fold', 'drug', 'product_name'}
    genes = [f for f in gene_features if f not in meta and not f.startswith(FP_PREFIX)]
    return genes + fp_cols
