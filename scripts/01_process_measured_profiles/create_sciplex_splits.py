"""Filter SciPlex to control + highest dose, annotate perturbation matches, and add splits.

Keeps vehicle/control cells and cells at the maximum observed dose (10 μM in
Srivatsan 2019). SciPlex obs use ``condition`` for the drug (formerly
``product_name``) and ``gene_target`` for the mapped gene label (formerly
``condition``). Drugs are matched to ``sciplex_drug_to_perturbation.csv``;
all cells are retained and ``perturbation_match`` flags whether a gene target was
found. Within each cell line, treated ``(cell_type, condition)`` pairs are
partitioned into 5 folds (~80/20 train/test). All cells of a drug in a line
share the same fold, so that pair is never in both train and test. Control cells have
``fold=-1`` and are ``train`` in every split. Expression is library-size
normalized, log1p-transformed, and aligned to the 19264-gene
``OS_scRNA_gene_index``. Cells with fewer than ``config.MIN_GENES_PER_CELL``
detected genes are dropped (same minimal QC as the original per-line SciPlex /
McFarland processing).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config, setup_project
from preprocess_utils import normalize_log1p_and_align_genes, process_merged_df, store_counts_layer
from qc_utils import annotate_dataset_qc
from utils import (
    ensure_directories_exist,
    log_script_end,
    log_script_start,
    setup_logging_for_script,
)

setup_project()
logger = setup_logging_for_script(__file__)

data_dir = str(config.DATA_DIR)
resources_dir = str(config.RESOURCES_DIR)
ensure_directories_exist(str(config.PROJECT_ROOT), data_dir)

RAW_H5AD_NAME = "Srivatsan_2019_raw.h5ad"
EXTERNAL_SCIPLEX_RAW_DIRS = (
    Path(r"N:/ewi/insy/DBL/nbrouwer1/perturbation_response_model/DATA/sciplex_raw"),
    Path("/tudelft.net/staff-bulk/ewi/insy/DBL/nbrouwer1/perturbation_response_model/DATA/sciplex_raw"),
)
RAW_H5AD_CANDIDATES = tuple(
    raw_dir / RAW_H5AD_NAME for raw_dir in EXTERNAL_SCIPLEX_RAW_DIRS
) + (Path(data_dir) / "sciplex_raw" / RAW_H5AD_NAME,)
PROCESSED_H5AD = Path(data_dir) / "sciplex_processed" / "Srivatsan_2019_raw_processed.h5ad"
DRUG_TO_PERTURBATION_CSV = Path(resources_dir) / "sciplex_drug_to_perturbation.csv"
SCIPLEX_SENSITIVITY_CSV = Path(resources_dir) / "sciplex_sensitivity_info.csv"
MCFARLAND_SENSITIVITY_CSV = Path(resources_dir) / "mcfarland_sensitivity_info.csv"
SENSITIVITY_COLUMNS = ("sens", "sens_label")
SCIPLEX_SENSITIVITY_THRESHOLD = 0.2
SPLIT_COLUMNS = [f"split_{i}" for i in range(config.DEFAULT_N_OUTER_SPLITS)]
ANNOTATION_COLUMNS = ("gene_target", "perturbation_match", "fold", *SPLIT_COLUMNS)
STRATUM_COLUMNS = ("cell_type",)
CONTROL_NAME_VALUES = {"vehicle", "control", "ctrl", "dmso"}


def canonicalize_sciplex_obs_columns(obs: pd.DataFrame) -> pd.DataFrame:
    """SciPlex: ``product_name`` → ``condition`` (drug); old ``condition`` → ``gene_target``.

    No-op when ``product_name`` is absent (already canonical, or McFarland).
    """
    if "product_name" not in obs.columns:
        return obs
    out = obs.copy()
    if "condition" in out.columns and "gene_target" not in out.columns:
        out = out.rename(columns={"condition": "gene_target"})
    elif "condition" in out.columns and "gene_target" in out.columns:
        out = out.drop(columns=["condition"])
    out = out.rename(columns={"product_name": "condition"})
    return out


def resolve_obs_column(obs: pd.DataFrame, canonical: str) -> str | None:
    aliases = {
        "cell_type": ("cell_type", "cell_line"),
        # Drug / product label before SciPlex rename. After canonicalize, use
        # ``condition`` via ``resolve_sciplex_drug_column``.
        "product_name": ("product_name", "perturbation", "drug"),
        "gene_target": ("gene_target",),
        "dose": ("dose", "dose_value"),
    }
    return next((col for col in aliases[canonical] if col in obs.columns), None)


def resolve_sciplex_drug_column(obs: pd.DataFrame) -> str | None:
    """Column holding the SciPlex drug name (``condition`` after rename, else ``product_name``)."""
    if "product_name" in obs.columns:
        return "product_name"
    if "condition" in obs.columns and (
        "gene_target" in obs.columns or "perturbation_match" in obs.columns
    ):
        return "condition"
    if "condition" in obs.columns and "gene_target" not in obs.columns:
        # Ambiguous if McFarland-like; prefer condition only when product_name gone
        # and this looks like SciPlex (dose / vehicle / replicate present).
        if any(c in obs.columns for c in ("dose", "vehicle", "replicate", "product_dose")):
            return "condition"
    return resolve_obs_column(obs, "product_name")


def resolve_stratum_columns(obs: pd.DataFrame) -> list[str]:
    resolved: list[str] = []
    for canonical in STRATUM_COLUMNS:
        match = resolve_obs_column(obs, canonical)
        if match is None:
            raise KeyError(
                f"Cannot stratify splits: no {canonical} column found in obs. "
                f"Available columns: {list(obs.columns)}"
            )
        resolved.append(match)
    return resolved


def resolve_holdout_columns(obs: pd.DataFrame) -> list[str]:
    """Hold-out unit: ``(cell_type, condition)`` (SciPlex drug or McFarland pert)."""
    cell_col = resolve_obs_column(obs, "cell_type")
    if cell_col is None:
        raise KeyError(
            "Cannot hold out pairs: no cell_type/cell_line column in obs. "
            f"Available columns: {list(obs.columns)}"
        )
    if "condition" not in obs.columns:
        raise KeyError(
            "Cannot hold out pairs: need condition column. "
            f"Available columns: {list(obs.columns)}"
        )
    return [cell_col, "condition"]


def is_control_cell(obs: pd.DataFrame) -> pd.Series:
    """Vehicle / dose-0 / ctrl-named cells. These stay in train for every split."""
    is_control = pd.Series(False, index=obs.index)

    if "vehicle" in obs.columns:
        is_control = is_control | obs["vehicle"].fillna(False).astype(bool)

    dose_col = resolve_obs_column(obs, "dose")
    if dose_col is not None:
        dose = pd.to_numeric(obs[dose_col], errors="coerce")
        is_control = is_control | (dose.fillna(-1) == 0)

    if "condition" in obs.columns:
        condition = obs["condition"].astype(str).str.strip().str.lower()
        is_control = is_control | condition.isin(CONTROL_NAME_VALUES) | condition.eq("ctrl")

    if "gene_target" in obs.columns:
        gene_target = obs["gene_target"].astype(str).str.strip().str.lower()
        is_control = is_control | gene_target.eq("ctrl")

    if "product_name" in obs.columns:
        product = obs["product_name"].astype(str).str.strip().str.lower()
        is_control = is_control | product.isin(CONTROL_NAME_VALUES)

    return is_control.fillna(False)


def sciplex_condition_from_product(obs: pd.DataFrame) -> pd.Series:
    """SciPlex grouping label: drug ``condition``, with vehicles collapsed to ``ctrl``."""
    frame = canonicalize_sciplex_obs_columns(obs)
    if "condition" not in frame.columns:
        raise KeyError("SciPlex obs lacks condition (drug) column")
    labels = frame["condition"].astype(str)
    return labels.mask(is_control_cell(frame), "ctrl").rename("condition")


def control_or_highest_dose_mask(obs: pd.DataFrame) -> tuple[pd.Series, float]:
    """Keep vehicle/control cells and cells at the highest observed dose.

    If there is no dose column (McFarland), keep every cell: the processed file
    is already a single concentration per condition.
    """
    is_control = is_control_cell(obs)
    dose_col = resolve_obs_column(obs, "dose")
    if dose_col is None:
        return pd.Series(True, index=obs.index), float("nan")

    dose = pd.to_numeric(obs[dose_col], errors="coerce")
    max_dose = float(dose.max())
    is_highest_dose = dose == max_dose
    return is_control | is_highest_dose, max_dose


def normalize_product_name(names: pd.Series) -> pd.Series:
    """Strip leading/trailing whitespace and remove all internal spaces for SciPlex drug joins."""
    return names.astype(str).str.strip().str.replace(r"\s+", "", regex=True)


def add_perturbation_match(
    obs: pd.DataFrame,
    mapping_path: Path = DRUG_TO_PERTURBATION_CSV,
) -> pd.DataFrame:
    """Left-join gene targets onto SciPlex drugs; flag whether a match was found."""
    if not mapping_path.exists():
        raise FileNotFoundError(f"Drug-to-perturbation map not found: {mapping_path}")

    obs = canonicalize_sciplex_obs_columns(obs)
    drug_col = resolve_sciplex_drug_column(obs)
    if drug_col is None:
        raise KeyError(
            f"Cannot match perturbations: no SciPlex drug column found in obs. "
            f"Available columns: {list(obs.columns)}"
        )

    mapping = pd.read_csv(mapping_path, index_col=0)
    mapping["product_name_key"] = normalize_product_name(mapping["product_name"])
    mapping = mapping.drop_duplicates(subset="product_name_key", keep="first")
    target_by_key = mapping.set_index("product_name_key")["target"]

    obs = obs.copy()
    product_keys = normalize_product_name(obs[drug_col])
    obs["gene_target"] = product_keys.map(target_by_key)
    obs["perturbation_match"] = obs["gene_target"].notna()

    n_matched_cells = int(obs["perturbation_match"].sum())
    n_products = int(product_keys.nunique())
    n_matched_products = int(product_keys.loc[obs["perturbation_match"]].nunique())
    logger.info(
        "Perturbation match: %d/%d cells, %d/%d product names",
        n_matched_cells,
        len(obs),
        n_matched_products,
        n_products,
    )
    return obs


def _cell_line_key(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.upper().str.split("_", n=1).str[0]


def merge_sensitivity_into_obs(obs: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Left-join ``sens`` / ``sens_label`` from resources onto cell-level obs."""
    obs = obs.copy().drop(columns=[c for c in SENSITIVITY_COLUMNS if c in obs.columns], errors="ignore")
    if "cell_type" not in obs.columns:
        raise KeyError("obs has no cell_type column for sensitivity merge")
    if "condition" not in obs.columns:
        raise KeyError("obs has no condition column for sensitivity merge")

    if dataset == "mcfarland":
        path = MCFARLAND_SENSITIVITY_CSV
        if not path.exists():
            raise FileNotFoundError(f"McFarland sensitivity table not found: {path}")
        sens = pd.read_csv(path)
        sens["sens"] = pd.to_numeric(sens["sens"], errors="coerce")
        if "sens_label" not in sens.columns:
            sens["sens_label"] = (sens["sens"] >= SCIPLEX_SENSITIVITY_THRESHOLD).astype("Int64")
        sens["cond_key"] = sens["target"].astype(str).str.strip()
    elif dataset == "sciplex":
        path = SCIPLEX_SENSITIVITY_CSV
        if not path.exists():
            raise FileNotFoundError(f"SciPlex sensitivity table not found: {path}")
        sens = pd.read_csv(path)
        if "sens" not in sens.columns:
            sens["sens"] = pd.to_numeric(sens["y"], errors="coerce")
        else:
            sens["sens"] = pd.to_numeric(sens["sens"], errors="coerce")
        if "sens_label" not in sens.columns:
            sens["sens_label"] = (sens["sens"] >= SCIPLEX_SENSITIVITY_THRESHOLD).astype("Int64")
        # Join on drug name (obs.condition after SciPlex rename).
        drug_col = "product_name" if "product_name" in sens.columns else "condition"
        sens["cond_key"] = normalize_product_name(sens[drug_col])
    else:
        raise ValueError(f"Unknown dataset for sensitivity merge: {dataset}")

    sens["cell_key"] = _cell_line_key(sens["cell_line"])
    right = (
        sens[["cell_key", "cond_key", "sens", "sens_label"]]
        .drop_duplicates(subset=["cell_key", "cond_key"], keep="first")
    )
    obs["_cell_key"] = _cell_line_key(obs["cell_type"])
    if dataset == "sciplex":
        obs["_cond_key"] = normalize_product_name(obs["condition"])
    else:
        obs["_cond_key"] = obs["condition"].astype(str).str.strip()
    n_before = len(obs)
    index = obs.index
    index_name = obs.index.name
    merged = obs.merge(right, left_on=["_cell_key", "_cond_key"], right_on=["cell_key", "cond_key"], how="left")
    if len(merged) != n_before:
        raise RuntimeError(
            f"Sensitivity merge changed row count ({n_before} -> {len(merged)}); check duplicate keys"
        )
    merged.index = index
    merged.index.name = index_name
    merged = merged.drop(columns=["_cell_key", "_cond_key", "cell_key", "cond_key"])
    control = is_control_cell(merged)
    merged.loc[control, "sens"] = 0.0
    merged.loc[control, "sens_label"] = 0
    treated = ~control
    n_matched = int(merged["sens"].notna().sum())
    n_treated = int(treated.sum())
    n_treated_matched = int(merged.loc[treated, "sens"].notna().sum())
    n_control = int(control.sum())
    logger.info(
        "Sensitivity merge (%s): %d/%d cells with sens (%d control set to 0; %d/%d treated matched)",
        dataset,
        n_matched,
        len(merged),
        n_control,
        n_treated_matched,
        n_treated,
    )
    return merged


def assign_stratified_folds(
    obs: pd.DataFrame,
    stratum_cols: list[str],
    n_splits: int,
    random_state: int,
    control_mask: pd.Series | None = None,
    holdout_cols: list[str] | None = None,
) -> np.ndarray:
    """Assign each treated hold-out pair to one test fold.

    SciPlex: ``(cell_type, condition)`` where condition is the drug.
    McFarland: ``(cell_type, condition)`` where condition is the perturbation.
    Within each stratum, unique pairs are shuffled and assigned round-robin so
    every cell of a pair shares the same fold. Controls stay at -1 and are
    train in every split.
    """
    rng = np.random.default_rng(random_state)
    fold_id = np.full(len(obs), fill_value=-1, dtype=np.int8)
    if control_mask is None:
        treated_pos = np.arange(len(obs))
        treated_obs = obs
    else:
        treated_pos = np.flatnonzero(~control_mask.to_numpy())
        treated_obs = obs.iloc[treated_pos]
    if holdout_cols is None:
        holdout_cols = resolve_holdout_columns(obs)

    grouped = treated_obs.groupby(list(stratum_cols), observed=True, sort=False)
    for _, sub_idx in grouped.indices.items():
        sub_idx = np.asarray(sub_idx)
        sub = treated_obs.iloc[sub_idx]
        pair_id = (
            sub.groupby(list(holdout_cols), sort=False, observed=True, dropna=False)
            .ngroup()
            .to_numpy()
        )
        pair_id = np.asarray(pair_id, dtype=float)
        nan_mask = np.isnan(pair_id)
        if nan_mask.any():
            fill = (np.nanmax(pair_id) + 1) if np.any(~nan_mask) else 0.0
            pair_id = pair_id.copy()
            pair_id[nan_mask] = fill
        pair_id = pair_id.astype(np.int64)
        unique_pairs = np.unique(pair_id)
        rng.shuffle(unique_pairs)
        pair_to_fold = {int(pid): i % n_splits for i, pid in enumerate(unique_pairs)}
        folds = np.fromiter(
            (pair_to_fold[int(p)] for p in pair_id),
            dtype=np.int8,
            count=len(pair_id),
        )
        fold_id[treated_pos[sub_idx]] = folds

    assigned = fold_id[treated_pos]
    if len(assigned) and np.any(assigned < 0):
        raise RuntimeError("Failed to assign a fold to every treated cell")
    return fold_id


def add_split_columns(
    obs: pd.DataFrame,
    n_splits: int = config.DEFAULT_N_OUTER_SPLITS,
    random_state: int = config.RANDOM_STATE,
    stratum_cols: list[str] | None = None,
) -> pd.DataFrame:
    obs = obs.copy()
    if stratum_cols is None:
        stratum_cols = resolve_stratum_columns(obs)
    else:
        missing = [col for col in stratum_cols if col not in obs.columns]
        if missing:
            raise KeyError(f"Missing stratum columns: {missing}")
    holdout_cols = resolve_holdout_columns(obs)
    control_mask = is_control_cell(obs)
    n_control = int(control_mask.sum())
    logger.info(
        "Group-aware 80/20 splits within %s; hold-out unit %s; %d control cells always in train",
        stratum_cols,
        holdout_cols,
        n_control,
    )

    fold_id = assign_stratified_folds(
        obs,
        stratum_cols,
        n_splits,
        random_state,
        control_mask=control_mask,
        holdout_cols=holdout_cols,
    )
    obs["fold"] = fold_id

    split_dtype = pd.CategoricalDtype(categories=["train", "test"])
    for split_idx, col in enumerate(SPLIT_COLUMNS[:n_splits]):
        labels = np.where(fold_id == split_idx, "test", "train")
        obs[col] = pd.Categorical(labels, dtype=split_dtype)
        n_test = int((fold_id == split_idx).sum())
        logger.info("%s: train=%d test=%d", col, len(obs) - n_test, n_test)

    return obs


def log_split_balance(obs: pd.DataFrame, stratum_cols: list[str], n_splits: int) -> None:
    holdout_cols = resolve_holdout_columns(obs)
    groups = obs.groupby(list(stratum_cols), observed=True)
    treated = obs.loc[obs["fold"] >= 0]
    n_pairs = treated.groupby(list(holdout_cols), observed=True, dropna=False).ngroups
    logger.info(
        "%d strata (%s), %d treated %s pairs, %d cells",
        groups.ngroups,
        stratum_cols,
        n_pairs,
        tuple(holdout_cols),
        len(obs),
    )

    pair_n_folds = treated.groupby(list(holdout_cols), observed=True, dropna=False)["fold"].nunique()
    leaked = int((pair_n_folds > 1).sum())
    if leaked:
        raise RuntimeError(f"{leaked} {tuple(holdout_cols)} pairs span more than one fold")

    for split_idx in range(n_splits):
        test_mask = obs["fold"] == split_idx
        n_test = int(test_mask.sum())
        n_treated = int((obs["fold"] >= 0).sum())
        frac = n_test / n_treated if n_treated else float("nan")
        test_treated = treated.loc[treated["fold"] == split_idx]
        train_treated = treated.loc[treated["fold"] != split_idx]
        n_test_pairs = test_treated.groupby(list(holdout_cols), observed=True, dropna=False).ngroups
        n_strata_with_test = test_treated.groupby(list(stratum_cols), observed=True).ngroups
        overlap = (
            train_treated[list(holdout_cols)]
            .drop_duplicates()
            .merge(test_treated[list(holdout_cols)].drop_duplicates(), on=list(holdout_cols))
        )
        if len(overlap):
            raise RuntimeError(
                f"split_{split_idx}: {len(overlap)} {tuple(holdout_cols)} pairs "
                "are in both train and test"
            )
        logger.info(
            "split_%d: test=%d cells (%.1f%% of treated), %d/%d pairs, %d/%d strata in test",
            split_idx,
            n_test,
            100 * frac,
            n_test_pairs,
            n_pairs,
            n_strata_with_test,
            groups.ngroups,
        )


def write_h5ad_with_obs_csv(adata: sc.AnnData, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.stem + ".tmp.h5ad")
    obs_csv_path = path.with_name(path.stem + "_obs.csv")
    logger.info("Writing dataset with splits to %s", path)
    adata.write(tmp_path)
    tmp_path.replace(path)
    adata.obs.to_csv(obs_csv_path)
    logger.info("Wrote obs CSV to %s", obs_csv_path)
    return obs_csv_path


def refresh_sensitivity_on_h5ad(
    path: Path,
    dataset: str = "sciplex",
    *,
    merge_sensitivity: bool = True,
) -> pd.DataFrame:
    """Canonicalize SciPlex obs and re-merge sensitivity; fold/split columns stay unchanged."""
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    logger.info(
        "Refreshing SciPlex obs on %s (splits unchanged%s)",
        path,
        ", sensitivity re-merged" if merge_sensitivity else "",
    )
    adata = sc.read_h5ad(path)
    split_cols = [c for c in ("fold", *SPLIT_COLUMNS) if c in adata.obs.columns]
    splits_before = adata.obs[split_cols].copy() if split_cols else None

    adata.obs = canonicalize_sciplex_obs_columns(adata.obs)
    if dataset == "sciplex" and "condition" in adata.obs.columns:
        adata.obs["condition"] = normalize_product_name(adata.obs["condition"])
    if merge_sensitivity:
        adata.obs = merge_sensitivity_into_obs(adata.obs, dataset)

    if splits_before is not None and not adata.obs[split_cols].equals(splits_before):
        raise RuntimeError(f"Split columns changed unexpectedly on {path}")

    write_h5ad_with_obs_csv(adata, path)
    return adata.obs


def refresh_splits_on_h5ad(
    path: Path,
    n_splits: int = config.DEFAULT_N_OUTER_SPLITS,
    random_state: int = config.RANDOM_STATE,
    stratum_cols: list[str] | None = None,
    dataset: str | None = None,
) -> pd.DataFrame:
    """Reassign fold/split columns on an existing processed h5ad; leave X unchanged."""
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    logger.info("Refreshing split columns on %s (expression unchanged)", path)
    adata = sc.read_h5ad(path)
    adata.obs = canonicalize_sciplex_obs_columns(adata.obs)
    drop = [col for col in ("fold", *SPLIT_COLUMNS) if col in adata.obs.columns]
    if drop:
        adata.obs = adata.obs.drop(columns=drop)
    if stratum_cols is None:
        stratum_cols = resolve_stratum_columns(adata.obs)
    adata.obs = add_split_columns(
        adata.obs, n_splits=n_splits, random_state=random_state, stratum_cols=stratum_cols
    )
    log_split_balance(adata.obs, stratum_cols, n_splits)
    if dataset is not None:
        adata.obs = merge_sensitivity_into_obs(adata.obs, dataset)
    write_h5ad_with_obs_csv(adata, path)
    return adata.obs


def resolve_raw_h5ad(path: Path | None = None) -> Path:
    if path is not None:
        if path.exists():
            return path
        raise FileNotFoundError(f"SciPlex raw dataset not found: {path}")
    for candidate in RAW_H5AD_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "SciPlex raw dataset not found. Tried: "
        + ", ".join(str(p) for p in RAW_H5AD_CANDIDATES)
    )


def create_sciplex_splits(
    path: Path | None = None,
    n_splits: int = config.DEFAULT_N_OUTER_SPLITS,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    path = resolve_raw_h5ad(path)

    logger.info("Reading %s", path)
    adata_backed = sc.read_h5ad(path, backed="r")
    keep, max_dose = control_or_highest_dose_mask(adata_backed.obs)
    n_keep = int(keep.sum())
    logger.info(
        "Filtering to control or highest dose (%.0f): keeping %d/%d cells",
        max_dose,
        n_keep,
        adata_backed.n_obs,
    )
    if n_keep == 0:
        adata_backed.file.close()
        raise RuntimeError("No cells left after control/highest-dose filter")

    adata = adata_backed[keep.to_numpy()].to_memory()
    adata_backed.file.close()

    drop_cols = [col for col in ANNOTATION_COLUMNS if col in adata.obs.columns]
    if drop_cols:
        logger.info("Replacing existing annotation columns: %s", drop_cols)
        adata.obs = adata.obs.drop(columns=drop_cols)

    adata.obs = canonicalize_sciplex_obs_columns(adata.obs)
    adata = store_counts_layer(adata)
    adata = process_merged_df(adata)

    adata.obs = add_perturbation_match(adata.obs)
    adata.obs = add_split_columns(adata.obs, n_splits=n_splits, random_state=random_state)
    log_split_balance(adata.obs, resolve_stratum_columns(adata.obs), n_splits)
    adata = normalize_log1p_and_align_genes(adata)
    adata = annotate_dataset_qc(
        adata,
        dataset="sciplex",
        group_cols=["cell_type", "condition"],
        out_dir=Path(config.RESULTS_01_DIR),
    )
    adata.obs = merge_sensitivity_into_obs(adata.obs, "sciplex")

    write_h5ad_with_obs_csv(adata, PROCESSED_H5AD)
    return adata.obs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-splits-only",
        action="store_true",
        help="Reassign fold/split columns on the existing processed h5ad; skip QC and normalize.",
    )
    parser.add_argument(
        "--refresh-sensitivity-only",
        action="store_true",
        help="Re-merge sensitivity on existing processed h5ad(s); leave fold/split unchanged.",
    )
    args = parser.parse_args()
    log_script_start(__file__, logger)
    try:
        if args.refresh_sensitivity_only:
            refresh_sensitivity_on_h5ad(PROCESSED_H5AD, dataset="sciplex")
            for cell_line in ("mcf7", "a549", "k562"):
                per_line = PROCESSED_H5AD.parent / f"sciplex{cell_line}.h5ad"
                if per_line.exists():
                    refresh_sensitivity_on_h5ad(per_line, dataset="sciplex")
            logger.info("SciPlex processed h5ad sensitivity merge refreshed (splits unchanged)")
        elif args.refresh_splits_only:
            refresh_splits_on_h5ad(PROCESSED_H5AD, dataset="sciplex")
            logger.info("SciPlex fold/split columns refreshed")
        else:
            create_sciplex_splits()
            logger.info("Filtered SciPlex dataset written with train/test splits")
    except Exception as e:
        logger.error("Error creating SciPlex splits: %s", e)
        raise
    log_script_end(__file__, logger)
