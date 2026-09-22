"""Annotate McFarland with perturbation matches and 5 train/test splits.

Matches ``condition`` to ``mcfarland_drug_to_perturbation.csv`` targets (DMSO is
``ctrl``). All cells are retained; ``perturbation_match`` flags whether a gene
target was found. Within each tissue, treated ``(cell_type, condition)`` pairs
are partitioned into 5 folds (~80/20 train/test). All cells of a pair share
the same fold, so a pair is never in both train and test. Control cells have
``fold=-1`` and are ``train`` in every split. Control and highest-dose cells
are kept (no dose column here, so that is DMSO/ctrl plus all treated cells).
Expression is library-size normalized, log1p-transformed (skipped if already
logged), and aligned to the 19264-gene ``OS_scRNA_gene_index``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import scanpy as sc

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent))
from config import config, setup_project
from create_sciplex_splits import (
    SPLIT_COLUMNS,
    add_split_columns,
    control_or_highest_dose_mask,
    log_split_balance,
    merge_sensitivity_into_obs,
    refresh_splits_on_h5ad,
    write_h5ad_with_obs_csv,
)
from preprocess_utils import normalize_log1p_and_align_genes
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

MCFARLAND_H5AD = Path(data_dir) / "mcfarland_processed" / "all_cell_lines.h5ad"
DRUG_TO_PERTURBATION_CSV = Path(resources_dir) / "mcfarland_drug_to_perturbation.csv"
STRATUM_COLUMNS = ["tissue"]
DROP_COLUMNS = ("drug", "tissue", "perturbation_match", "fold", *SPLIT_COLUMNS)
CELL_LINE_METADATA = Path(data_dir) / "mcfarland_raw" / "cell_line_features" / "metadata.csv"


def add_perturbation_match(
    obs: pd.DataFrame,
    mapping_path: Path = DRUG_TO_PERTURBATION_CSV,
) -> pd.DataFrame:
    """Flag whether ``condition`` matches a mapped gene target; keep every cell."""
    if not mapping_path.exists():
        raise FileNotFoundError(f"Drug-to-perturbation map not found: {mapping_path}")
    if "condition" not in obs.columns:
        raise KeyError(
            f"Cannot match perturbations: no condition column in obs. "
            f"Available columns: {list(obs.columns)}"
        )

    mapping = pd.read_csv(mapping_path)
    mapping["target_key"] = mapping["target"].astype(str).str.replace(" ", "", regex=False)
    mapping = mapping.drop_duplicates(subset="target_key", keep="first")
    drug_by_target = mapping.set_index("target_key")["drug"]
    valid_targets = set(mapping["target_key"].dropna()) | {"ctrl"}

    obs = obs.copy()
    condition_key = obs["condition"].astype(str).str.replace(" ", "", regex=False)
    obs["drug"] = condition_key.map(drug_by_target)
    obs.loc[condition_key.eq("ctrl"), "drug"] = "DMSO"
    obs["perturbation_match"] = condition_key.isin(valid_targets)

    n_matched_cells = int(obs["perturbation_match"].sum())
    n_conditions = int(condition_key.nunique())
    n_matched_conditions = int(condition_key.loc[obs["perturbation_match"]].nunique())
    logger.info(
        "Perturbation match: %d/%d cells, %d/%d conditions",
        n_matched_cells,
        len(obs),
        n_matched_conditions,
        n_conditions,
    )
    return obs


def add_tissue_column(
    obs: pd.DataFrame,
    metadata_path: Path = CELL_LINE_METADATA,
) -> pd.DataFrame:
    """Map ``cell_type`` to DepMap ``Disease`` tissue labels."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"McFarland cell-line metadata not found: {metadata_path}")
    if "cell_type" not in obs.columns:
        raise KeyError(
            f"Cannot assign tissue: no cell_type column in obs. "
            f"Available columns: {list(obs.columns)}"
        )

    meta = pd.read_csv(metadata_path)
    if "CCLE_ID" not in meta.columns or "Disease" not in meta.columns:
        raise KeyError(
            f"metadata.csv missing CCLE_ID / Disease (got: {list(meta.columns)})"
        )

    ccle = meta["CCLE_ID"].astype(str).str.split("_", n=1, expand=True)
    tissue_map = (
        pd.DataFrame(
            {
                "cell_type": ccle[0].astype(str).str.strip().str.upper(),
                "tissue": meta["Disease"]
                .astype(str)
                .str.strip()
                .str.replace(" ", "_", regex=False)
                .str.upper(),
            }
        )
        .dropna(subset=["cell_type", "tissue"])
        .drop_duplicates(subset="cell_type", keep="first")
        .set_index("cell_type")["tissue"]
    )

    obs = obs.copy()
    cell_keys = obs["cell_type"].astype(str).str.strip().str.upper()
    obs["tissue"] = cell_keys.map(tissue_map)
    missing = obs["tissue"].isna()
    n_missing = int(missing.sum())
    if n_missing:
        obs.loc[missing, "tissue"] = cell_keys.loc[missing]
        logger.info(
            "Tissue map missing for %d/%d cells (%d cell lines); using cell_type as tissue",
            n_missing,
            len(obs),
            int(cell_keys.loc[missing].nunique()),
        )
    logger.info("%d unique tissues", int(obs["tissue"].nunique()))
    return obs


def create_mcfarland_splits(
    path: Path = MCFARLAND_H5AD,
    n_splits: int = config.DEFAULT_N_OUTER_SPLITS,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"McFarland dataset not found: {path}")

    logger.info("Reading %s", path)
    adata = sc.read_h5ad(path)
    keep, max_dose = control_or_highest_dose_mask(adata.obs)
    n_keep = int(keep.sum())
    if pd.isna(max_dose):
        logger.info(
            "No dose column; keeping control and all treated cells: %d/%d",
            n_keep,
            adata.n_obs,
        )
    else:
        logger.info(
            "Filtering to control or highest dose (%.0f): keeping %d/%d cells",
            max_dose,
            n_keep,
            adata.n_obs,
        )
    if n_keep == 0:
        raise RuntimeError("No cells left after control/highest-dose filter")
    adata = adata[keep.to_numpy()].copy()
    drop_cols = [col for col in DROP_COLUMNS if col in adata.obs.columns]
    if drop_cols:
        logger.info("Replacing existing annotation columns: %s", drop_cols)
        adata.obs = adata.obs.drop(columns=drop_cols)

    adata.obs = add_perturbation_match(adata.obs)
    adata.obs = add_tissue_column(adata.obs)
    adata.obs = add_split_columns(
        adata.obs, n_splits=n_splits, random_state=random_state, stratum_cols=STRATUM_COLUMNS
    )
    log_split_balance(adata.obs, STRATUM_COLUMNS, n_splits)
    adata = normalize_log1p_and_align_genes(adata)
    adata = annotate_dataset_qc(
        adata,
        dataset="mcfarland",
        group_cols=["cell_type", "condition"],
        out_dir=Path(config.RESULTS_01_DIR),
        mcfarland_raw_dir=Path(data_dir) / "mcfarland_raw",
    )
    adata.obs = merge_sensitivity_into_obs(adata.obs, "mcfarland")
    write_h5ad_with_obs_csv(adata, path)
    return adata.obs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-splits-only",
        action="store_true",
        help="Reassign fold/split columns on the existing processed h5ad; skip QC and normalize.",
    )
    args = parser.parse_args()
    log_script_start(__file__, logger)
    try:
        if args.refresh_splits_only:
            refresh_splits_on_h5ad(MCFARLAND_H5AD, stratum_cols=STRATUM_COLUMNS, dataset="mcfarland")
            logger.info("McFarland fold/split columns refreshed")
        else:
            create_mcfarland_splits()
            logger.info("McFarland train/test splits written to obs")
    except Exception as e:
        logger.error("Error creating McFarland splits: %s", e)
        raise
    log_script_end(__file__, logger)
