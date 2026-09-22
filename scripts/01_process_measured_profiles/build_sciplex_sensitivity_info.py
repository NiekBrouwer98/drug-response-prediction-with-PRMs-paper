"""Build ``resources/sciplex_sensitivity_info.csv`` from SciPlex3 viability mapping.

Uses ``sciplex3_response_variable_mapping.csv`` (all drug–cell–dose rows with
viability). Keeps the highest dose (10 µM) to match SciPlex scRNA processing.
Sensitivity is ``1 - viability``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config, setup_project
from create_sciplex_splits import normalize_product_name
from utils import log_script_end, log_script_start, setup_logging_for_script

setup_project()
logger = setup_logging_for_script(__file__)

RESOURCES = Path(config.RESOURCES_DIR)
MAPPING_CSV = RESOURCES / "sciplex3_response_variable_mapping.csv"
DRUG_TO_PERT_CSV = RESOURCES / "sciplex_drug_to_perturbation.csv"
OUTPUT_CSV = RESOURCES / "sciplex_sensitivity_info.csv"
HIGHEST_DOSE_UM = 10.0


def build_sciplex_sensitivity_info() -> pd.DataFrame:
    if not MAPPING_CSV.exists():
        raise FileNotFoundError(f"SciPlex3 mapping not found: {MAPPING_CSV}")

    mapping = pd.read_csv(MAPPING_CSV)
    at_max_dose = pd.to_numeric(mapping["dose"], errors="coerce") == HIGHEST_DOSE_UM
    subset = mapping.loc[at_max_dose].copy()
    if subset.empty:
        raise RuntimeError(f"No rows at dose {HIGHEST_DOSE_UM} µM in {MAPPING_CSV}")

    subset["viability"] = pd.to_numeric(subset["viability"], errors="coerce")
    subset["y"] = 1.0 - subset["viability"]
    subset["sens"] = subset["y"]
    subset["cell_line"] = subset["cell_type"].astype(str).str.strip().str.upper()
    subset["product_name"] = subset["product_name"].astype(str).str.strip()
    subset["condition"] = normalize_product_name(subset["product_name"])

    if DRUG_TO_PERT_CSV.exists():
        pert = pd.read_csv(DRUG_TO_PERT_CSV, index_col=0)
        pert["product_name_key"] = normalize_product_name(pert["product_name"])
        pert = pert.drop_duplicates(subset="product_name_key", keep="first")
        target_by_key = pert.set_index("product_name_key")["target"]
        subset["target"] = subset["condition"].map(target_by_key)
        n_missing_target = int(subset["target"].isna().sum())
        if n_missing_target:
            logger.warning(
                "%d/%d rows have no gene target in %s",
                n_missing_target,
                len(subset),
                DRUG_TO_PERT_CSV.name,
            )
    else:
        logger.warning("Missing %s; target column will be empty", DRUG_TO_PERT_CSV)
        subset["target"] = pd.NA

    out = (
        subset[["condition", "y", "sens", "cell_line", "product_name", "target"]]
        .drop_duplicates(subset=["cell_line", "condition"], keep="first")
        .sort_values(["cell_line", "condition"])
        .reset_index(drop=True)
    )
    return out


def main() -> None:
    out = build_sciplex_sensitivity_info()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    logger.info(
        "Wrote %s (%d rows; %d cell lines; sens range %.4f–%.4f)",
        OUTPUT_CSV,
        len(out),
        out["cell_line"].nunique(),
        out["y"].min(),
        out["y"].max(),
    )


if __name__ == "__main__":
    log_script_start(__file__, logger)
    main()
    log_script_end(__file__, logger)
