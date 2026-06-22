import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config, setup_project
from utils import (
    ensure_directories_exist,
    log_script_end,
    log_script_start,
    setup_logging_for_script,
)
from compute_de_genes import main as compute_de_genes_main
from systema_reference import (
    leave_one_split_reference_vectors,
    leave_one_split_reference_vectors_per_cell_line,
    observation_profiles_with_split,
    reference_vectors_per_cell_line_non_control,
    resolve_cell_line_column,
    resolve_split_column,
)

setup_project()
logger = setup_logging_for_script(__file__)

home_dir = str(config.PROJECT_ROOT)
data_dir = str(config.DATA_DIR)
resources_dir = str(config.RESOURCES_DIR)
results_dir = str(config.RESULTS_03_DIR)
figures_dir = str(config.FIGURES_03_DIR)

ensure_directories_exist(home_dir, data_dir, resources_dir, results_dir, figures_dir)

CONTROL_CONDITION = "ctrl"
SCIPLEX_KEY_COLS = ("cell_type", "condition")


def _get_common_gene_columns(
    predictions: pd.DataFrame, observations: pd.DataFrame, excluded_cols: Iterable[str] = ("condition", "cell_type")
) -> list[str]:
    excluded = set(excluded_cols)
    common_columns = [c for c in predictions.columns if c in observations.columns and c not in excluded]
    if not common_columns:
        raise ValueError("No common gene columns found between predictions and observations.")
    return common_columns


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    val = pearsonr(x, y)[0]
    if np.isnan(val):
        return 0.0
    return float(val)


def _systema_reference_all_non_control(
    observations: pd.DataFrame, gene_columns: list[str]
) -> pd.Series:
    """Fallback when profiles lack fold/split labels (current SciPlex pseudobulks)."""
    observations_no_ctrl = observations[observations["condition"] != CONTROL_CONDITION]
    if observations_no_ctrl.empty:
        raise ValueError("No non-control observations available to compute Systema reference vector.")
    return observations_no_ctrl[gene_columns].mean(axis=0)


def compute_metrics_with_systema(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
    de_genes: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    metrics: dict[str, list[float]] = {
        "mse": [],
        "pearson": [],
        "pearson_systema": [],
        "pearson_systema_cellline": [],
        "mse_de": [],
        "pearson_de": [],
        "pearson_systema_de": [],
        "pearson_systema_cellline_de": [],
    }
    metrics_pert: dict[str, dict[str, float]] = {}

    gene_columns = _get_common_gene_columns(predictions, observations)
    split_col = resolve_split_column(predictions)
    cell_line_col = resolve_cell_line_column(observations)
    if split_col is not None:
        ref_profiles = observation_profiles_with_split(
            observations, predictions, key_cols=SCIPLEX_KEY_COLS
        )
        reference_by_split = leave_one_split_reference_vectors(
            ref_profiles, gene_columns, split_col
        )
        reference_by_split_cellline = leave_one_split_reference_vectors_per_cell_line(
            ref_profiles, gene_columns, split_col, cell_line_col=cell_line_col
        )
        reference_by_cellline = None
        reference_vector = None
    else:
        reference_by_split = None
        reference_by_split_cellline = None
        reference_vector = _systema_reference_all_non_control(observations, gene_columns)
        reference_by_cellline = reference_vectors_per_cell_line_non_control(
            observations, gene_columns, cell_line_col=cell_line_col
        )

    for pert in np.unique(predictions["condition"]):
        predicted_rows = predictions[predictions["condition"] == pert]
        observed_rows = observations[observations["condition"] == pert]
        if predicted_rows.empty or observed_rows.empty:
            continue

        predicted_expression = predicted_rows[gene_columns].iloc[0].to_numpy(dtype=float)
        true_expression = observed_rows[gene_columns].iloc[0].to_numpy(dtype=float)
        cell_line_value = str(predicted_rows[cell_line_col].iloc[0])

        if reference_by_split is not None:
            split_value = int(predicted_rows[split_col].iloc[0])
            reference_expression = reference_by_split[split_value].to_numpy(dtype=float)
            ref_key = (split_value, cell_line_value)
            ref_expr_cl = (
                reference_by_split_cellline[ref_key].to_numpy(dtype=float)
                if ref_key in reference_by_split_cellline
                else None
            )
        else:
            reference_expression = reference_vector.to_numpy(dtype=float)
            ref_expr_cl = (
                reference_by_cellline[cell_line_value].to_numpy(dtype=float)
                if cell_line_value in reference_by_cellline
                else None
            )

        centered_pred = predicted_expression - reference_expression
        centered_true = true_expression - reference_expression

        metrics_pert[pert] = {
            "mse": float(mse(predicted_expression, true_expression)),
            "pearson": _safe_pearson(predicted_expression, true_expression),
            "pearson_systema": _safe_pearson(centered_pred, centered_true),
            "pearson_systema_cellline": (
                _safe_pearson(
                    predicted_expression - ref_expr_cl,
                    true_expression - ref_expr_cl,
                )
                if ref_expr_cl is not None
                else 0.0
            ),
        }
        metrics["mse"].append(metrics_pert[pert]["mse"])
        metrics["pearson"].append(metrics_pert[pert]["pearson"])
        metrics["pearson_systema"].append(metrics_pert[pert]["pearson_systema"])
        metrics["pearson_systema_cellline"].append(metrics_pert[pert]["pearson_systema_cellline"])

        if pert == CONTROL_CONDITION:
            metrics_pert[pert]["mse_de"] = 0.0
            metrics_pert[pert]["pearson_de"] = 0.0
            metrics_pert[pert]["pearson_systema_de"] = 0.0
            metrics_pert[pert]["pearson_systema_cellline_de"] = 0.0
            continue

        de_gene_subset = (
            de_genes[de_genes["condition"] == pert]
            .drop(columns=["cell_type", "condition"], errors="ignore")
            .iloc[0, 0:20]
            .tolist()
        )
        de_gene_subset = [g for g in de_gene_subset if g in gene_columns]
        if not de_gene_subset:
            metrics_pert[pert]["mse_de"] = 0.0
            metrics_pert[pert]["pearson_de"] = 0.0
            metrics_pert[pert]["pearson_systema_de"] = 0.0
            metrics_pert[pert]["pearson_systema_cellline_de"] = 0.0
            continue

        pred_de = predicted_rows[de_gene_subset].iloc[0].to_numpy(dtype=float)
        true_de = observed_rows[de_gene_subset].iloc[0].to_numpy(dtype=float)
        if reference_by_split is not None:
            ref_de = reference_by_split[split_value][de_gene_subset].to_numpy(dtype=float)
            ref_de_cl = (
                reference_by_split_cellline[ref_key][de_gene_subset].to_numpy(dtype=float)
                if ref_expr_cl is not None
                else None
            )
        else:
            ref_de = reference_vector[de_gene_subset].to_numpy(dtype=float)
            ref_de_cl = (
                reference_by_cellline[cell_line_value][de_gene_subset].to_numpy(dtype=float)
                if ref_expr_cl is not None
                else None
            )

        centered_pred_de = pred_de - ref_de
        centered_true_de = true_de - ref_de

        metrics_pert[pert]["mse_de"] = float(mse(pred_de, true_de))
        metrics_pert[pert]["pearson_de"] = _safe_pearson(pred_de, true_de)
        metrics_pert[pert]["pearson_systema_de"] = _safe_pearson(centered_pred_de, centered_true_de)
        metrics_pert[pert]["pearson_systema_cellline_de"] = (
            _safe_pearson(pred_de - ref_de_cl, true_de - ref_de_cl)
            if ref_de_cl is not None
            else 0.0
        )

        metrics["mse_de"].append(metrics_pert[pert]["mse_de"])
        metrics["pearson_de"].append(metrics_pert[pert]["pearson_de"])
        metrics["pearson_systema_de"].append(metrics_pert[pert]["pearson_systema_de"])
        metrics["pearson_systema_cellline_de"].append(
            metrics_pert[pert]["pearson_systema_cellline_de"]
        )

    metrics_mean = {
        metric_name: float(np.mean(values)) if values else 0.0 for metric_name, values in metrics.items()
    }
    return metrics_mean, metrics_pert


def _load_de_genes_for_cell_line(cell_line: str) -> pd.DataFrame:
    de_genes = pd.read_csv(os.path.join(results_dir, f"sciplex{cell_line}_de_genes.csv"))
    pert_to_drug = pd.read_csv(os.path.join(resources_dir, "sciplex_drug_to_perturbation.csv"), index_col=0)
    pert_to_drug["product_name"] = pert_to_drug["product_name"].str.replace(" ", "")

    de_genes = (
        pd.merge(de_genes, pert_to_drug, left_on=["condition"], right_on=["product_name"], how="left")
        .drop(columns=["product_name", "condition"])
        .rename(columns={"target": "condition"})
    )
    de_genes["cell_type"] = cell_line.upper()
    return de_genes


def _write_perturbation_metrics(
    pert_metrics: dict[str, dict[str, float]], cell_line: str, model_name: str, output_name: str
) -> None:
    pert_metrics_df = pd.DataFrame.from_dict(pert_metrics, orient="index")
    pert_metrics_df["condition"] = pert_metrics_df.index
    pert_metrics_df = pert_metrics_df.melt(id_vars="condition", var_name="metric", value_name="value")
    pert_metrics_df["cell_type"] = cell_line.upper()
    pert_metrics_df["model"] = model_name
    pert_metrics_df.to_csv(os.path.join(results_dir, output_name), index=False)


def _evaluate_and_save(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
    de_genes: pd.DataFrame,
    cell_line: str,
    model_name: str,
    output_suffix: str,
) -> None:
    _, pert_metrics = compute_metrics_with_systema(
        predictions=predictions, observations=observations, de_genes=de_genes
    )
    _write_perturbation_metrics(
        pert_metrics=pert_metrics,
        cell_line=cell_line,
        model_name=model_name,
        output_name=f"sciplex{cell_line}_{output_suffix}_systema_outcomes.csv",
    )


def evaluate_average_effect_predictions() -> None:
    for cell_line in ["mcf7", "a549", "k562"]:
        predictions = pd.read_csv(
            os.path.join(data_dir, "average_effect_predictions", f"sciplex_mean_post_{cell_line}.csv")
        )
        observations = pd.read_csv(
            os.path.join(data_dir, "observed_pseudobulk", f"sciplex_mean_post_{cell_line}.csv"), index_col=0
        )
        de_genes = _load_de_genes_for_cell_line(cell_line=cell_line)
        _evaluate_and_save(predictions, observations, de_genes, cell_line, "Average effect", "average_effect")


def evaluate_no_effect_predictions() -> None:
    for cell_line in ["mcf7", "a549", "k562"]:
        predictions = pd.read_csv(os.path.join(data_dir, "no_effect_predictions", f"sciplex_mean_post_{cell_line}.csv"))
        observations = pd.read_csv(
            os.path.join(data_dir, "observed_pseudobulk", f"sciplex_mean_post_{cell_line}.csv"), index_col=0
        )
        de_genes = _load_de_genes_for_cell_line(cell_line=cell_line)
        _evaluate_and_save(predictions, observations, de_genes, cell_line, "No effect", "no_effect")


def evaluate_gears_predictions() -> None:
    for cell_line in ["mcf7", "a549", "k562"]:
        predictions = pd.read_csv(os.path.join(data_dir, "GEARS_predictions", f"sciplex_mean_post_{cell_line}.csv"), index_col=0)
        predictions = predictions.rename(columns={"perturbation": "condition"})
        predictions = predictions[predictions["condition"] != CONTROL_CONDITION]
        observations = pd.read_csv(
            os.path.join(data_dir, "observed_pseudobulk", f"sciplex_mean_post_{cell_line}.csv"), index_col=0
        )
        de_genes = _load_de_genes_for_cell_line(cell_line=cell_line)
        _evaluate_and_save(predictions, observations, de_genes, cell_line, "GEARS", "GEARS")


def evaluate_gears_noreg_predictions() -> None:
    for cell_line in ["mcf7", "a549", "k562"]:
        predictions = pd.read_csv(
            os.path.join(data_dir, "GEARS_noreg_predictions", f"sciplex_mean_post_{cell_line}.csv"), index_col=0
        )
        predictions = predictions.rename(columns={"perturbation": "condition"})
        predictions = predictions[predictions["condition"] != CONTROL_CONDITION]
        observations = pd.read_csv(
            os.path.join(data_dir, "observed_pseudobulk", f"sciplex_mean_post_{cell_line}.csv"), index_col=0
        )
        de_genes = _load_de_genes_for_cell_line(cell_line=cell_line)
        _evaluate_and_save(predictions, observations, de_genes, cell_line, "GEARS_noreg", "GEARS_noreg")


def evaluate_scfoundation_predictions() -> None:
    for cell_line in ["mcf7", "a549", "k562"]:
        predictions = pd.read_csv(
            os.path.join(data_dir, "scfoundation_predictions", f"sciplex_mean_post_{cell_line}.csv"), index_col=0
        )
        predictions = predictions.rename(columns={"perturbation": "condition"})
        predictions = predictions[predictions["condition"] != CONTROL_CONDITION]
        observations = pd.read_csv(
            os.path.join(data_dir, "observed_pseudobulk", f"sciplex_mean_post_{cell_line}.csv"), index_col=0
        )
        de_genes = _load_de_genes_for_cell_line(cell_line=cell_line)
        _evaluate_and_save(predictions, observations, de_genes, cell_line, "scfoundation", "scfoundation")


def evaluate_cpa_predictions() -> None:
    for cell_line in ["mcf7", "a549", "k562"]:
        predictions = pd.read_csv(os.path.join(data_dir, "CPA_predictions", f"sciplex_mean_post_{cell_line}.csv"))
        predictions = predictions[predictions["condition"] != CONTROL_CONDITION]
        observations = pd.read_csv(
            os.path.join(data_dir, "observed_pseudobulk", f"sciplex_mean_post_{cell_line}.csv"), index_col=0
        )
        de_genes = _load_de_genes_for_cell_line(cell_line=cell_line)
        _evaluate_and_save(predictions, observations, de_genes, cell_line, "CPA", "CPA")


def main() -> None:
    log_script_start(__file__, logger)
    try:
        for cell_line in ["mcf7", "a549", "k562"]:
            pd.read_csv(os.path.join(results_dir, f"sciplex{cell_line}_de_genes.csv"))
    except Exception:
        logger.info("DE genes have not been computed yet")
        logger.info("Computing DE genes...")
        compute_de_genes_main()

    try:
        logger.info("Evaluating average effect predictions (Systema metrics)...")
        evaluate_average_effect_predictions()
        logger.info("Evaluating no effect predictions (Systema metrics)...")
        evaluate_no_effect_predictions()
        logger.info("Evaluating GEARS predictions (Systema metrics)...")
        evaluate_gears_predictions()
        logger.info("Evaluating GEARS noreg predictions (Systema metrics)...")
        evaluate_gears_noreg_predictions()
        logger.info("Evaluating scfoundation predictions (Systema metrics)...")
        evaluate_scfoundation_predictions()
        logger.info("Evaluating CPA predictions (Systema metrics)...")
        evaluate_cpa_predictions()
        logger.info("Systema evaluation completed successfully")
    except Exception as exc:
        logger.error(f"Error in evaluate_predictions_sciplex_systema: {str(exc)}")
        raise
    log_script_end(__file__, logger)


if __name__ == "__main__":
    main()
