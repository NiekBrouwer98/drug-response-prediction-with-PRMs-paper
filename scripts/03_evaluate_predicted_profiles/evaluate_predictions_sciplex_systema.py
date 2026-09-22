import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse

# Add project root to path for imports
_eval_dir = Path(__file__).parent
sys.path.insert(0, str(_eval_dir))
sys.path.append(str(_eval_dir.parent.parent))
from config import config, setup_project
from utils import (
    ensure_directories_exist,
    log_script_end,
    log_script_start,
    setup_logging_for_script,
)
from compute_de_genes import main as compute_de_genes_main
from mcfarland_profile_metrics import (
    compute_metrics_with_systema_mcfarland,
    load_sciplex_de_genes,
    map_sciplex_conditions_to_gene_targets,
    pair_metrics_to_long_df,
    prepare_predictions_for_systema,
    _get_common_gene_columns,
    _iter_matched_pairs,
    _profiles_for_matching,
)
from prediction_io import (
    SCIPLEX_CELL_LINES,
    align_sciplex_profile_conditions,
    drop_control_rows,
    filter_sciplex_cell_line,
    load_cpa_post,
    load_chemcpa_post,
    load_prnet_post,
    load_sciplex_observed_post,
    load_sciplex_observed_post_all,
)
from systema_reference import (
    reference_vector_all_non_control,
    reference_vectors_per_cell_line_non_control,
    resolve_cell_line_column,
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


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    val = pearsonr(x, y)[0]
    if np.isnan(val):
        return 0.0
    return float(val)


def _load_sciplex_reference_observations() -> pd.DataFrame:
    """Pooled observed post profiles across all SciPlex lines (564 pairs when fully measured)."""
    return align_sciplex_profile_conditions(load_sciplex_observed_post_all(data_dir))


def compute_metrics_with_systema(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
    de_genes: pd.DataFrame,
    *,
    reference_observations: pd.DataFrame | None = None,
    sciplex_normalize_condition: bool = True,
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

    predictions = _profiles_for_matching(
        predictions, sciplex_normalize_condition=sciplex_normalize_condition
    )
    observations = _profiles_for_matching(
        observations, sciplex_normalize_condition=sciplex_normalize_condition
    )
    de_genes = _profiles_for_matching(
        de_genes, sciplex_normalize_condition=sciplex_normalize_condition
    )
    ref_source = reference_observations if reference_observations is not None else observations
    ref_source = _profiles_for_matching(
        ref_source, sciplex_normalize_condition=sciplex_normalize_condition
    )

    gene_columns = _get_common_gene_columns(predictions, observations)
    cell_line_col = resolve_cell_line_column(ref_source)
    reference_vector = reference_vector_all_non_control(
        ref_source, gene_columns, cell_line_col=cell_line_col
    )
    reference_by_cellline = reference_vectors_per_cell_line_non_control(
        ref_source, gene_columns, cell_line_col=cell_line_col
    )

    for cell_line, pert, pred_row, obs_row in _iter_matched_pairs(
        predictions,
        observations,
        sciplex_normalize_condition=False,
    ):
        predicted_expression = pred_row[gene_columns].to_numpy(dtype=float)
        true_expression = obs_row[gene_columns].to_numpy(dtype=float)
        cell_line_value = str(cell_line)
        ref_expr_cl = (
            reference_by_cellline[cell_line_value].to_numpy(dtype=float)
            if cell_line_value in reference_by_cellline
            else None
        )

        centered_pred = predicted_expression - reference_vector.to_numpy(dtype=float)
        centered_true = true_expression - reference_vector.to_numpy(dtype=float)

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

        de_rows = de_genes[
            (de_genes["condition"] == pert)
            & (de_genes[cell_line_col].astype(str).str.upper() == cell_line_value)
        ]
        if de_rows.empty:
            metrics_pert[pert]["mse_de"] = 0.0
            metrics_pert[pert]["pearson_de"] = 0.0
            metrics_pert[pert]["pearson_systema_de"] = 0.0
            metrics_pert[pert]["pearson_systema_cellline_de"] = 0.0
            continue

        de_gene_subset = (
            de_rows.drop(columns=["cell_type", "cell_line", "condition"], errors="ignore")
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

        pred_de = pred_row[de_gene_subset].to_numpy(dtype=float)
        true_de = obs_row[de_gene_subset].to_numpy(dtype=float)
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


def _load_de_genes_for_cell_line(cell_line: str, *, condition_key: str = 'product_name') -> pd.DataFrame:
    return load_sciplex_de_genes(
        results_dir,
        cell_line,
        resources_dir,
        condition_key=condition_key,
    )


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
    *,
    reference_observations: pd.DataFrame,
) -> None:
    predictions = align_sciplex_profile_conditions(predictions)
    observations = align_sciplex_profile_conditions(observations)
    _, pert_metrics = compute_metrics_with_systema(
        predictions=predictions,
        observations=observations,
        de_genes=de_genes,
        reference_observations=reference_observations,
    )
    _write_perturbation_metrics(
        pert_metrics=pert_metrics,
        cell_line=cell_line,
        model_name=model_name,
        output_name=f"sciplex{cell_line}_{output_suffix}_systema_outcomes.csv",
    )


def evaluate_average_effect_predictions() -> None:
    reference_observations = _load_sciplex_reference_observations()
    for cell_line in ["mcf7", "a549", "k562"]:
        predictions = pd.read_csv(
            os.path.join(data_dir, "average_effect_predictions", f"sciplex_mean_post_{cell_line}.csv")
        )
        observations = pd.read_csv(
            os.path.join(data_dir, "observed_pseudobulk", f"sciplex_mean_post_{cell_line}.csv"), index_col=0
        )
        de_genes = _load_de_genes_for_cell_line(cell_line=cell_line)
        _evaluate_and_save(
            predictions,
            observations,
            de_genes,
            cell_line,
            "Average effect",
            "average_effect",
            reference_observations=reference_observations,
        )


def evaluate_no_effect_predictions() -> None:
    reference_observations = _load_sciplex_reference_observations()
    for cell_line in ["mcf7", "a549", "k562"]:
        predictions = pd.read_csv(os.path.join(data_dir, "no_effect_predictions", f"sciplex_mean_post_{cell_line}.csv"))
        observations = pd.read_csv(
            os.path.join(data_dir, "observed_pseudobulk", f"sciplex_mean_post_{cell_line}.csv"), index_col=0
        )
        de_genes = _load_de_genes_for_cell_line(cell_line=cell_line)
        _evaluate_and_save(
            predictions,
            observations,
            de_genes,
            cell_line,
            "No effect",
            "no_effect",
            reference_observations=reference_observations,
        )


def evaluate_gears_predictions() -> None:
    reference_observations = _load_sciplex_reference_observations()
    for cell_line in ["mcf7", "a549", "k562"]:
        predictions = pd.read_csv(os.path.join(data_dir, "GEARS_predictions", f"sciplex_mean_post_{cell_line}.csv"), index_col=0)
        predictions = predictions.rename(columns={"perturbation": "condition"})
        predictions = predictions[predictions["condition"] != CONTROL_CONDITION]
        observations = pd.read_csv(
            os.path.join(data_dir, "observed_pseudobulk", f"sciplex_mean_post_{cell_line}.csv"), index_col=0
        )
        observations = map_sciplex_conditions_to_gene_targets(observations, resources_dir)
        de_genes = _load_de_genes_for_cell_line(cell_line=cell_line, condition_key='gene_target')
        _evaluate_and_save(
            predictions,
            observations,
            de_genes,
            cell_line,
            "GEARS",
            "GEARS",
            reference_observations=reference_observations,
        )


def evaluate_gears_noreg_predictions() -> None:
    reference_observations = _load_sciplex_reference_observations()
    for cell_line in ["mcf7", "a549", "k562"]:
        predictions = pd.read_csv(
            os.path.join(data_dir, "GEARS_noreg_predictions", f"sciplex_mean_post_{cell_line}.csv"), index_col=0
        )
        predictions = predictions.rename(columns={"perturbation": "condition"})
        predictions = predictions[predictions["condition"] != CONTROL_CONDITION]
        observations = pd.read_csv(
            os.path.join(data_dir, "observed_pseudobulk", f"sciplex_mean_post_{cell_line}.csv"), index_col=0
        )
        observations = map_sciplex_conditions_to_gene_targets(observations, resources_dir)
        de_genes = _load_de_genes_for_cell_line(cell_line=cell_line, condition_key='gene_target')
        _evaluate_and_save(
            predictions,
            observations,
            de_genes,
            cell_line,
            "GEARS_noreg",
            "GEARS_noreg",
            reference_observations=reference_observations,
        )


def evaluate_scfoundation_predictions() -> None:
    reference_observations = _load_sciplex_reference_observations()
    for cell_line in ["mcf7", "a549", "k562"]:
        predictions = pd.read_csv(
            os.path.join(data_dir, "scfoundation_predictions", f"sciplex_mean_post_{cell_line}.csv"), index_col=0
        )
        predictions = predictions.rename(columns={"perturbation": "condition"})
        predictions = predictions[predictions["condition"] != CONTROL_CONDITION]
        observations = pd.read_csv(
            os.path.join(data_dir, "observed_pseudobulk", f"sciplex_mean_post_{cell_line}.csv"), index_col=0
        )
        observations = map_sciplex_conditions_to_gene_targets(observations, resources_dir)
        de_genes = _load_de_genes_for_cell_line(cell_line=cell_line, condition_key='gene_target')
        _evaluate_and_save(
            predictions,
            observations,
            de_genes,
            cell_line,
            "scFoundation",
            "scfoundation",
            reference_observations=reference_observations,
        )


def evaluate_cpa_predictions() -> None:
    predictions = drop_control_rows(load_cpa_post(data_dir, "sciplex"))
    _evaluate_combined_sciplex_systema(predictions, "CPA", "CPA")


def evaluate_chemcpa_predictions() -> None:
    predictions = drop_control_rows(load_chemcpa_post(data_dir, "sciplex"))
    _evaluate_combined_sciplex_systema(predictions, "chemCPA", "chemCPA")


def evaluate_prnet_predictions() -> None:
    predictions = drop_control_rows(load_prnet_post(data_dir, "sciplex"))
    _evaluate_combined_sciplex_systema(predictions, "PRnet", "PRnet")


def _evaluate_combined_sciplex_systema(
    predictions: pd.DataFrame,
    model_name: str,
    output_suffix: str,
) -> None:
    reference_observations = _load_sciplex_reference_observations()
    predictions = align_sciplex_profile_conditions(predictions)
    for cell_line in SCIPLEX_CELL_LINES:
        pred = filter_sciplex_cell_line(predictions, cell_line)
        observations = align_sciplex_profile_conditions(load_sciplex_observed_post(data_dir, cell_line))
        de_genes = align_sciplex_profile_conditions(_load_de_genes_for_cell_line(cell_line=cell_line))
        pred = prepare_predictions_for_systema(pred, observations, split_template=pred)
        _, pair_metrics = compute_metrics_with_systema_mcfarland(
            predictions=pred,
            observations=observations,
            de_genes=de_genes,
            split_template=pred,
            reference_observations=reference_observations,
            sciplex_normalize_condition=False,
        )
        out = pair_metrics_to_long_df(pair_metrics, model_name=model_name)
        out = out.rename(columns={"cell_line": "cell_type"})
        out.to_csv(
            os.path.join(results_dir, f"sciplex{cell_line}_{output_suffix}_systema_outcomes.csv"),
            index=False,
        )


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
        logger.info("Evaluating chemCPA predictions (Systema metrics)...")
        evaluate_chemcpa_predictions()
        logger.info("Evaluating PRnet predictions (Systema metrics)...")
        evaluate_prnet_predictions()
        logger.info("Systema evaluation completed successfully")
    except Exception as exc:
        logger.error(f"Error in evaluate_predictions_sciplex_systema: {str(exc)}")
        raise
    log_script_end(__file__, logger)


if __name__ == "__main__":
    main()
