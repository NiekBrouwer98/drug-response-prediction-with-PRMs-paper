"""Evaluate McFarland predicted post-treatment profiles (MSE and Pearson)."""

import os
import sys
from pathlib import Path

import pandas as pd

_eval_dir = Path(__file__).parent
sys.path.insert(0, str(_eval_dir))
sys.path.append(str(_eval_dir.parent.parent))
from config import config, setup_project
from mcfarland_profile_metrics import (
    compute_metrics_from_means_mcfarland,
    compute_mcfarland_de_genes_from_lfc,
    load_mcfarland_gears_post,
    load_mcfarland_scfoundation_post,
    load_mcfarland_observed_lfc,
    load_mcfarland_observed_post,
    pair_metrics_to_long_df,
    prepare_predictions_for_evaluation,
)
from utils import (
    ensure_directories_exist,
    log_script_end,
    log_script_start,
    setup_logging_for_script,
)

setup_project()
logger = setup_logging_for_script(__file__)

data_dir = str(config.DATA_DIR)
results_dir = str(config.RESULTS_03_DIR)
figures_dir = str(config.FIGURES_03_DIR)

ensure_directories_exist(data_dir, results_dir, figures_dir)

DE_GENES_PATH = os.path.join(results_dir, 'mcfarland_de_genes.csv')


def _resolve_cpa_post_path() -> str:
    for subdir in ('CPA_predictions', 'cpa'):
        path = os.path.join(data_dir, subdir, 'mcfarland_mean_post_all.csv')
        if os.path.exists(path):
            return path
    raise FileNotFoundError('McFarland CPA post file mcfarland_mean_post_all.csv not found')


def _load_or_compute_de_genes() -> pd.DataFrame:
    if os.path.exists(DE_GENES_PATH):
        return pd.read_csv(DE_GENES_PATH)
    logger.info('Computing McFarland DE genes from observed LFC...')
    de_genes = compute_mcfarland_de_genes_from_lfc(load_mcfarland_observed_lfc(data_dir))
    de_genes.to_csv(DE_GENES_PATH, index=False)
    return de_genes


def _evaluate_and_save(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
    de_genes: pd.DataFrame,
    model_name: str,
    output_name: str,
) -> None:
    predictions = prepare_predictions_for_evaluation(predictions, observations)
    _, pair_metrics = compute_metrics_from_means_mcfarland(
        predictions=predictions,
        observations=observations,
        de_genes=de_genes,
    )
    out = pair_metrics_to_long_df(pair_metrics, model_name=model_name)
    out.to_csv(os.path.join(results_dir, output_name), index=False)
    logger.info('Wrote %s (%d rows)', output_name, len(out))


def evaluate_cpa_predictions() -> None:
    predictions = pd.read_csv(_resolve_cpa_post_path())
    observations = load_mcfarland_observed_post(data_dir)
    de_genes = _load_or_compute_de_genes()
    _evaluate_and_save(predictions, observations, de_genes, 'CPA', 'mcfarland_CPA_outcomes.csv')


def evaluate_gears_predictions() -> None:
    predictions = load_mcfarland_gears_post(data_dir)
    observations = load_mcfarland_observed_post(data_dir)
    de_genes = _load_or_compute_de_genes()
    _evaluate_and_save(predictions, observations, de_genes, 'GEARS', 'mcfarland_GEARS_outcomes.csv')


def evaluate_scfoundation_predictions() -> None:
    predictions = load_mcfarland_scfoundation_post(data_dir)
    observations = load_mcfarland_observed_post(data_dir)
    de_genes = _load_or_compute_de_genes()
    _evaluate_and_save(
        predictions, observations, de_genes, 'scFoundation', 'mcfarland_scFoundation_outcomes.csv'
    )


def evaluate_no_effect_predictions() -> None:
    path = os.path.join(data_dir, 'no_effect_predictions', 'mcfarland_mean_post_all_celllines.csv')
    predictions = pd.read_csv(path)
    observations = load_mcfarland_observed_post(data_dir)
    de_genes = _load_or_compute_de_genes()
    _evaluate_and_save(predictions, observations, de_genes, 'No effect', 'mcfarland_no_effect_outcomes.csv')


def evaluate_average_effect_predictions() -> None:
    path = os.path.join(data_dir, 'average_effect_predictions', 'mcfarland_mean_post_all_celllines.csv')
    predictions = pd.read_csv(path)
    observations = load_mcfarland_observed_post(data_dir)
    de_genes = _load_or_compute_de_genes()
    _evaluate_and_save(
        predictions, observations, de_genes, 'Average effect', 'mcfarland_average_effect_outcomes.csv'
    )


def main() -> None:
    log_script_start(__file__, logger)
    try:
        logger.info('Evaluating McFarland CPA predictions (MSE / Pearson)...')
        evaluate_cpa_predictions()
        logger.info('Evaluating McFarland GEARS predictions...')
        evaluate_gears_predictions()
        logger.info('Evaluating McFarland scFoundation predictions...')
        evaluate_scfoundation_predictions()
        logger.info('Evaluating McFarland no-effect predictions...')
        evaluate_no_effect_predictions()
        logger.info('Evaluating McFarland average-effect predictions...')
        evaluate_average_effect_predictions()
        logger.info('McFarland profile evaluation completed successfully')
    except Exception as exc:
        logger.error('Error in evaluate_predictions_mcfarland: %s', exc)
        raise
    log_script_end(__file__, logger)


if __name__ == '__main__':
    main()
