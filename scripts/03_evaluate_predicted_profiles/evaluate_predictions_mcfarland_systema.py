"""Evaluate McFarland predicted profiles with MSE, Pearson, and Systema metrics."""

import os
import sys
from pathlib import Path

import pandas as pd

_eval_dir = Path(__file__).parent
sys.path.insert(0, str(_eval_dir))
sys.path.append(str(_eval_dir.parent.parent))
from config import config, setup_project
from mcfarland_profile_metrics import (
    compute_metrics_with_systema_mcfarland,
    compute_mcfarland_de_genes_from_lfc,
    load_mcfarland_gears_post,
    load_mcfarland_scfoundation_post,
    load_mcfarland_observed_lfc,
    load_mcfarland_observed_post,
    pair_metrics_to_long_df,
    prepare_predictions_for_systema,
)
from prediction_io import drop_control_rows, load_cpa_post, load_chemcpa_post, load_prnet_post
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


def _load_or_compute_de_genes() -> pd.DataFrame:
    if os.path.exists(DE_GENES_PATH):
        return pd.read_csv(DE_GENES_PATH)
    logger.info('Computing McFarland DE genes from observed LFC...')
    de_genes = compute_mcfarland_de_genes_from_lfc(load_mcfarland_observed_lfc(data_dir))
    de_genes.to_csv(DE_GENES_PATH, index=False)
    return de_genes


def _split_template_cpa() -> pd.DataFrame:
    return load_cpa_post(data_dir, 'mcfarland')


def _evaluate_and_save(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
    de_genes: pd.DataFrame,
    model_name: str,
    output_name: str,
    split_template: pd.DataFrame | None = None,
) -> None:
    template = split_template if split_template is not None else _split_template_cpa()
    predictions = prepare_predictions_for_systema(predictions, observations, split_template=template)
    _, pair_metrics = compute_metrics_with_systema_mcfarland(
        predictions=predictions,
        observations=observations,
        de_genes=de_genes,
        split_template=template,
    )
    out = pair_metrics_to_long_df(pair_metrics, model_name=model_name)
    out.to_csv(os.path.join(results_dir, output_name), index=False)
    logger.info('Wrote %s (%d rows)', output_name, len(out))


def evaluate_cpa_predictions() -> None:
    predictions = drop_control_rows(load_cpa_post(data_dir, 'mcfarland'))
    observations = load_mcfarland_observed_post(data_dir)
    de_genes = _load_or_compute_de_genes()
    template = _split_template_cpa()
    _evaluate_and_save(
        predictions,
        observations,
        de_genes,
        'CPA',
        'mcfarland_CPA_systema_outcomes.csv',
        split_template=template,
    )


def evaluate_chemcpa_predictions() -> None:
    predictions = drop_control_rows(load_chemcpa_post(data_dir, 'mcfarland'))
    observations = load_mcfarland_observed_post(data_dir)
    de_genes = _load_or_compute_de_genes()
    _evaluate_and_save(
        predictions,
        observations,
        de_genes,
        'chemCPA',
        'mcfarland_chemCPA_systema_outcomes.csv',
        split_template=predictions,
    )


def evaluate_prnet_predictions() -> None:
    predictions = drop_control_rows(load_prnet_post(data_dir, 'mcfarland'))
    observations = load_mcfarland_observed_post(data_dir)
    de_genes = _load_or_compute_de_genes()
    _evaluate_and_save(
        predictions,
        observations,
        de_genes,
        'PRnet',
        'mcfarland_PRnet_systema_outcomes.csv',
        split_template=predictions,
    )


def evaluate_gears_predictions() -> None:
    predictions = load_mcfarland_gears_post(data_dir)
    observations = load_mcfarland_observed_post(data_dir)
    de_genes = _load_or_compute_de_genes()
    _evaluate_and_save(
        predictions,
        observations,
        de_genes,
        'GEARS',
        'mcfarland_GEARS_systema_outcomes.csv',
        split_template=predictions,
    )


def evaluate_scfoundation_predictions() -> None:
    predictions = load_mcfarland_scfoundation_post(data_dir)
    observations = load_mcfarland_observed_post(data_dir)
    de_genes = _load_or_compute_de_genes()
    _evaluate_and_save(
        predictions,
        observations,
        de_genes,
        'scFoundation',
        'mcfarland_scFoundation_systema_outcomes.csv',
        split_template=predictions,
    )


def evaluate_no_effect_predictions() -> None:
    path = os.path.join(data_dir, 'no_effect_predictions', 'mcfarland_mean_post_all_celllines.csv')
    predictions = pd.read_csv(path)
    observations = load_mcfarland_observed_post(data_dir)
    de_genes = _load_or_compute_de_genes()
    _evaluate_and_save(
        predictions, observations, de_genes, 'No effect', 'mcfarland_no_effect_systema_outcomes.csv'
    )


def evaluate_average_effect_predictions() -> None:
    path = os.path.join(data_dir, 'average_effect_predictions', 'mcfarland_mean_post_all_celllines.csv')
    predictions = pd.read_csv(path)
    observations = load_mcfarland_observed_post(data_dir)
    de_genes = _load_or_compute_de_genes()
    _evaluate_and_save(
        predictions,
        observations,
        de_genes,
        'Average effect',
        'mcfarland_average_effect_systema_outcomes.csv',
    )


def main() -> None:
    log_script_start(__file__, logger)
    try:
        logger.info('Evaluating McFarland CPA predictions (Systema metrics)...')
        evaluate_cpa_predictions()
        logger.info('Evaluating McFarland chemCPA predictions (Systema)...')
        evaluate_chemcpa_predictions()
        logger.info('Evaluating McFarland PRnet predictions (Systema)...')
        evaluate_prnet_predictions()
        logger.info('Evaluating McFarland GEARS predictions (Systema)...')
        evaluate_gears_predictions()
        logger.info('Evaluating McFarland scFoundation predictions (Systema)...')
        evaluate_scfoundation_predictions()
        logger.info('Evaluating McFarland no-effect predictions (Systema)...')
        evaluate_no_effect_predictions()
        logger.info('Evaluating McFarland average-effect predictions (Systema)...')
        evaluate_average_effect_predictions()
        logger.info('McFarland Systema evaluation completed successfully')
    except Exception as exc:
        logger.error('Error in evaluate_predictions_mcfarland_systema: %s', exc)
        raise
    log_script_end(__file__, logger)


if __name__ == '__main__':
    main()
