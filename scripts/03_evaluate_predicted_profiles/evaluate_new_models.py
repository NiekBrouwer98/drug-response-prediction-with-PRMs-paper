"""Evaluate only CPA (new files), chemCPA, and PRnet."""

from __future__ import annotations

import sys
from pathlib import Path

_eval_dir = Path(__file__).parent
sys.path.insert(0, str(_eval_dir))
sys.path.append(str(_eval_dir.parent.parent))

from evaluate_predictions_sciplex import (  # noqa: E402
    evaluate_CPA_predictions,
    evaluate_chemcpa_predictions,
    evaluate_prnet_predictions,
)
from evaluate_predictions_sciplex_systema import (  # noqa: E402
    evaluate_cpa_predictions as evaluate_sciplex_cpa_systema,
    evaluate_chemcpa_predictions as evaluate_sciplex_chemcpa_systema,
    evaluate_prnet_predictions as evaluate_sciplex_prnet_systema,
)
from evaluate_predictions_mcfarland import (  # noqa: E402
    evaluate_cpa_predictions as evaluate_mcf_cpa,
    evaluate_chemcpa_predictions as evaluate_mcf_chemcpa,
    evaluate_prnet_predictions as evaluate_mcf_prnet,
)
from evaluate_predictions_mcfarland_systema import (  # noqa: E402
    evaluate_cpa_predictions as evaluate_mcf_cpa_systema,
    evaluate_chemcpa_predictions as evaluate_mcf_chemcpa_systema,
    evaluate_prnet_predictions as evaluate_mcf_prnet_systema,
)
from utils import log_script_end, log_script_start, setup_logging_for_script

logger = setup_logging_for_script(__file__)


def main() -> None:
    log_script_start(__file__, logger)
    logger.info('SciPlex MSE/Pearson: CPA, chemCPA, PRnet')
    evaluate_CPA_predictions()
    evaluate_chemcpa_predictions()
    evaluate_prnet_predictions()
    logger.info('SciPlex Systema: CPA, chemCPA, PRnet')
    evaluate_sciplex_cpa_systema()
    evaluate_sciplex_chemcpa_systema()
    evaluate_sciplex_prnet_systema()
    logger.info('McFarland MSE/Pearson: CPA, chemCPA, PRnet')
    evaluate_mcf_cpa()
    evaluate_mcf_chemcpa()
    evaluate_mcf_prnet()
    logger.info('McFarland Systema: CPA, chemCPA, PRnet')
    evaluate_mcf_cpa_systema()
    evaluate_mcf_chemcpa_systema()
    evaluate_mcf_prnet_systema()
    logger.info('New-model evaluation finished')
    log_script_end(__file__, logger)


if __name__ == '__main__':
    main()
