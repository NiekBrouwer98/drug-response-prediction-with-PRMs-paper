"""Entry point: run SciPlex + McFarland measured-profile T1–T4 task CV."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config, setup_project
from task_cv import (
    DEFAULT_TWO_STAGE_THRESHOLDS,
    TASK_DEFINITIONS,
    combine_measured_task_outputs,
    resolve_measured_model_keys,
    run_all_measured_task_head_sweep,
    run_all_measured_tasks,
)
from utils import (
    ensure_directories_exist,
    log_script_end,
    log_script_start,
    setup_logging_for_script,
)

setup_project()
logger = setup_logging_for_script(__file__)

results_dir = str(config.RESULTS_04_DIR)
figures_dir = str(config.FIGURES_04_DIR)
ensure_directories_exist(results_dir, figures_dir)


def _parse_thresholds(value: str) -> tuple[float, ...]:
    thresholds = tuple(float(v.strip()) for v in value.split(',') if v.strip())
    if not thresholds:
        raise ValueError('At least one threshold is required')
    return thresholds


def _parse_tasks(value: str) -> tuple[str, ...]:
    tasks = tuple(tok.strip().upper() for tok in value.split(',') if tok.strip())
    if not tasks:
        raise ValueError('At least one task is required (e.g. T1 or T1,T2,T3,T4)')
    unknown = [t for t in tasks if t not in TASK_DEFINITIONS]
    if unknown:
        raise ValueError(f'Unknown tasks {unknown}; expected subset of {list(TASK_DEFINITIONS)}')
    return tasks


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--mode',
        choices=['continuous', 'two-stage', 'threshold-sweep'],
        default='continuous',
        help='Prediction head: continuous ElasticNet, one two-stage threshold, or threshold sweep.',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.02,
        help='Sensitivity threshold for --mode two-stage.',
    )
    parser.add_argument(
        '--thresholds',
        default=','.join(str(x) for x in DEFAULT_TWO_STAGE_THRESHOLDS),
        help='Comma-separated sensitivity thresholds for --mode threshold-sweep.',
    )
    parser.add_argument(
        '--pseudobulk',
        choices=['mean', 'count'],
        default='mean',
        help='Use log-mean pseudobulks (default) or count-based size-factor log1p profiles.',
    )
    parser.add_argument(
        '--models',
        default='Pre+SMILES,Post+SMILES,LFC+SMILES',
        help=(
            'Comma-separated measured modalities. Default: Pre+SMILES,Post+SMILES,LFC+SMILES. '
            'Also accepts keys or aliases (pre+smiles / Pre / post_treatment, …).'
        ),
    )
    parser.add_argument(
        '--tasks',
        default='T1,T2,T3,T4',
        help='Comma-separated tasks to run (default: T1,T2,T3,T4). Use T1 to rerun only T1.',
    )
    parser.add_argument(
        '--t1-scheme',
        choices=['both', 'predefined_fold', 'exhaustive'],
        default='predefined_fold',
        help=(
            'T1 CV scheme: predefined_fold (default, 5 folds matching PRM splits), '
            'exhaustive leave-one-(drug, context), or both. T2–T4 always exhaustive.'
        ),
    )
    parser.add_argument(
        '--output-tag',
        default='',
        help='Optional tag inserted into output filenames (e.g. smiles).',
    )
    parser.add_argument(
        '--dataset',
        choices=['both', 'sciplex', 'mcfarland'],
        default='both',
        help='Which measured-profile dataset(s) to run (default: both).',
    )
    parser.add_argument(
        '--combine-only',
        action='store_true',
        help='Assemble task shards (if any) and merge SciPlex/McFarland CSVs into combined summaries.',
    )
    parser.add_argument(
        '--output-shard',
        default='',
        help=(
            'Write per-task shard CSVs (…__T1.csv) instead of merging into shared files. '
            'Used by parallel Slurm array workers; finalize with --combine-only.'
        ),
    )
    args = parser.parse_args()
    model_keys = resolve_measured_model_keys(
        [tok.strip() for tok in args.models.split(',') if tok.strip()]
    )
    logger.info('Running measured modalities: %s', list(model_keys))
    tasks = _parse_tasks(args.tasks)
    if args.t1_scheme == 'both':
        t1_schemes = ('exhaustive', 'predefined_fold')
    else:
        t1_schemes = (args.t1_scheme,)
    output_shard = args.output_shard.strip().upper()
    if output_shard and len(tasks) != 1:
        raise ValueError('--output-shard requires exactly one --tasks value')
    if output_shard and output_shard != tasks[0]:
        raise ValueError(f'--output-shard {output_shard!r} must match --tasks {tasks[0]!r}')
    logger.info('Running tasks: %s', list(tasks))
    logger.info('T1 scheme(s): %s', list(t1_schemes))
    logger.info('Dataset: %s', args.dataset)
    if output_shard:
        logger.info('Output shard: %s', output_shard)
    log_script_start(__file__, logger)
    try:
        if args.combine_only:
            summary = combine_measured_task_outputs(
                results_dir,
                pseudobulk=args.pseudobulk,
                output_tag=args.output_tag,
                dataset=args.dataset,
            )
        elif args.mode == 'continuous':
            summary = run_all_measured_tasks(
                results_dir,
                n_features=1000,
                n_repeats=5,
                random_state=1,
                pseudobulk=args.pseudobulk,
                head='continuous',
                output_tag=args.output_tag,
                model_keys=model_keys,
                tasks=tasks,
                dataset=args.dataset,
                t1_schemes=t1_schemes,
                output_shard=output_shard,
            )
        elif args.mode == 'two-stage':
            summary = run_all_measured_tasks(
                results_dir,
                n_features=1000,
                n_repeats=5,
                random_state=1,
                pseudobulk=args.pseudobulk,
                head='two_stage',
                threshold=args.threshold,
                output_tag=(
                    args.output_tag
                    or f'two_stage_t{str(args.threshold).replace("-", "m").replace(".", "p")}'
                ),
                model_keys=model_keys,
                tasks=tasks,
                dataset=args.dataset,
                t1_schemes=t1_schemes,
                output_shard=output_shard,
            )
        else:
            summary = run_all_measured_task_head_sweep(
                results_dir,
                thresholds=_parse_thresholds(args.thresholds),
                n_features=1000,
                n_repeats=5,
                random_state=1,
                pseudobulk=args.pseudobulk,
                model_keys=model_keys,
                dataset=args.dataset,
            )
        if summary is not None and not summary.empty:
            logger.info('Combined task summary:\n%s', summary.to_string(index=False))
        elif output_shard:
            logger.info('Shard %s written; combine deferred to finalize job.', output_shard)
    except Exception as exc:
        logger.error('Error in run_measured_task_cv: %s', exc)
        raise
    log_script_end(__file__, logger)


if __name__ == '__main__':
    main()
