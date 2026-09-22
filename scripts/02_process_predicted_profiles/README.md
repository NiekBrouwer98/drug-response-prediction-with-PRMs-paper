# 02: Process Predicted Profiles

To facilitate reproduction, we have made all processed files available here: https://surfdrive.surf.nl/s/Fdg4spN2zbtkwMa.

Creates log-mean observed pseudobulks and baseline (average-effect / no-effect) predictions.

## Outputs

```
data/observed_pseudobulk/          # *_mean_* log-mean pseudobulks (+ n_cells)
data/average_effect_predictions/
data/no_effect_predictions/
```

Count-based pseudobulks (`*_count_*`) are written by
`scripts/01_process_measured_profiles/create_qc_and_count_pseudobulk.py`.

## Scripts

| Script | Role |
|--------|------|
| `create_pseudobulk.py` | Log-mean observed pseudobulks (SciPlex + McFarland) |
| `create_baseline_predictions.py` | Average-effect and no-effect baselines |

## Slurm (from repo root)

```bash
sbatch run_rerun_pseudobulk.sh
```
