# 02: Process Predicted Profiles

Processed files for reproduction: https://surfdrive.surf.nl/s/DbRwLCbCXcbiC2E.

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

## Run

```bash
python scripts/02_process_predicted_profiles/create_pseudobulk.py
python scripts/02_process_predicted_profiles/create_baseline_predictions.py
```
