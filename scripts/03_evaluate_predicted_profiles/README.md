# 03: Evaluate Predicted Profiles

Evaluates reconstruction accuracy of profiles predicted by baseline and perturbation response models.

DE genes are saved under `results/03_evaluate_predicted_profiles/`.

## Prediction layouts under `data/`

- **CPA / chemCPA**: one combined CSV per dataset (`*_mean_post.csv` / `*_cv_pred_post.csv`) with gene columns plus `cell_type`, `condition`, and `split`
- **PRnet**: one combined h5ad per dataset; post expression is `layers['predicted_mean']`, LFC is `layers['predicted_lfc']`
- **GEARS / scFoundation / baselines**: per-cell-line SciPlex CSVs and combined McFarland CSVs

## Scripts

### SciPlex
- **`evaluate_predictions_sciplex.py`**: MSE and Pearson vs observed post pseudobulk (CPA, chemCPA, PRnet, GEARS, scFoundation, average effect, no effect)
- **`evaluate_predictions_sciplex_systema.py`**: adds Systema Pearson (dataset-wide or per cell line)
- Outputs: `results/03_evaluate_predicted_profiles/sciplex{a549,k562,mcf7}_{model}_outcomes.csv` and `*_systema_outcomes.csv`

### McFarland
- **`evaluate_predictions_mcfarland.py`**: MSE and Pearson per `(cell_line, condition)`
- **`evaluate_predictions_mcfarland_systema.py`**: Systema Pearson variants
- **`mcfarland_profile_metrics.py`**: shared metric helpers
- **`prediction_io.py`**: loaders for combined CPA / chemCPA / PRnet files (also used by step 02 baselines)
- Outputs: `results/03_evaluate_predicted_profiles/mcfarland_{CPA,chemCPA,PRnet,GEARS,scFoundation,no_effect,average_effect}_outcomes.csv` and `*_systema_outcomes.csv`
- DE genes: `mcfarland_de_genes.csv` (auto-computed via `compute_de_genes.py` if missing)

### Orchestration / figures
- **`evaluate_new_models.py`**: run only CPA / chemCPA / PRnet evaluations
- **`create_figures.py`**: SciPlex | McFarland boxplots and reconstruction supplementary tables
- **`create_figures.ipynb`**: same plots for interactive manuscript figure creation

```bash
python scripts/03_evaluate_predicted_profiles/evaluate_new_models.py
python scripts/03_evaluate_predicted_profiles/create_figures.py
```

Figure prefixes:
- CPA family: `performance_boxplot_cpa_family_*` (CPA, chemCPA, PRnet, Average effect, No effect)
- GEARS family: `performance_boxplot_gears_family_*`
- Systema: `performance_boxplot_systema_{cpa,gears}_family_*`
- Tables: `reconstruction_metrics_summary.csv`, `prm_vs_baseline_significance.csv` (+ `_systema` variants) and matching TeX under `figures/03_evaluate_predicted_profiles/`

## Slurm (from repo root)

```bash
sbatch run_rerun_profile_eval.sh
```
