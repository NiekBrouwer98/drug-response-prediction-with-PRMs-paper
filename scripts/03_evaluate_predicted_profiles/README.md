# 03: Evaluate Predicted Profiles

This directory contains scripts for evaluating and comparing the reconstruction accuracy of profiles predicted by baseline and perturbation response models.

DE genes are saved in the results directory.

### McFarland
- **`evaluate_predictions_mcfarland.py`**: MSE and Pearson per `(cell_line, condition)` vs observed post pseudobulk
- **`evaluate_predictions_mcfarland_systema.py`**: adds Systema-centered Pearson (reference = mean of first 200 observed pairs)
- **`mcfarland_profile_metrics.py`**: shared metric helpers
- Outputs: `results/03_evaluate_predicted_profiles/mcfarland_{CPA,no_effect,average_effect}_outcomes.csv` and `*_systema_outcomes.csv`
- DE genes: `mcfarland_de_genes.csv` (top LFC genes per pair; auto-computed if missing)