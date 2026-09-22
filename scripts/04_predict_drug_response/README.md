# 04: Predict Drug Response

Predicts drug sensitivity from measured and predicted gene-expression profiles (ElasticNet / two-part models), with SMILES (ECFP) chemical features and T1–T4 / predefined-fold CV designs.

## Feature types

1. **Pre / Post / LFC** measured profiles
2. **Predicted Post / LFC** from CPA, chemCPA, PRnet, GEARS, scFoundation (+ average/no-effect baselines)
3. **+SMILES**: same expression features concatenated with Morgan fingerprints from `resources/drug_smiles.csv` (build via `build_drug_smiles.py` → also writes `drug_ecfp_2048.npz`)

## Experimental designs

1. **T1–T4 measured tasks** (`task_cv.py`, `run_measured_task_cv.py`): group-aware CV
   - **T1** seen drug + seen tissue/line — exhaustive leave-one-(drug, context); also writes predefined 5-fold T1 (`*_t1_5fold_*`) for predicted-profile figures
   - **T2** unseen tissue (McFarland) or cell line (SciPlex)
   - **T3** unseen drug
   - **T4** unseen drug and tissue/line
2. **Predicted-profile split CV** (`response_prediction.py`): predefined 5-fold keys from the PRM / chemical splits
3. **Train Measured → test predicted**: same folds, Measured train features, predicted test features

## Core scripts

| Script | Role |
|--------|------|
| `run_measured_task_cv.py` | SciPlex + McFarland T1–T4 measured-profile CV |
| `task_cv.py` | Shared group-aware task CV |
| `response_prediction.py` | Predicted-profile / transfer split-CV CLI |
| `prediction_utils.py` | Profile loaders for downstream steps |
| `evaluate_predictions.py` | Metrics helpers for predicted-profile figures |
| `drug_fingerprints.py` / `build_drug_smiles.py` | ECFP + SMILES cache |
| `feature_importance_io.py` | Feature-importance I/O for GSEA / correlations |
| `compute_correlations.py` / `compute_GSEA.py` | Gene–gene and pathway analyses |
| `create_figures_tasks.py` | Measured T1–T4 figures + supplementary tables |
| `create_figures_predictions.py` | Predicted-profile / transfer / threshold-sweep figures |
| `create_figures_correlations.py` | Gene–gene correlation figures |
| `create_figures_GSEA.py` | GSEA figures |
| `create_figures_observations.ipynb` | Manuscript measured-profile panels (calls `create_figures_tasks`) |
| `create_figures_predictions.ipynb` | Manuscript predicted-profile panels (calls `create_figures_predictions`) |
| `create_figures_correlations.ipynb` | Manuscript gene–gene figures |
| `create_figures_GSEA.ipynb` | Manuscript GSEA figures |

## Measured-task CV

```bash
# Default: Pre+SMILES / Post+SMILES / LFC+SMILES, T1–T4, both datasets
python scripts/04_predict_drug_response/run_measured_task_cv.py --dataset both

# Expression-only modalities
python scripts/04_predict_drug_response/run_measured_task_cv.py \
  --dataset both --models Pre,Post,LFC --output-tag nosmiles

# Optional: combine shards after parallel per-task runs
python scripts/04_predict_drug_response/run_measured_task_cv.py --combine-only
```

Useful flags: `--tasks T1,T2,T3,T4`, `--t1-scheme predefined_fold|exhaustive|both`, `--pseudobulk mean|count`, `--mode continuous|two-stage|threshold-sweep`.

## Predicted-profile split CV

Chemical PRMs (CPA, chemCPA, PRnet) share CPA / SciPlex fold keys. McFarland GEARS / scFoundation use native folds; SciPlex genetic PRMs are expanded onto chemical SciPlex fold keys. With `--include-predicted-smiles` (default), Post+SMILES / LFC+SMILES heads are added for predicted models and baselines.

```bash
# Train AND test on predicted profiles (+ baselines), SMILES heads only
python scripts/04_predict_drug_response/response_prediction.py \
  --dataset both --smiles-only --no-include-observed --results-suffix _smiles \
  --models CPA,chemCPA,PRnet,GEARS,scFoundation

# Train on Measured, test on predicted
python scripts/04_predict_drug_response/response_prediction.py \
  --dataset both --smiles-only --train-on-measured-test-on-predicted \
  --results-suffix _train_measured_test_predicted \
  --models CPA,chemCPA,PRnet,GEARS,scFoundation

# Optional: gene-only heads / two-part threshold sweep
python scripts/04_predict_drug_response/response_prediction.py \
  --dataset both --no-include-predicted-smiles --results-suffix ''
python scripts/04_predict_drug_response/response_prediction.py \
  --dataset both --mode threshold-sweep --smiles-only --results-suffix _smiles
```

## Figures

```bash
python scripts/04_predict_drug_response/create_figures_tasks.py
python scripts/04_predict_drug_response/create_figures_predictions.py
python scripts/04_predict_drug_response/create_figures_correlations.py
python scripts/04_predict_drug_response/create_figures_GSEA.py
```

Interactive manuscript notebooks (run from `scripts/04_predict_drug_response/`):

- `create_figures_observations.ipynb`
- `create_figures_predictions.ipynb`
- `create_figures_correlations.ipynb`
- `create_figures_GSEA.ipynb`

## Key outputs

- `results/04_predict_drug_response/{sciplex,mcfarland}_*measured_tasks*.csv`
- `results/04_predict_drug_response/{sciplex,mcfarland}_split_cv_smiles_*.csv`
- `results/04_predict_drug_response/{sciplex,mcfarland}_split_cv_train_measured_test_predicted_*.csv`
- `results/04_predict_drug_response/predictions_*_{fold_metrics,paired_comparisons,...}.csv`
- `figures/04_predict_drug_response/measured_tasks_*.pdf` / `predictions_*.pdf`
- Supplementary TeX under `figures/04_predict_drug_response/` (included by `figures/submission_figures/Supplementary_Data.tex`)
