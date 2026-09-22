[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22026963.svg)](https://doi.org/10.5281/zenodo.22026963)

# Downstream Drug Response Prediction Reveals Biological Utility of Perturbation Response Models

This repository contains the code to reproduce the analyses and figures for the study "Downstream Drug Response Prediction Reveals Biological Utility of Perturbation Response Models" ([under review](link)).

## Project Structure

```
drug-response-prediction/
├── data/                          # Processed datasets and predictions
├── figures/                       # Generated figures and manuscript files
├── resources/                     # Reference tables (SMILES, splits, sensitivity)
├── results/                       # Analysis outputs
├── scripts/                       # Pipeline stages
│   ├── 01_process_measured_profiles/
│   ├── 02_process_predicted_profiles/
│   ├── 03_evaluate_predicted_profiles/
│   └── 04_predict_drug_response/
├── config.py                      # Paths and analysis parameters
└── utils.py                       # Shared logging / directory helpers
```

## Dependencies

Python 3.10 and packages listed in `environment.yml`.

## Installation

### Option 1: Apptainer container (recommended)

The container activates the `drug-response-prediction` conda environment automatically.

```bash
git clone <repository-url>
cd drug-response-prediction
apptainer build drug-response-prediction.sif drug-response-prediction.def
cp config_example.yaml config.yaml   # optional path overrides
```

### Option 2: Conda

```bash
git clone <repository-url>
cd drug-response-prediction
conda env create -f environment.yml
conda activate drug-response-prediction
cp config_example.yaml config.yaml   # optional
```

## Configuration

- **`config.py`**: Project paths and defaults (`setup_project()`, `Config`)
- **`config_example.yaml`**: Optional overrides
- **`utils.py`**: Logging and directory helpers used by scripts

```python
from config import setup_project
config = setup_project()
```

## Datasets

Processed files for reproduction: https://surfdrive.surf.nl/s/DbRwLCbCXcbiC2E.

Raw sources:
- [McFarland et al. (2020)](https://figshare.com/s/139f64b495dea9d88c70)
- [SciPlex3 / Srivatsan et al. (2020)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE139944)

## Usage

Stages are sequential. Commands below assume the repo root and a working Python environment (or wrap with `apptainer exec drug-response-prediction.sif …`). See `scripts/*/README.md` for options and outputs.

```bash
# 01 — annotate / split measured profiles (+ optional QC count pseudobulks)
python scripts/01_process_measured_profiles/create_sciplex_splits.py
python scripts/01_process_measured_profiles/create_mcfarland_splits.py
python scripts/01_process_measured_profiles/create_qc_and_count_pseudobulk.py

# 02 — log-mean observed pseudobulks + average/no-effect baselines
python scripts/02_process_predicted_profiles/create_pseudobulk.py
python scripts/02_process_predicted_profiles/create_baseline_predictions.py

# 03 — reconstruction metrics
python scripts/03_evaluate_predicted_profiles/evaluate_predictions_sciplex.py
python scripts/03_evaluate_predicted_profiles/evaluate_predictions_sciplex_systema.py
python scripts/03_evaluate_predicted_profiles/evaluate_predictions_mcfarland.py
python scripts/03_evaluate_predicted_profiles/evaluate_predictions_mcfarland_systema.py
python scripts/03_evaluate_predicted_profiles/create_figures.py

# 04 — measured T1–T4 task CV
python scripts/04_predict_drug_response/run_measured_task_cv.py --dataset both

# 04 — predicted-profile split CV (train/test on predicted; SMILES heads)
python scripts/04_predict_drug_response/response_prediction.py \
  --dataset both --smiles-only --no-include-observed --results-suffix _smiles

# 04 — train Measured → test predicted
python scripts/04_predict_drug_response/response_prediction.py \
  --dataset both --smiles-only --train-on-measured-test-on-predicted \
  --results-suffix _train_measured_test_predicted

# 04 — figures
python scripts/04_predict_drug_response/create_figures_tasks.py
python scripts/04_predict_drug_response/create_figures_predictions.py
python scripts/04_predict_drug_response/create_figures_correlations.py
python scripts/04_predict_drug_response/create_figures_GSEA.py
```

Manuscript figure notebooks live under `scripts/03_evaluate_predicted_profiles/` and `scripts/04_predict_drug_response/` (see stage READMEs). Reference tables: `resources/README.md`.

## Citation

```bibtex
@article{brouwer2026drug,
  title={Downstream Drug Response Prediction Reveals Biological Utility of Perturbation Response Models},
  author={Brouwer, Niek and Damyanov, Martin and Argelo, Jonas and Vis, Dani{\"e}l J and Reinders, Marcel JT and Wessels, Lodewyk FA},
  journal={[Journal Name]},
  year={2026}
}
```

## License

MIT — see `LICENSE`.

## Contact

Niek Brouwer: n.brouwer-1@tudelft.nl

## Acknowledgments

We thank the developers of CPA, GEARS, and scFoundation for making their models publicly available. This research is part of the Oncode Accelerator Project funded by the Dutch National Growth Fund (NGF).
