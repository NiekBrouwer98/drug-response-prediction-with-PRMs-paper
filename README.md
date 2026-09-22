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
├── utils.py                       # Shared logging / directory helpers
└── run_*.sh                       # Slurm entrypoints (submit from repo root)
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
- [SciPlex / Srivatsan et al. (2020)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE139944) (this repo reads the [scPerturb](http://projects.sanderlab.org/scperturb/) `Srivatsan_2019_raw.h5ad`)

## Usage

Stages are sequential. Prefer the Slurm wrappers at the repo root; each stage README lists the Python entrypoints.

```bash
# 01 — annotate / split measured profiles (+ optional QC count pseudobulks)
sbatch run_create_sciplex_splits.sh
sbatch run_create_mcfarland_splits.sh
sbatch run_qc_and_count_pseudobulk.sh

# 02 — log-mean observed pseudobulks + average/no-effect baselines
sbatch run_rerun_pseudobulk.sh

# 03 — reconstruction metrics (CPA / chemCPA / PRnet / GEARS / scFoundation / baselines)
sbatch run_rerun_profile_eval.sh

# 04 — measured T1–T4 task CV + predicted-profile split CV + figures
bash scripts/04_predict_drug_response/submit_measured_task_cv.sh
DATASET=both sbatch scripts/04_predict_drug_response/run_predicted_smiles_split_cv.sh
DATASET=both sbatch scripts/04_predict_drug_response/run_train_measured_test_predicted_split_cv.sh
sbatch run_rerun_drug_response.sh   # or run figure scripts directly (see stage READMEs)
```

Interactive / single-script runs:

```bash
apptainer exec drug-response-prediction.sif \
  python scripts/01_process_measured_profiles/create_sciplex_splits.py
```

See `scripts/*/README.md` and `resources/README.md` for file-level detail.

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
