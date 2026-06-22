# Downstream Drug Response Prediction Reveals Biological Utility of Perturbation Response Models

This repository contains the code to reproduce the analyses and figures for the study "Downstream Drug Response Prediction Reveals Biological Utility of Perturbation Response Models" ([under review](link)).

## Project Structure

```
drug_response_prediction/
├── data/                          # Processed datasets and predictions
│   ├── sciplex_raw/               # Raw sciplex3 dataset
│   ├── sciplex_processed/         # Processed sciplex3 dataset
│   ├── mcfarland_raw/             # Raw McFarland dataset
│   ├── mcfarland_processed/       # Processed McFarland dataset
│   ├── no_effect_predictions/     # No Effect Baseline model predictions
│   ├── average_effect_predictions/# Average Effect Baseline model 
│   ├── CPA_predictions/           # CPA model predictions
│   ├── GEARS_predictions/         # GEARS model predictions
│   ├── scfoundation_predictions/  # scFoundation predictions
│   └── measured_pseudobulk/       # Measured gene expression data as pseudobulks
├── figures/                       # Generated figures and manuscript files
├── resources/                     # Reference data and annotations
├── results/                       # Analysis results and outputs
└── scripts/                       # Analysis scripts organized by workflow stage
    ├── 01_process_mesured_profiles/    # Data preprocessing and loading
    ├── 02_process_predicted_profiles/   # Processing model predictions
    ├── 03_evaluate_predicted_profiles/  # Model evaluation and comparison
    └── 04_predict_drug_response/        # Drug response prediction
```

## Dependencies

The project requires Python 3.10 and the following key listed in `environment.yml`.

## Installation

### Option 1: Using the Provided Container (Recommended)
**Note**: The container automatically activates the `drug-response-prediction` conda environment, so you don't need to manually activate it.

1. Clone the repository:
```bash
git clone <repository-url>
cd drug_response_prediction
```

2. Build the Singularity/Apptainer container:
```bash
# Build the container from the definition file
apptainer build drug-response-prediction.sif drug-response-prediction.def
```

3. Setup configuration (optional):
```bash
# Copy example configuration
cp config_example.yaml config.yaml
# Edit config.yaml to match your system paths
```

### Option 2: Manual Conda Installation

If you prefer to use conda directly:

1. Clone the repository:
```bash
git clone <repository-url>
cd drug_response_prediction
```

2. Create the conda environment:
```bash
conda env create -f environment.yml
conda activate drug-response-prediction
```

3. Setup configuration (optional):
```bash
# Copy example configuration
cp config_example.yaml config.yaml
# Edit config.yaml to match your system paths
```

## Configuration

The project uses a centralized configuration system for managing paths and parameters:

- **`config.py`**: Main configuration module with default settings
- **`config_example.yaml`**: Example configuration file for customization
- **`scripts/utils.py`**: Utility functions for backward compatibility

### Quick Setup

```python
from config import setup_project
config = setup_project()  # Creates directories and sets up logging
```

### Custom Configuration

```python
from config import Config
config = Config('config.yaml')  # Load custom configuration
```

## Datasets
To facilitate reproduction, we have made all processed files available here: https://surfdrive.surf.nl/s/Fdg4spN2zbtkwMa.

The raw datasets were retrieved from their original source:
- [McFarland et al. (2020)](https://figshare.com/s/139f64b495dea9d88c70): Large-scale drug sensitivity screening across cancer cell lines 
- [SciPlex, Srivatsan et al. (2020)](https://figshare.com/s/139f64b495dea9d88c70): Single-cell perturbation dataset with drug treatments

## Usage

The analysis follows a sequential workflow:

#### 1. Process Measured Profiles
**Using the container:**
```bash
apptainer exec drug-response-prediction.sif python scripts/01_process_measured_profiles/mcfarland_data_loaders.py
apptainer exec drug-response-prediction.sif python scripts/01_process_measured_profiles/sciplex_data_loaders.py
```

#### 2. Process Predicted Profiles
**Using the container:**

```bash
apptainer exec drug-response-prediction.sif python scripts/02_process_predicted_profiles/process_predictions_sciplex.py
```

#### 3. Evaluate Predicted Profiles
**Using the container:**
```bash
apptainer exec drug-response-prediction.sif python scripts/03_evaluate_predicted_profiles/evaluate_predictions_sciplex.py
```

#### 4. Predict Drug Response
**Using the container:**
```bash
apptainer exec drug-response-prediction.sif python scripts/04_predict_drug_response/response_prediction.py
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{brouwer2025drug,
  title={Drug Response Prediction Provides a Biologically Relevant Benchmark for Perturbation Response Models},
  author={Brouwer, Niek and Damyanov, Martin and Argelo, Jonas and Vis, Dani{\"e}l J and Reinders, Marcel JT and Wessels, Lodewyk FA},
  journal={[Journal Name]},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please contact:
- Niek Brouwer: n.brouwer-1@tudelft.nl

## Acknowledgments

We thank the developers of CPA, GEARS, and scFoundation for making their models publicly available. This research is part of the Oncode Accelerator Project that has received funding from the Dutch National Growth Fund (NGF).
