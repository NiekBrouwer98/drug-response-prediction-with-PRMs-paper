# 04: Predict Drug Response

This directory contains scripts for predicting drug sensitivity using both measured and predicted gene expression profiles as features.

## Overview

The scripts in this directory implement the core drug response prediction pipeline, including:
- Feature selection and preprocessing for machine learning models
- Cross-validation and model training with different feature sets
- Drug sensitivity prediction using ElasticNet and logistic regression
- Comprehensive evaluation across different experimental designs
- Model interpretation and biological pathway analysis

## Prediction Strategies

### Feature Types
1. **Pre-treatment profiles**: Baseline gene expression before treatment
2. **Post-treatment profiles**: Gene expression after treatment (observed or predicted)
3. **Log-fold change (LFC)**: Difference between post- and pre-treatment expression

### Experimental Designs
1. **Leave-one-out (LOO)**: Train on all but one cell line, test on held-out cell line
2. **Leave-drug-out (LDO)**: Train on all but one drug, test on held-out drug
3. **Leave-tissue-out (LTO)**: Train on all but one tissue type, test on held-out tissue
4. **Independent test set**: Train on one dataset, test on completely independent dataset

### Model Types
1. **ElasticNet regression**: Regularized linear regression for continuous sensitivity
2. **Logistic regression**: Binary classification for sensitive vs. resistant
3. **Two-part model**: Binary classification + regression for zero-inflated data

## Files

### Main Prediction Scripts
- **`response_prediction.py`**: Core drug sensitivity prediction pipeline
  - Implements multiple prediction strategies (pre-treatment, post-treatment, LFC)
  - Cross-validation with different experimental designs (LOO, LDO, LTO)
  - Feature selection and model training with ElasticNet
  - Comprehensive evaluation across cell lines and treatments

### Analysis Notebooks
- **`create_figures_mcfarland.ipynb`**: Generate prediction-related figures for McFarland dataset
- **`create_figures_sciplex.ipynb`**: Generate prediction-related figures for Sciplex dataset
- **`create_figures_correlations.ipynb`**: Generate figures for interpretation of gene-gene correlations
- **`create_figures_GSEA.ipynb`**: Generate figures for GSEA