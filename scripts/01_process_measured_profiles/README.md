# 01: Process Measured Profiles

Processed files for reproduction: https://surfdrive.surf.nl/s/DbRwLCbCXcbiC2E.

This stage builds processed AnnData objects with QC flags, perturbation matches, sensitivity labels, and 5-fold train/test splits.

## Outputs

```
data/sciplex_processed/Srivatsan_2019_raw_processed.h5ad
data/mcfarland_processed/all_cell_lines.h5ad
results/01_process_measured_profiles/   # QC tables, gene assay-missing masks
data/observed_pseudobulk/sciplex_count_*.csv
data/observed_pseudobulk/mcfarland_count_*.csv
resources/{sciplex,mcfarland}_split_identifiers.csv   # optional obs exports
```

## Scripts

| Script | Role |
|--------|------|
| `create_sciplex_splits.py` | Raw SciPlex → filter control+10 µM, QC, normalize, align genes, assign folds |
| `create_mcfarland_splits.py` | Annotate existing `all_cell_lines.h5ad`, normalize, assign folds |
| `mcfarland_data_loaders.py` | **Bootstrap only**: merge MIX-seq raw mtx → `all_cell_lines.h5ad` (skip if using Surfdrive processed data) |
| `create_qc_and_count_pseudobulk.py` | Cell-level QC tables + count-based (sum UMI → size-factor log1p → mean) pseudobulks |
| `build_sciplex_sensitivity_info.py` | Rebuild `resources/sciplex_sensitivity_info.csv` |
| `preprocess_utils.py` / `qc_utils.py` | Shared helpers |

## Run

```bash
python scripts/01_process_measured_profiles/create_sciplex_splits.py
python scripts/01_process_measured_profiles/create_mcfarland_splits.py
python scripts/01_process_measured_profiles/create_qc_and_count_pseudobulk.py
```

Reassign folds on an existing processed h5ad with `--refresh-splits-only` (skip QC and normalize).

## Splits

Treated pairs are the hold-out unit: all cells of a pair share a fold, so a pair is never in both train and test.

- **SciPlex**: `(cell_type, condition)` within each cell line (`condition` = drug; gene labels in `gene_target`)
- **McFarland**: `(cell_type, condition)` within each tissue
- Controls: `fold=-1` (always train)

`create_qc_and_count_pseudobulk.py`: SciPlex groups by `condition` (vehicles as `ctrl`); McFarland by `condition`. SciPlex count-level input must be raw UMIs (`layers['counts']` on the processed file, or the original `sciplex_raw` h5ad). McFarland counts are read from staff-bulk `mcfarland_raw/*/matrix.mtx`.
