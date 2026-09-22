# Resources

Reference tables used across preprocessing, profile evaluation, and drug-response prediction. Paths are resolved via `config.RESOURCES_DIR`.

| File | Role |
|------|------|
| [`OS_scRNA_gene_index.19264.tsv`](OS_scRNA_gene_index.19264.tsv) | 19,264-gene vocabulary for expression alignment |
| [`mcfarland_drug_to_perturbation.csv`](mcfarland_drug_to_perturbation.csv) | MIX-seq experiment → drug → gene-target |
| [`sciplex_drug_to_perturbation.csv`](sciplex_drug_to_perturbation.csv) | SciPlex3 product name → gene-target |
| [`mcfarland_sensitivity_info.csv`](mcfarland_sensitivity_info.csv) | Cell-line × target sensitivity labels (McFarland) |
| [`sciplex_sensitivity_info.csv`](sciplex_sensitivity_info.csv) | Cell-line × drug sensitivity at 10 µM (SciPlex3) |
| [`sciplex3_response_variable_mapping.csv`](sciplex3_response_variable_mapping.csv) | Dose-level SciPlex3 viability plus matched screens |
| [`drug_smiles.csv`](drug_smiles.csv) | Canonical SMILES for McFarland and SciPlex3 drugs |
| [`drug_ecfp_2048.npz`](drug_ecfp_2048.npz) | Precomputed Morgan fingerprints (radius 2, 2048 bits) |
| [`sciplex_split_identifiers.csv`](sciplex_split_identifiers.csv) | Cell-level SciPlex3 obs with CV folds |
| [`mcfarland_split_identifiers.csv`](mcfarland_split_identifiers.csv) | Cell-level McFarland obs with CV folds |

---

## Gene vocabulary

### `OS_scRNA_gene_index.19264.tsv`

Ordered list of 19,264 gene symbols (`gene_name`, `index`) used by scFoundation. Processed SciPlex3 and McFarland profiles are aligned to this list; genes absent from the assay are padded with zeros and marked `var['mask']==1`.

**Source:** scFoundation gene vocabulary (Hao et al., *Nat Methods* 2024). Canonical copy: [biomap-research/scFoundation `OS_scRNA_gene_index.19264.tsv`](https://github.com/biomap-research/scFoundation/blob/main/OS_scRNA_gene_index.19264.tsv).

---

## Drug → perturbation maps

Gene targets use the GEARS-style `GENE+ctrl` encoding (or `GENE1+GENE2` for dual-target compounds). These maps join chemical labels to the gene-level condition names used by GEARS and scFoundation. Targets are taken from the original dataset when present; otherwise they were retrieved from [DrugBank](https://go.drugbank.com/).

### `mcfarland_drug_to_perturbation.csv`

13 MIX-seq treatments: 11 small molecules plus two CRISPR guides (`sgGPX4`, `sgOR2J2`).

| Column | Meaning |
|--------|---------|
| `dataset` | MIX-seq experiment folder name (e.g. `Afatinib_expt10`) |
| `drug` | Compound or guide name |
| `target` | Canonical gene target |

**Source:** Experiment names follow the MIX-seq figshare layout (McFarland et al., *Nat Commun* 2020; [figshare](https://figshare.com/s/139f64b495dea9d88c70)). Gene targets come from the original MIX-seq annotations when available, otherwise from DrugBank (e.g. Afatinib → `EGFR+ctrl`).

### `sciplex_drug_to_perturbation.csv`

189 SciPlex3 product names mapped to 78 unique gene targets. Used to set `gene_target` / `perturbation_match` on SciPlex cells and to match GEARS/scFoundation predictions (gene-target `condition`) to observed drug names.

**Source:** Product names are the SciPlex3 compound list (Srivatsan et al., *Science* 2020). The raw SciPlex obs `target` field is often a protein-family label (JAK, HIF, …), not a gene id; when a gene-level target was missing it was retrieved from DrugBank and stored here in GEARS format.

---

## Sensitivity labels

### `mcfarland_sensitivity_info.csv`

8,999 rows: DepMap/CCLE cell lines × 11 chemical targets (`cell_line`, `target`, `sens`, `sens_label`). `sens` is a continuous viability-derived score (higher = more sensitive), clipped away from 0/1. `sens_label` is the binary call at threshold 0.2. CRISPR treatments (`GPX4+ctrl`, `OR2J2+ctrl`) have no chemical-screen AUC and are absent.

**Source:** Per-drug tables under `cell_line_features/` in the MIX-seq figshare dump. McFarland compiled those scores from [GDSC](https://www.cancerrxgene.org/) and [PRISM](https://depmap.org/) AUCs (quantile-normalized and averaged when both were available; see McFarland et al. Methods). This file is the curated join of those tables onto `mcfarland_drug_to_perturbation.csv` targets. Tissue labels at runtime come from DepMap `metadata.csv` (`Disease`), not this file.

### `sciplex3_response_variable_mapping.csv`

2,259 rows covering all SciPlex3 `(cell_type, product_name, dose)` combinations (A549, K562, MCF7; 0–10 µM). Primary viability is the SciPlex-inferred score (`viability` / `sciplex_signal_remaining`; complete). Additional columns match the same treatments to large-scale screens where a compound–line pair exists:

- **CTRP** — `cpd_conc_umol`, `cpd_avg_pv`, `ctrp_signal_remaining`
- **PRISM / DepMap** — `prism_lfc`, `prism_lfc_dose`, `prism_lfc_screen_id` (`HTS`, `HTS002`, `MTS*`), `prism_lfc_source` (primary/secondary), `prism_signal_remaining`, `depmap_id`
- **GDSC** — `gdsc_signal_remaining`, `gdsc_dose` (columns present; currently unused / empty)
- **Identifiers** — Sanger `SIDM` and DepMap `ACH-*` for the three lines; SciPlex SMILES (`sciplex_smiles`)

`signal_remaining` is a filled remaining-signal column (SciPlex, else PRISM/CTRP when available).

**Sources:** SciPlex3 viability from recovered-cell / nuclear-hash scores in Srivatsan et al. Compound–line matches to [CTRP](https://portals.broadinstitute.org/ctrp/), [PRISM](https://depmap.org/), and [GDSC](https://www.cancerrxgene.org/). Cell-line ids: MCF7 `SIDM00148` / `ACH-000019`, A549 `SIDM00903` / `ACH-000681`, K562 `SIDM00791` / `ACH-000006`.

### `sciplex_sensitivity_info.csv`

564 rows: all drug × cell-line pairs at **10 µM** (highest dose kept in scRNA processing). `y` and `sens` are `1 − viability`. `condition` is the space-stripped product name used for joins; `target` is filled from `sciplex_drug_to_perturbation.csv` where a mapping exists (39 rows have no target).

**Source:** Built from `sciplex3_response_variable_mapping.csv` (SciPlex viability) plus the SciPlex drug→target map.

```bash
python scripts/01_process_measured_profiles/build_sciplex_sensitivity_info.py
```

---

## Chemical structure features

### `drug_smiles.csv`

Canonical SMILES for every McFarland and SciPlex3 drug key (`drug_key`, `smiles`, `source`). Lookups go through the PubChem PUG REST name endpoint. `source` is `pubchem`, `missing`, or `non_small_molecule` (Vehicle, `sgGPX4`, `sgOR2J2`).

**Source:** [PubChem](https://pubchem.ncbi.nlm.nih.gov/). Drug names from the two `*_drug_to_perturbation.csv` tables.

```bash
python scripts/04_predict_drug_response/build_drug_smiles.py
```

### `drug_ecfp_2048.npz`

Compressed NumPy archive: `drug_keys`, `fps` (202 × 2048 `float32`), `n_bits=2048`, `radius=2`. Morgan (ECFP) bit vectors from RDKit `GetMorganFingerprintAsBitVect`. Drugs without SMILES get a zero vector. Written by the same script (requires RDKit). Loaded at prediction time so the cluster image does not need RDKit.

---

## Split identifier tables

Cell-level AnnData `obs` exports from the processed h5ads after QC, perturbation matching, and 5-fold assignment (`create_sciplex_splits.py`, `create_mcfarland_splits.py`). Same content as the `*_obs.csv` written next to each processed h5ad. Controls have `fold=-1` and are `train` in every `split_*` column. Treated hold-out units are `(cell_type, condition)` (within cell line for SciPlex3, within tissue for McFarland).

These CSVs are not read by the pipeline; they are the shareable split tables (expression lives in the h5ads).

### `sciplex_split_identifiers.csv`

SciPlex3 cells at vehicle or 10 µM. Original SciPlex/scperturb columns (`pathway`, `g1s_score`, `g2m_score`, `replicate`, `vehicle`, …) plus this project's `gene_target`, `perturbation_match`, `fold`, `split_0`–`split_4`, `sens`, `sens_label`, and QC flags.

**Sources:** Cell barcodes and original metadata from SciPlex3 (Srivatsan et al.; GEO [GSE139944](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE139944); this repo reads `Srivatsan_2019_raw.h5ad`, the [scPerturb](http://projects.sanderlab.org/scperturb/) distribution). Folds, QC, and sensitivity are added here.

### `mcfarland_split_identifiers.csv`

MIX-seq cells with `drug`, `tissue`, folds, sensitivity, and MIX-seq QC (`cell_quality`, `doublet_GMM_prob`, `singlet_ID`, …).

**Sources:** Cell barcodes and MIX-seq metadata from McFarland et al. ([figshare](https://figshare.com/s/139f64b495dea9d88c70)). Tissue is DepMap `Disease`. Folds, QC, and sensitivity are added here.

---

## Dataset citations

- **SciPlex3:** Srivatsan et al. (2020) *Science* — [doi:10.1126/science.aax6234](https://doi.org/10.1126/science.aax6234)
- **MIX-seq / McFarland:** McFarland et al. (2020) *Nat Commun* — [doi:10.1038/s41467-020-17440-w](https://doi.org/10.1038/s41467-020-17440-w)
- **scFoundation gene index:** Hao et al. (2024) *Nat Methods* — [doi:10.1038/s41592-024-02305-9](https://doi.org/10.1038/s41592-024-02305-9)
- **Sensitivity screens:** GDSC (Yang et al. / Iorio et al.), PRISM (Corsello et al.), CTRP (Seashore-Ludlow et al. / Rees et al.)
- **Drug targets (when missing from the original dataset):** [DrugBank](https://go.drugbank.com/)
- **SMILES:** PubChem
