import numpy as np
import pandas as pd
import os
import logging
import scanpy as sc

from config import config

logger = logging.getLogger(__name__)

data_dir = str(config.DATA_DIR)
resources_dir = str(config.RESOURCES_DIR)

'''Data getters'''
def get_sensitivity_information(expt_files):
    cell_line_info = pd.DataFrame(columns = ['DEPMAP_ID', 'CCLE_ID', 'sens', 'drug','time'])

    for file in expt_files:
        single_drug_info = pd.read_csv(file)
        single_drug_info = single_drug_info[['DEPMAP_ID', 'CCLE_ID', 'sens']]
        meta_data = str(file).replace(os.path.join(data_dir, 'mcfarland_raw','cell_line_features'),'')
        meta_data = meta_data.replace('\\','')
        meta_data = meta_data.replace('/','')
        drug_info = meta_data.split('_')
        single_drug_info['drug'] = drug_info[0]
        if len(drug_info) > 1:
            single_drug_info['experiment'] = '_'.join(drug_info[1:]).removesuffix('.csv')

        cell_line_info = pd.concat([cell_line_info, single_drug_info])

    cell_line_info= cell_line_info.drop(columns=['time','DEPMAP_ID'])
    # Parse CCLE_ID robustly: some entries may not contain an underscore.
    split_cols = cell_line_info['CCLE_ID'].astype(str).str.split('_', expand=True, n=1)
    cell_line_info['cell_line'] = split_cols[0].replace({'': np.nan, 'nan': np.nan})
    if 1 in split_cols.columns:
        cell_line_info['tissue'] = split_cols[1]
    else:
        cell_line_info['tissue'] = np.nan
    cell_line_info = cell_line_info.drop_duplicates()

    return(cell_line_info)

def get_AUCs():
    # Use curated sensitivity labels from resources (single source of truth).
    filepath = os.path.join(resources_dir, 'mcfarland_sensitivity_info.csv')
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing required sensitivity file: {filepath}")
    sensitivity_info = pd.read_csv(filepath)

    # Cap values to mitigate outliers
    sensitivity_info['sens'] = np.where(sensitivity_info['sens'] < 0 , 0, sensitivity_info['sens'])

    return sensitivity_info

def get_sens_labels(threshold):
    sensitivity_info = get_AUCs()
    
    sensitivity_info['sens_label'] = np.where(sensitivity_info['sens'] < threshold, 0, 1)

    return(sensitivity_info)


def _load_mcfarland_depmap_cell_line_tissue_map() -> pd.DataFrame:
    """
    Map DepMap/CCLE-style cell line name -> tissue (DepMap `Disease` field) from
    data/mcfarland_raw/cell_line_features/metadata.csv.

    CCLE_ID is like ``HCC827_LUNG``; we take the first underscore token as ``cell_line``
    (matching ``mcfarland_sensitivity_info.csv``) and use ``Disease`` as tissue label.
    """
    meta_path = os.path.join(data_dir, 'mcfarland_raw', 'cell_line_features', 'metadata.csv')
    if not os.path.isfile(meta_path):
        logger.warning("DepMap cell-line metadata not found: %s", meta_path)
        return pd.DataFrame(columns=['cell_line', 'tissue'])

    meta = pd.read_csv(meta_path)
    if 'CCLE_ID' not in meta.columns or 'Disease' not in meta.columns:
        logger.error(
            "McFarland metadata.csv missing expected columns CCLE_ID / Disease (got: %s)",
            list(meta.columns),
        )
        return pd.DataFrame(columns=['cell_line', 'tissue'])

    sp = meta['CCLE_ID'].astype(str).str.split('_', n=1, expand=True)
    meta = meta.assign(
        cell_line=sp[0].astype(str).str.strip().str.upper(),
        tissue=meta['Disease'].astype(str).str.strip().str.replace(' ', '_', regex=False).str.upper(),
    )
    out = meta[['cell_line', 'tissue']].dropna(subset=['cell_line', 'tissue'])
    out = out[out['cell_line'].ne('') & out['tissue'].ne('')]
    out = out.drop_duplicates(subset=['cell_line'], keep='first')
    logger.info(
        "_load_mcfarland_depmap_cell_line_tissue_map: %d unique cell_line -> tissue from %s",
        len(out),
        meta_path,
    )
    return out


def get_McFarland_sensitivityinfo():
    filepath = os.path.join(resources_dir, 'mcfarland_sensitivity_info.csv')
    if os.path.exists(filepath):
        cellline_sensitivity_info = pd.read_csv(filepath)
    else:
        drug_to_perturbation = pd.read_csv(os.path.join(resources_dir, 'mcfarland_drug_to_perturbation.csv'))
        sensitivity_info = get_sens_labels(0.2)

        cellline_sensitivity_info = sensitivity_info.copy()
        cellline_sensitivity_info = pd.merge(cellline_sensitivity_info, drug_to_perturbation, on='drug', how='left')
        cellline_sensitivity_info = cellline_sensitivity_info[['cell_line', 'target', 'sens','sens_label' ]]
        cellline_sensitivity_info.drop_duplicates(inplace=True)

        cellline_sensitivity_info = cellline_sensitivity_info.dropna(axis=0)
        cellline_sensitivity_info['sens'] = np.where(cellline_sensitivity_info['sens'] >= 1, 0.999, cellline_sensitivity_info['sens'])
        cellline_sensitivity_info['sens'] = np.where(cellline_sensitivity_info['sens'] <= 0, 0.001, cellline_sensitivity_info['sens'])

    cellline_sensitivity_info['cell_line'] = (
        cellline_sensitivity_info['cell_line'].astype(str).str.strip().str.upper()
    )
    cellline_sensitivity_info['target'] = cellline_sensitivity_info['target'].astype(str).str.strip()

    tmap = _load_mcfarland_depmap_cell_line_tissue_map()
    if len(tmap) > 0 and 'tissue' not in cellline_sensitivity_info.columns:
        cellline_sensitivity_info = cellline_sensitivity_info.merge(tmap, on='cell_line', how='left')
        logger.info(
            "get_McFarland_sensitivityinfo: merged DepMap Disease as tissue for %d / %d rows",
            int(cellline_sensitivity_info['tissue'].notna().sum()),
            len(cellline_sensitivity_info),
        )

    logger.info(
        "get_McFarland_sensitivityinfo: %d rows (cell_line, target, sens; + tissue from DepMap metadata when available)",
        len(cellline_sensitivity_info),
    )

    return cellline_sensitivity_info


def get_McFarland_sensitivityinfo_for_profile_merge() -> pd.DataFrame:
    """Sensitivity columns for merging onto pseudobulk; drops ``tissue`` to avoid duplicate columns (tissue from ``get_McFarland_mean_data``)."""
    df = get_McFarland_sensitivityinfo()
    return df.drop(columns=['tissue'], errors='ignore')


def _normalize_sciplex_product_name(names: pd.Series) -> pd.Series:
    """Strip leading/trailing whitespace and remove all internal spaces for SciPlex drug joins."""
    out = names.astype(str).str.strip().str.replace(r'\s+', '', regex=True)
    # chemCPA / PubChem-style aliases → names used in observed / split identifiers
    out = out.str.replace('(plus)', '(+)', regex=False)
    out = out.replace({'JQ1': '(+)-JQ1'})
    return out


def merge_sciplex_sensitivity(
    profiles: pd.DataFrame,
    sensitivity: pd.DataFrame,
    *,
    value_col: str = 'y',
) -> pd.DataFrame:
    """Merge sensitivity onto profiles using normalized product_name + cell_line.

    Falls back to ``gene_target``/``target`` join when product names do not match
    (e.g. pseudobulk ``condition`` still holds perturbation labels).
    """
    if profiles.empty:
        logger.warning('merge_sciplex_sensitivity: profiles dataframe is empty')
        return profiles.copy()

    left = profiles.copy()
    if 'condition' in left.columns:
        left['_profile_key'] = _normalize_sciplex_product_name(left['condition'])
    else:
        raise KeyError('SciPlex profiles missing condition column')
    if 'cell_line' not in left.columns and 'cell_type' in left.columns:
        left['cell_line'] = left['cell_type']
    if 'cell_line' in left.columns:
        left['cell_line'] = left['cell_line'].astype(str).str.strip().str.upper()

    right = sensitivity.copy()
    key_col = 'condition' if 'condition' in right.columns else value_col
    if 'condition' in right.columns:
        right['_sens_key'] = _normalize_sciplex_product_name(right['condition'])
    else:
        right['_sens_key'] = _normalize_sciplex_product_name(right[key_col])
    if 'cell_line' in right.columns:
        right['cell_line'] = right['cell_line'].astype(str).str.strip().str.upper()

    val_out = value_col if value_col in right.columns else ('y' if 'y' in right.columns else value_col)
    sens_cols = ['_sens_key', 'cell_line']
    if val_out in right.columns:
        sens_cols.append(val_out)
    right = right[sens_cols].drop_duplicates(subset=['_sens_key', 'cell_line'], keep='first')

    merged = left.merge(
        right,
        left_on=['_profile_key', 'cell_line'],
        right_on=['_sens_key', 'cell_line'],
        how='left',
    )
    if val_out not in merged.columns and val_out != 'y' and 'y' in merged.columns:
        merged[val_out] = merged['y']

    n_hit = int(merged[val_out].notna().sum()) if val_out in merged.columns else 0
    if n_hit == 0:
        logger.warning(
            'SciPlex product-name merge matched 0/%d rows; trying gene_target merge',
            len(merged),
        )
        by_target = _load_sciplex_sensitivity_by_target()
        target_val = value_col if value_col in by_target.columns else 'y'
        merged = left.merge(
            by_target,
            left_on=['_profile_key', 'cell_line'],
            right_on=['target', 'cell_line'],
            how='left',
            suffixes=('', '_target'),
        )
        if target_val in merged.columns:
            merged[val_out] = merged[target_val]
        if 'condition_product' in merged.columns:
            merged['condition'] = merged['condition_product'].fillna(merged['condition'])
        n_hit = int(merged[val_out].notna().sum()) if val_out in merged.columns else 0

    logger.info(
        'merge_sciplex_sensitivity: matched %d / %d rows on %s',
        n_hit,
        len(merged),
        val_out,
    )
    if n_hit == 0 and len(merged) > 0:
        prof_sample = left[['_profile_key', 'cell_line']].drop_duplicates().head(5)
        logger.error('Profile key sample:\n%s', prof_sample.to_string(index=False))

    merged = merged.drop(columns=['_profile_key', '_sens_key'], errors='ignore')
    if 'condition' not in merged.columns:
        merged['condition'] = merged.get('_profile_key', left['condition'])
    return merged


def _load_sciplex_sensitivity_by_target() -> pd.DataFrame:
    """Sensitivity keyed by perturbation target (``BRD4+ctrl``) and cell_line."""
    filepath = os.path.join(resources_dir, 'sciplex_sensitivity_info.csv')
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'SciPlex sensitivity table not found: {filepath}')
    df = pd.read_csv(filepath)
    value_col = 'y' if 'y' in df.columns else 'sens'
    if 'target' not in df.columns:
        raise ValueError('sciplex_sensitivity_info.csv missing target column for fallback merge')
    out = df.copy()
    out['target'] = out['target'].astype(str).str.strip()
    out['cell_line'] = out['cell_line'].astype(str).str.strip().str.upper()
    out[value_col] = pd.to_numeric(out[value_col], errors='coerce')
    if 'product_name' in out.columns:
        out['condition_product'] = _normalize_sciplex_product_name(out['product_name'])
    else:
        out['condition_product'] = out['target']
    cols = ['target', 'cell_line', value_col, 'condition_product']
    return out[cols].dropna(subset=['target', 'cell_line', value_col]).drop_duplicates(
        subset=['target', 'cell_line'], keep='first'
    )


def _load_sciplex_sensitivity_table() -> pd.DataFrame:
    """SciPlex sensitivity keyed by normalized ``product_name`` (``condition`` column)."""
    filepath = os.path.join(resources_dir, 'sciplex_sensitivity_info.csv')
    if os.path.exists(filepath):
        sensitivity_df = pd.read_csv(filepath)
    else:
        frames = []
        for cell_line in ['mcf7', 'k562', 'a549']:
            sensitivity = pd.read_csv(
                os.path.join(results_dir, '01_process_observed_profiles', f'sciplex{cell_line}_msd.csv')
            )
            sensitivity = sensitivity[(sensitivity['dose'] == 10000) | (sensitivity['dose'] == 0)][
                ['condition', '1-viability']
            ].drop_duplicates()
            sensitivity['cell_line'] = cell_line.upper()
            sensitivity = sensitivity.rename(columns={'1-viability': 'y', 'condition': 'product_name'})

            pert_to_drug = pd.read_csv(
                os.path.join(resources_dir, 'sciplex_drug_to_perturbation.csv'), index_col=0
            )
            pert_to_drug['product_name_key'] = _normalize_sciplex_product_name(pert_to_drug['product_name'])
            sensitivity['product_name_key'] = _normalize_sciplex_product_name(sensitivity['product_name'])
            sensitivity = pd.merge(
                sensitivity,
                pert_to_drug[['product_name', 'target', 'product_name_key']],
                on='product_name_key',
                how='left',
            )
            frames.append(sensitivity)

        sensitivity_df = pd.concat(frames, axis=0, ignore_index=True)
        sensitivity_df.to_csv(filepath, index=False)

    drug_col = 'product_name' if 'product_name' in sensitivity_df.columns else 'condition'
    if drug_col not in sensitivity_df.columns:
        raise ValueError(
            'SciPlex sensitivity table must contain product_name or condition; '
            f'got columns {list(sensitivity_df.columns)}'
        )

    value_col = 'y' if 'y' in sensitivity_df.columns else 'sens'
    out = sensitivity_df.copy()
    out['condition'] = _normalize_sciplex_product_name(out[drug_col])
    out['cell_line'] = out['cell_line'].astype(str).str.strip().str.upper()
    out['y'] = pd.to_numeric(out[value_col], errors='coerce')
    out = out.dropna(subset=['condition', 'cell_line', 'y'])
    return out.drop_duplicates(subset=['condition', 'cell_line'], keep='first')[['condition', 'cell_line', 'y']]


def get_sciplex_AUCs():
    return _load_sciplex_sensitivity_table()


def get_sciplex_pair_fold_keys() -> pd.DataFrame:
    """Official SciPlex ``(cell_line, condition, fold)`` keys for all treated pairs (~188×3).

    Built from ``resources/sciplex_split_identifiers.csv`` (controls have fold=-1 and are
    dropped). Prefer this over CPA fold tables when expanding measured / baseline
    profiles so the full drug grid is retained.
    """
    path = os.path.join(resources_dir, 'sciplex_split_identifiers.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Missing {path}; run create_sciplex_splits.py to write fold identifiers'
        )
    raw = pd.read_csv(path, usecols=['cell_type', 'condition', 'fold'])
    treated = raw.loc[pd.to_numeric(raw['fold'], errors='coerce').fillna(-1).ge(0)].copy()
    treated['cell_line'] = treated['cell_type'].astype(str).str.strip().str.upper()
    treated['condition'] = _normalize_sciplex_product_name(treated['condition'])
    treated['fold'] = pd.to_numeric(treated['fold'], errors='coerce').astype(int)
    keys = treated[['cell_line', 'condition', 'fold']].drop_duplicates()
    logger.info(
        'get_sciplex_pair_fold_keys: %d pairs across %d conditions × %d lines',
        len(keys),
        keys['condition'].nunique(),
        keys['cell_line'].nunique(),
    )
    return keys


def _mcfarland_profile_csv_paths(kind: str) -> tuple[str, str, str]:
    pb = os.path.join(data_dir, "observed_pseudobulk")
    if kind == "count":
        return (
            os.path.join(pb, "mcfarland_count_pre.csv"),
            os.path.join(pb, "mcfarland_count_post.csv"),
            os.path.join(pb, "mcfarland_count_LFC.csv"),
        )
    return (
        os.path.join(pb, "mcfarland_mean_pre_all_celllines.csv"),
        os.path.join(pb, "mcfarland_mean_post_all_celllines.csv"),
        os.path.join(pb, "mcfarland_mean_LFC_all_celllines.csv"),
    )


def _drop_assay_missing_gene_columns(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    mask_path = os.path.join(str(config.RESULTS_01_DIR), f"{dataset}_gene_assay_mask.csv")
    if not os.path.exists(mask_path):
        return df
    mask = pd.read_csv(mask_path, index_col=0).iloc[:, 0].astype(bool)
    missing = [gene for gene in mask.index[mask] if gene in df.columns]
    if missing:
        logger.info("Dropped %d assay-missing genes from %s profiles", len(missing), dataset)
        return df.drop(columns=missing)
    return df


def get_McFarland_mean_data(kind: str = "mean"):
    pre_path, post_path, lfc_path = _mcfarland_profile_csv_paths(kind)
    mean_observed_pre_treatment = pd.read_csv(pre_path, index_col=0)
    mean_observed_post_treatment = pd.read_csv(post_path, index_col=0)
    mean_observed_LFC = pd.read_csv(lfc_path, index_col=0)

    logger.info(
        "get_McFarland_mean_data: read CSV rows pre=%d post=%d LFC=%d; pre ncols=%d",
        len(mean_observed_pre_treatment),
        len(mean_observed_post_treatment),
        len(mean_observed_LFC),
        mean_observed_pre_treatment.shape[1],
    )
    mean_observed_pre_treatment = mean_observed_pre_treatment.rename(columns={'cell_type':'cell_line'})
    mean_observed_post_treatment = mean_observed_post_treatment.rename(columns={'cell_type':'cell_line'})
    mean_observed_LFC = mean_observed_LFC.rename(columns={'cell_type':'cell_line'})

    for _df in (mean_observed_pre_treatment, mean_observed_post_treatment, mean_observed_LFC):
        _df['cell_line'] = (
            _df['cell_line'].astype(str).str.strip().str.split('_').str[0].str.strip().str.upper()
        )
        if 'condition' in _df.columns:
            _df['condition'] = _df['condition'].astype(str).str.strip()

    tissues = get_tissue_labels()

    def _merge_tissue_lookup(df: pd.DataFrame, label: str) -> pd.DataFrame:
        file_tissue = df['tissue'].copy() if 'tissue' in df.columns else None
        out = df.drop(columns=['tissue'], errors='ignore').merge(tissues, on='cell_line', how='left')
        if file_tissue is not None:
            out['tissue'] = file_tissue.combine_first(out['tissue'])
            n_from_file = int(file_tissue.notna().sum())
            if n_from_file > 0:
                logger.info(
                    "get_McFarland_mean_data [%s]: combined tissue from CSV column where present (%d non-null on file)",
                    label,
                    n_from_file,
                )
        return out

    mean_observed_pre_treatment_with_tissue = _merge_tissue_lookup(mean_observed_pre_treatment, 'pre')
    mean_observed_post_treatment_with_tissue = _merge_tissue_lookup(mean_observed_post_treatment, 'post')
    mean_observed_LFC_with_tissue = _merge_tissue_lookup(mean_observed_LFC, 'LFC')

    for name, dfm in (
        ('pre', mean_observed_pre_treatment_with_tissue),
        ('post', mean_observed_post_treatment_with_tissue),
        ('LFC', mean_observed_LFC_with_tissue),
    ):
        n = len(dfm)
        n_tissue = int(dfm['tissue'].notna().sum()) if 'tissue' in dfm.columns else 0
        n_cond = dfm['condition'].nunique() if 'condition' in dfm.columns else 0
        n_cl = dfm['cell_line'].nunique() if 'cell_line' in dfm.columns else 0
        logger.info(
            "get_McFarland_mean_data merge tissue [%s]: rows=%d non_null_tissue=%d (%.1f%%) unique cell_line=%d unique condition=%d",
            name,
            n,
            n_tissue,
            100.0 * n_tissue / max(n, 1),
            n_cl,
            n_cond,
        )
        if n_tissue < n and 'tissue' in dfm.columns:
            miss = dfm.loc[dfm['tissue'].isna(), 'cell_line'].drop_duplicates().head(12).tolist()
            logger.warning(
                "get_McFarland_mean_data [%s]: %d rows lack tissue after merge; example cell_line with no tissue map: %s",
                name,
                n - n_tissue,
                miss,
            )
        if 'condition' in dfm.columns and n:
            samp = dfm['condition'].drop_duplicates().head(8).tolist()
            logger.info("get_McFarland_mean_data [%s]: sample condition values: %s", name, samp)

    if kind == "count":
        mean_observed_pre_treatment_with_tissue = _drop_assay_missing_gene_columns(
            mean_observed_pre_treatment_with_tissue, "mcfarland"
        )
        mean_observed_post_treatment_with_tissue = _drop_assay_missing_gene_columns(
            mean_observed_post_treatment_with_tissue, "mcfarland"
        )
        mean_observed_LFC_with_tissue = _drop_assay_missing_gene_columns(
            mean_observed_LFC_with_tissue, "mcfarland"
        )

    return mean_observed_pre_treatment_with_tissue, mean_observed_post_treatment_with_tissue, mean_observed_LFC_with_tissue


def get_McFarland_count_data():
    return get_McFarland_mean_data(kind="count")

def get_sciplex_pre_treatment_data():
    result = []
    for cell_line in ['mcf7', 'k562', 'a549']:
        adata = sc.read_h5ad(os.path.join(data_dir, 'sciplex_processed', f'sciplex{cell_line}.h5ad'))
        adata_df = adata.to_df()
        drug_col = 'condition' if 'condition' in adata.obs.columns else 'product_name'
        adata_df['condition'] = adata.obs[drug_col].astype(str).to_numpy()
        ctrl_data = adata_df[adata_df['condition'].str.lower().isin(['vehicle', 'ctrl', 'control', 'dmso'])]
        result.append(ctrl_data)

    pre_data = pd.concat(result,axis=0)
    pre_data = pre_data.drop(columns=['condition'], errors='ignore')
    return(pre_data)

def get_sciplex_mean_data(kind: str = "mean"):
    infix = "mean" if kind == "mean" else "count"
    mean_pre_df = []
    mean_post_df = []
    mean_LFC_df = []

    for cell_line in ['mcf7', 'k562', 'a549']:
        mean_pre = pd.read_csv(os.path.join(data_dir, 'observed_pseudobulk', f'sciplex_{infix}_pre_{cell_line}.csv'), index_col=0).drop(columns=['cell_type'], errors='ignore')
        mean_post = pd.read_csv(os.path.join(data_dir, 'observed_pseudobulk', f'sciplex_{infix}_post_{cell_line}.csv'), index_col=0).drop(columns=['cell_type'], errors='ignore')
        mean_LFC = pd.read_csv(os.path.join(data_dir, 'observed_pseudobulk', f'sciplex_{infix}_LFC_{cell_line}.csv'), index_col=0)
        mean_pre['cell_line'] = cell_line.upper()
        mean_post['cell_line'] = cell_line.upper()
        mean_LFC['cell_line'] = cell_line.upper()
        mean_pre_df.append(mean_pre)
        mean_post_df.append(mean_post)
        mean_LFC_df.append(mean_LFC)

    mean_pre_df = pd.concat(mean_pre_df, axis=0)
    mean_post_df = pd.concat(mean_post_df, axis=0)
    mean_LFC_df = pd.concat(mean_LFC_df, axis=0)
    for _df in (mean_pre_df, mean_post_df, mean_LFC_df):
        if 'condition' in _df.columns:
            _df['condition'] = _normalize_sciplex_product_name(_df['condition'])
        if 'cell_line' in _df.columns:
            _df['cell_line'] = _df['cell_line'].astype(str).str.strip().str.upper()
    if kind == "count":
        mean_pre_df = _drop_assay_missing_gene_columns(mean_pre_df, "sciplex")
        mean_post_df = _drop_assay_missing_gene_columns(mean_post_df, "sciplex")
        mean_LFC_df = _drop_assay_missing_gene_columns(mean_LFC_df, "sciplex")

    n_cond = int(mean_post_df['condition'].nunique()) if 'condition' in mean_post_df.columns else 0
    n_pairs = (
        mean_post_df.groupby(['cell_line', 'condition']).ngroups
        if {'cell_line', 'condition'}.issubset(mean_post_df.columns)
        else len(mean_post_df)
    )
    logger.info(
        "get_sciplex_%s_data: post rows=%d unique conditions=%d (cell_line,condition) pairs=%d",
        infix,
        len(mean_post_df),
        n_cond,
        n_pairs,
    )
    if kind == "mean" and n_cond < 150:
        logger.warning(
            "SciPlex mean post has only %d conditions (expected ~188). "
            "Re-run scripts/02_process_predicted_profiles/create_pseudobulk.py "
            "against Srivatsan_2019_raw_processed.h5ad, or use kind='count'.",
            n_cond,
        )

    return mean_pre_df, mean_post_df, mean_LFC_df


def get_sciplex_count_data():
    return get_sciplex_mean_data(kind="count")

def _remap_sciplex_gene_target_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Map SciPlex GEARS/scFoundation gene-target ``condition`` onto product names."""
    mapping_path = os.path.join(resources_dir, 'sciplex_drug_to_perturbation.csv')
    if not os.path.exists(mapping_path):
        logger.warning(
            '_remap_sciplex_gene_target_conditions: missing %s; leaving conditions unchanged',
            mapping_path,
        )
        return df
    mapping = pd.read_csv(mapping_path, index_col=0)
    if 'target' not in mapping.columns or 'product_name' not in mapping.columns:
        raise ValueError(f'Expected target/product_name in {mapping_path}')
    target_to_label = (
        mapping.dropna(subset=['target', 'product_name'])
        .drop_duplicates(subset='target', keep='first')
        .set_index('target')['product_name']
    )
    out = df.copy()
    mapped = out['condition'].astype(str).map(target_to_label)
    out['condition'] = mapped.fillna(out['condition'].astype(str))
    out['condition'] = _normalize_sciplex_product_name(out['condition'])
    return out


def _finalize_sciplex_genetic_prm(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonize SciPlex genetic-PRM tables; preserve native ``fold`` when present."""
    out = df.copy()
    if 'cell_type' in out.columns and 'cell_line' not in out.columns:
        out = out.rename(columns={'cell_type': 'cell_line'})
    if 'perturbation' in out.columns and 'condition' not in out.columns:
        out = out.rename(columns={'perturbation': 'condition'})
    if 'cell_line' in out.columns:
        out['cell_line'] = out['cell_line'].astype(str).str.strip().str.upper()
    out = _remap_sciplex_gene_target_conditions(out)
    if 'tissue' not in out.columns:
        out['tissue'] = out['cell_line']
    out = _drop_non_gene_extra_columns(out)
    out = _normalize_split_to_fold(out)
    return out


def get_GEARS_predictions():
    """Load SciPlex GEARS post/LFC tables (expanded onto chemical fold keys downstream)."""
    post_predictions = []
    lfc_predictions = []
    for cell_line in ['mcf7', 'k562', 'a549']:
        predicted_cell_line = pd.read_csv(
            os.path.join(data_dir, 'GEARS_predictions', f'sciplex_mean_post_{cell_line}.csv'),
            index_col=0,
        )
        predicted_LFC_cell_line = pd.read_csv(
            os.path.join(data_dir, 'GEARS_predictions', f'sciplex_mean_LFC_{cell_line}.csv'),
        )
        post_predictions.append(_finalize_sciplex_genetic_prm(predicted_cell_line))
        lfc_predictions.append(_finalize_sciplex_genetic_prm(predicted_LFC_cell_line))

    post_predictions = pd.concat(post_predictions, axis=0)
    lfc_predictions = pd.concat(lfc_predictions, axis=0)
    return post_predictions, lfc_predictions


def get_GEARS_noreg_predictions():
    post_predictions = []
    lfc_predictions = []
    for cell_line in ['mcf7', 'k562', 'a549']:
        predicted_cell_line = pd.read_csv(
            os.path.join(data_dir, 'GEARS_noreg_predictions', f'sciplex_mean_post_{cell_line}.csv'),
            index_col=0,
        )
        predicted_LFC_cell_line = pd.read_csv(
            os.path.join(data_dir, 'GEARS_noreg_predictions', f'sciplex_mean_LFC_{cell_line}.csv'),
        )
        post_predictions.append(_finalize_sciplex_genetic_prm(predicted_cell_line))
        lfc_predictions.append(_finalize_sciplex_genetic_prm(predicted_LFC_cell_line))

    post_predictions = pd.concat(post_predictions, axis=0)
    lfc_predictions = pd.concat(lfc_predictions, axis=0)
    return post_predictions, lfc_predictions


def get_scfoundation_predictions():
    """Load SciPlex scFoundation post/LFC tables (expanded onto chemical fold keys downstream)."""
    post_predictions = []
    lfc_predictions = []
    for cell_line in ['mcf7', 'k562', 'a549']:
        predicted_cell_line = pd.read_csv(
            os.path.join(data_dir, 'scfoundation_predictions', f'sciplex_mean_post_{cell_line}.csv'),
            index_col=0,
        )
        predicted_LFC_cell_line = pd.read_csv(
            os.path.join(data_dir, 'scfoundation_predictions', f'sciplex_mean_LFC_{cell_line}.csv'),
        )
        post_predictions.append(_finalize_sciplex_genetic_prm(predicted_cell_line))
        lfc_predictions.append(_finalize_sciplex_genetic_prm(predicted_LFC_cell_line))

    post_predictions = pd.concat(post_predictions, axis=0)
    lfc_predictions = pd.concat(lfc_predictions, axis=0)
    return post_predictions, lfc_predictions

PROFILE_META_COLS = {
    'cell_line', 'cell_type', 'condition', 'tissue', 'fold', 'split',
    'n_cells', 'sens', 'target', 'sens_label', 'y', 'drug', 'dose',
    'cpa_pert', 'dataset', 'Unnamed: 0', 'perturbation', 'index', 'gene_target',
    'product_name',
}


def _normalize_split_to_fold(df: pd.DataFrame) -> pd.DataFrame:
    """Map ``split`` (int or ``split_0``) onto integer ``fold`` 0–4 used by fold CV."""
    out = df.copy()
    if 'fold' not in out.columns and 'split' in out.columns:
        split = out['split']
        if pd.api.types.is_numeric_dtype(split):
            out['fold'] = pd.to_numeric(split, errors='coerce')
        else:
            extracted = split.astype(str).str.extract(r'(\d+)', expand=False)
            out['fold'] = pd.to_numeric(extracted, errors='coerce')
        n_missing = int(out['fold'].isna().sum())
        if n_missing:
            logger.warning("Dropping %d rows with unparsable split/fold labels", n_missing)
            out = out.loc[out['fold'].notna()].copy()
        out['fold'] = out['fold'].astype(int)
    out = out.drop(columns=['split'], errors='ignore')
    return out


def _drop_non_gene_extra_columns(df: pd.DataFrame, extra: set[str] | None = None) -> pd.DataFrame:
    # Keep ``drug`` — needed for ECFP attachment after fold expansion / normalization.
    drop = {'Unnamed: 0', 'index', 'cpa_pert', 'dataset', 'dose', 'perturbation', 'gene_target'}
    if extra:
        drop |= extra
    return df.drop(columns=[c for c in drop if c in df.columns], errors='ignore')


def _aggregate_duplicate_profile_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Average gene columns for duplicate (cell_line, condition, fold) rows."""
    keys = [c for c in ('cell_line', 'condition', 'fold') if c in df.columns]
    if len(keys) < 2:
        return df
    n_dup = int(df.duplicated(subset=keys).sum())
    if n_dup == 0:
        return df
    meta = [c for c in df.columns if c in PROFILE_META_COLS]
    gene_cols = [c for c in df.columns if c not in PROFILE_META_COLS]
    gene_cols = [c for c in gene_cols if pd.api.types.is_numeric_dtype(df[c])]
    grouped = df.groupby(keys, as_index=False, sort=False)
    out = grouped[gene_cols].mean()
    extra_meta = [c for c in meta if c not in keys]
    if extra_meta:
        out = out.merge(grouped[extra_meta].first(), on=keys, how='left')
    logger.info(
        "_aggregate_duplicate_profile_rows: averaged %d duplicate rows -> %d unique keys",
        n_dup,
        len(out),
    )
    return out


def _normalize_predicted_profile_table(df: pd.DataFrame, mcfarland_cell_line: bool = False) -> pd.DataFrame:
    """Harmonize predicted-profile tables: cell_line, condition, integer fold."""
    out = df.copy()
    if 'cell_type' in out.columns and 'cell_line' not in out.columns:
        out = out.rename(columns={'cell_type': 'cell_line'})
    if 'cell_line' in out.columns:
        cell = out['cell_line'].astype(str).str.strip()
        if mcfarland_cell_line:
            cell = cell.str.split('_').str[0].str.strip()
        out['cell_line'] = cell.str.upper()
    if 'condition' in out.columns:
        if mcfarland_cell_line:
            out['condition'] = out['condition'].astype(str).str.strip()
        else:
            out['condition'] = _normalize_sciplex_product_name(out['condition'])
    out = _normalize_split_to_fold(out)
    out = _drop_non_gene_extra_columns(out)
    out = _aggregate_duplicate_profile_rows(out)
    return out


def _adata_layer_to_frame(adata, layer: str) -> pd.DataFrame:
    """Build a profile DataFrame from an AnnData layer plus obs metadata."""
    values = adata.layers[layer]
    if hasattr(values, 'toarray'):
        values = values.toarray()
    expr = pd.DataFrame(values, columns=adata.var_names.astype(str))
    obs = adata.obs.reset_index(drop=True)
    return pd.concat([obs, expr], axis=1)


def get_CPA_predictions():
    post_predictions = pd.read_csv(os.path.join(data_dir, 'CPA_predictions', 'sciplex_mean_post.csv'))
    lfc_predictions = pd.read_csv(os.path.join(data_dir, 'CPA_predictions', 'sciplex_mean_LFC.csv'))
    post_predictions = _normalize_predicted_profile_table(post_predictions)
    lfc_predictions = _normalize_predicted_profile_table(lfc_predictions)
    return post_predictions, lfc_predictions


def get_chemCPA_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load SciPlex chemCPA post-treatment and LFC tables (with fold from ``split``).

    chemCPA CSVs contain predictions for every pair under all 5 split labels. Keep only
    the held-out fold matching ``get_sciplex_pair_fold_keys`` so drug-response CV does
    not train on the same (cell_line, condition) identities that appear in test.
    """
    post_predictions = pd.read_csv(os.path.join(data_dir, 'chemCPA_predictions', 'sciplex_cv_pred_post.csv'))
    lfc_predictions = pd.read_csv(os.path.join(data_dir, 'chemCPA_predictions', 'sciplex_cv_pred_lfc.csv'))
    post_predictions = _normalize_predicted_profile_table(post_predictions)
    lfc_predictions = _normalize_predicted_profile_table(lfc_predictions)
    fold_keys = get_sciplex_pair_fold_keys()
    n_post, n_lfc = len(post_predictions), len(lfc_predictions)
    post_predictions = align_mcfarland_profiles_to_fold_keys(post_predictions, fold_keys)
    lfc_predictions = align_mcfarland_profiles_to_fold_keys(lfc_predictions, fold_keys)
    logger.info(
        "get_chemCPA_predictions: post %d→%d LFC %d→%d (held-out fold only); folds=%s",
        n_post,
        len(post_predictions),
        n_lfc,
        len(lfc_predictions),
        sorted(post_predictions['fold'].dropna().unique().tolist()) if 'fold' in post_predictions.columns else [],
    )
    return post_predictions, lfc_predictions


def get_McFarland_chemCPA_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load McFarland chemCPA post-treatment and LFC tables (fold from ``split``).

    Same held-out-fold filter as SciPlex chemCPA: use CPA fold keys so each pair
    appears in exactly one CV fold.
    """
    post_path = os.path.join(data_dir, 'chemCPA_predictions', 'mcfarland_cv_pred_post.csv')
    lfc_path = os.path.join(data_dir, 'chemCPA_predictions', 'mcfarland_cv_pred_lfc.csv')
    post_predictions = _normalize_mcfarland_profile_columns(pd.read_csv(post_path))
    lfc_predictions = _normalize_mcfarland_profile_columns(pd.read_csv(lfc_path))
    post_predictions, lfc_predictions = _attach_mcfarland_tissue(post_predictions, lfc_predictions)
    if 'fold' not in post_predictions.columns or 'fold' not in lfc_predictions.columns:
        raise ValueError("McFarland chemCPA predictions must contain a fold/split column")
    # CPA fold table is the McFarland reference for held-out pairs.
    cpa_post, _ = get_McFarland_CPA_predictions()
    fold_keys = cpa_post[['cell_line', 'condition', 'fold']].drop_duplicates()
    n_post, n_lfc = len(post_predictions), len(lfc_predictions)
    post_predictions = align_mcfarland_profiles_to_fold_keys(post_predictions, fold_keys)
    lfc_predictions = align_mcfarland_profiles_to_fold_keys(lfc_predictions, fold_keys)
    logger.info(
        "get_McFarland_chemCPA_predictions: post %d→%d LFC %d→%d (held-out fold only); folds=%s",
        n_post,
        len(post_predictions),
        n_lfc,
        len(lfc_predictions),
        sorted(post_predictions['fold'].dropna().unique().tolist()),
    )
    return post_predictions, lfc_predictions


def get_PRnet_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load SciPlex PRnet post (``predicted_mean``) and LFC (``predicted_lfc``) layers."""
    path = os.path.join(data_dir, 'PRnet_predictions', 'sciplex_all_splits_predicted_pseudobulk.h5ad')
    adata = sc.read_h5ad(path)
    post_predictions = _normalize_predicted_profile_table(_adata_layer_to_frame(adata, 'predicted_mean'))
    lfc_predictions = _normalize_predicted_profile_table(_adata_layer_to_frame(adata, 'predicted_lfc'))
    logger.info(
        "get_PRnet_predictions: post rows=%d LFC rows=%d folds=%s",
        len(post_predictions),
        len(lfc_predictions),
        sorted(post_predictions['fold'].dropna().unique().tolist()) if 'fold' in post_predictions.columns else [],
    )
    return post_predictions, lfc_predictions


def _resolve_mcfarland_cpa_path(filename: str) -> str:
    """Resolve McFarland CPA pseudobulk CSV (``CPA_predictions/`` or ``cpa/``)."""
    name_variants = [filename]
    if filename.lower() != filename:
        name_variants.append(filename.lower())
    if 'lfc' in filename.lower():
        name_variants.extend(
            {filename.replace('lfc', 'LFC'), filename.replace('LFC', 'lfc')}
        )
    # Prefer the current single-file export, then the older ``*_all`` names.
    if filename.endswith('_all.csv'):
        name_variants.append(filename.replace('_all.csv', '.csv'))
    elif filename.endswith('.csv') and '_all' not in filename:
        stem, ext = filename.rsplit('.', 1)
        name_variants.append(f'{stem}_all.{ext}')
    for subdir in ('CPA_predictions', 'cpa'):
        for name in dict.fromkeys(name_variants):
            path = os.path.join(data_dir, subdir, name)
            if os.path.exists(path):
                return path
    raise FileNotFoundError(
        f"McFarland CPA file '{filename}' not found under {data_dir}/CPA_predictions or {data_dir}/cpa"
    )


def _normalize_mcfarland_profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    return _normalize_predicted_profile_table(df, mcfarland_cell_line=True)


def _attach_mcfarland_tissue(post_predictions: pd.DataFrame, lfc_predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tissues = get_tissue_labels()
    for name, frame in (('post', post_predictions), ('lfc', lfc_predictions)):
        frame = frame.drop(columns=['tissue'], errors='ignore').merge(tissues, on='cell_line', how='left')
        if name == 'post':
            post_predictions = frame
        else:
            lfc_predictions = frame
    return post_predictions, lfc_predictions


def get_McFarland_CPA_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load McFarland CPA post-treatment and LFC pseudobulk tables.

    Each row includes a ``fold`` column (0–4) from the CPA cross-validation splits.
    """
    post_predictions = pd.read_csv(_resolve_mcfarland_cpa_path('mcfarland_mean_post.csv'))
    lfc_predictions = pd.read_csv(_resolve_mcfarland_cpa_path('mcfarland_mean_LFC.csv'))
    post_predictions = _normalize_mcfarland_profile_columns(post_predictions)
    lfc_predictions = _normalize_mcfarland_profile_columns(lfc_predictions)
    post_predictions, lfc_predictions = _attach_mcfarland_tissue(post_predictions, lfc_predictions)

    if 'fold' not in post_predictions.columns:
        raise ValueError("McFarland CPA post predictions must contain a 'fold' column")
    if 'fold' not in lfc_predictions.columns:
        raise ValueError("McFarland CPA LFC predictions must contain a 'fold' column")

    logger.info(
        "get_McFarland_CPA_predictions: post rows=%d LFC rows=%d folds=%s",
        len(post_predictions),
        len(lfc_predictions),
        sorted(post_predictions['fold'].dropna().unique().tolist()),
    )
    return post_predictions, lfc_predictions


def get_McFarland_PRnet_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load McFarland PRnet post (``predicted_mean``) and LFC (``predicted_lfc``) layers."""
    path = os.path.join(data_dir, 'PRnet_predictions', 'mcfarland_all_splits_predicted_pseudobulk.h5ad')
    adata = sc.read_h5ad(path)
    post_predictions = _normalize_mcfarland_profile_columns(_adata_layer_to_frame(adata, 'predicted_mean'))
    lfc_predictions = _normalize_mcfarland_profile_columns(_adata_layer_to_frame(adata, 'predicted_lfc'))
    post_predictions, lfc_predictions = _attach_mcfarland_tissue(post_predictions, lfc_predictions)
    if 'fold' not in post_predictions.columns or 'fold' not in lfc_predictions.columns:
        raise ValueError("McFarland PRnet predictions must contain a fold/split column")
    logger.info(
        "get_McFarland_PRnet_predictions: post rows=%d LFC rows=%d folds=%s",
        len(post_predictions),
        len(lfc_predictions),
        sorted(post_predictions['fold'].dropna().unique().tolist()),
    )
    return post_predictions, lfc_predictions


def get_McFarland_GEARS_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load McFarland GEARS post-treatment and LFC pseudobulk tables.

    Each row includes a ``fold`` column (0–4) from the GEARS cross-validation splits.
    """
    post_path = os.path.join(data_dir, 'GEARS_predictions', 'mcfarland_mean_post_all.csv')
    lfc_path = os.path.join(data_dir, 'GEARS_predictions', 'mcfarland_mean_LFC_all.csv')
    if not os.path.exists(post_path):
        raise FileNotFoundError(f'McFarland GEARS post file not found: {post_path}')
    if not os.path.exists(lfc_path):
        raise FileNotFoundError(f'McFarland GEARS LFC file not found: {lfc_path}')

    post_predictions = pd.read_csv(post_path)
    lfc_predictions = pd.read_csv(lfc_path)
    if 'perturbation' in post_predictions.columns:
        post_predictions = post_predictions.rename(columns={'perturbation': 'condition'})
    if 'perturbation' in lfc_predictions.columns:
        lfc_predictions = lfc_predictions.rename(columns={'perturbation': 'condition'})

    post_predictions = _normalize_mcfarland_profile_columns(post_predictions)
    lfc_predictions = _normalize_mcfarland_profile_columns(lfc_predictions)

    tissues = get_tissue_labels()
    for name, frame in (('post', post_predictions), ('lfc', lfc_predictions)):
        frame = frame.drop(columns=['tissue'], errors='ignore').merge(tissues, on='cell_line', how='left')
        if name == 'post':
            post_predictions = frame
        else:
            lfc_predictions = frame

    if 'fold' not in post_predictions.columns:
        raise ValueError("McFarland GEARS post predictions must contain a 'fold' column")
    if 'fold' not in lfc_predictions.columns:
        raise ValueError("McFarland GEARS LFC predictions must contain a 'fold' column")

    logger.info(
        "get_McFarland_GEARS_predictions: post rows=%d LFC rows=%d folds=%s",
        len(post_predictions),
        len(lfc_predictions),
        sorted(post_predictions['fold'].dropna().unique().tolist()),
    )
    return post_predictions, lfc_predictions


def compute_mcfarland_lfc_from_post_and_pre(
    post_predictions: pd.DataFrame,
    pre_treatment: pd.DataFrame,
) -> pd.DataFrame:
    """LFC = post minus per-cell-line pre-treatment mean."""
    post_predictions = _normalize_mcfarland_profile_columns(post_predictions)
    pre_treatment = _normalize_mcfarland_profile_columns(pre_treatment)

    meta_cols = set(PROFILE_META_COLS)
    gene_cols = [
        c
        for c in post_predictions.columns
        if c not in meta_cols and c in pre_treatment.columns
    ]
    if not gene_cols:
        raise ValueError('No shared gene columns between post predictions and pre-treatment profiles.')
    cell_col = 'cell_line' if 'cell_line' in post_predictions.columns else 'cell_type'

    combined_gene_df = post_predictions[[cell_col] + gene_cols].copy()
    pre_gene_df = pre_treatment[[cell_col] + gene_cols].copy().drop_duplicates()
    lfc_df = combined_gene_df.merge(
        pre_gene_df, on=cell_col, how='left', suffixes=('', '_pre')
    )
    for gene in gene_cols:
        lfc_df[gene] = lfc_df[gene] - lfc_df[f'{gene}_pre']

    lfc_result = lfc_df[[cell_col] + gene_cols].copy()
    for col in ('condition', 'fold'):
        if col in post_predictions.columns:
            lfc_result[col] = post_predictions[col].values
    return _normalize_mcfarland_profile_columns(lfc_result)


def get_McFarland_scFoundation_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load McFarland scFoundation post-treatment and LFC pseudobulk tables.

    LFC is read from ``scfoundation_predictions/mcfarland_mean_LFC_all.csv`` when present;
    otherwise computed from post minus observed pre-treatment pseudobulk.
    """
    post_path = os.path.join(data_dir, 'scfoundation_predictions', 'mcfarland_mean_post_all.csv')
    lfc_path = os.path.join(data_dir, 'scfoundation_predictions', 'mcfarland_mean_LFC_all.csv')
    if not os.path.exists(post_path):
        raise FileNotFoundError(f'McFarland scFoundation post file not found: {post_path}')

    post_predictions = pd.read_csv(post_path)
    if 'test_condition' in post_predictions.columns:
        post_predictions = post_predictions.rename(columns={'test_condition': 'condition'})
    if 'perturbation' in post_predictions.columns:
        post_predictions = post_predictions.rename(columns={'perturbation': 'condition'})
    post_predictions = _normalize_mcfarland_profile_columns(post_predictions)

    if os.path.exists(lfc_path):
        lfc_predictions = pd.read_csv(lfc_path)
        if 'perturbation' in lfc_predictions.columns:
            lfc_predictions = lfc_predictions.rename(columns={'perturbation': 'condition'})
        lfc_predictions = _normalize_mcfarland_profile_columns(lfc_predictions)
    else:
        pre_path = os.path.join(data_dir, 'observed_pseudobulk', 'mcfarland_mean_pre_all_celllines.csv')
        pre_treatment = pd.read_csv(pre_path, index_col=0)
        lfc_predictions = compute_mcfarland_lfc_from_post_and_pre(post_predictions, pre_treatment)
        logger.info(
            'get_McFarland_scFoundation_predictions: computed LFC from post and %s',
            pre_path,
        )

    tissues = get_tissue_labels()
    for name, frame in (('post', post_predictions), ('lfc', lfc_predictions)):
        frame = frame.drop(columns=['tissue'], errors='ignore').merge(tissues, on='cell_line', how='left')
        if name == 'post':
            post_predictions = frame
        else:
            lfc_predictions = frame

    if 'fold' not in post_predictions.columns:
        raise ValueError("McFarland scFoundation post predictions must contain a 'fold' column")
    if 'fold' not in lfc_predictions.columns:
        raise ValueError("McFarland scFoundation LFC predictions must contain a 'fold' column")

    logger.info(
        "get_McFarland_scFoundation_predictions: post rows=%d LFC rows=%d folds=%s",
        len(post_predictions),
        len(lfc_predictions),
        sorted(post_predictions['fold'].dropna().unique().tolist()),
    )
    return post_predictions, lfc_predictions


def expand_mcfarland_profiles_with_folds(
    profiles: pd.DataFrame,
    fold_reference: pd.DataFrame,
) -> pd.DataFrame:
    """
    Replicate observed pseudobulk rows across CPA CV folds.

    Fold labels are taken from ``fold_reference`` (typically CPA predicted post profiles).
    """
    profiles = _normalize_mcfarland_profile_columns(profiles)
    fold_reference = _normalize_mcfarland_profile_columns(fold_reference)
    # SciPlex baseline CSVs keep spaces in product names; fold keys strip them.
    if 'condition' in profiles.columns:
        profiles['condition'] = _normalize_sciplex_product_name(profiles['condition'])
    if 'condition' in fold_reference.columns:
        fold_reference['condition'] = _normalize_sciplex_product_name(fold_reference['condition'])
    fold_keys = fold_reference[['cell_line', 'condition', 'fold']].drop_duplicates()
    profiles = profiles.drop(columns=['fold'], errors='ignore')
    expanded = fold_keys.merge(profiles, on=['cell_line', 'condition'], how='inner')
    logger.info(
        "expand_mcfarland_profiles_with_folds: %d unique fold keys -> %d rows (from %d profile rows)",
        len(fold_keys),
        len(expanded),
        len(profiles),
    )
    return expanded


def align_mcfarland_profiles_to_fold_keys(
    profiles: pd.DataFrame,
    fold_keys: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only rows matching ``(cell_line, condition, fold)`` from CPA fold assignments."""
    profiles = _normalize_mcfarland_profile_columns(profiles)
    fold_keys = _normalize_mcfarland_profile_columns(fold_keys.copy())
    if 'condition' in profiles.columns:
        profiles['condition'] = _normalize_sciplex_product_name(profiles['condition'])
    if 'condition' in fold_keys.columns:
        fold_keys['condition'] = _normalize_sciplex_product_name(fold_keys['condition'])
    fold_keys = fold_keys[['cell_line', 'condition', 'fold']].drop_duplicates()
    if 'fold' in profiles.columns:
        aligned = fold_keys.merge(
            profiles, on=['cell_line', 'condition', 'fold'], how='inner'
        )
    else:
        profiles = profiles.drop(columns=['fold'], errors='ignore')
        aligned = fold_keys.merge(profiles, on=['cell_line', 'condition'], how='inner')
    logger.info(
        "align_mcfarland_profiles_to_fold_keys: %d / %d fold keys matched",
        len(aligned),
        len(fold_keys),
    )
    return aligned


def log_mcfarland_profile_expression_difference(
    observed: pd.DataFrame,
    predicted: pd.DataFrame,
    label: str,
    sample_genes: int = 500,
) -> None:
    """Log whether gene expression differs between observed and predicted profile tables."""
    meta = set(PROFILE_META_COLS)
    merge_keys = ['cell_line', 'condition', 'fold']
    if not all(k in observed.columns and k in predicted.columns for k in merge_keys):
        merge_keys = ['cell_line', 'condition']

    obs = _normalize_mcfarland_profile_columns(observed)
    pred = _normalize_mcfarland_profile_columns(predicted)
    if 'condition' in obs.columns:
        obs['condition'] = _normalize_sciplex_product_name(obs['condition'])
    if 'condition' in pred.columns:
        pred['condition'] = _normalize_sciplex_product_name(pred['condition'])
    gene_cols = sorted(set(obs.columns) & set(pred.columns) - meta)
    if not gene_cols:
        logger.warning("%s: no shared gene columns to compare", label)
        return

    sample_cols = gene_cols[:sample_genes]
    merged = obs[merge_keys + sample_cols].merge(
        pred[merge_keys + sample_cols],
        on=merge_keys,
        suffixes=('_obs', '_pred'),
        how='inner',
    )
    if merged.empty:
        logger.warning("%s: no overlapping rows for expression comparison", label)
        return

    diff = np.abs(
        merged[[f'{g}_obs' for g in sample_cols]].to_numpy()
        - merged[[f'{g}_pred' for g in sample_cols]].to_numpy()
    )
    max_diff = float(np.nanmax(diff))
    mean_diff = float(np.nanmean(diff))
    logger.info(
        "%s vs observed: %d paired rows, %d genes sampled, max|diff|=%.6g, mean|diff|=%.6g",
        label,
        len(merged),
        len(sample_cols),
        max_diff,
        mean_diff,
    )
    if max_diff < 1e-12:
        logger.warning(
            "%s expression is numerically identical to observed on sampled genes — "
            "check CPA export / input files.",
            label,
        )


def _load_mcfarland_baseline_profiles(
    subdir: str,
    post_filename: str = 'mcfarland_mean_post_all_celllines.csv',
    include_lfc: bool = True,
    lfc_filename: str = 'mcfarland_mean_LFC_all_celllines.csv',
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load McFarland baseline post (and optional LFC) tables with tissue labels."""
    post_path = os.path.join(data_dir, subdir, post_filename)
    post = _normalize_mcfarland_profile_columns(pd.read_csv(post_path))

    tissues = get_tissue_labels()
    post = post.drop(columns=['tissue'], errors='ignore').merge(tissues, on='cell_line', how='left')

    lfc = None
    if include_lfc:
        lfc_path = os.path.join(data_dir, subdir, lfc_filename)
        if not os.path.exists(lfc_path):
            alt = lfc_path.replace('LFC', 'lfc') if 'LFC' in lfc_filename else lfc_path
            lfc_path = alt if os.path.exists(alt) else lfc_path
        lfc = _normalize_mcfarland_profile_columns(pd.read_csv(lfc_path))
        lfc = lfc.drop(columns=['tissue'], errors='ignore').merge(tissues, on='cell_line', how='left')

    return post, lfc


def get_McFarland_no_effect_predictions() -> pd.DataFrame:
    """
    McFarland no-effect profiles (cell-line mean pre-treatment expression per drug pair).

    Saved by ``create_baseline_predictions`` as ``no_effect_predictions/mcfarland_mean_post_all_celllines.csv``.
    """
    post, _ = _load_mcfarland_baseline_profiles('no_effect_predictions', include_lfc=False)
    logger.info("get_McFarland_no_effect_predictions: %d rows", len(post))
    return post


def get_McFarland_average_effect_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    McFarland average-effect profiles (cell-line mean post/LFC across drugs).

    Saved by ``create_baseline_predictions`` under ``average_effect_predictions/``.
    """
    post, lfc = _load_mcfarland_baseline_profiles('average_effect_predictions')
    if lfc is None:
        raise FileNotFoundError(
            f"McFarland average-effect LFC not found under {data_dir}/average_effect_predictions/"
        )
    logger.info(
        "get_McFarland_average_effect_predictions: post rows=%d LFC rows=%d",
        len(post),
        len(lfc),
    )
    return post, lfc


def get_average_effect_predictions():
    post_predictions_df = []
    lfc_predictions_df = []
    for cell_line in ['mcf7', 'k562', 'a549']:
        post_predictions = pd.read_csv(os.path.join(data_dir, 'average_effect_predictions', f'sciplex_mean_post_{cell_line}.csv')).rename(columns={'cell_type':'cell_line'})
        lfc_predictions = pd.read_csv(os.path.join(data_dir, 'average_effect_predictions', f'sciplex_mean_LFC_{cell_line}.csv')).rename(columns={'cell_type':'cell_line'})
        post_predictions_df.append(post_predictions)
        lfc_predictions_df.append(lfc_predictions)

    post_predictions_df = _normalize_predicted_profile_table(pd.concat(post_predictions_df, axis=0))
    lfc_predictions_df = _normalize_predicted_profile_table(pd.concat(lfc_predictions_df, axis=0))
    return post_predictions_df, lfc_predictions_df

def get_no_effect_predictions():
    # There are no LFCs for no-effect predictions, because these are always 0
    result = []
    for cell_line in ['mcf7', 'k562', 'a549']:
        post_data = pd.read_csv(os.path.join(data_dir, 'no_effect_predictions', f'sciplex_mean_post_{cell_line}.csv'), index_col=0).rename(columns={'cell_type':'cell_line'})
        post_data['cell_line'] = cell_line.upper()
        result.append(post_data)

    return _normalize_predicted_profile_table(pd.concat(result, axis=0)) 

'''Helper functions'''	
def add_y_and_normalize(df_with_sensitivity, y, normalize=True, groupby=['condition'], keep=[]):
    df_with_sensitivity = df_with_sensitivity.copy()

    n_start = len(df_with_sensitivity)
    if n_start == 0:
        logger.warning("add_y_and_normalize: input dataframe is empty (y=%s, keep=%s)", y, keep)

    if normalize:
        df_with_sensitivity['y'] = df_with_sensitivity.groupby(groupby)[y].transform(lambda x: (x - x.mean()) / x.std())
    else:
        df_with_sensitivity['y'] = df_with_sensitivity[y]

    columns_to_remove = list(set(['sens', 'sens_label', 'target', 'condition', 'cell_line', 'tissue', 'product_name']).difference(set(keep)))
    columns_to_remove = [c for c in columns_to_remove if c in df_with_sensitivity.columns]
    df_with_sensitivity = df_with_sensitivity.drop(columns=columns_to_remove, axis=1)
    n_after_dropcols = len(df_with_sensitivity)
    df_with_sensitivity = df_with_sensitivity[df_with_sensitivity['y'].notna()]
    n_after_y = len(df_with_sensitivity)
    # Do not drop rows for NaNs in gene columns — sparse/CSV artifacts often leave scattered
    # gene NaNs and `dropna(axis=0)` would remove every row when any gene is missing.
    required_non_null = ['y'] + [c for c in keep if c in df_with_sensitivity.columns]
    df_with_sensitivity = df_with_sensitivity.dropna(axis=0, subset=required_non_null)
    n_end = len(df_with_sensitivity)

    if n_start > 0:
        logger.info(
            "add_y_and_normalize: rows start=%d after_dropcols=%d after_y_notna=%d -> end=%d (dropna subset=%s)",
            n_start,
            n_after_dropcols,
            n_after_y,
            n_end,
            required_non_null,
        )
    if n_start > 0 and n_end == 0:
        logger.error(
            "add_y_and_normalize: all rows removed (check sensitivity merge / NaN in y or keep columns %s)",
            required_non_null,
        )

    return(df_with_sensitivity)

def get_tissue_labels():
    sensitivity_info = get_sens_labels(0.2)
    if 'drug' in sensitivity_info.columns:
        drug_to_perturbation = pd.read_csv(os.path.join(resources_dir, 'mcfarland_drug_to_perturbation.csv'))
        tissues = pd.merge(sensitivity_info, drug_to_perturbation, on='drug', how='left')[['cell_line', 'tissue']].drop_duplicates()
    else:
        # Curated sensitivity has no tissue; resolve from DepMap metadata.csv first, then pseudobulk.
        pre_path = os.path.join(data_dir, 'observed_pseudobulk', 'mcfarland_mean_pre_all_celllines.csv')
        observed_pre = pd.read_csv(pre_path, index_col=0)
        cell_col = 'cell_type' if 'cell_type' in observed_pre.columns else 'cell_line'
        cl = observed_pre[cell_col].astype(str).str.strip().str.split('_').str[0].str.strip().str.upper()

        depmap_tissues = _load_mcfarland_depmap_cell_line_tissue_map()

        if len(depmap_tissues) > 0:
            tissues = depmap_tissues.copy()
            if 'tissue' in observed_pre.columns and observed_pre['tissue'].notna().any():
                file_map = (
                    pd.DataFrame({'cell_line': cl, 'tissue': observed_pre['tissue']})
                    .dropna(subset=['tissue'])
                    .drop_duplicates(subset=['cell_line'], keep='first')
                )
                only_file = file_map[~file_map['cell_line'].isin(tissues['cell_line'])]
                if len(only_file) > 0:
                    tissues = pd.concat([tissues, only_file], ignore_index=True)
                    logger.info(
                        "get_tissue_labels: added %d cell_line tissues from pseudobulk not present in DepMap metadata",
                        len(only_file),
                    )
        elif 'tissue' in observed_pre.columns and observed_pre['tissue'].notna().any():
            tissues = (
                pd.DataFrame({'cell_line': cl, 'tissue': observed_pre['tissue']})
                .dropna(subset=['tissue'])
                .drop_duplicates()
            )
            logger.info(
                "get_tissue_labels: using `tissue` column from pseudobulk CSV (%d non-null rows)",
                int(observed_pre['tissue'].notna().sum()),
            )
        else:
            split_cols = observed_pre[cell_col].astype(str).str.split('_', expand=True, n=1)
            if 1 in split_cols.columns:
                tissues = pd.DataFrame({
                    'cell_line': split_cols[0].astype(str).str.strip().str.upper(),
                    'tissue': split_cols[1],
                }).dropna(subset=['tissue']).drop_duplicates()
            else:
                logger.warning(
                    "get_tissue_labels: no DepMap metadata map, no pseudobulk tissue column, and no "
                    "CELL_LINE_TISSUE pattern in %s; using tissue=UNKNOWN per cell_line.",
                    cell_col,
                )
                tissues = pd.DataFrame({'cell_line': cl.drop_duplicates(), 'tissue': 'UNKNOWN'})

        cl_obs = cl.drop_duplicates()
        missing = cl_obs[~cl_obs.isin(tissues['cell_line'])]
        if len(missing) > 0:
            tissues = pd.concat(
                [tissues, pd.DataFrame({'cell_line': missing.values, 'tissue': 'UNKNOWN'})],
                ignore_index=True,
            )
            logger.warning(
                "get_tissue_labels: %d pseudobulk cell_line values not in tissue map; assigned UNKNOWN (sample: %s)",
                len(missing),
                missing.head(20).tolist(),
            )

    # Ensure string operations are safe even if tissue inferred as non-string dtype
    tissues['tissue'] = tissues['tissue'].where(tissues['tissue'].notna(), np.nan)
    tissues['tissue'] = tissues['tissue'].astype('string')
    tissues['tissue'] = np.where(tissues['tissue'].str.contains('SKIN', na=False), 'SKIN', tissues['tissue'])
    tissues['tissue'] = np.where(
        tissues['tissue'].str.contains('TO_', na=False),
        tissues['tissue'].str.split('_', n=1).str[1],
        tissues['tissue'],
    )
    tissues = tissues.drop_duplicates()
    tissues = tissues[tissues['tissue']!='MATCHED_NORMAL_TISSUE']
    tissues['cell_line'] = tissues['cell_line'].astype(str).str.strip().str.upper()

    logger.info(
        "get_tissue_labels: %d unique (cell_line, tissue) rows (source=%s)",
        len(tissues),
        "drug_to_perturbation merge"
        if 'drug' in sensitivity_info.columns
        else "DepMap metadata.csv (+ pseudobulk fallbacks)",
    )

    return(tissues)

def filter_on_coefficient_of_variation(df_with_sensitivity, groupby=['tissue', 'condition'], threshold=0.5):
    if df_with_sensitivity.empty:
        logger.warning('filter_on_coefficient_of_variation: empty input')
        return df_with_sensitivity.copy()
    coefficient_of_variation = (df_with_sensitivity.groupby(groupby)['sens'].std() / df_with_sensitivity.groupby(groupby)['sens'].mean()).reset_index().rename(columns={'sens': 'coefficient_of_variation'})
    df_with_sensitivity = pd.merge(df_with_sensitivity, coefficient_of_variation, left_on=groupby, right_on=groupby, how='left')
    df_with_sensitivity = df_with_sensitivity.dropna(axis=0)
    print("Filtering on coefficient of variation drops:")
    print(df_with_sensitivity[df_with_sensitivity['coefficient_of_variation']<threshold][groupby].drop_duplicates())

    df_with_sensitivity = df_with_sensitivity[df_with_sensitivity['coefficient_of_variation']>threshold]
    df_with_sensitivity.drop(columns=['coefficient_of_variation'], inplace=True)
    
    return(df_with_sensitivity)

'''Prediction functions'''
def feature_selection(X_train, n_features, feature_subset=[]):
    remove_columns = [
        'y', 'tissue', 'cell_line', 'target', 'drug', 'condition', 'perturbation',
        'cell_type', 'fold', 'split', 'n_cells', 'n_replicates', 'replicate',
        'dose', 'cpa_pert', 'dataset', 'sens', 'sens_label', 'product_name',
    ]
    remove_columns.extend(
        c for c in X_train.columns if str(c).startswith('ecfp_')
    )
    remove_columns = list(set(remove_columns).difference(feature_subset))
    for c in remove_columns:
        if c in X_train.columns:
            X_train = X_train.drop(columns=c)

    numeric = X_train.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        logger.warning("feature_selection: no numeric columns after dropping metadata")
        selected_features = list(feature_subset) if feature_subset else []
    elif len(feature_subset) == 0:
        variances = numeric.var(axis=0)
        selected_features = variances.nlargest(min(n_features, len(variances))).index.tolist()
    else:
        selected_features = feature_subset

    selected_features = list(set(selected_features).intersection(set(X_train.columns)))
   
    # Only keep condition as drug indicator
    # Keep cell line and tissue to identify test instances 
    selected_features = selected_features + ['tissue', 'condition', 'cell_line']

    logger.info(
        "feature_selection: rows=%d cols_for_variance=%d returned_n_features=%d (includes tissue, condition, cell_line)",
        len(X_train),
        X_train.shape[1],
        len(selected_features),
    )

    return selected_features
