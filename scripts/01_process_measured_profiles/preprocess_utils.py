import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse
import os
import sys
from pathlib import Path
import logging

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config, setup_project
from utils import setup_logging_for_script, log_script_start, log_script_end, ensure_directories_exist

# Setup project and logging
setup_project()
logger = setup_logging_for_script(__file__)

# Set project directories using configuration
path = os.getcwd()
home_dir = str(config.PROJECT_ROOT)
data_dir = str(config.DATA_DIR)
resources_dir = str(config.RESOURCES_DIR)

# Ensure directories exist
ensure_directories_exist(home_dir,data_dir,resources_dir)

def get_gene_set():
    gene_list_df = pd.read_csv(os.path.join(resources_dir, 'OS_scRNA_gene_index.19264.tsv'), header=0, delimiter='\t')
    gene_list = list(gene_list_df['gene_name'])
    return gene_list


def main_gene_selection(X_df, gene_list):
    """
    Describe:
        rebuild the input adata to unified gene symbol
    """
    remove_columns = list(set(X_df.columns) - set(gene_list))
    X_df = X_df.drop(columns=remove_columns)
    to_fill_columns = list(set(gene_list) - set(X_df.columns))
    print(f'No. fill columns: {len(to_fill_columns)}')
    padding_df = pd.DataFrame(np.zeros((X_df.shape[0], len(to_fill_columns))), 
                              columns=to_fill_columns, 
                              index=X_df.index)
    X_df = pd.DataFrame(np.concatenate([df.values for df in [X_df, padding_df]], axis=1), 
                        index=X_df.index, 
                        columns=list(X_df.columns) + list(padding_df.columns))
    X_df = X_df[gene_list]
    
    var = pd.DataFrame(index=X_df.columns)
    var['mask'] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
    return X_df, to_fill_columns,var


def basic_filter(merged_df, qc_min_genes=None, qc_min_cells=0):
    """Drop cells with too few detected genes. Gene filter default is a no-op (min_cells=0)."""
    if qc_min_genes is None:
        qc_min_genes = int(config.MIN_GENES_PER_CELL)
    n_before = merged_df.n_obs
    sc.pp.filter_cells(merged_df, min_genes=qc_min_genes)
    sc.pp.filter_genes(merged_df, min_cells=qc_min_cells)
    logger.info(
        "basic_filter min_genes=%d min_cells=%d: %d -> %d cells, %d genes",
        qc_min_genes,
        qc_min_cells,
        n_before,
        merged_df.n_obs,
        merged_df.n_vars,
    )
    return merged_df


def process_merged_df(merged_df):
    """Minimal QC used for McFarland and the SciPlex A549/MCF7/K562 subsets."""
    if "gene_name" not in merged_df.var.columns:
        merged_df.var["gene_name"] = (
            pd.Index(merged_df.var_names.astype(str)).str.replace(r"\.\d+$", "", regex=True)
        )
    merged_df = basic_filter(merged_df)
    qc_metrics = sc.pp.calculate_qc_metrics(merged_df)
    merged_df.obs["total_count"] = qc_metrics[0]["total_counts"]
    merged_df.var.index = merged_df.var.index.str.replace(r"\.\d+", "", regex=True)
    merged_df.var["gene_name"] = merged_df.var.index
    return merged_df


def _strip_gene_version(names: pd.Index) -> pd.Index:
    return pd.Index(names.astype(str)).str.replace(r"\.\d+$", "", regex=True)


def collapse_duplicate_genes(X, var_names: pd.Index):
    """Sum columns that share a gene symbol (count-safe; do not use on log1p X)."""
    names = _strip_gene_version(pd.Index(var_names))
    if not names.duplicated().any():
        return X, names
    unique = pd.Index(names.unique())
    name_to_new = {name: i for i, name in enumerate(unique)}
    n_old = X.shape[1]
    n_new = len(unique)
    transform = sparse.csr_matrix(
        (
            np.ones(n_old, dtype=np.float32),
            (np.arange(n_old), [name_to_new[n] for n in names]),
        ),
        shape=(n_old, n_new),
        dtype=np.float32,
    )
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    logger.info("Collapsed %d duplicate gene columns -> %d unique symbols", n_old - n_new, n_new)
    return X.dot(transform).tocsr(), unique


def align_matrix_to_gene_list(X, var_names: pd.Index, gene_list: list[str], collapse: bool):
    """Restrict/pad a cells x genes matrix to ``gene_list``.

    ``mask`` is 1 for assay-missing genes (zero-padded), 0 for genes present in
    the assay. Duplicate symbols are summed when ``collapse`` is True.
    """
    names = _strip_gene_version(pd.Index(var_names))
    if collapse:
        X, names = collapse_duplicate_genes(X, names)
    elif names.duplicated().any():
        keep = ~names.duplicated()
        X = X[:, np.flatnonzero(keep)]
        names = names[keep]
        logger.info("Dropped duplicate gene columns (keep first); collapse=False")

    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    X = X.tocsr().astype("float32")
    lookup = {name: i for i, name in enumerate(names)}
    present = [g for g in gene_list if g in lookup]
    missing = [g for g in gene_list if g not in lookup]
    logger.info("Aligning to gene index: %d present, %d assay-missing (padded)", len(present), len(missing))
    X_present = (
        X[:, [lookup[g] for g in present]]
        if present
        else sparse.csr_matrix((X.shape[0], 0), dtype=np.float32)
    )
    if missing:
        X_present = sparse.hstack(
            [X_present, sparse.csr_matrix((X.shape[0], len(missing)), dtype=np.float32)],
            format="csr",
        )
    ordered = present + missing
    col_of = {g: i for i, g in enumerate(ordered)}
    X_out = X_present[:, [col_of[g] for g in gene_list]]
    mask = np.array([int(g in set(missing)) for g in gene_list], dtype=np.int8)
    return X_out, mask


def store_counts_layer(adata: sc.AnnData) -> sc.AnnData:
    """Copy integer X into ``layers['counts']`` before normalize/log1p."""
    from qc_utils import looks_like_counts

    if "counts" in adata.layers:
        return adata
    if looks_like_counts(adata.X):
        X = adata.X
        adata.layers["counts"] = X.tocsr() if sparse.issparse(X) else sparse.csr_matrix(X)
        logger.info("Stored counts layer (%d cells x %d genes)", adata.n_obs, adata.n_vars)
    return adata


def normalize_log1p_and_align_genes(
    adata: sc.AnnData,
    skip_transform_if_logged: bool = True,
) -> sc.AnnData:
    """Library-size normalize, log1p, and restrict/pad genes to the 19264-gene index."""
    gene_list = get_gene_set()
    already_logged = "log1p" in adata.uns
    adata = store_counts_layer(adata)

    var_names = pd.Index(adata.var_names.astype(str))
    if "gene_name" in adata.var.columns:
        gene_name = adata.var["gene_name"].astype(str)
        if gene_name.isin(gene_list).sum() > _strip_gene_version(var_names).isin(gene_list).sum():
            var_names = gene_name

    counts_layer = None
    mask = None
    if "counts" in adata.layers:
        counts_layer, mask = align_matrix_to_gene_list(
            adata.layers["counts"], var_names, gene_list, collapse=True
        )

    if skip_transform_if_logged and already_logged:
        logger.info("Skipping normalize/log1p; log1p already present in uns")
        X_out, x_mask = align_matrix_to_gene_list(adata.X, var_names, gene_list, collapse=False)
        if mask is None:
            mask = x_mask
    elif counts_layer is not None:
        logger.info("normalize_total + log1p from counts layer (%d cells)", adata.n_obs)
        tmp = sc.AnnData(counts_layer.copy(), obs=adata.obs.copy())
        sc.pp.normalize_total(tmp)
        sc.pp.log1p(tmp)
        X_out = tmp.X
        adata.uns.update(dict(tmp.uns))
    else:
        logger.info("normalize_total + log1p on %d cells x %d genes", adata.n_obs, adata.n_vars)
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        X_out, mask = align_matrix_to_gene_list(adata.X, var_names, gene_list, collapse=False)

    var = pd.DataFrame({"gene_name": gene_list, "mask": mask}, index=gene_list)
    out = sc.AnnData(X_out, obs=adata.obs.copy(), var=var)
    out.uns.update(dict(adata.uns))
    if counts_layer is not None:
        out.layers["counts"] = counts_layer
    return out


def scfoundation_preprocessing(merged_df):
    obs_df = merged_df.obs
    gene_list = get_gene_set()

    merged_df_pd = merged_df.to_df()
    X_df, to_fill_columns, var = main_gene_selection(merged_df_pd, gene_list)
    X_df = X_df.astype('float32')

    adata = sc.AnnData(sparse.csr_matrix(X_df), obs=obs_df, var=X_df.columns.to_frame() )
    adata.var['gene_name'] = adata.var.index
    adata.var = adata.var.drop(columns=adata.var.columns[0])

    return(adata)