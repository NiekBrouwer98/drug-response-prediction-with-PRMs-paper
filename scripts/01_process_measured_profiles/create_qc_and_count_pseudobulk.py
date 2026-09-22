"""QC tables and replicate-aware count-based pseudobulks for measured profiles.

Sums raw UMI counts per (cell_line, condition, replicate), library-size
normalizes, log1p-transforms, then averages replicate profiles to the same
grain as the existing log-mean CSVs.

SciPlex groups by ``condition`` (drug; vehicles → ``ctrl``) and ``obs['replicate']``.
McFarland uses MIX-seq experiment folders as replicates. Assay-missing genes (``var['mask']``) are padded with zero and
written to a mask table; they are not treated as biological zeros.

Submit:
    sbatch run_qc_and_count_pseudobulk.sh
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent))
from config import config, setup_project
from create_mcfarland_splits import add_tissue_column
from create_sciplex_splits import (
    PROCESSED_H5AD,
    RAW_H5AD_CANDIDATES,
    sciplex_condition_from_product,
    canonicalize_sciplex_obs_columns,
)
from preprocess_utils import align_matrix_to_gene_list, get_gene_set
from qc_utils import (
    annotate_dataset_qc,
    counts_matrix,
    looks_like_counts,
)
from utils import (
    ensure_directories_exist,
    log_script_end,
    log_script_start,
    setup_logging_for_script,
)

setup_project()
logger = setup_logging_for_script(__file__)

data_dir = Path(config.DATA_DIR)
resources_dir = Path(config.RESOURCES_DIR)
qc_dir = Path(config.RESULTS_01_DIR)
pseudobulk_dir = data_dir / "observed_pseudobulk"
ensure_directories_exist(str(config.PROJECT_ROOT), str(data_dir), str(qc_dir), str(pseudobulk_dir))

SCIPLEX_PROCESSED = PROCESSED_H5AD
SCIPLEX_RAW_CANDIDATES = RAW_H5AD_CANDIDATES
MCFARLAND_PROCESSED = data_dir / "mcfarland_processed" / "all_cell_lines.h5ad"
REPO_MCFARLAND_RAW = data_dir / "mcfarland_raw"
EXTERNAL_MCFARLAND_RAW_DIRS = (
    Path(r"N:/ewi/insy/DBL/nbrouwer1/perturbation_response_model/DATA/mcfarland_raw"),
    Path("/tudelft.net/staff-bulk/ewi/insy/DBL/nbrouwer1/perturbation_response_model/DATA/mcfarland_raw"),
)


def _to_csr(X):
    if not sparse.issparse(X):
        return sparse.csr_matrix(X, dtype=np.float32)
    return X.tocsr().astype(np.float32)


def size_factor_log1p(count_df: pd.DataFrame, gene_cols: list[str]) -> pd.DataFrame:
    """Median-library-size normalize count pseudobulks, then log1p."""
    counts = count_df[gene_cols].to_numpy(dtype=np.float64)
    lib = counts.sum(axis=1, keepdims=True)
    empty = lib.ravel() <= 0
    lib = np.clip(lib, 1.0, None)
    target = float(np.median(lib[~empty])) if (~empty).any() else 1.0
    normed = np.log1p(counts * (target / lib))
    normed[empty] = 0.0
    out = count_df.copy()
    out[gene_cols] = normed
    logger.info("Size-factor log1p: median library size %.1f over %d pseudobulks", target, len(out))
    return out


def _sum_groups(X, obs: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    X = _to_csr(X)
    cols = [c for c in group_cols if c in obs.columns]
    records = []
    grouped = obs.groupby(cols, observed=True, sort=False)
    for key, idx in grouped.indices.items():
        if not isinstance(key, tuple):
            key = (key,)
        summed = np.asarray(X[idx].sum(axis=0)).ravel()
        rec = {col: val for col, val in zip(cols, key)}
        rec["n_cells"] = int(len(idx))
        rec["_counts"] = summed
        records.append(rec)
    return pd.DataFrame.from_records(records)


def counts_frame_from_records(records: pd.DataFrame, gene_list: list[str]) -> pd.DataFrame:
    counts = np.vstack(records["_counts"].to_list())
    out = records.drop(columns=["_counts"]).reset_index(drop=True)
    gene_df = pd.DataFrame(counts, columns=gene_list)
    return pd.concat([out, gene_df], axis=1)


def average_replicates(
    log_df: pd.DataFrame,
    gene_cols: list[str],
    sample_cols: list[str],
) -> pd.DataFrame:
    """Mean of size-factor log1p replicate profiles; n_cells is summed."""
    grouped = log_df.groupby(sample_cols, observed=True, sort=False)
    mean_expr = grouped[gene_cols].mean().reset_index()
    n_cells = grouped["n_cells"].sum().rename("n_cells").reset_index()
    n_reps = grouped.size().rename("n_replicates").reset_index()
    out = mean_expr.merge(n_cells, on=sample_cols).merge(n_reps, on=sample_cols)
    return out


def split_pre_post_lfc(
    profiles: pd.DataFrame,
    gene_cols: list[str],
    extra_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pre = profiles.loc[profiles["condition"].astype(str).eq("ctrl")].copy()
    post = profiles.loc[~profiles["condition"].astype(str).eq("ctrl")].copy()
    if pre.empty or post.empty:
        raise RuntimeError("Count pseudobulk missing ctrl or treated profiles")

    cell_col = "cell_type"
    treated_cells = set(post[cell_col].astype(str))
    ctrl_cells = set(pre[cell_col].astype(str))
    keep = treated_cells & ctrl_cells
    pre = pre.loc[pre[cell_col].astype(str).isin(keep)]
    post = post.loc[post[cell_col].astype(str).isin(keep)].copy()

    pairs = post[[cell_col, "condition"]].drop_duplicates()
    pre_expr = pre.drop(columns=["condition"], errors="ignore")
    keep_pre = [cell_col] + [c for c in extra_cols if c in pre_expr.columns and c != cell_col] + gene_cols
    keep_pre = list(dict.fromkeys([c for c in keep_pre if c in pre_expr.columns]))
    pre_expr = pre_expr[keep_pre].drop_duplicates(subset=[cell_col])
    pre_aligned = pairs.merge(pre_expr, on=cell_col, how="left")

    ctrl_genes = pre.drop_duplicates(subset=[cell_col]).set_index(cell_col)[gene_cols]
    lfc_values = post[gene_cols].to_numpy(dtype=np.float64) - ctrl_genes.loc[post[cell_col].astype(str)].to_numpy(
        dtype=np.float64
    )
    meta = post.drop(columns=gene_cols, errors="ignore").reset_index(drop=True)
    lfc = pd.concat([meta, pd.DataFrame(lfc_values, columns=gene_cols)], axis=1)
    return pre_aligned, post, lfc


def filter_min_cells(df: pd.DataFrame, label: str) -> pd.DataFrame:
    min_cells = int(config.MIN_CELLS_PER_CONDITION)
    keep = df["n_cells"] >= min_cells
    n_drop = int((~keep).sum())
    if n_drop:
        logger.info("Dropped %d %s pseudobulks with n_cells < %d", n_drop, label, min_cells)
    return df.loc[keep].copy()


def write_profile_tables(
    pre: pd.DataFrame,
    post: pd.DataFrame,
    lfc: pd.DataFrame,
    prefix: str,
) -> None:
    pseudobulk_dir.mkdir(parents=True, exist_ok=True)
    pre.to_csv(pseudobulk_dir / f"{prefix}_pre.csv")
    post.to_csv(pseudobulk_dir / f"{prefix}_post.csv")
    lfc.to_csv(pseudobulk_dir / f"{prefix}_LFC.csv")
    logger.info("Wrote %s pre/post/LFC (%d post profiles)", prefix, len(post))


def profiles_from_count_frame(
    count_df: pd.DataFrame,
    gene_list: list[str],
    sample_cols: list[str],
    extra_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    count_df = filter_min_cells(count_df, "replicate")
    log_df = size_factor_log1p(count_df, gene_list)
    aggregated = average_replicates(log_df, gene_list, sample_cols)
    aggregated = filter_min_cells(aggregated, "aggregated")
    pre, post, lfc = split_pre_post_lfc(aggregated, gene_list, extra_cols)
    return log_df, pre, post, lfc


def resolve_sciplex_counts(
    processed: sc.AnnData,
    raw_path: Path | None,
) -> sc.AnnData | None:
    X_counts = counts_matrix(processed)
    if X_counts is not None:
        out = processed.copy()
        out.X = X_counts
        logger.info("SciPlex counts from processed AnnData (%s)", "layers/counts" if "counts" in processed.layers else "X")
        return out

    candidates = []
    if raw_path is not None:
        candidates.append(raw_path)
    candidates.extend(SCIPLEX_RAW_CANDIDATES)
    for path in candidates:
        if path is None or not path.exists():
            continue
        logger.info("Reading SciPlex raw counts from %s", path)
        raw = sc.read_h5ad(path)
        if not looks_like_counts(raw.X) and "counts" not in raw.layers:
            logger.warning("%s does not look like counts; skipping", path)
            continue
        if "counts" in raw.layers:
            raw.X = raw.layers["counts"]
        shared = processed.obs_names.intersection(raw.obs_names)
        logger.info("Matched %d/%d processed SciPlex barcodes to raw counts", len(shared), processed.n_obs)
        raw = raw[shared].copy()
        raw.obs = processed.obs.loc[shared].copy()
        return raw
    return None


def sciplex_count_pseudobulk(processed: sc.AnnData, raw_path: Path | None = None) -> None:
    counts_ad = resolve_sciplex_counts(processed, raw_path)
    if counts_ad is None:
        logger.error(
            "SciPlex raw counts are not available. Re-run create_sciplex_splits.py "
            "on the original UMIs (layers['counts'] will be stored), or pass "
            "--sciplex-raw pointing at Srivatsan_2019_raw.h5ad."
        )
        return

    keep = counts_ad.obs["qc_pass"].to_numpy() if "qc_pass" in counts_ad.obs.columns else np.ones(counts_ad.n_obs, dtype=bool)
    counts_ad = counts_ad[keep].copy()
    gene_list = get_gene_set()
    var_names = counts_ad.var_names
    if "gene_name" in counts_ad.var.columns:
        var_names = counts_ad.var["gene_name"].astype(str)
    X_aligned, mask = align_matrix_to_gene_list(counts_ad.X, pd.Index(var_names), gene_list, collapse=True)
    counts_ad.var = pd.DataFrame(
        {"gene_name": gene_list, "mask": mask},
        index=gene_list,
    )
    mask_series = pd.Series(mask.astype(bool), index=gene_list, name="assay_missing")
    mask_series.to_csv(qc_dir / "sciplex_gene_assay_mask.csv", header=True)

    obs = canonicalize_sciplex_obs_columns(counts_ad.obs.copy())
    obs["condition"] = sciplex_condition_from_product(obs)
    group_cols = ["cell_type", "condition"]
    if "replicate" in obs.columns:
        group_cols.append("replicate")
    records = _sum_groups(X_aligned, obs, group_cols)
    count_df = counts_frame_from_records(records, gene_list)
    sample_cols = ["cell_type", "condition"]
    extra = ["n_replicates"]
    log_df, pre, post, lfc = profiles_from_count_frame(count_df, gene_list, sample_cols, extra)
    write_profile_tables(pre, post, lfc, "sciplex_count")
    log_df.to_csv(pseudobulk_dir / "sciplex_count_replicate.csv")
    logger.info("Wrote SciPlex replicate-level count pseudobulk (%d rows)", len(log_df))

    for cell_line, suffix in (("A549", "a549"), ("K562", "k562"), ("MCF7", "mcf7")):
        for kind, frame in (("pre", pre), ("post", post), ("LFC", lfc)):
            subset = frame.loc[frame["cell_type"].astype(str).eq(cell_line)].copy()
            subset.to_csv(pseudobulk_dir / f"sciplex_count_{kind}_{suffix}.csv")


def _mcfarland_drug_to_target() -> pd.DataFrame:
    path = Path(resources_dir) / "mcfarland_drug_to_perturbation.csv"
    mapping = pd.read_csv(path)[["drug", "target"]]
    mapping = pd.concat(
        [mapping, pd.DataFrame({"drug": ["DMSO"], "target": ["ctrl"]})],
        ignore_index=True,
    )
    return mapping.drop_duplicates(subset="drug")


def _n_mtx(raw_dir: Path) -> int:
    return sum(1 for _ in raw_dir.glob("*/matrix.mtx"))


def resolve_mcfarland_raw(override: Path | None = None) -> Path:
    """Prefer staff-bulk MIX-seq 10x folders; the umbrella copy is incomplete."""
    candidates: list[Path] = []
    if override is not None:
        candidates.append(override)
    candidates.extend(EXTERNAL_MCFARLAND_RAW_DIRS)
    candidates.append(REPO_MCFARLAND_RAW)
    ranked: list[tuple[int, Path]] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen or not path.exists():
            continue
        seen.add(key)
        n = _n_mtx(path)
        logger.info("McFarland raw candidate %s (%d matrix.mtx)", path, n)
        ranked.append((n, path))
    if not ranked:
        return REPO_MCFARLAND_RAW
    ranked.sort(key=lambda item: item[0], reverse=True)
    chosen = ranked[0][1]
    logger.info("Using McFarland 10x dir %s", chosen)
    return chosen


def _mcfarland_experiment_rows(raw_dir: Path) -> pd.DataFrame:
    """List MIX-seq 10x folders without ``mcfarland_dataset`` (needs cell_lines.csv)."""
    mapping = _mcfarland_drug_to_target()
    drugs = mapping["drug"].dropna().astype(str).unique().tolist()
    rows = []
    for drug in drugs:
        for path in raw_dir.glob(f"{drug}*expt*"):
            if not path.is_dir():
                continue
            name = path.name
            if any(token in name for token in (".zip", "6hr", "expt5")):
                continue
            if not (path / "matrix.mtx").exists():
                continue
            experiment = name.split("_", 1)[1] if "_" in name else name
            rows.append({"filename": name, "drug": drug, "experiment": experiment})
    files = pd.DataFrame(rows)
    if files.empty:
        logger.warning("No McFarland experiment folders with matrix.mtx under %s", raw_dir)
        return files
    files = files.merge(mapping, on="drug", how="left")
    n_unmapped = int(files["target"].isna().sum())
    if n_unmapped:
        logger.warning("Dropping %d experiment folders with unmapped drug names", n_unmapped)
        files = files.dropna(subset=["target"])
    files = files.drop_duplicates(subset="filename").reset_index(drop=True)
    logger.info("McFarland count load: %d experiment folders", len(files))
    return files


def mcfarland_count_pseudobulk(processed: sc.AnnData, raw_dir: Path) -> None:
    gene_list = get_gene_set()
    n_genes = len(gene_list)
    processed_index = pd.Index(processed.obs_names.astype(str))
    qc_pass = (
        processed.obs["qc_pass"].to_numpy()
        if "qc_pass" in processed.obs.columns
        else np.ones(processed.n_obs, dtype=bool)
    )
    pass_index = set(processed_index[qc_pass].astype(str))
    assay_present = np.zeros(n_genes, dtype=bool)
    sums: dict[tuple, np.ndarray] = {}
    n_cells: dict[tuple, int] = {}
    n_matched = 0
    n_experiments_used = 0

    files = _mcfarland_experiment_rows(raw_dir)
    for _, row in files.iterrows():
        filename = str(row["filename"])
        condition = str(row["target"])
        mtx_dir = raw_dir / filename
        if not (mtx_dir / "matrix.mtx").exists():
            logger.warning("Skipping %s: no matrix.mtx", mtx_dir)
            continue
        logger.info("Reading 10x counts %s (%s)", filename, condition)
        try:
            adata = sc.read_10x_mtx(mtx_dir)
        except Exception as exc:
            logger.warning("Skipping %s: failed to read 10x matrix (%s)", mtx_dir, exc)
            continue
        keep = adata.obs_names.astype(str).isin(pass_index)
        n_keep = int(keep.sum())
        if n_keep == 0:
            logger.info("%s: 0 barcodes overlap processed qc_pass cells", filename)
            continue
        adata = adata[keep].copy()
        n_matched += n_keep
        n_experiments_used += 1

        obs = processed.obs.loc[adata.obs_names]
        cell_type = obs["cell_type"].astype(str)
        replicate = filename
        X_aligned, mask = align_matrix_to_gene_list(
            adata.X, adata.var_names, gene_list, collapse=True
        )
        assay_present |= mask == 0
        X_aligned = _to_csr(X_aligned)
        meta = pd.DataFrame(
            {
                "cell_type": cell_type.to_numpy(),
                "condition": condition,
                "replicate": replicate,
            },
            index=adata.obs_names,
        )
        for key, idx in meta.groupby(["cell_type", "condition", "replicate"], sort=False).indices.items():
            block = np.asarray(X_aligned[idx].sum(axis=0)).ravel()
            if key in sums:
                sums[key] += block
                n_cells[key] += len(idx)
            else:
                sums[key] = block
                n_cells[key] = len(idx)
        del adata

    logger.info(
        "McFarland counts: %d cells from %d experiments (processed qc_pass=%d)",
        n_matched,
        n_experiments_used,
        int(qc_pass.sum()),
    )
    if n_matched < 0.5 * int(qc_pass.sum()):
        logger.warning(
            "Only %d/%d qc_pass cells had 10x counts. Count pseudobulks are incomplete "
            "(missing experiment folders under %s).",
            n_matched,
            int(qc_pass.sum()),
            raw_dir,
        )
    if not sums:
        logger.error(
            "No McFarland 10x counts matched processed cells. Expected experiment "
            "folders under %s (currently %d with matrix.mtx).",
            raw_dir,
            _n_mtx(raw_dir),
        )
        return

    records = []
    for key, vec in sums.items():
        cell_type, condition, replicate = key
        records.append(
            {
                "cell_type": cell_type,
                "condition": condition,
                "replicate": replicate,
                "n_cells": n_cells[key],
                "_counts": vec,
            }
        )
    count_df = counts_frame_from_records(pd.DataFrame.from_records(records), gene_list)
    mask_series = pd.Series(~assay_present, index=gene_list, name="assay_missing")
    mask_series.to_csv(qc_dir / "mcfarland_gene_assay_mask.csv", header=True)

    log_df, pre, post, lfc = profiles_from_count_frame(
        count_df,
        gene_list,
        sample_cols=["cell_type", "condition"],
        extra_cols=["n_replicates", "tissue"],
    )
    pre = add_tissue_column(pre)
    post = add_tissue_column(post)
    lfc = add_tissue_column(lfc)
    write_profile_tables(pre, post, lfc, "mcfarland_count")
    log_df = add_tissue_column(log_df)
    log_df.to_csv(pseudobulk_dir / "mcfarland_count_replicate.csv")
    logger.info("Wrote McFarland replicate-level count pseudobulk (%d rows)", len(log_df))


def run_qc(
    dataset: str,
    adata: sc.AnnData,
    group_cols: list[str],
    mcfarland_raw_dir: Path | None = None,
) -> sc.AnnData:
    return annotate_dataset_qc(
        adata,
        dataset=dataset,
        group_cols=group_cols,
        out_dir=qc_dir,
        mcfarland_raw_dir=mcfarland_raw_dir if dataset == "mcfarland" else None,
    )


def main(sciplex_raw: Path | None = None, mcfarland_raw: Path | None = None) -> None:
    if SCIPLEX_PROCESSED.exists():
        logger.info("QC SciPlex %s", SCIPLEX_PROCESSED)
        sciplex = sc.read_h5ad(SCIPLEX_PROCESSED)
        sciplex.obs = canonicalize_sciplex_obs_columns(sciplex.obs)
        group_cols = ["cell_type", "condition"]
        if "replicate" in sciplex.obs.columns:
            group_cols.append("replicate")
        sciplex = run_qc("sciplex", sciplex, group_cols)
        obs_csv = SCIPLEX_PROCESSED.with_name(SCIPLEX_PROCESSED.stem + "_obs.csv")
        sciplex.obs.to_csv(obs_csv)
        logger.info("Wrote SciPlex obs CSV with QC columns to %s", obs_csv)
        sciplex_count_pseudobulk(sciplex, raw_path=sciplex_raw)
    else:
        logger.warning("SciPlex processed file not found: %s", SCIPLEX_PROCESSED)

    if MCFARLAND_PROCESSED.exists():
        raw_dir = resolve_mcfarland_raw(mcfarland_raw)
        logger.info("QC McFarland %s", MCFARLAND_PROCESSED)
        mcf = sc.read_h5ad(MCFARLAND_PROCESSED)
        group_cols = ["cell_type", "condition"]
        mcf = run_qc("mcfarland", mcf, group_cols, mcfarland_raw_dir=raw_dir)
        obs_csv = MCFARLAND_PROCESSED.with_name(MCFARLAND_PROCESSED.stem + "_obs.csv")
        mcf.obs.to_csv(obs_csv)
        logger.info("Wrote McFarland obs CSV with QC columns to %s", obs_csv)
        mcfarland_count_pseudobulk(mcf, raw_dir=raw_dir)
    else:
        logger.warning("McFarland processed file not found: %s", MCFARLAND_PROCESSED)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sciplex-raw",
        type=Path,
        default=None,
        help="Path to SciPlex count-level h5ad (barcode-matched to processed obs).",
    )
    parser.add_argument(
        "--mcfarland-raw",
        type=Path,
        default=None,
        help="MIX-seq 10x directory (defaults to staff-bulk mcfarland_raw if present).",
    )
    args = parser.parse_args()
    log_script_start(__file__, logger)
    try:
        main(sciplex_raw=args.sciplex_raw, mcfarland_raw=args.mcfarland_raw)
        logger.info("QC and count-based pseudobulks finished")
    except Exception as exc:
        logger.error("Error in create_qc_and_count_pseudobulk: %s", exc)
        raise
    log_script_end(__file__, logger)
