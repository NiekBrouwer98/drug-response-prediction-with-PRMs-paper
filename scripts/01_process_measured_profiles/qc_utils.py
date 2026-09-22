"""Essential QC for measured SciPlex / McFarland profiles.

Reports cell-level metrics, condition-level cell counts, and a qc_pass flag.
Does not drop cells from the processed AnnData used by existing models; count
pseudobulk filters on qc_pass.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from config import config

logger = logging.getLogger(__name__)

CONTROL_CONDITION = "ctrl"
MITO_PREFIXES = ("MT-", "mt-")
BARCODE_CHANNEL_RE = r"_ch\d+$"


def looks_like_counts(X, n_check: int = 5000) -> bool:
    """True when X is non-negative and integer-valued with a count-like range."""
    if X is None:
        return False
    if sparse.issparse(X):
        data = np.asarray(X.data[:n_check], dtype=np.float64)
    else:
        data = np.asarray(X, dtype=np.float64).ravel()[:n_check]
    if data.size == 0:
        return False
    if np.any(data < -1e-6):
        return False
    if not np.allclose(data, np.round(data), atol=1e-4):
        return False
    return float(np.nanmax(data)) > 20.0


def counts_matrix(adata: sc.AnnData):
    """Prefer ``layers['counts']``, else integer X."""
    if "counts" in adata.layers:
        return adata.layers["counts"]
    if looks_like_counts(adata.X):
        return adata.X
    return None


def mito_gene_mask(var_names: pd.Index) -> np.ndarray:
    names = pd.Index(var_names.astype(str))
    mask = np.zeros(len(names), dtype=bool)
    for prefix in MITO_PREFIXES:
        mask |= np.asarray(names.str.startswith(prefix), dtype=bool)
    return mask


def _row_sums(X) -> np.ndarray:
    if sparse.issparse(X):
        return np.asarray(X.sum(axis=1)).ravel()
    return np.asarray(X).sum(axis=1).ravel()


def _row_detected(X) -> np.ndarray:
    if sparse.issparse(X):
        return np.asarray((X > 0).sum(axis=1)).ravel()
    return np.asarray(X > 0).sum(axis=1).ravel()


def annotate_expression_qc(adata: sc.AnnData) -> sc.AnnData:
    """Add n_genes, total_counts, and pct_counts_mt from counts when available."""
    X_counts = counts_matrix(adata)
    source = "counts" if X_counts is not None else "X"
    X = X_counts if X_counts is not None else adata.X
    adata.obs["n_genes"] = _row_detected(X).astype(np.int32)
    adata.obs["total_counts"] = _row_sums(X).astype(np.float32)
    if "total_count" in adata.obs.columns and "total_counts" in adata.obs.columns:
        # Keep the historical McFarland column; prefer the newly computed counts total.
        pass

    mito = mito_gene_mask(adata.var_names)
    n_mito = int(mito.sum())
    if n_mito and X_counts is not None:
        adata.obs["pct_counts_mt"] = np.where(
            adata.obs["total_counts"].to_numpy() > 0,
            100.0 * _row_sums(X_counts[:, mito]) / np.clip(adata.obs["total_counts"].to_numpy(), 1, None),
            np.nan,
        )
        adata.obs["qc_metrics_source"] = source
    else:
        adata.obs["pct_counts_mt"] = np.nan
        adata.obs["qc_metrics_source"] = f"{source}_no_mito" if n_mito == 0 else f"{source}_not_counts"
        if n_mito == 0:
            logger.info("No MT- genes in var_names; pct_counts_mt left as NA")
        elif X_counts is None:
            logger.info("Expression is not counts; pct_counts_mt left as NA (not a valid QC metric on log1p)")
    logger.info(
        "QC metrics from %s: n_genes median=%.0f total_counts median=%.1f",
        adata.obs["qc_metrics_source"].iloc[0],
        float(adata.obs["n_genes"].median()),
        float(adata.obs["total_counts"].median()),
    )
    return adata


def normalize_barcode(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(BARCODE_CHANNEL_RE, "", regex=True)


def load_mcfarland_classifications(raw_dir: Path) -> pd.DataFrame:
    """Concatenate MIX-seq ``classifications.csv`` files under ``mcfarland_raw``."""
    files = sorted(Path(raw_dir).glob("*/classifications.csv"))
    if not files:
        logger.warning("No classifications.csv under %s", raw_dir)
        return pd.DataFrame()

    frames = []
    for path in files:
        df = pd.read_csv(path)
        if "barcode" not in df.columns:
            logger.warning("Skipping %s: no barcode column", path)
            continue
        df = df.copy()
        df["experiment"] = path.parent.name
        df["barcode"] = df["barcode"].astype(str)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d classification rows from %d experiments", len(out), out["experiment"].nunique())
    return out


def join_mcfarland_classifications(obs: pd.DataFrame, classif: pd.DataFrame) -> pd.DataFrame:
    """Attach MIX-seq QC fields to processed obs by barcode."""
    obs = obs.copy()
    if classif.empty:
        obs["experiment"] = pd.NA
        obs["percent_mito"] = np.nan
        obs["cell_quality"] = pd.NA
        obs["doublet_GMM_prob"] = np.nan
        obs["singlet_ID"] = pd.NA
        return obs

    keep_cols = [
        c
        for c in (
            "barcode",
            "experiment",
            "percent.mito",
            "cell_quality",
            "doublet_GMM_prob",
            "singlet_ID",
            "tot_reads",
        )
        if c in classif.columns
    ]
    classif = classif[keep_cols].drop_duplicates(subset="barcode", keep="first")
    classif = classif.rename(columns={"percent.mito": "percent_mito"})
    drop_existing = [c for c in classif.columns if c != "barcode" and c in obs.columns]
    obs = obs.drop(columns=drop_existing, errors="ignore")

    left = obs.reset_index()
    index_col = left.columns[0]
    left["barcode_key"] = left[index_col].astype(str)
    classif = classif.copy()
    classif["barcode_key"] = classif["barcode"].astype(str)
    merged = left.merge(classif.drop(columns=["barcode"]), on="barcode_key", how="left")
    n_matched = int(merged["experiment"].notna().sum()) if "experiment" in merged.columns else 0
    if n_matched == 0:
        left["barcode_key"] = normalize_barcode(left[index_col])
        classif["barcode_key"] = normalize_barcode(classif["barcode"])
        merged = left.merge(
            classif.drop(columns=["barcode"]).drop_duplicates("barcode_key"),
            on="barcode_key",
            how="left",
        )
        n_matched = int(merged["experiment"].notna().sum()) if "experiment" in merged.columns else 0
    logger.info("Joined MIX-seq classifications for %d/%d cells", n_matched, len(merged))
    merged = merged.drop(columns=["barcode_key"])
    merged = merged.set_index(index_col)
    merged.index.name = obs.index.name
    return merged


def add_qc_pass_flag(obs: pd.DataFrame) -> pd.DataFrame:
    """Flag cells that fail essential QC. Missing MIX-seq fields do not fail the cell."""
    obs = obs.copy()
    min_genes = int(config.MIN_GENES_PER_CELL)
    max_doublet = float(getattr(config, "DOUBLET_GMM_PROB_MAX", 0.5))
    max_mito = float(getattr(config, "MAX_PCT_MITO", 20.0))

    reasons: list[pd.Series] = []
    qc_pass = pd.Series(True, index=obs.index)

    if "n_genes" in obs.columns:
        low_genes = obs["n_genes"].fillna(min_genes) < min_genes
        qc_pass &= ~low_genes
        reasons.append(low_genes.map({True: "min_genes", False: ""}))

    if "cell_quality" in obs.columns:
        quality = obs["cell_quality"].fillna("").astype(str).str.lower()
        low_q = quality.isin(["low_quality", "doublet", "empty_droplet"])
        qc_pass &= ~low_q
        reasons.append(low_q.map({True: "cell_quality", False: ""}))

    if "doublet_GMM_prob" in obs.columns:
        doublet = obs["doublet_GMM_prob"].notna() & (obs["doublet_GMM_prob"] >= max_doublet)
        qc_pass &= ~doublet
        reasons.append(doublet.map({True: "doublet", False: ""}))

    mito_col = None
    if "percent_mito" in obs.columns and obs["percent_mito"].notna().any():
        mito_col = "percent_mito"
    elif "pct_counts_mt" in obs.columns and obs["pct_counts_mt"].notna().any():
        mito_col = "pct_counts_mt"
    mito_flag = pd.Series(False, index=obs.index)
    if mito_col is not None:
        mito_vals = pd.to_numeric(obs[mito_col], errors="coerce")
        mito_flag = mito_vals.notna() & (mito_vals > max_mito)
        obs["qc_high_mito"] = mito_flag
        logger.info(
            "High mito (>%.1f%%) in %d cells (flagged, not used to fail qc_pass)",
            max_mito,
            int(mito_flag.sum()),
        )
    else:
        obs["qc_high_mito"] = False

    obs["qc_pass"] = qc_pass.to_numpy()
    if reasons:
        stacked = pd.concat(reasons, axis=1).to_numpy()
        obs["qc_fail_reason"] = [
            ";".join(v for v in row if v) for row in stacked
        ]
    else:
        obs["qc_fail_reason"] = ""
    logger.info(
        "qc_pass=%d/%d (min_genes=%d, doublet_GMM<%.2f); mito is reported not filtered",
        int(obs["qc_pass"].sum()),
        len(obs),
        min_genes,
        max_doublet,
    )
    return obs


def condition_cell_counts(
    obs: pd.DataFrame,
    group_cols: list[str],
    qc_pass_only: bool = False,
) -> pd.DataFrame:
    """n_cells per condition (and replicate when present)."""
    frame = obs
    if qc_pass_only and "qc_pass" in obs.columns:
        frame = obs.loc[obs["qc_pass"]]
    cols = [c for c in group_cols if c in frame.columns]
    if not cols:
        raise KeyError(f"None of {group_cols} found in obs")
    counts = (
        frame.groupby(cols, observed=True, dropna=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    if "qc_pass" in obs.columns:
        n_pass = (
            obs.groupby(cols, observed=True, dropna=False)["qc_pass"]
            .sum()
            .rename("n_cells_qc_pass")
            .reset_index()
        )
        counts = counts.merge(n_pass, on=cols, how="left")
    counts["below_min_cells"] = counts["n_cells"] < int(config.MIN_CELLS_PER_CONDITION)
    return counts.sort_values("n_cells")


def add_n_cells_to_obs(obs: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    counts = condition_cell_counts(obs, group_cols)
    cols = [c for c in group_cols if c in obs.columns]
    index = obs.index
    index_name = obs.index.name
    obs = obs.copy().drop(columns=["n_cells"], errors="ignore")
    out = obs.merge(counts[cols + ["n_cells"]], on=cols, how="left")
    out.index = index
    out.index.name = index_name
    return out


def write_qc_tables(
    obs: pd.DataFrame,
    out_dir: Path,
    dataset: str,
    group_cols: list[str],
    gene_mask: pd.Series | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_path = out_dir / f"{dataset}_cell_qc.csv"
    cond_path = out_dir / f"{dataset}_condition_counts.csv"
    summary_path = out_dir / f"{dataset}_qc_summary.csv"

    cell_cols = [
        c
        for c in (
            "cell_type",
            "tissue",
            "condition",
            "drug",
            "replicate",
            "experiment",
            "n_genes",
            "total_counts",
            "total_count",
            "pct_counts_mt",
            "percent_mito",
            "qc_high_mito",
            "cell_quality",
            "doublet_GMM_prob",
            "qc_pass",
            "qc_fail_reason",
            "qc_metrics_source",
            "n_cells",
            "perturbation_match",
            "fold",
        )
        if c in obs.columns
    ]
    obs[cell_cols].to_csv(cell_path)
    logger.info("Wrote %s (%d cells)", cell_path, len(obs))

    counts = condition_cell_counts(obs, group_cols)
    counts.to_csv(cond_path, index=False)
    logger.info("Wrote %s (%d groups)", cond_path, len(counts))

    rep_extra = [c for c in ("replicate", "experiment") if c in obs.columns and c not in group_cols]
    if rep_extra:
        rep_counts = condition_cell_counts(obs, group_cols + rep_extra)
        rep_path = out_dir / f"{dataset}_replicate_counts.csv"
        rep_counts.to_csv(rep_path, index=False)
        logger.info("Wrote %s (%d groups)", rep_path, len(rep_counts))

    summary = {
        "dataset": dataset,
        "n_cells": len(obs),
        "n_qc_pass": int(obs["qc_pass"].sum()) if "qc_pass" in obs.columns else len(obs),
        "n_conditions": (
            int(obs["condition"].nunique())
            if "condition" in obs.columns
            else pd.NA
        ),
        "n_cell_types": int(obs["cell_type"].nunique()) if "cell_type" in obs.columns else pd.NA,
        "n_groups_below_min_cells": int(counts["below_min_cells"].sum()),
        "min_cells_per_condition": int(config.MIN_CELLS_PER_CONDITION),
        "min_genes_per_cell": int(config.MIN_GENES_PER_CELL),
        "median_n_genes": float(obs["n_genes"].median()) if "n_genes" in obs.columns else pd.NA,
        "median_total_counts": (
            float(obs["total_counts"].median()) if "total_counts" in obs.columns else pd.NA
        ),
        "n_high_mito": int(obs["qc_high_mito"].sum()) if "qc_high_mito" in obs.columns else 0,
        "n_assay_missing_genes": int(gene_mask.sum()) if gene_mask is not None else pd.NA,
        "n_assayed_genes": int((~gene_mask).sum()) if gene_mask is not None else pd.NA,
    }
    if "replicate" in obs.columns:
        summary["n_replicates"] = int(obs["replicate"].nunique())
    if "experiment" in obs.columns:
        summary["n_experiments"] = int(obs["experiment"].nunique())
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    logger.info("Wrote %s", summary_path)

    if gene_mask is not None:
        mask_path = out_dir / f"{dataset}_gene_assay_mask.csv"
        gene_mask.rename("assay_missing").to_csv(mask_path, header=True)
        logger.info(
            "Wrote %s (%d assay-missing / %d genes)",
            mask_path,
            int(gene_mask.sum()),
            len(gene_mask),
        )


def gene_assay_mask(adata: sc.AnnData) -> pd.Series | None:
    if "mask" in adata.var.columns:
        return adata.var["mask"].astype(bool).rename("assay_missing")
    return None


def annotate_dataset_qc(
    adata: sc.AnnData,
    dataset: str,
    group_cols: list[str],
    out_dir: Path,
    mcfarland_raw_dir: Path | None = None,
) -> sc.AnnData:
    """Annotate obs with QC fields, write tables, keep all cells."""
    adata.obs = adata.obs.copy()
    annotate_expression_qc(adata)
    if dataset == "mcfarland" and mcfarland_raw_dir is not None:
        classif = load_mcfarland_classifications(mcfarland_raw_dir)
        adata.obs = join_mcfarland_classifications(adata.obs, classif)
    adata.obs = add_qc_pass_flag(adata.obs)
    adata.obs = add_n_cells_to_obs(adata.obs, group_cols)
    write_qc_tables(
        adata.obs,
        out_dir,
        dataset,
        group_cols,
        gene_mask=gene_assay_mask(adata),
    )
    return adata
