import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path for imports (must be before importing project modules)
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config, setup_project
from prediction_utils import (
    add_y_and_normalize,
    feature_selection,
    filter_on_coefficient_of_variation,
    get_sciplex_AUCs,
    get_sciplex_mean_data,
)
from response_prediction import (
    out_of_distribution_elasticnet,
    prediction_pipeline,
    within_distribution_elasticnet,
)
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from utils import (
    ensure_directories_exist,
    log_script_end,
    log_script_start,
    setup_logging_for_script,
)

# Setup project and logging
setup_project()
logger = setup_logging_for_script(__file__)

# Set project directories using configuration
home_dir = str(config.PROJECT_ROOT)
data_dir = str(config.DATA_DIR)
resources_dir = str(config.RESOURCES_DIR)
results_dir = str(config.RESULTS_04_DIR)
figures_dir = str(config.FIGURES_04_DIR)

# Ensure directories exist
ensure_directories_exist(results_dir, figures_dir, data_dir, resources_dir)


def _get_sciplex_data_with_sensitivity() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mean_pre, mean_post, mean_lfc = get_sciplex_mean_data()
    sensitivities = get_sciplex_AUCs().rename(columns={"y": "sens"})

    mean_pre = pd.merge(mean_pre, sensitivities, on=["condition", "cell_line"], how="left").dropna()
    mean_post = pd.merge(mean_post, sensitivities, on=["condition", "cell_line"], how="left").dropna()
    mean_lfc = pd.merge(mean_lfc, sensitivities, on=["condition", "cell_line"], how="left").dropna()

    # Keep compatibility with add_y_and_normalize(), which expects these columns.
    for df in (mean_pre, mean_post, mean_lfc):
        df["target"] = df["condition"]
        df["sens_label"] = (df["sens"] >= 0.2).astype(int)

    # Keep compatibility with generic modeling code that expects a tissue column.
    mean_pre["tissue"] = mean_pre["cell_line"]
    mean_post["tissue"] = mean_post["cell_line"]
    mean_lfc["tissue"] = mean_lfc["cell_line"]

    return mean_pre, mean_post, mean_lfc


def predictions_sciplex(n_features: int = 1000, feature_subset: list[str] | None = None) -> None:
    if feature_subset is None:
        feature_subset = []

    logger.info(f"Performing SciPlex predictions with {n_features} features")
    mean_pre, mean_post, mean_lfc = _get_sciplex_data_with_sensitivity()

    mean_pre = filter_on_coefficient_of_variation(mean_pre, groupby=["condition"])
    mean_post = filter_on_coefficient_of_variation(mean_post, groupby=["condition"])
    mean_lfc = filter_on_coefficient_of_variation(mean_lfc, groupby=["condition"])

    feature_to_keep = feature_subset + ["condition", "tissue", "cell_line"]
    mean_pre_with_y = add_y_and_normalize(mean_pre, "sens", normalize=False, keep=feature_to_keep)
    mean_post_with_y = add_y_and_normalize(mean_post, "sens", normalize=False, keep=feature_to_keep)
    mean_lfc_with_y = add_y_and_normalize(mean_lfc, "sens", normalize=False, keep=feature_to_keep)

    pre_outcomes, pre_predictions, pre_feature_importance = prediction_pipeline(
        mean_pre_with_y, mean_pre, "pre_treatment", n_features=n_features, feature_subset=feature_subset
    )
    post_outcomes, post_predictions, post_feature_importance = prediction_pipeline(
        mean_post_with_y, mean_pre, "post_treatment", n_features=n_features, feature_subset=feature_subset
    )
    lfc_outcomes, lfc_predictions, lfc_feature_importance = prediction_pipeline(
        mean_lfc_with_y, mean_pre, "LFC", n_features=n_features, feature_subset=feature_subset
    )

    results = pd.concat([pre_outcomes, post_outcomes, lfc_outcomes])
    predictions = pd.concat([pre_predictions, post_predictions, lfc_predictions])
    feature_importance = pd.concat(
        [
            pre_feature_importance.assign(model="Pre"),
            post_feature_importance.assign(model="Post"),
            lfc_feature_importance.assign(model="Log(fold change)"),
        ]
    )

    if len(feature_subset) == 0:
        results.to_csv(os.path.join(results_dir, f"sciplex_regression_{n_features}_features_results.csv"))
        predictions.to_csv(os.path.join(results_dir, f"sciplex_regression_{n_features}_features_predictions.csv"))
        feature_importance.to_csv(
            os.path.join(results_dir, f"sciplex_regression_{n_features}_features_feature_importance.csv")
        )
    else:
        results.to_csv(os.path.join(results_dir, "sciplex_regression_manual_features_results.csv"))
        predictions.to_csv(os.path.join(results_dir, "sciplex_regression_manual_features_predictions.csv"))
        feature_importance.to_csv(os.path.join(results_dir, "sciplex_regression_manual_features_feature_importance.csv"))


def predictions_sciplex_per_treatment(n_features: int = 1000, outersplits: int = 5) -> None:
    mean_pre, mean_post, mean_lfc = _get_sciplex_data_with_sensitivity()

    results = pd.DataFrame()
    predictions_df = pd.DataFrame()
    feature_importance_pre: dict[str, pd.DataFrame] = {}
    feature_importance_post: dict[str, pd.DataFrame] = {}
    feature_importance_lfc: dict[str, pd.DataFrame] = {}

    features = feature_selection(mean_pre, n_features)

    mean_pre = filter_on_coefficient_of_variation(mean_pre, groupby=["condition"])
    mean_post = filter_on_coefficient_of_variation(mean_post, groupby=["condition"])
    mean_lfc = filter_on_coefficient_of_variation(mean_lfc, groupby=["condition"])

    conditions = mean_pre["condition"].unique().tolist()
    for condition in conditions:
        pre_subset = mean_pre[mean_pre["condition"] == condition]
        post_subset = mean_post[mean_post["condition"] == condition]
        lfc_subset = mean_lfc[mean_lfc["condition"] == condition]

        pre_subset = add_y_and_normalize(pre_subset, "sens", normalize=False, keep=["condition", "tissue", "cell_line"])
        post_subset = add_y_and_normalize(post_subset, "sens", normalize=False, keep=["condition", "tissue", "cell_line"])
        lfc_subset = add_y_and_normalize(lfc_subset, "sens", normalize=False, keep=["condition", "tissue", "cell_line"])

        if pre_subset.shape[0] < 20:
            logger.info(f"Too few samples for {condition} ({pre_subset.shape[0]} samples)")
            continue

        pre_outcomes, pre_predictions, pre_feature_importance = prediction_pipeline(
            pre_subset, mean_pre, "pre_treatment", feature_subset=features, n_outer_splits=outersplits
        )
        post_outcomes, post_predictions, post_feature_importance = prediction_pipeline(
            post_subset, mean_pre, "post_treatment", feature_subset=features, n_outer_splits=outersplits
        )
        lfc_outcomes, lfc_predictions, lfc_feature_importance = prediction_pipeline(
            lfc_subset, mean_pre, "LFC", feature_subset=features, n_outer_splits=outersplits
        )

        pre_outcomes = pre_outcomes.assign(condition=condition).assign(n_genes=n_features)
        post_outcomes = post_outcomes.assign(condition=condition).assign(n_genes=n_features)
        lfc_outcomes = lfc_outcomes.assign(condition=condition).assign(n_genes=n_features)
        results = pd.concat([results, pre_outcomes, post_outcomes, lfc_outcomes])

        pre_predictions = pre_predictions.assign(condition=condition)
        post_predictions = post_predictions.assign(condition=condition)
        lfc_predictions = lfc_predictions.assign(condition=condition)
        predictions_df = pd.concat([predictions_df, pre_predictions, post_predictions, lfc_predictions])

        feature_importance_pre[condition] = pre_feature_importance
        feature_importance_post[condition] = post_feature_importance
        feature_importance_lfc[condition] = lfc_feature_importance

    results.to_csv(os.path.join(results_dir, "sciplex_treatmentmodels_regression_results.csv"))
    predictions_df.to_csv(os.path.join(results_dir, "sciplex_treatmentmodels_regression_predictions.csv"))
    with open(os.path.join(results_dir, "sciplex_treatmentmodels_regression_feature_importance_on_pre.pkl"), "wb") as file:
        pickle.dump(feature_importance_pre, file)
    with open(os.path.join(results_dir, "sciplex_treatmentmodels_regression_feature_importance_on_post.pkl"), "wb") as file:
        pickle.dump(feature_importance_post, file)
    with open(os.path.join(results_dir, "sciplex_treatmentmodels_regression_feature_importance_on_lfc.pkl"), "wb") as file:
        pickle.dump(feature_importance_lfc, file)


def predictions_sciplex_per_tissue(n_features: int = 1000, outersplits: int = 5) -> None:
    mean_pre, mean_post, mean_lfc = _get_sciplex_data_with_sensitivity()

    results = pd.DataFrame()
    predictions_df = pd.DataFrame()
    feature_importance_pre: dict[str, pd.DataFrame] = {}
    feature_importance_post: dict[str, pd.DataFrame] = {}
    feature_importance_lfc: dict[str, pd.DataFrame] = {}

    features = feature_selection(mean_pre, n_features)

    mean_pre = filter_on_coefficient_of_variation(mean_pre, groupby=["tissue"])
    mean_post = filter_on_coefficient_of_variation(mean_post, groupby=["tissue"])
    mean_lfc = filter_on_coefficient_of_variation(mean_lfc, groupby=["tissue"])

    tissues = mean_pre["tissue"].unique().tolist()
    for tissue in tissues:
        pre_subset = mean_pre[mean_pre["tissue"] == tissue]
        post_subset = mean_post[mean_post["tissue"] == tissue]
        lfc_subset = mean_lfc[mean_lfc["tissue"] == tissue]

        pre_subset = add_y_and_normalize(pre_subset, "sens", normalize=False, keep=["condition", "tissue", "cell_line"])
        post_subset = add_y_and_normalize(post_subset, "sens", normalize=False, keep=["condition", "tissue", "cell_line"])
        lfc_subset = add_y_and_normalize(lfc_subset, "sens", normalize=False, keep=["condition", "tissue", "cell_line"])

        if pre_subset.shape[0] < 20:
            logger.info(f"Too few samples for {tissue} ({pre_subset.shape[0]} samples)")
            continue

        pre_outcomes, pre_predictions, pre_feature_importance = prediction_pipeline(
            pre_subset, mean_pre, "pre_treatment", feature_subset=features, n_outer_splits=outersplits
        )
        post_outcomes, post_predictions, post_feature_importance = prediction_pipeline(
            post_subset, mean_pre, "post_treatment", feature_subset=features, n_outer_splits=outersplits
        )
        lfc_outcomes, lfc_predictions, lfc_feature_importance = prediction_pipeline(
            lfc_subset, mean_pre, "LFC", feature_subset=features, n_outer_splits=outersplits
        )

        pre_outcomes = pre_outcomes.assign(tissue=tissue).assign(n_genes=n_features)
        post_outcomes = post_outcomes.assign(tissue=tissue).assign(n_genes=n_features)
        lfc_outcomes = lfc_outcomes.assign(tissue=tissue).assign(n_genes=n_features)
        results = pd.concat([results, pre_outcomes, post_outcomes, lfc_outcomes])

        pre_predictions = pre_predictions.assign(tissue=tissue)
        post_predictions = post_predictions.assign(tissue=tissue)
        lfc_predictions = lfc_predictions.assign(tissue=tissue)
        predictions_df = pd.concat([predictions_df, pre_predictions, post_predictions, lfc_predictions])

        feature_importance_pre[tissue] = pre_feature_importance
        feature_importance_post[tissue] = post_feature_importance
        feature_importance_lfc[tissue] = lfc_feature_importance

    results.to_csv(os.path.join(results_dir, "sciplex_tissuemodels_regression_results.csv"))
    predictions_df.to_csv(os.path.join(results_dir, "sciplex_tissuemodels_regression_predictions.csv"))
    with open(os.path.join(results_dir, "sciplex_tissuemodels_regression_feature_importance_on_pre.pkl"), "wb") as file:
        pickle.dump(feature_importance_pre, file)
    with open(os.path.join(results_dir, "sciplex_tissuemodels_regression_feature_importance_on_post.pkl"), "wb") as file:
        pickle.dump(feature_importance_post, file)
    with open(os.path.join(results_dir, "sciplex_tissuemodels_regression_feature_importance_on_lfc.pkl"), "wb") as file:
        pickle.dump(feature_importance_lfc, file)


def predictions_sciplex_seen_seen() -> None:
    """
    5-fold CV for the "seen drug, seen cell line" scenario.

    Avoids exhaustive leave-(condition,tissue)-out (nested CV) which is brittle for small
    numbers of tissues (here: only 3 cell lines).
    """

    def _seen_seen_cv(
        df_with_y: pd.DataFrame,
        selected_features: list[str],
        n_folds: int = 5,
        random_state: int = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
        df_with_y = df_with_y.reset_index(drop=True).dropna(axis=0)
        rng = np.random.default_rng(random_state)

        fold_id = np.full(len(df_with_y), fill_value=-1, dtype=int)
        # Assign folds within each tissue to ensure the cell line remains supported in training.
        for tissue in df_with_y["tissue"].unique().tolist():
            idx = df_with_y.index[df_with_y["tissue"] == tissue].to_numpy()
            order = idx.copy()
            if len(order) > n_folds:
                rng.shuffle(order)
            fold_id[idx] = np.arange(len(order)) % n_folds

        param_grid = {
            "elasticnet__alpha": [0.1, 1.0, 10.0],
            "elasticnet__l1_ratio": [0.0, 0.1, 0.5, 1.0],
        }

        outer_scores: list[pd.DataFrame] = []
        outcomes: list[pd.DataFrame] = []
        feature_importance_dict: dict[str, pd.Series] = {}

        for fold_idx in range(n_folds):
            test_mask = fold_id == fold_idx
            if test_mask.sum() == 0:
                continue

            train_df = df_with_y.loc[~test_mask].copy()
            test_df = df_with_y.loc[test_mask].copy()

            y_train = train_df["y"].astype(float)
            y_test = test_df["y"].astype(float)

            X_train_df = train_df[selected_features].copy()
            X_test_df = test_df[selected_features].copy()

            identifiers = X_test_df[["tissue", "cell_line", "condition"]].reset_index(drop=True)

            X_train_mat = X_train_df.drop(columns=["tissue", "cell_line", "condition"]).astype(float)
            X_test_mat = X_test_df.drop(columns=["tissue", "cell_line", "condition"]).astype(float)

            if len(X_train_mat) < 2 or len(X_test_mat) < 1:
                continue

            cv_folds = min(5, len(X_train_mat))
            if cv_folds < 2:
                continue

            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("elasticnet", ElasticNet(max_iter=1000)),
                ]
            )

            grid_search = GridSearchCV(pipeline, param_grid, cv=cv_folds)
            grid_search.fit(X_train_mat, y_train)

            best_model = grid_search.best_estimator_.named_steps["elasticnet"]
            y_pred = grid_search.predict(X_test_mat)
            outer_acc = float(np.sqrt(mean_squared_error(y_test, y_pred)))

            fold_name = f"fold={fold_idx}"
            feature_importance_dict[fold_name] = pd.Series(
                index=X_train_mat.columns, data=best_model.coef_, name=fold_name
            )

            test_outcomes = pd.concat(
                [
                    identifiers,
                    pd.DataFrame({"pred": y_pred, "true": y_test.reset_index(drop=True)}),
                ],
                axis=1,
            )

            # Keep the same test_condition format used by the plotting notebook.
            test_outcomes["test_condition"] = (
                "condition="
                + test_outcomes["condition"].astype(str)
                + " AND tissue="
                + test_outcomes["tissue"].astype(str)
            )

            outcomes.append(test_outcomes.assign(split=fold_idx))
            outer_scores.append(
                pd.DataFrame({"accuracy": [outer_acc], "test_condition": [fold_name], "split": [fold_idx]})
            )

        outer_scores_df = (
            pd.concat(outer_scores, ignore_index=True)
            if outer_scores
            else pd.DataFrame(columns=["accuracy", "test_condition", "split"])
        )
        outcomes_df = (
            pd.concat(outcomes, ignore_index=True)
            if outcomes
            else pd.DataFrame(
                columns=["tissue", "cell_line", "condition", "pred", "true", "test_condition", "split"]
            )
        )
        return outer_scores_df, outcomes_df, feature_importance_dict

    mean_pre, mean_post, mean_lfc = _get_sciplex_data_with_sensitivity()
    mean_pre_with_y = add_y_and_normalize(mean_pre, "sens", normalize=False, keep=["condition", "tissue", "cell_line"])
    mean_post_with_y = add_y_and_normalize(mean_post, "sens", normalize=False, keep=["condition", "tissue", "cell_line"])
    mean_lfc_with_y = add_y_and_normalize(mean_lfc, "sens", normalize=False, keep=["condition", "tissue", "cell_line"])

    features = feature_selection(mean_pre_with_y.drop(columns=["y"]), 1000)

    pre_scores, pre_predictions, fi_pre = _seen_seen_cv(mean_pre_with_y, features)
    post_scores, post_predictions, fi_post = _seen_seen_cv(mean_post_with_y, features)
    lfc_scores, lfc_predictions, fi_lfc = _seen_seen_cv(mean_lfc_with_y, features)

    pre_scores = pre_scores.assign(model="pre-treatment").assign(feature_selection="top1000 highest variance")
    post_scores = post_scores.assign(model="post-treatment").assign(feature_selection="top1000 highest variance")
    lfc_scores = lfc_scores.assign(model="LFC").assign(feature_selection="top1000 highest variance")
    pre_predictions = pre_predictions.assign(model="pre-treatment").assign(feature_selection="top1000 highest variance")
    post_predictions = post_predictions.assign(model="post-treatment").assign(feature_selection="top1000 highest variance")
    lfc_predictions = lfc_predictions.assign(model="LFC").assign(feature_selection="top1000 highest variance")

    pd.concat([pre_scores, post_scores, lfc_scores]).to_csv(
        os.path.join(results_dir, "sciplex_regression_seen_seen.csv")
    )
    pd.concat([pre_predictions, post_predictions, lfc_predictions]).to_csv(
        os.path.join(results_dir, "sciplex_regression_predictions_seen_seen.csv")
    )

    with open(os.path.join(results_dir, "sciplex_regression_seen_seen_feature_importance_on_pre.pkl"), "wb") as file:
        pickle.dump(fi_pre, file)
    with open(os.path.join(results_dir, "sciplex_regression_seen_seen_feature_importance_on_post.pkl"), "wb") as file:
        pickle.dump(fi_post, file)
    with open(os.path.join(results_dir, "sciplex_regression_seen_seen_feature_importance_on_lfc.pkl"), "wb") as file:
        pickle.dump(fi_lfc, file)


def predictions_sciplex_ldo() -> None:
    """
    5-fold CV for the "unseen drug, seen cell line" scenario.

    This replaces exhaustive leave-drug-out, which can be unstable when only 3
    cell lines exist (test sets become too small for the nested modeling).
    """

    def _drug_out_cv(
        df_with_y: pd.DataFrame,
        selected_features: list[str],
        n_folds: int = 5,
        random_state: int = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
        df_with_y = df_with_y.reset_index().dropna(axis=0)

        all_drugs = df_with_y["condition"].drop_duplicates().tolist()
        all_drugs = sorted(all_drugs)
        rng = np.random.default_rng(random_state)
        rng.shuffle(all_drugs)

        drug_folds = np.array_split(np.array(all_drugs, dtype=object), n_folds)

        outer_scores: list[pd.DataFrame] = []
        outcomes: list[pd.DataFrame] = []
        feature_importance_dict: dict[str, pd.Series] = {}

        param_grid = {
            "elasticnet__alpha": [0.1, 1.0, 10.0],
            "elasticnet__l1_ratio": [0.0, 0.1, 0.5, 1.0],
        }

        for fold_idx, test_drugs in enumerate(drug_folds, start=1):
            test_mask = df_with_y["condition"].isin(test_drugs.tolist())
            if test_mask.sum() == 0:
                continue

            X_train_df = df_with_y.loc[~test_mask].copy()
            X_test_df = df_with_y.loc[test_mask].copy()

            y_train = X_train_df["y"].astype(float)
            y_test = X_test_df["y"].astype(float)

            X_train_df = X_train_df.drop(columns=["y"])
            X_test_df = X_test_df.drop(columns=["y"])

            identifiers = X_test_df[["tissue", "cell_line", "condition"]].reset_index(drop=True)

            X_train_df = X_train_df[selected_features]
            X_test_df = X_test_df[selected_features]

            # Remove metadata columns from the feature matrix
            X_train_mat = X_train_df.drop(columns=["tissue", "cell_line", "condition"]).astype(float)
            X_test_mat = X_test_df.drop(columns=["tissue", "cell_line", "condition"]).astype(float)

            # Guard against too-small folds
            if len(X_train_mat) < 2 or len(X_test_mat) < 1:
                continue

            cv_folds = min(5, len(X_train_mat))
            if cv_folds < 2:
                continue

            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("elasticnet", ElasticNet(max_iter=1000)),
                ]
            )

            grid_search = GridSearchCV(pipeline, param_grid, cv=cv_folds)
            grid_search.fit(X_train_mat, y_train)

            best_model = grid_search.best_estimator_.named_steps["elasticnet"]
            y_pred = grid_search.predict(X_test_mat)
            outer_acc = float(np.sqrt(mean_squared_error(y_test, y_pred)))

            test_condition_name = f"fold={fold_idx};n_drugs={len(test_drugs)}"
            feature_importance = pd.Series(index=X_train_mat.columns, data=best_model.coef_, name=test_condition_name)
            feature_importance_dict[test_condition_name] = feature_importance

            split_outcomes = pd.concat(
                [
                    identifiers,
                    pd.DataFrame({"pred": y_pred, "true": y_test.reset_index(drop=True)}),
                ],
                axis=1,
            )
            split_outcomes["test_condition"] = test_condition_name
            split_outcomes["split"] = fold_idx
            outcomes.append(split_outcomes)
            outer_scores.append(
                pd.DataFrame({"accuracy": [outer_acc], "test_condition": [test_condition_name], "split": [fold_idx]})
            )

        if outer_scores:
            outer_scores_df = pd.concat(outer_scores, ignore_index=True)
        else:
            outer_scores_df = pd.DataFrame(columns=["accuracy", "test_condition", "split"])

        if outcomes:
            outcomes_df = pd.concat(outcomes, ignore_index=True)
        else:
            outcomes_df = pd.DataFrame(
                columns=["tissue", "cell_line", "condition", "pred", "true", "test_condition", "split"]
            )

        return outer_scores_df, outcomes_df, feature_importance_dict

    mean_pre, mean_post, mean_lfc = _get_sciplex_data_with_sensitivity()
    mean_pre_with_y = add_y_and_normalize(mean_pre, "sens", normalize=False, keep=["condition", "tissue", "cell_line"])
    mean_post_with_y = add_y_and_normalize(mean_post, "sens", normalize=False, keep=["condition", "tissue", "cell_line"])
    mean_lfc_with_y = add_y_and_normalize(mean_lfc, "sens", normalize=False, keep=["condition", "tissue", "cell_line"])

    features = feature_selection(mean_pre_with_y.drop(columns=["y"]), 1000)

    pre_scores, pre_predictions, fi_pre = _drug_out_cv(mean_pre_with_y, features)
    post_scores, post_predictions, fi_post = _drug_out_cv(mean_post_with_y, features)
    lfc_scores, lfc_predictions, fi_lfc = _drug_out_cv(mean_lfc_with_y, features)

    pre_scores = pre_scores.assign(model="pre-treatment").assign(feature_selection="top1000 highest variance")
    post_scores = post_scores.assign(model="post-treatment").assign(feature_selection="top1000 highest variance")
    lfc_scores = lfc_scores.assign(model="LFC").assign(feature_selection="top1000 highest variance")

    pre_predictions = pre_predictions.assign(model="pre-treatment").assign(feature_selection="top1000 highest variance")
    post_predictions = post_predictions.assign(model="post-treatment").assign(feature_selection="top1000 highest variance")
    lfc_predictions = lfc_predictions.assign(model="LFC").assign(feature_selection="top1000 highest variance")

    pd.concat([pre_scores, post_scores, lfc_scores]).to_csv(os.path.join(results_dir, "sciplex_regression_LDO.csv"))
    pd.concat([pre_predictions, post_predictions, lfc_predictions]).to_csv(
        os.path.join(results_dir, "sciplex_regression_LDO_predictions.csv")
    )

    with open(os.path.join(results_dir, "sciplex_regression_LDO_feature_importance_on_pre.pkl"), "wb") as file:
        pickle.dump(fi_pre, file)
    with open(os.path.join(results_dir, "sciplex_regression_LDO_feature_importance_on_post.pkl"), "wb") as file:
        pickle.dump(fi_post, file)
    with open(os.path.join(results_dir, "sciplex_regression_LDO_feature_importance_on_lfc.pkl"), "wb") as file:
        pickle.dump(fi_lfc, file)


def predictions_sciplex_lto() -> None:
    """
    Leave-one-cell-line-out CV for the "unseen cell line, seen drug" scenario.

    SciPlex has only 3 cell lines, so this is a 3-fold CV: each fold holds out
    one full cell line and trains on the other two. Drugs are not removed from
    training, so the drug axis remains "seen".
    """

    def _unseen_cellline_seen_drug_cv(
        df_with_y: pd.DataFrame,
        selected_features: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
        df_with_y = df_with_y.reset_index().dropna(axis=0)

        cell_lines = sorted(df_with_y["tissue"].drop_duplicates().tolist())
        if len(cell_lines) < 2:
            return (
                pd.DataFrame(columns=["accuracy", "test_condition"]),
                pd.DataFrame(columns=["tissue", "cell_line", "condition", "pred", "true", "test_condition"]),
                {},
            )

        param_grid = {
            "elasticnet__alpha": [0.1, 1.0, 10.0],
            "elasticnet__l1_ratio": [0.0, 0.1, 0.5, 1.0],
        }

        outer_scores: list[pd.DataFrame] = []
        outcomes: list[pd.DataFrame] = []
        feature_importance_dict: dict[str, pd.Series] = {}

        for fold_idx, test_tissue in enumerate(cell_lines):
            test_mask = df_with_y["tissue"] == test_tissue
            if test_mask.sum() == 0:
                continue

            # Important: include *all* drugs in training (so drugs are "seen").
            train_mask = df_with_y["tissue"] != test_tissue
            if train_mask.sum() < 2:
                continue

            X_train_df = df_with_y.loc[train_mask].copy()
            X_test_df = df_with_y.loc[test_mask].copy()

            y_train = X_train_df["y"].astype(float)
            y_test = X_test_df["y"].astype(float)

            X_train_df = X_train_df.drop(columns=["y"])
            X_test_df = X_test_df.drop(columns=["y"])

            identifiers = X_test_df[["tissue", "cell_line", "condition"]].reset_index(drop=True)

            X_train_df = X_train_df[selected_features]
            X_test_df = X_test_df[selected_features]

            X_train_mat = X_train_df.drop(columns=["tissue", "cell_line", "condition"]).astype(float)
            X_test_mat = X_test_df.drop(columns=["tissue", "cell_line", "condition"]).astype(float)

            if len(X_train_mat) < 2 or len(X_test_mat) < 1:
                continue

            cv_folds = min(5, len(X_train_mat))
            if cv_folds < 2:
                continue

            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("elasticnet", ElasticNet(max_iter=1000)),
                ]
            )
            grid_search = GridSearchCV(pipeline, param_grid, cv=cv_folds)
            grid_search.fit(X_train_mat, y_train)

            best_model = grid_search.best_estimator_.named_steps["elasticnet"]
            y_pred = grid_search.predict(X_test_mat)

            outer_acc = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            fold_name = f"fold={fold_idx};tissue={test_tissue}"

            feature_importance_dict[fold_name] = pd.Series(
                index=X_train_mat.columns, data=best_model.coef_, name=fold_name
            )

            split_outcomes = pd.concat(
                [
                    identifiers,
                    pd.DataFrame({"pred": y_pred, "true": y_test.reset_index(drop=True)}),
                ],
                axis=1,
            )

            split_outcomes["test_condition"] = (
                "tissue="
                + split_outcomes["tissue"].astype(str)
                + " OR condition="
                + split_outcomes["condition"].astype(str)
            )
            split_outcomes["split"] = fold_idx
            outcomes.append(split_outcomes)

            outer_scores.append(pd.DataFrame({"accuracy": [outer_acc], "test_condition": [fold_name], "split": [fold_idx]}))

        outer_scores_df = pd.concat(outer_scores, ignore_index=True) if outer_scores else pd.DataFrame(
            columns=["accuracy", "test_condition", "split"]
        )
        outcomes_df = (
            pd.concat(outcomes, ignore_index=True)
            if outcomes
            else pd.DataFrame(columns=["tissue", "cell_line", "condition", "pred", "true", "test_condition", "split"])
        )
        return outer_scores_df, outcomes_df, feature_importance_dict

    mean_pre, mean_post, mean_lfc = _get_sciplex_data_with_sensitivity()
    mean_pre_with_y = add_y_and_normalize(mean_pre, "sens", normalize=False, keep=["condition", "tissue", "cell_line"])
    mean_post_with_y = add_y_and_normalize(mean_post, "sens", normalize=False, keep=["condition", "tissue", "cell_line"])
    mean_lfc_with_y = add_y_and_normalize(mean_lfc, "sens", normalize=False, keep=["condition", "tissue", "cell_line"])

    features = feature_selection(mean_pre_with_y.drop(columns=["y"]), 1000)

    pre_scores, pre_predictions, fi_pre = _unseen_cellline_seen_drug_cv(mean_pre_with_y, features)
    post_scores, post_predictions, fi_post = _unseen_cellline_seen_drug_cv(mean_post_with_y, features)
    lfc_scores, lfc_predictions, fi_lfc = _unseen_cellline_seen_drug_cv(mean_lfc_with_y, features)

    pre_scores = pre_scores.assign(model="pre-treatment").assign(feature_selection="top1000 highest variance")
    post_scores = post_scores.assign(model="post-treatment").assign(feature_selection="top1000 highest variance")
    lfc_scores = lfc_scores.assign(model="LFC").assign(feature_selection="top1000 highest variance")
    pre_predictions = pre_predictions.assign(model="pre-treatment").assign(feature_selection="top1000 highest variance")
    post_predictions = post_predictions.assign(model="post-treatment").assign(feature_selection="top1000 highest variance")
    lfc_predictions = lfc_predictions.assign(model="LFC").assign(feature_selection="top1000 highest variance")

    pd.concat([pre_scores, post_scores, lfc_scores]).to_csv(
        os.path.join(results_dir, "sciplex_regression_LTO.csv")
    )
    pd.concat([pre_predictions, post_predictions, lfc_predictions]).to_csv(
        os.path.join(results_dir, "sciplex_regression_LTO_predictions.csv")
    )

    with open(os.path.join(results_dir, "sciplex_regression_LTO_feature_importance_on_pre.pkl"), "wb") as file:
        pickle.dump(fi_pre, file)
    with open(os.path.join(results_dir, "sciplex_regression_LTO_feature_importance_on_post.pkl"), "wb") as file:
        pickle.dump(fi_post, file)
    with open(os.path.join(results_dir, "sciplex_regression_LTO_feature_importance_on_lfc.pkl"), "wb") as file:
        pickle.dump(fi_lfc, file)


def predictions_sciplex_loo() -> None:
    """
    Leave-one-cell-line-out CV for the "unseen drug, unseen cell line" scenario.

    SciPlex has only 3 cell lines, so this is a 3-fold CV over cell lines. In
    each fold, we also hold out one matched drug subset and remove that subset
    from training, so both the cell-line and drug axes are unseen.
    """

    def _unseen_cellline_drug_cv(
        df_with_y: pd.DataFrame,
        selected_features: list[str],
        random_state: int = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
        df_with_y = df_with_y.reset_index().dropna(axis=0)

        cell_lines = sorted(df_with_y["tissue"].drop_duplicates().tolist())
        if len(cell_lines) < 2:
            # Not enough to define unseen cell line
            return (
                pd.DataFrame(columns=["accuracy", "test_condition"]),
                pd.DataFrame(columns=["tissue", "cell_line", "condition", "pred", "true", "test_condition"]),
                {},
            )

        all_drugs = sorted(df_with_y["condition"].drop_duplicates().tolist())
        rng = np.random.default_rng(random_state)
        rng.shuffle(all_drugs)
        drug_folds = np.array_split(np.array(all_drugs, dtype=object), len(cell_lines))

        param_grid = {
            "elasticnet__alpha": [0.1, 1.0, 10.0],
            "elasticnet__l1_ratio": [0.0, 0.1, 0.5, 1.0],
        }

        outer_scores: list[pd.DataFrame] = []
        outcomes: list[pd.DataFrame] = []
        feature_importance_dict: dict[str, pd.Series] = {}

        for fold_idx, test_tissue in enumerate(cell_lines):
            test_drugs = drug_folds[fold_idx]
            if len(test_drugs) == 0:
                continue

            test_mask = (df_with_y["tissue"] == test_tissue) & df_with_y["condition"].isin(test_drugs.tolist())
            if test_mask.sum() == 0:
                continue

            train_mask = (df_with_y["tissue"] != test_tissue) & (~df_with_y["condition"].isin(test_drugs.tolist()))
            if train_mask.sum() < 2:
                continue

            X_train_df = df_with_y.loc[train_mask].copy()
            X_test_df = df_with_y.loc[test_mask].copy()

            y_train = X_train_df["y"].astype(float)
            y_test = X_test_df["y"].astype(float)

            X_train_df = X_train_df.drop(columns=["y"])
            X_test_df = X_test_df.drop(columns=["y"])

            identifiers = X_test_df[["tissue", "cell_line", "condition"]].reset_index(drop=True)

            X_train_df = X_train_df[selected_features]
            X_test_df = X_test_df[selected_features]

            X_train_mat = X_train_df.drop(columns=["tissue", "cell_line", "condition"]).astype(float)
            X_test_mat = X_test_df.drop(columns=["tissue", "cell_line", "condition"]).astype(float)

            cv_folds = min(5, len(X_train_mat))
            if cv_folds < 2:
                continue

            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("elasticnet", ElasticNet(max_iter=1000)),
                ]
            )
            grid_search = GridSearchCV(pipeline, param_grid, cv=cv_folds)
            grid_search.fit(X_train_mat, y_train)

            best_model = grid_search.best_estimator_.named_steps["elasticnet"]
            y_pred = grid_search.predict(X_test_mat)

            test_fold_name = f"fold={fold_idx};tissue={test_tissue}"
            feature_importance = pd.Series(index=X_train_mat.columns, data=best_model.coef_, name=test_fold_name)
            feature_importance_dict[test_fold_name] = feature_importance

            split_outcomes = pd.concat(
                [
                    identifiers,
                    pd.DataFrame({"pred": y_pred, "true": y_test.reset_index(drop=True)}),
                ],
                axis=1,
            )

            # Important: set test_condition per row so the plotting notebook can parse it.
            split_outcomes["test_condition"] = (
                "tissue=" + split_outcomes["tissue"].astype(str) + " OR condition=" + split_outcomes["condition"].astype(str)
            )
            split_outcomes["split"] = fold_idx
            outcomes.append(split_outcomes)

            # Create per-drug accuracy entries (so loo_scores can be parsed consistently).
            tmp = split_outcomes.copy()
            for cond in tmp["condition"].unique().tolist():
                sub = tmp[tmp["condition"] == cond]
                if len(sub) < 2:
                    acc_val = 0.0
                else:
                    acc_val = float(np.sqrt(mean_squared_error(sub["true"], sub["pred"])))
                outer_scores.append(
                    pd.DataFrame(
                        {
                            "accuracy": [acc_val],
                            "test_condition": [f"tissue={test_tissue} OR condition={cond}"],
                            "split": [fold_idx],
                        }
                    )
                )

        if outer_scores:
            outer_scores_df = pd.concat(outer_scores, ignore_index=True)
        else:
            outer_scores_df = pd.DataFrame(columns=["accuracy", "test_condition", "split"])

        if outcomes:
            outcomes_df = pd.concat(outcomes, ignore_index=True)
        else:
            outcomes_df = pd.DataFrame(
                columns=["tissue", "cell_line", "condition", "pred", "true", "test_condition", "split"]
            )

        return outer_scores_df, outcomes_df, feature_importance_dict

    mean_pre, mean_post, mean_lfc = _get_sciplex_data_with_sensitivity()
    mean_pre_with_y = add_y_and_normalize(mean_pre, "sens", normalize=False, keep=["condition", "tissue", "cell_line"])
    mean_post_with_y = add_y_and_normalize(mean_post, "sens", normalize=False, keep=["condition", "tissue", "cell_line"])
    mean_lfc_with_y = add_y_and_normalize(mean_lfc, "sens", normalize=False, keep=["condition", "tissue", "cell_line"])

    features = feature_selection(mean_pre_with_y.drop(columns=["y"]), 1000)

    pre_scores, pre_predictions, fi_pre = _unseen_cellline_drug_cv(mean_pre_with_y, features)
    post_scores, post_predictions, fi_post = _unseen_cellline_drug_cv(mean_post_with_y, features)
    lfc_scores, lfc_predictions, fi_lfc = _unseen_cellline_drug_cv(mean_lfc_with_y, features)

    pre_scores = pre_scores.assign(model="pre-treatment").assign(feature_selection="top1000 highest variance")
    post_scores = post_scores.assign(model="post-treatment").assign(feature_selection="top1000 highest variance")
    lfc_scores = lfc_scores.assign(model="LFC").assign(feature_selection="top1000 highest variance")

    pre_predictions = pre_predictions.assign(model="pre-treatment").assign(feature_selection="top1000 highest variance")
    post_predictions = post_predictions.assign(model="post-treatment").assign(feature_selection="top1000 highest variance")
    lfc_predictions = lfc_predictions.assign(model="LFC").assign(feature_selection="top1000 highest variance")

    pd.concat([pre_scores, post_scores, lfc_scores]).to_csv(os.path.join(results_dir, "sciplex_regression_LOO.csv"))
    pd.concat([pre_predictions, post_predictions, lfc_predictions]).to_csv(
        os.path.join(results_dir, "sciplex_regression_LOO_predictions.csv")
    )

    with open(os.path.join(results_dir, "sciplex_regression_LOO_feature_importance_on_pre.pkl"), "wb") as file:
        pickle.dump(fi_pre, file)
    with open(os.path.join(results_dir, "sciplex_regression_LOO_feature_importance_on_post.pkl"), "wb") as file:
        pickle.dump(fi_post, file)
    with open(os.path.join(results_dir, "sciplex_regression_LOO_feature_importance_on_lfc.pkl"), "wb") as file:
        pickle.dump(fi_lfc, file)


if __name__ == "__main__":
    log_script_start(__file__, logger)
    try:
        logger.info(f"Project root: {home_dir}")
        logger.info(f"Results directory: {results_dir}")
        logger.info(f"Resources directory: {resources_dir}")

        logger.info("Starting SciPlex observed pseudobulk prediction pipeline...")
        # predictions_sciplex_per_treatment()
        # predictions_sciplex_per_tissue()
        # predictions_sciplex()
        predictions_sciplex_seen_seen()
        predictions_sciplex_ldo()
        predictions_sciplex_lto()
        predictions_sciplex_loo()
        logger.info("SciPlex observed pseudobulk prediction pipeline completed successfully")
    except Exception as error:
        logger.error(f"Error in SciPlex observed pseudobulk prediction: {str(error)}")
        raise
    finally:
        log_script_end(__file__, logger)
