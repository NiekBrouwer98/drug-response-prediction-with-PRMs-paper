import numpy as np
import pandas as pd
import os
import pickle
import sys
from pathlib import Path

from sklearn.linear_model import ElasticNet, LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

# Add project root to path for imports (must be before importing prediction_utils)
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config, setup_project
from prediction_utils import *
from utils import setup_logging_for_script, log_script_start, log_script_end, ensure_directories_exist

# Setup project and logging
setup_project()
logger = setup_logging_for_script(__file__)

# Set project directories using configuration
path = os.getcwd()
home_dir = str(config.PROJECT_ROOT)
data_dir = str(config.DATA_DIR)
resources_dir = str(config.RESOURCES_DIR)
results_dir = str(config.RESULTS_04_DIR)
figures_dir = str(config.FIGURES_04_DIR)

# Ensure directories exist
ensure_directories_exist(results_dir, figures_dir, data_dir, resources_dir)


def one_fold_elasticnet(X_train, X_test, selected_features):

    X_train = X_train.reset_index().dropna(axis=0)
    X_test = X_test.reset_index().dropna(axis=0)

    y_train = X_train['y']
    X_train = X_train.drop(columns=['y'])
    y_test = X_test['y']
    X_test = X_test.drop(columns=['y'])

    # Split the data into train and test sets
    identifiers = X_test[['tissue','cell_line', 'condition']].copy()
    X_train = X_train.drop(columns=['tissue','cell_line', 'condition']).astype(float)
    X_test = X_test.drop(columns=['tissue','cell_line', 'condition']).astype(float)
    y_train = y_train.astype(float)
    y_test = y_test.astype(float)

    X_train = X_train[list(set(selected_features).intersection(set(X_train.columns)))]
    X_test = X_test[list(set(selected_features).intersection(set(X_test.columns)))]	

    assert X_test.shape[1] == X_train.shape[1], "Mismatch in number of features between train and test sets."

    param_grid = {
        'elasticnet__alpha': [0.1, 1.0, 10.0],
        'elasticnet__l1_ratio': [0.0, 0.1, 0.5, 1.0]
    }

    # Train model
    # Regression pipeline with GridSearchCV
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('elasticnet', ElasticNet(max_iter=1000))
    ])
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Extract model and predictions
    best_model = grid_search.best_estimator_.named_steps['elasticnet']
    feature_importance = pd.Series(index=X_train.columns, data=best_model.coef_)
    y_pred = grid_search.predict(X_test)  # Predict using entire pipeline
    score = mean_squared_error(y_test, y_pred)

    outcomes = pd.DataFrame({'pred': y_pred, 'true': y_test})
    outcomes = pd.concat([identifiers.reset_index(drop=True), outcomes], axis=1)

    print(f"\nFinal Model MSE: {score:.4f}")

    return pd.DataFrame({'accuracy':[score]}), outcomes, feature_importance

def cross_validation_elasticnet(df,selected_features, n_outer_splits=5, metric='roc_auc',gridsearch=True, classification_bool=None):
    metrics = {'roc_auc': roc_auc_score, 'f1': f1_score, 'accuracy': accuracy_score, 'precision': precision_score, 'recall': recall_score, 'RMSE':mean_squared_error}
    feature_importances = []

    if n_outer_splits == 1:
        raise ValueError("n_outer_splits must be greater than 1 for cross-validation.")

    df = df.reset_index()
    
    df = df.dropna(axis=0)
    y = df['y']
    X = df.drop(columns=['y'])

    if classification_bool is None:
        if y.nunique() < 3:
            classification_bool = True
            print("Classification task")
        else:
            classification_bool = False
            print("Regression task")

    if classification_bool:
        outer_cv = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=42)
    else:
        outer_cv = KFold(n_splits=n_outer_splits, shuffle=True, random_state=42)

    outer_scores = []
    outcomes = []

    for split_num, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        identifiers = X_test[['tissue','cell_line', 'condition']].copy()
        X_train = X_train.drop(columns=['tissue','cell_line', 'condition']).astype(float).reset_index(drop=True)
        X_test = X_test.drop(columns=['tissue','cell_line', 'condition']).astype(float).reset_index(drop=True)
        y_train = y_train.astype(float).reset_index(drop=True)
        y_test = y_test.astype(float).reset_index(drop=True)

        if (y_train.nunique() == 1) & (y_test.nunique() == 1):
            outer_acc = 1
        else:
            selected_features = list(set(selected_features).intersection(set(X_train.columns)))

            X_train = X_train[selected_features]
            X_test = X_test[selected_features]


            param_grid = {
                'elasticnet__alpha': [0.1, 1.0, 10.0],
                'elasticnet__l1_ratio': [0.0, 0.1, 0.5, 1.0]
            }

            if classification_bool:
                grid_search = Pipeline([
                        ('scaler', StandardScaler()),
                        ('logreg', LogisticRegressionCV(
                            Cs=10,           # number of Cs or list of regularization strengths
                            cv=5,            # number of CV folds
                            penalty='elasticnet',  # or 'l1', 'l2'
                            solver='saga',   # required for elasticnet or l1
                            l1_ratios=[0.0, 0.5, 1.0],  # only used for elasticnet
                            max_iter=1000,
                            scoring='accuracy',  # or 'roc_auc', 'f1', etc.
                            refit=True
                        ))
                    ])
                grid_search.fit(X_train, y_train)

                feature_importance = pd.Series(index=X_train.columns, data=grid_search.coef_[0])
                feature_importance_df = feature_importance.reset_index()
                feature_importance_df.columns = ['feature', 'importance']
                feature_importance_df['split'] = split_num
                feature_importances.append(feature_importance_df)
                y_pred = grid_search.predict(X_test)

                for m, fct in metrics.items():
                    if m == metric:
                        outer_acc = fct(y_test, y_pred)

                split_outcome = pd.concat([identifiers.reset_index(drop=True), pd.DataFrame({'pred': y_pred, 'true': y_test}).reset_index(drop=True)], axis=1)
                outcomes.append(split_outcome.assign(split=split_num))

            else:
                if gridsearch==False:
                    # Pipeline with scaling
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('elasticnet', ElasticNet(alpha=0.5, l1_ratio=0.7, max_iter=1000))
                    ])

                    # Fit the model
                    pipeline.fit(X_train, y_train)
                    fitted_model = pipeline.named_steps['elasticnet']

                    feature_importance = pd.Series(index=X_train.columns, data=fitted_model.coef_)
                    feature_importance_df = feature_importance.reset_index()
                    feature_importance_df.columns = ['feature', 'importance']
                    feature_importance_df['split'] = split_num
                    feature_importances.append(feature_importance_df)

                    y_pred = pipeline.predict(X_test)
                else:
                    pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('elasticnet', ElasticNet(max_iter=1000))
                    ])
                    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
                    grid_search.fit(X_train, y_train)

                    best_model = grid_search.best_estimator_.named_steps['elasticnet']
                    feature_importance = pd.Series(index=X_train.columns, data=best_model.coef_)
                    feature_importance_df = feature_importance.reset_index()
                    feature_importance_df.columns = ['feature', 'importance']
                    feature_importance_df['split'] = split_num
                    feature_importances.append(feature_importance_df)

                    y_pred = best_model.predict(X_test)

                outer_acc = mean_squared_error(y_test, y_pred)
                split_outcome = pd.concat([identifiers.reset_index(drop=True), pd.DataFrame({'pred': y_pred, 'true': y_test})], axis=1)
                outcomes.append(split_outcome.assign(split=split_num))

        outer_scores.append(pd.DataFrame({'acc':[outer_acc], 'split':[split_num]}))

        print(f"Outer Fold {split_num} {metric}: {outer_acc:.4f}")

    avg_auc = np.mean(outer_scores)
    std_auc = np.std(outer_scores)

    outcomes = pd.concat(outcomes)
    outer_scores = pd.concat(outer_scores)

    print(f"\nNested CV Results: Mean {metric} = {avg_auc:.4f}, Std = {std_auc:.4f}")

    feature_importances = pd.concat(feature_importances, ignore_index=True)
    return outer_scores, outcomes, feature_importances


def out_of_distribution_elasticnet(df, selected_features, condition):
    if not isinstance(condition, list) or len(condition) < 1:
        raise ValueError("Condition must be a list with at least one column name.")

    df = df.reset_index()

    # Get all unique combinations of values across condition columns
    unique_combinations = df[condition].drop_duplicates().values.tolist()

    outer_scores = []
    outcomes = []
    feature_importance_dict = {}

    for split_num, combo in enumerate(unique_combinations, start=1):

        # Construct test mask with OR logic
        test_mask = pd.Series(False, index=df.index)
        label_parts = []
        for col, val in zip(condition, combo):
            test_mask |= (df[col] == val)
            label_parts.append(f"{col}={val}")
        test_condition_name = " OR ".join(label_parts)

        if test_mask.sum() <= 20:
            print(f"Skipping {test_condition_name}: only {test_mask.sum()} samples.")
            continue

        # Split data
        X_train = df[~test_mask]
        X_test = df[test_mask]
        y_train = X_train['y']
        y_test = X_test['y']
        X_train = X_train.drop(columns=['y'])
        X_test = X_test.drop(columns=['y'])

        # Store identifiers for outcomes
        identifiers = X_test[['tissue', 'cell_line', 'condition']].copy()

        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        X_train = X_train.drop(columns=['tissue', 'cell_line', 'condition'])
        X_test = X_test.drop(columns=['tissue', 'cell_line', 'condition'])

        # Handle degenerate case
        if (y_train.nunique() == 1) & (y_test.nunique() == 1):
            outer_scores.append(pd.DataFrame({'accuracy': [1], 'test_condition': [test_condition_name], 'split': [split_num]}))
            continue

        classification_bool = y_test.nunique() < 3
        print("Classification task" if classification_bool else "Regression task")


        param_grid = {
            'elasticnet__alpha': [0.1, 1.0, 10.0],
            'elasticnet__l1_ratio': [0.0, 0.1, 0.5, 1.0]
        }  

        # Train model
        if classification_bool:
            grid_search = Pipeline([
                ('scaler'[str], StandardScaler()),
                ('logreg', LogisticRegressionCV(
                    Cs=10,           # number of Cs or list of regularization strengths
                    cv=5,            # number of CV folds
                    penalty='elasticnet',  # or 'l1', 'l2'
                    solver='saga',   # required for elasticnet or l1
                    l1_ratios=[0.0, 0.5, 1.0],  # only used for elasticnet
                    max_iter=1000,
                    scoring='accuracy',  # or 'roc_auc', 'f1', etc.
                    refit=True
                ))
            ])
            grid_search.fit(X_train, y_train)
            y_pred = grid_search.predict(X_test)
            outer_acc = f1_score(y_test, y_pred)
            feature_importance = pd.Series(index=X_train.columns, data=grid_search.coef_[0], name=test_condition_name)
        else:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('elasticnet', ElasticNet(max_iter=1000))
            ])

            grid_search = GridSearchCV(pipeline, param_grid, cv=5)
            grid_search.fit(X_train, y_train)

            y_pred = grid_search.predict(X_test)

            best_model = grid_search.best_estimator_.named_steps['elasticnet']
            outer_acc = np.sqrt(mean_squared_error(y_test, y_pred))
            feature_importance = pd.Series(index=X_train.columns, data=best_model.coef_, name=test_condition_name)

        # Store results
        outer_scores.append(pd.DataFrame({'accuracy': [outer_acc], 'test_condition': [test_condition_name], 'split': [split_num]}))
        split_outcomes = pd.concat([identifiers, pd.DataFrame({'pred': y_pred, 'true': y_test})],axis=1)
        outcomes.append(split_outcomes.assign(test_condition=test_condition_name).assign(split=split_num))
        feature_importance_dict[test_condition_name] = feature_importance

        print(f"Outer Fold accuracy: {outer_acc:.4f} for {test_condition_name}")

    if outer_scores:
        outer_scores = pd.concat(outer_scores, ignore_index=True)
    else:
        outer_scores = pd.DataFrame(columns=['accuracy', 'test_condition', 'split'])

    if outcomes:
        outcomes = pd.concat(outcomes, ignore_index=True)
    else:
        outcomes = pd.DataFrame(columns=['tissue', 'cell_line', 'condition', 'pred', 'true', 'test_condition', 'split'])

    return outer_scores, outcomes, feature_importance_dict


def within_distribution_elasticnet(df, selected_features, condition1, condition2):
    """
    Perform seen/seen evaluation where test instances are specific combinations of two condition variables,
    but these individual values exist in the training data in other combinations.
    
    Args:
        df: DataFrame containing the data
        selected_features: List of features to use for prediction
        condition1: First condition column name
        condition2: Second condition column name
    """
    if not isinstance(condition1, str) or not isinstance(condition2, str):
        raise ValueError("condition1 and condition2 must be column names (strings)")
    
    df = df.reset_index()

    # Get all unique combinations of the two conditions
    unique_combinations = df[[condition1, condition2]].drop_duplicates().values.tolist()

    outer_scores = []
    outcomes = []
    feature_importance_dict = {}

    for split_num, combo in enumerate(unique_combinations, start=1):
        val1, val2 = combo
        test_condition_name = f"{condition1}={val1} AND {condition2}={val2}"
        print(f"Evaluating {test_condition_name}")

        # Create test mask for this specific combination
        test_mask = (df[condition1] == val1) & (df[condition2] == val2)

        X_train = df[~test_mask]
        X_test = df[test_mask]
        y_train = X_train['y']
        y_test = X_test['y']
        X_train = X_train.drop(columns=['y'])
        X_test = X_test.drop(columns=['y'])

        # Store identifiers for outcomes
        identifiers = X_test[['tissue', 'cell_line', 'condition']].copy()

        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        X_train = X_train.drop(columns=['tissue', 'cell_line', 'condition'])
        X_test = X_test.drop(columns=['tissue', 'cell_line', 'condition'])
        
        # Handle degenerate case
        if (y_train.nunique() == 1) & (y_test.nunique() == 1):
            outer_scores.append(pd.DataFrame({'accuracy': [1], 'test_condition': [test_condition_name], 'split': [split_num]}))
            continue

        param_grid = {
            'elasticnet__alpha': [0.1, 1.0, 10.0],
            'elasticnet__l1_ratio': [0.0, 0.1, 0.5, 1.0]
        }

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('elasticnet', ElasticNet(max_iter=1000))
        ])

        grid_search = GridSearchCV(pipeline, param_grid, cv=5)
        grid_search.fit(X_train, y_train)

        y_pred = grid_search.predict(X_test)

        best_model = grid_search.best_estimator_.named_steps['elasticnet']
        outer_acc = np.sqrt(mean_squared_error(y_test, y_pred))
        feature_importance = pd.Series(index=X_train.columns, data=best_model.coef_, name=test_condition_name)

        # Store results
        outer_scores.append(pd.DataFrame({'accuracy': [outer_acc], 'test_condition': [test_condition_name], 'split': [split_num]}))
        split_outcomes = pd.concat([identifiers, pd.DataFrame({'pred': y_pred, 'true': y_test})],axis=1)
        outcomes.append(split_outcomes.assign(test_condition=test_condition_name).assign(split=split_num))
        feature_importance_dict[test_condition_name] = feature_importance

        print(f"Outer Fold accuracy: {outer_acc:.4f} for {test_condition_name}")

    if outer_scores:
        outer_scores = pd.concat(outer_scores, ignore_index=True)
    else:
        outer_scores = pd.DataFrame(columns=['accuracy', 'test_condition', 'split'])

    if outcomes:
        outcomes = pd.concat(outcomes, ignore_index=True)
    else:
        outcomes = pd.DataFrame(columns=['pred', 'true', 'test_condition', 'split'])

    return outer_scores, outcomes, feature_importance_dict


def prediction_pipeline(df_with_y, df_for_feature_selection,label,result_label='',feature_subset=[], n_features=1000, metric='roc_auc',n_outer_splits=5, gridsearch=True):

    features = feature_selection(df_for_feature_selection, n_features=n_features,feature_subset=feature_subset)
    print(f'using {len(features)} features')

    scores, predictions, feature_importance = cross_validation_elasticnet(df_with_y, features, n_outer_splits=n_outer_splits, metric=metric, gridsearch=gridsearch)

    if result_label == '':
        scores_df = scores.assign(model=label).assign(feature_selection=f'top{n_features} HVG')
        predictions = predictions.assign(model=label).assign(feature_selection=f'top{n_features} HVG')
    else:
        scores_df = scores.assign(model=label).assign(feature_selection=result_label)
        predictions = predictions.assign(model=label).assign(feature_selection=result_label)

    return scores_df, predictions, feature_importance


def predictions_McFarland(n_features=1000, feature_subset=[]):
    print(f'Performing predictions with {n_features} features')
    mean_observed_pre_treatment, mean_observed_post_treatment, mean_observed_LFC = get_McFarland_mean_data()
    cellline_sensitivity_info = get_McFarland_sensitivityinfo_for_profile_merge()

    mean_observed_pre_treatment = pd.merge(mean_observed_pre_treatment, cellline_sensitivity_info, left_on=['cell_line', 'condition'], right_on=['cell_line', 'target'], how='left')
    mean_observed_post_treatment = pd.merge(mean_observed_post_treatment, cellline_sensitivity_info, left_on=['cell_line', 'condition'], right_on=['cell_line', 'target'], how='left')
    mean_observed_LFC = pd.merge(mean_observed_LFC, cellline_sensitivity_info, left_on=['cell_line', 'condition'], right_on=['cell_line', 'target'], how='left')

    mean_observed_pre_treatment = filter_on_coefficient_of_variation(mean_observed_pre_treatment, groupby=['condition'])
    mean_observed_post_treatment = filter_on_coefficient_of_variation(mean_observed_post_treatment, groupby=['condition'])
    mean_observed_LFC = filter_on_coefficient_of_variation(mean_observed_LFC, groupby=['condition'])

    feature_to_keep = feature_subset + ['condition', 'tissue', 'cell_line']
    mean_observed_pre_treatment_with_y = add_y_and_normalize(mean_observed_pre_treatment, 'sens', normalize=False, keep=feature_to_keep)
    mean_observed_post_treatment_with_y = add_y_and_normalize(mean_observed_post_treatment, 'sens', normalize=False, keep=feature_to_keep)
    mean_observed_LFC_with_y= add_y_and_normalize(mean_observed_LFC, 'sens',normalize=False, keep=feature_to_keep)

    pre_outcomes, pre_predictions, pre_feature_importance = prediction_pipeline(mean_observed_pre_treatment_with_y, mean_observed_pre_treatment, 'pre_treatment', n_features=n_features,feature_subset=feature_subset)
    post_outcomes, post_predictions, post_feature_importance = prediction_pipeline(mean_observed_post_treatment_with_y, mean_observed_pre_treatment, 'post_treatment',n_features=n_features,feature_subset=feature_subset)
    LFC_outcomes, LFC_predictions, LFC_feature_importance = prediction_pipeline(mean_observed_LFC_with_y, mean_observed_pre_treatment, 'LFC', n_features=n_features,feature_subset=feature_subset)

    results = pd.concat([pre_outcomes, post_outcomes, LFC_outcomes])
    predictions = pd.concat([pre_predictions, post_predictions, LFC_predictions])
    feature_importance = pd.concat([pre_feature_importance.assign(model='Pre'), post_feature_importance.assign(model='Post'), LFC_feature_importance.assign(model='Log(fold change)')])
    
    if feature_subset==[]:
        results.to_csv(os.path.join(results_dir, f'mcfarland_regression_{n_features}_features_results.csv'))
        predictions.to_csv(os.path.join(results_dir, f'mcfarland_regression_{n_features}_features_predictions.csv'))
        feature_importance.to_csv(os.path.join(results_dir, f'mcfarland_regression_{n_features}_features_feature_importance.csv'))

    else:
        results.to_csv(os.path.join(results_dir, 'mcfarland_regression_manual_features_results.csv'))
        predictions.to_csv(os.path.join(results_dir, 'mcfarland_regression_manual_features_predictions.csv'))
        feature_importance.to_csv(os.path.join(results_dir, 'mcfarland_regression_manual_features_feature_importance.csv'))


def predictions_McFarland_per_treatment(n_features=1000, outersplits=5):
    mean_observed_pre_treatment, mean_observed_post_treatment, mean_observed_LFC = get_McFarland_mean_data()
    cellline_sensitivity_info = get_McFarland_sensitivityinfo_for_profile_merge()

    mean_observed_pre_treatment = pd.merge(mean_observed_pre_treatment, cellline_sensitivity_info, left_on=['cell_line', 'condition'], right_on=['cell_line', 'target'], how='left')
    mean_observed_post_treatment = pd.merge(mean_observed_post_treatment, cellline_sensitivity_info, left_on=['cell_line', 'condition'], right_on=['cell_line', 'target'], how='left')
    mean_observed_LFC = pd.merge(mean_observed_LFC, cellline_sensitivity_info, left_on=['cell_line', 'condition'], right_on=['cell_line', 'target'], how='left')

    results = pd.DataFrame()
    predictions_df = pd.DataFrame()
    feature_importance_pre = {}
    feature_importance_post = {}
    feature_importance_lfc = {}

    features = feature_selection(mean_observed_pre_treatment, n_features)

    mean_observed_pre_treatment = filter_on_coefficient_of_variation(mean_observed_pre_treatment, groupby=['condition'])
    mean_observed_post_treatment = filter_on_coefficient_of_variation(mean_observed_post_treatment, groupby=['condition'])
    mean_observed_LFC = filter_on_coefficient_of_variation(mean_observed_LFC, groupby=['condition'])

    conditions = mean_observed_pre_treatment['condition'].unique().tolist()

    for i, c in enumerate(conditions):
        pre_subset = mean_observed_pre_treatment[mean_observed_pre_treatment['condition'] == c]
        post_subset = mean_observed_post_treatment[mean_observed_post_treatment['condition'] == c]
        lfc_subset = mean_observed_LFC[mean_observed_LFC['condition'] == c]

        pre_subset = add_y_and_normalize(pre_subset, 'sens', normalize=False, keep=['condition', 'tissue', 'cell_line'])
        post_subset = add_y_and_normalize(post_subset, 'sens',  normalize=False, keep=['condition', 'tissue', 'cell_line'])
        lfc_subset = add_y_and_normalize(lfc_subset, 'sens', normalize=False, keep=['condition', 'tissue', 'cell_line'])

        if pre_subset.shape[0] < 20:
            print(f'Too few samples for {c} ({pre_subset.shape[0]} samples)')
            continue
        else:
            print(f'Predicting for {c}')
            tissues = mean_observed_pre_treatment[mean_observed_pre_treatment['condition'] == c]['tissue'].unique().tolist()
            pre_outcomes, pre_predictions, pre_feature_importance = prediction_pipeline(pre_subset, mean_observed_pre_treatment, 'pre_treatment', feature_subset=features, n_outer_splits=outersplits)
            post_outcomes, post_predictions, post_feature_importance = prediction_pipeline(post_subset, mean_observed_pre_treatment, 'post_treatment',feature_subset=features,n_outer_splits=outersplits)
            LFC_outcomes, LFC_predictions, LFC_feature_importance = prediction_pipeline(lfc_subset, mean_observed_pre_treatment, 'LFC', feature_subset=features, n_outer_splits=outersplits)

            pre_outcomes = pre_outcomes.assign(condition=c).assign(n_genes = n_features)
            post_outcomes = post_outcomes.assign(condition=c).assign(n_genes = n_features)
            LFC_outcomes = LFC_outcomes.assign(condition=c).assign(n_genes = n_features)
            results = pd.concat([results, pre_outcomes, post_outcomes, LFC_outcomes])

            pre_predictions = pre_predictions.assign(condition=c)
            post_predictions = post_predictions.assign(condition=c)
            LFC_predictions = LFC_predictions.assign(condition=c)
            predictions_df = pd.concat([predictions_df, pre_predictions, post_predictions, LFC_predictions])

            feature_importance_pre[c] = pre_feature_importance
            feature_importance_post[c] = post_feature_importance
            feature_importance_lfc[c] = LFC_feature_importance

    results.to_csv(os.path.join(results_dir,'treatmentmodels_regression_results.csv'))
    predictions_df.to_csv(os.path.join(results_dir,'treatmentmodels_regression_predictions.csv'))

    with open(os.path.join(results_dir,'treatmentmodels_regression_feature_importance_on_pre.pkl'), "wb") as f:
        pickle.dump(feature_importance_pre, f)
    with open(os.path.join(results_dir,'treatmentmodels_regression_feature_importance_on_post.pkl'), "wb") as f:
        pickle.dump(feature_importance_post, f)
    with open(os.path.join(results_dir,'treatmentmodels_regression_feature_importance_on_lfc.pkl'), "wb") as f:
        pickle.dump(feature_importance_lfc, f)

def predictions_McFarland_per_tissue(n_features=1000, outersplits=5):
    mean_observed_pre_treatment, mean_observed_post_treatment, mean_observed_LFC = get_McFarland_mean_data()
    cellline_sensitivity_info = get_McFarland_sensitivityinfo_for_profile_merge()

    mean_observed_pre_treatment = pd.merge(mean_observed_pre_treatment, cellline_sensitivity_info, left_on=['cell_line', 'condition'], right_on=['cell_line', 'target'], how='left')
    mean_observed_post_treatment = pd.merge(mean_observed_post_treatment, cellline_sensitivity_info, left_on=['cell_line', 'condition'], right_on=['cell_line', 'target'], how='left')
    mean_observed_LFC = pd.merge(mean_observed_LFC, cellline_sensitivity_info, left_on=['cell_line', 'condition'], right_on=['cell_line', 'target'], how='left')

    results = pd.DataFrame()
    predictions_df = pd.DataFrame()
    feature_importance_pre = {}
    feature_importance_post = {}
    feature_importance_lfc = {}

    features = feature_selection(mean_observed_pre_treatment, n_features)

    mean_observed_pre_treatment = filter_on_coefficient_of_variation(mean_observed_pre_treatment, groupby=['tissue'])
    mean_observed_post_treatment = filter_on_coefficient_of_variation(mean_observed_post_treatment, groupby=['tissue'])
    mean_observed_LFC = filter_on_coefficient_of_variation(mean_observed_LFC, groupby=['tissue'])

    tissues = mean_observed_pre_treatment['tissue'].unique().tolist()

    for i, t in enumerate(tissues):
        pre_subset = mean_observed_pre_treatment[mean_observed_pre_treatment['tissue'] == t]
        post_subset = mean_observed_post_treatment[mean_observed_post_treatment['tissue'] == t]
        lfc_subset = mean_observed_LFC[mean_observed_LFC['tissue'] == t]

        pre_subset = add_y_and_normalize(pre_subset, 'sens', normalize=False, keep=['condition', 'tissue', 'cell_line'])
        post_subset = add_y_and_normalize(post_subset, 'sens',  normalize=False, keep=['condition', 'tissue', 'cell_line'])
        lfc_subset = add_y_and_normalize(lfc_subset, 'sens', normalize=False, keep=['condition', 'tissue', 'cell_line'])

        if pre_subset.shape[0] < 20:
            print(f'Too few samples for {t} ({pre_subset.shape[0]} samples)')
            continue
        else:
            print(f'Predicting for {t}')
            pre_outcomes, pre_predictions, pre_feature_importance = prediction_pipeline(pre_subset, mean_observed_pre_treatment, 'pre_treatment', feature_subset=features, n_outer_splits=outersplits)
            post_outcomes, post_predictions, post_feature_importance = prediction_pipeline(post_subset, mean_observed_pre_treatment, 'post_treatment',feature_subset=features,n_outer_splits=outersplits)
            LFC_outcomes, LFC_predictions, LFC_feature_importance = prediction_pipeline(lfc_subset, mean_observed_pre_treatment, 'LFC', feature_subset=features, n_outer_splits=outersplits)

            pre_outcomes = pre_outcomes.assign(tissue=t).assign(n_genes = n_features)
            post_outcomes = post_outcomes.assign(tissue=t).assign(n_genes = n_features)
            LFC_outcomes = LFC_outcomes.assign(tissue=t).assign(n_genes = n_features)
            results = pd.concat([results, pre_outcomes, post_outcomes, LFC_outcomes])

            pre_predictions = pre_predictions.assign(tissue=t)
            post_predictions = post_predictions.assign(tissue=t)
            LFC_predictions = LFC_predictions.assign(tissue=t)
            predictions_df = pd.concat([predictions_df, pre_predictions, post_predictions, LFC_predictions])

            feature_importance_pre[t] = pre_feature_importance
            feature_importance_post[t] = post_feature_importance
            feature_importance_lfc[t] = LFC_feature_importance


    results.to_csv(os.path.join(results_dir,'tissuemodels_regression_results.csv'))
    predictions_df.to_csv(os.path.join(results_dir,'tissuemodels_regression_predictions.csv'))

    with open(os.path.join(results_dir,'tissuemodels_regression_feature_importance_on_pre.pkl'), "wb") as f:
        pickle.dump(feature_importance_pre, f)
    with open(os.path.join(results_dir,'tissuemodels_regression_feature_importance_on_post.pkl'), "wb") as f:
        pickle.dump(feature_importance_post, f)
    with open(os.path.join(results_dir,'tissuemodels_regression_feature_importance_on_lfc.pkl'), "wb") as f:
        pickle.dump(feature_importance_lfc, f)

def _mcfarland_scenario_cv(df_with_y, selected_features, scenario='seen_seen', n_folds=5, random_state=1):
    """
    5-fold CV evaluators for McFarland scenario analyses.
    Scenarios:
      - seen_seen
      - unseen_drug_seen_tissue
      - seen_drug_unseen_tissue
      - unseen_drug_unseen_tissue
    """
    n_input = len(df_with_y)
    n_features_requested = len(selected_features)
    _meta_cols = [c for c in ('y', 'tissue', 'cell_line', 'condition') if c in df_with_y.columns]
    df_with_y = df_with_y.reset_index(drop=True)
    if _meta_cols:
        n_before_meta_na = len(df_with_y)
        df_with_y = df_with_y.dropna(axis=0, subset=_meta_cols)
        if len(df_with_y) != n_before_meta_na:
            logger.info(
                "_mcfarland_scenario_cv: dropped %d/%d rows with NaN in metadata columns %s (not gene-wide dropna)",
                n_before_meta_na - len(df_with_y),
                n_before_meta_na,
                _meta_cols,
            )
    else:
        df_with_y = df_with_y.dropna(axis=0)
        logger.warning("_mcfarland_scenario_cv: no y/tissue/cell_line/condition in frame; applied full dropna")

    selected_features = list(set(selected_features).intersection(set(df_with_y.columns)))
    rng = np.random.default_rng(random_state)

    logger.info(
        "_mcfarland_scenario_cv: scenario=%s rows_in=%d rows_after_meta_dropna=%d features_requested=%d features_after_col_intersection=%d n_folds=%d",
        scenario,
        n_input,
        len(df_with_y),
        n_features_requested,
        len(selected_features),
        n_folds,
    )

    if len(selected_features) == 0:
        logger.error(
            "_mcfarland_scenario_cv: no selected_features intersect df columns (df columns sample: %s)",
            list(df_with_y.columns[:24]),
        )
        return (
            pd.DataFrame(columns=['accuracy', 'test_condition', 'split']),
            pd.DataFrame(columns=['tissue', 'cell_line', 'condition', 'pred', 'true', 'test_condition', 'split']),
            {}
        )

    tissues = sorted(df_with_y['tissue'].drop_duplicates().tolist())
    conditions = sorted(df_with_y['condition'].drop_duplicates().tolist())
    shuffled_tissues = tissues.copy()
    shuffled_conditions = conditions.copy()
    rng.shuffle(shuffled_tissues)
    rng.shuffle(shuffled_conditions)
    tissue_folds = np.array_split(np.array(shuffled_tissues, dtype=object), n_folds)
    condition_folds = np.array_split(np.array(shuffled_conditions, dtype=object), n_folds)

    # used for seen_seen
    fold_id = np.full(len(df_with_y), fill_value=-1, dtype=int)
    if scenario == 'seen_seen':
        for tissue in tissues:
            if pd.isna(tissue):
                continue
            idx = df_with_y.index[df_with_y['tissue'] == tissue].to_numpy()
            order = idx.copy()
            if len(order) > n_folds:
                rng.shuffle(order)
            fold_id[order] = np.arange(len(order)) % n_folds
        # Rows with missing tissue never match `tissue == np.nan`; without this, fold_id stays -1
        # and every fold is skipped (empty CSVs with headers only).
        unassigned = fold_id < 0
        if unassigned.any():
            logger.warning(
                "_mcfarland_scenario_cv seen_seen: %d/%d rows had no tissue fold assignment; "
                "assigning folds by shuffled row index.",
                int(unassigned.sum()),
                len(df_with_y),
            )
            idx_unassigned = df_with_y.index[unassigned].to_numpy()
            order = idx_unassigned.copy()
            if len(order) > n_folds:
                rng.shuffle(order)
            fold_id[order] = np.arange(len(order)) % n_folds

    param_grid = {
        'elasticnet__alpha': [0.1, 1.0, 10.0],
        'elasticnet__l1_ratio': [0.0, 0.1, 0.5, 1.0]
    }

    outer_scores = []
    outcomes = []
    feature_importance_dict = {}
    skip_reasons: list[str] = []

    for fold_idx in range(n_folds):
        if scenario == 'seen_seen':
            test_mask = fold_id == fold_idx
            train_mask = ~test_mask
        elif scenario == 'unseen_drug_seen_tissue':
            test_conditions = condition_folds[fold_idx].tolist()
            test_mask = df_with_y['condition'].isin(test_conditions)
            train_mask = ~test_mask
        elif scenario == 'seen_drug_unseen_tissue':
            test_tissues = tissue_folds[fold_idx].tolist()
            test_mask = df_with_y['tissue'].isin(test_tissues)
            train_mask = ~test_mask
        elif scenario == 'unseen_drug_unseen_tissue':
            test_tissues = tissue_folds[fold_idx].tolist()
            test_conditions = condition_folds[fold_idx].tolist()
            test_mask = df_with_y['tissue'].isin(test_tissues) & df_with_y['condition'].isin(test_conditions)
            train_mask = (~df_with_y['tissue'].isin(test_tissues)) & (~df_with_y['condition'].isin(test_conditions))
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        n_test = int(test_mask.sum())
        n_train = int(train_mask.sum())
        if n_test == 0 or n_train < 2:
            skip_reasons.append(
                f"fold={fold_idx} scenario={scenario} n_test={n_test} n_train={n_train} (need test>=1 and train>=2)"
            )
            logger.info("_mcfarland_scenario_cv skip: %s", skip_reasons[-1])
            continue

        X_train = df_with_y.loc[train_mask].copy()
        X_test = df_with_y.loc[test_mask].copy()

        y_train = X_train['y'].astype(float)
        y_test = X_test['y'].astype(float).reset_index(drop=True)
        X_train = X_train.drop(columns=['y'])
        X_test = X_test.drop(columns=['y'])

        identifiers = X_test[['tissue', 'cell_line', 'condition']].reset_index(drop=True)
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        X_train = X_train.drop(columns=['tissue', 'cell_line', 'condition']).astype(float)
        X_test = X_test.drop(columns=['tissue', 'cell_line', 'condition']).astype(float)
        X_train = X_train.fillna(0.0)
        X_test = X_test.fillna(0.0)

        if X_train.shape[0] < 2 or X_test.shape[0] < 1:
            skip_reasons.append(
                f"fold={fold_idx} post-mask shapes train={X_train.shape} test={X_test.shape}"
            )
            logger.info("_mcfarland_scenario_cv skip: %s", skip_reasons[-1])
            continue

        cv_folds = min(5, X_train.shape[0])
        if cv_folds < 2:
            skip_reasons.append(f"fold={fold_idx} cv_folds={cv_folds} (need >=2 inner CV folds)")
            logger.info("_mcfarland_scenario_cv skip: %s", skip_reasons[-1])
            continue

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('elasticnet', ElasticNet(max_iter=1000))
        ])
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv_folds)
        try:
            grid_search.fit(X_train, y_train)
        except Exception as exc:
            logger.exception(
                "_mcfarland_scenario_cv: GridSearchCV failed fold=%d scenario=%s train_shape=%s n_features=%d: %s",
                fold_idx,
                scenario,
                X_train.shape,
                X_train.shape[1],
                exc,
            )
            skip_reasons.append(f"fold={fold_idx} GridSearchCV error: {exc}")
            continue
        y_pred = grid_search.predict(X_test)
        best_model = grid_search.best_estimator_.named_steps['elasticnet']
        outer_acc = np.sqrt(mean_squared_error(y_test, y_pred))

        fold_name = f'fold={fold_idx}'
        feature_importance_dict[fold_name] = pd.Series(index=X_train.columns, data=best_model.coef_, name=fold_name)

        split_outcomes = pd.concat([identifiers, pd.DataFrame({'pred': y_pred, 'true': y_test})], axis=1)
        if scenario == 'seen_seen':
            split_outcomes['test_condition'] = (
                'condition=' + split_outcomes['condition'].astype(str) + ' AND tissue=' + split_outcomes['tissue'].astype(str)
            )
            outer_scores.append(pd.DataFrame({'accuracy': [outer_acc], 'test_condition': [fold_name], 'split': [fold_idx]}))
        elif scenario == 'unseen_drug_seen_tissue':
            split_outcomes['test_condition'] = 'condition=' + split_outcomes['condition'].astype(str)
            outer_scores.append(pd.DataFrame({'accuracy': [outer_acc], 'test_condition': [fold_name], 'split': [fold_idx]}))
        elif scenario == 'seen_drug_unseen_tissue':
            split_outcomes['test_condition'] = 'tissue=' + split_outcomes['tissue'].astype(str)
            outer_scores.append(pd.DataFrame({'accuracy': [outer_acc], 'test_condition': [fold_name], 'split': [fold_idx]}))
        else:
            split_outcomes['test_condition'] = (
                'tissue=' + split_outcomes['tissue'].astype(str) + ' OR condition=' + split_outcomes['condition'].astype(str)
            )
            # per-combination scores so downstream parsing remains compatible
            for tc, sub in split_outcomes.groupby('test_condition'):
                if sub.shape[0] < 2:
                    acc_val = 0.0
                else:
                    acc_val = np.sqrt(mean_squared_error(sub['true'], sub['pred']))
                outer_scores.append(pd.DataFrame({'accuracy': [acc_val], 'test_condition': [tc], 'split': [fold_idx]}))

        outcomes.append(split_outcomes.assign(split=fold_idx))
        logger.info(
            "_mcfarland_scenario_cv ok: fold=%d scenario=%s RMSE_sqrt=%.4f n_train=%d n_test=%d n_feat=%d",
            fold_idx,
            scenario,
            outer_acc,
            X_train.shape[0],
            X_test.shape[0],
            X_train.shape[1],
        )

    logger.info(
        "_mcfarland_scenario_cv summary: scenario=%s folds_completed=%d folds_skipped=%d",
        scenario,
        len(outer_scores),
        len(skip_reasons),
    )

    if len(df_with_y) > 0 and not outer_scores:
        logger.warning(
            "_mcfarland_scenario_cv produced no outer scores. Skip reasons (up to 20): %s",
            skip_reasons[:20],
        )

    if outer_scores:
        outer_scores = pd.concat(outer_scores, ignore_index=True)
    else:
        outer_scores = pd.DataFrame(columns=['accuracy', 'test_condition', 'split'])

    if outcomes:
        outcomes = pd.concat(outcomes, ignore_index=True)
    else:
        outcomes = pd.DataFrame(columns=['tissue', 'cell_line', 'condition', 'pred', 'true', 'test_condition', 'split'])

    return outer_scores, outcomes, feature_importance_dict

def predictions_McFarland_seen_seen():
    mean_observed_pre_treatment, mean_observed_post_treatment, mean_observed_LFC = get_McFarland_mean_data()
    cellline_sensitivity_info = get_McFarland_sensitivityinfo_for_profile_merge()

    mean_observed_pre_treatment = pd.merge(mean_observed_pre_treatment, cellline_sensitivity_info, left_on=['cell_line', 'condition'], right_on=['cell_line', 'target'], how='left')
    mean_observed_post_treatment = pd.merge(mean_observed_post_treatment, cellline_sensitivity_info, left_on=['cell_line', 'condition'], right_on=['cell_line', 'target'], how='left')
    mean_observed_LFC = pd.merge(mean_observed_LFC, cellline_sensitivity_info, left_on=['cell_line', 'condition'], right_on=['cell_line', 'target'], how='left')

    if 'cell_line' in mean_observed_pre_treatment.columns and 'condition' in mean_observed_pre_treatment.columns:
        obs_pairs = set(zip(mean_observed_pre_treatment['cell_line'], mean_observed_pre_treatment['condition']))
        sens_pairs = set(zip(cellline_sensitivity_info['cell_line'], cellline_sensitivity_info['target']))
        only_obs = obs_pairs - sens_pairs
        only_sens = sens_pairs - obs_pairs
        logger.info(
            "McFarland merge key sets: unique (cell_line, condition) in pre=%d; (cell_line, target) in sensitivity=%d; "
            "in_obs_not_in_sens=%d sample=%s; in_sens_not_in_obs=%d sample=%s",
            len(obs_pairs),
            len(sens_pairs),
            len(only_obs),
            list(only_obs)[:15],
            len(only_sens),
            list(only_sens)[:10],
        )

    if 'sens' in mean_observed_pre_treatment.columns:
        n_hit = int(mean_observed_pre_treatment['sens'].notna().sum())
        logger.info(
            "McFarland sensitivity merge: pre rows=%d with matched sens=%d (%.1f%%)",
            len(mean_observed_pre_treatment),
            n_hit,
            100.0 * n_hit / max(len(mean_observed_pre_treatment), 1),
        )
        if n_hit == 0:
            logger.error(
                "No rows matched mcfarland_sensitivity_info on (cell_line, condition) vs "
                "(cell_line, target). Check condition strings in pseudobulk vs target in resources."
            )

    mean_observed_pre_treatment_with_y = add_y_and_normalize(mean_observed_pre_treatment, 'sens', normalize=False, keep=['condition', 'tissue', 'cell_line'])
    mean_observed_post_treatment_with_y = add_y_and_normalize(mean_observed_post_treatment, 'sens', normalize=False, keep=['condition', 'tissue', 'cell_line'])
    mean_observed_LFC_with_y= add_y_and_normalize(mean_observed_LFC, 'sens',normalize=False, keep=['condition', 'tissue', 'cell_line'])

    logger.info(
        "McFarland seen_seen row counts after add_y: pre=%d post=%d LFC=%d",
        len(mean_observed_pre_treatment_with_y),
        len(mean_observed_post_treatment_with_y),
        len(mean_observed_LFC_with_y),
    )

    features = feature_selection(mean_observed_pre_treatment_with_y.drop(columns=['y']), 1000)
    pre_scores, pre_predictions, feature_importance_pre = _mcfarland_scenario_cv(
        mean_observed_pre_treatment_with_y, features, scenario='seen_seen', n_folds=5, random_state=1
    )
    pre_scores_df = pre_scores.assign(model='pre-treatment').assign(feature_selection='top1000 highest variance')
    pre_predictions = pre_predictions.assign(model='pre-treatment').assign(feature_selection='top1000 highest variance')

    post_scores, post_predictions, feature_importance_post  = _mcfarland_scenario_cv(
        mean_observed_post_treatment_with_y, features, scenario='seen_seen', n_folds=5, random_state=1
    )
    post_scores_df = post_scores.assign(model='post-treatment').assign(feature_selection='top1000 highest variance')
    post_predictions = post_predictions.assign(model='post-treatment').assign(feature_selection='top1000 highest variance')    

    lfc_scores, lfc_predictions, feature_importance_lfc = _mcfarland_scenario_cv(
        mean_observed_LFC_with_y, features, scenario='seen_seen', n_folds=5, random_state=1
    )
    lfc_scores_df = lfc_scores.assign(model='LFC').assign(feature_selection='top1000 highest variance')
    lfc_predictions = lfc_predictions.assign(model='LFC').assign(feature_selection='top1000 highest variance')

    all_scores_leave_drug_out = pd.concat([pre_scores_df, post_scores_df, lfc_scores_df])
    all_predictions_leave_drug_out = pd.concat([pre_predictions, post_predictions, lfc_predictions])

    all_scores_leave_drug_out.to_csv(os.path.join(results_dir,'mcfarland_regression_seen_seen_cv5.csv'))
    all_predictions_leave_drug_out.to_csv(os.path.join(results_dir,'mcfarland_regression_predictions_seen_seen_cv5.csv'))

    with open(os.path.join(results_dir,'mcfarland_regression_seen_seen_cv5_feature_importance_on_pre.pkl'), "wb") as f:
        pickle.dump(feature_importance_pre, f)
    with open(os.path.join(results_dir,'mcfarland_regression_seen_seen_cv5_feature_importance_on_post.pkl'), "wb") as f:
        pickle.dump(feature_importance_post, f)
    with open(os.path.join(results_dir,'mcfarland_regression_seen_seen_cv5_feature_importance_on_lfc.pkl'), "wb") as f:
        pickle.dump(feature_importance_lfc, f)

def predictions_McFarland_LDO():
    """
    Perform unseen/seen evaluation by leaving out entire drugs.
    For each drug:
    - Test set: All samples from that drug
    - Training set: All samples from other drugs
    This tests how well the model generalizes to completely unseen drugs.

    Design decisions:
    - Uses filter_on_coefficient_of_variation to remove conditions with high variability
    - Selects features once using all data rather than per test case
    - Uses the same feature set across all test cases for consistency
    """
    mean_observed_pre_treatment, mean_observed_post_treatment, mean_observed_LFC = get_McFarland_mean_data()
    cellline_sensitivity_info = get_McFarland_sensitivityinfo_for_profile_merge()

    mean_observed_pre_treatment = pd.merge(mean_observed_pre_treatment, cellline_sensitivity_info, left_on=['cell_line', 'condition'], right_on=['cell_line', 'target'], how='left')
    mean_observed_post_treatment = pd.merge(mean_observed_post_treatment, cellline_sensitivity_info, left_on=['cell_line', 'condition'], right_on=['cell_line', 'target'], how='left')
    mean_observed_LFC = pd.merge(mean_observed_LFC, cellline_sensitivity_info, left_on=['cell_line', 'condition'], right_on=['cell_line', 'target'], how='left')
    mean_observed_pre_treatment_with_y = add_y_and_normalize(mean_observed_pre_treatment, 'sens', normalize=False, keep=['condition', 'tissue', 'cell_line'])
    mean_observed_post_treatment_with_y = add_y_and_normalize(mean_observed_post_treatment, 'sens', normalize=False, keep=['condition', 'tissue', 'cell_line'])
    mean_observed_LFC_with_y= add_y_and_normalize(mean_observed_LFC, 'sens',normalize=False, keep=['condition', 'tissue', 'cell_line'])

    features = feature_selection(mean_observed_pre_treatment_with_y.drop(columns=['y']), 1000)
    pre_scores, pre_predictions, feature_importance_pre = _mcfarland_scenario_cv(
        mean_observed_pre_treatment_with_y, features, scenario='unseen_drug_seen_tissue', n_folds=5, random_state=1
    )
    pre_scores_df = pre_scores.assign(model='pre-treatment').assign(feature_selection='top1000 highest variance')
    pre_predictions = pre_predictions.assign(model='pre-treatment').assign(feature_selection='top1000 highest variance')

    post_scores, post_predictions, feature_importance_post  = _mcfarland_scenario_cv(
        mean_observed_post_treatment_with_y, features, scenario='unseen_drug_seen_tissue', n_folds=5, random_state=1
    )
    post_scores_df = post_scores.assign(model='post-treatment').assign(feature_selection='top1000 highest variance')
    post_predictions = post_predictions.assign(model='post-treatment').assign(feature_selection='top1000 highest variance')    

    lfc_scores, lfc_predictions, feature_importance_lfc = _mcfarland_scenario_cv(
        mean_observed_LFC_with_y, features, scenario='unseen_drug_seen_tissue', n_folds=5, random_state=1
    )
    lfc_scores_df = lfc_scores.assign(model='LFC').assign(feature_selection='top1000 highest variance')
    lfc_predictions = lfc_predictions.assign(model='LFC').assign(feature_selection='top1000 highest variance')

    all_scores_leave_drug_out = pd.concat([pre_scores_df, post_scores_df, lfc_scores_df])
    all_predictions_leave_drug_out = pd.concat([pre_predictions, post_predictions, lfc_predictions])

    all_scores_leave_drug_out.to_csv(os.path.join(results_dir,'mcfarland_regression_LDO_cv5.csv'))
    all_predictions_leave_drug_out.to_csv(os.path.join(results_dir,'mcfarland_regression_LDO_cv5_predictions.csv'))

    with open(os.path.join(results_dir,'mcfarland_regression_LDO_cv5_feature_importance_on_pre.pkl'), "wb") as f:
        pickle.dump(feature_importance_pre, f)
    with open(os.path.join(results_dir,'mcfarland_regression_LDO_cv5_feature_importance_on_post.pkl'), "wb") as f:
        pickle.dump(feature_importance_post, f)
    with open(os.path.join(results_dir,'mcfarland_regression_LDO_cv5_feature_importance_on_lfc.pkl'), "wb") as f:
        pickle.dump(feature_importance_lfc, f)


def predictions_McFarland_LTO():
    """
    Perform unseen/seen evaluation by leaving out entire tissues.
    For each tissue:
    - Test set: All samples from that tissue
    - Training set: All samples from other tissues
    This tests how well the model generalizes to completely unseen tissues.

    Design decisions:
    - Uses filter_on_coefficient_of_variation to remove tissues with high variability
    - Selects features once using all data rather than per test case
    - Uses the same feature set across all test cases for consistency
    """
    mean_observed_pre_treatment, mean_observed_post_treatment, mean_observed_LFC = get_McFarland_mean_data()
    cellline_sensitivity_info = get_McFarland_sensitivityinfo_for_profile_merge()

    mean_observed_pre_treatment = pd.merge(mean_observed_pre_treatment, cellline_sensitivity_info, left_on=['cell_line', 'condition'], right_on=['cell_line', 'target'], how='left')
    mean_observed_post_treatment = pd.merge(mean_observed_post_treatment, cellline_sensitivity_info, left_on=['cell_line', 'condition'], right_on=['cell_line', 'target'], how='left')
    mean_observed_LFC = pd.merge(mean_observed_LFC, cellline_sensitivity_info, left_on=['cell_line', 'condition'], right_on=['cell_line', 'target'], how='left')

    mean_observed_pre_treatment_with_y = add_y_and_normalize(mean_observed_pre_treatment, 'sens', normalize=False, keep=['condition', 'tissue', 'cell_line'])
    mean_observed_post_treatment_with_y = add_y_and_normalize(mean_observed_post_treatment, 'sens', normalize=False, keep=['condition', 'tissue', 'cell_line'])
    mean_observed_LFC_with_y= add_y_and_normalize(mean_observed_LFC, 'sens',normalize=False, keep=['condition', 'tissue', 'cell_line'])

    features = feature_selection(mean_observed_pre_treatment_with_y.drop(columns=['y']), 1000)
    pre_scores, pre_predictions, feature_importance_pre = _mcfarland_scenario_cv(
        mean_observed_pre_treatment_with_y, features, scenario='seen_drug_unseen_tissue', n_folds=5, random_state=1
    )
    pre_scores_df = pre_scores.assign(model='pre-treatment').assign(feature_selection='top1000 highest variance')
    pre_predictions = pre_predictions.assign(model='pre-treatment').assign(feature_selection='top1000 highest variance')

    post_scores, post_predictions, feature_importance_post = _mcfarland_scenario_cv(
        mean_observed_post_treatment_with_y, features, scenario='seen_drug_unseen_tissue', n_folds=5, random_state=1
    )
    post_scores_df = post_scores.assign(model='post-treatment').assign(feature_selection='top1000 highest variance')
    post_predictions = post_predictions.assign(model='post-treatment').assign(feature_selection='top1000 highest variance')

    lfc_scores, lfc_predictions, feature_importance_lfc = _mcfarland_scenario_cv(
        mean_observed_LFC_with_y, features, scenario='seen_drug_unseen_tissue', n_folds=5, random_state=1
    )
    lfc_scores_df = lfc_scores.assign(model='LFC').assign(feature_selection='top1000 highest variance')
    lfc_predictions = lfc_predictions.assign(model='LFC').assign(feature_selection='top1000 highest variance')

    all_scores_leave_tissue_out = pd.concat([pre_scores_df, post_scores_df, lfc_scores_df])
    all_predictions_leave_tissue_out = pd.concat([pre_predictions, post_predictions, lfc_predictions])

    all_scores_leave_tissue_out.to_csv(os.path.join(results_dir,'mcfarland_regression_LTO_cv5.csv'))
    all_predictions_leave_tissue_out.to_csv(os.path.join(results_dir,'mcfarland_regression_LTO_cv5_predictions.csv'))

    with open(os.path.join(results_dir,'mcfarland_regression_LTO_cv5_feature_importance_on_pre.pkl'), "wb") as f:
        pickle.dump(feature_importance_pre, f)
    with open(os.path.join(results_dir,'mcfarland_regression_LTO_cv5_feature_importance_on_post.pkl'), "wb") as f:
        pickle.dump(feature_importance_post, f)
    with open(os.path.join(results_dir,'mcfarland_regression_LTO_cv5_feature_importance_on_lfc.pkl'), "wb") as f:
        pickle.dump(feature_importance_lfc, f)


def predictions_McFarland_LOO():
    mean_observed_pre_treatment, mean_observed_post_treatment, mean_observed_LFC = get_McFarland_mean_data()
    cellline_sensitivity_info = get_McFarland_sensitivityinfo_for_profile_merge()

    mean_observed_pre_treatment = pd.merge(mean_observed_pre_treatment, cellline_sensitivity_info, left_on=['cell_line', 'condition'], right_on=['cell_line', 'target'], how='left')
    mean_observed_post_treatment = pd.merge(mean_observed_post_treatment, cellline_sensitivity_info, left_on=['cell_line', 'condition'], right_on=['cell_line', 'target'], how='left')
    mean_observed_LFC = pd.merge(mean_observed_LFC, cellline_sensitivity_info, left_on=['cell_line', 'condition'], right_on=['cell_line', 'target'], how='left')

    mean_observed_pre_treatment_with_y = add_y_and_normalize(mean_observed_pre_treatment, 'sens', normalize=False, keep=['condition', 'tissue', 'cell_line'])
    mean_observed_post_treatment_with_y = add_y_and_normalize(mean_observed_post_treatment, 'sens', normalize=False, keep=['condition', 'tissue', 'cell_line'])
    mean_observed_LFC_with_y= add_y_and_normalize(mean_observed_LFC, 'sens',normalize=False, keep=['condition', 'tissue', 'cell_line'])

    features = feature_selection(mean_observed_pre_treatment_with_y.drop(columns=['y']), 1000)
    pre_scores, pre_predictions, feature_importance_pre = _mcfarland_scenario_cv(
        mean_observed_pre_treatment_with_y, features, scenario='unseen_drug_unseen_tissue', n_folds=5, random_state=1
    )
    pre_scores_df = pre_scores.assign(model='pre-treatment').assign(feature_selection='top1000 highest variance')
    pre_predictions = pre_predictions.assign(model='pre-treatment').assign(feature_selection='top1000 highest variance')

    post_scores, post_predictions, feature_importance_post = _mcfarland_scenario_cv(
        mean_observed_post_treatment_with_y, features, scenario='unseen_drug_unseen_tissue', n_folds=5, random_state=1
    )
    post_scores_df = post_scores.assign(model='post-treatment').assign(feature_selection='top1000 highest variance')
    post_predictions = post_predictions.assign(model='post-treatment').assign(feature_selection='top1000 highest variance')

    lfc_scores, lfc_predictions, feature_importance_lfc = _mcfarland_scenario_cv(
        mean_observed_LFC_with_y, features, scenario='unseen_drug_unseen_tissue', n_folds=5, random_state=1
    )
    lfc_scores_df = lfc_scores.assign(model='LFC').assign(feature_selection='top1000 highest variance')
    lfc_predictions = lfc_predictions.assign(model='LFC').assign(feature_selection='top1000 highest variance')

    all_scores_leave_tissue_out = pd.concat([pre_scores_df, post_scores_df, lfc_scores_df])
    all_predictions_leave_tissue_out = pd.concat([pre_predictions, post_predictions, lfc_predictions])

    all_scores_leave_tissue_out.to_csv(os.path.join(results_dir,'mcfarland_regression_LOO_cv5.csv'))
    all_predictions_leave_tissue_out.to_csv(os.path.join(results_dir,'mcfarland_regression_LOO_cv5_predictions.csv'))

    with open(os.path.join(results_dir,'mcfarland_regression_LOO_cv5_feature_importance_on_pre.pkl'), "wb") as f:
        pickle.dump(feature_importance_pre, f)
    with open(os.path.join(results_dir,'mcfarland_regression_LOO_cv5_feature_importance_on_post.pkl'), "wb") as f:
        pickle.dump(feature_importance_post, f)
    with open(os.path.join(results_dir, 'mcfarland_regression_LOO_cv5_feature_importance_on_lfc.pkl'), "wb") as f:
        pickle.dump(feature_importance_lfc, f)


def predictions_indepedent_test_set_twopart(X_train, X_test, features, two_part=True, save_feature_importance=False, prefix=''):
    y_train = X_train['y'].astype(float)
    y_test = X_test['y'].astype(float)
    X_train = X_train.drop(columns=X_train.columns.intersection(['tissue', 'y', 'cell_line', 'condition_cell_line', 'condition', 'binned']))
    X_test = X_test.drop(columns=X_test.columns.intersection(['tissue', 'y', 'cell_line', 'condition_cell_line', 'condition', 'binned']))
    
    features = list(set(features).intersection(set(X_test.columns)))
    X_train = X_train[features]
    X_test = X_test[features]
    
    if two_part:
        # ----- PART 1: Binary classification (y > 0.02)
        y_train_binary = (y_train > 0.02).astype(int)
        clf_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000))
        ])
        clf_pipeline.fit(X_train, y_train_binary)
        y_test_binary_pred = clf_pipeline.predict(X_test)
        
        # ----- PART 2: Regression on non-zero y
        mask_nonzero = (y_train > 0.02)
        X_train = X_train[mask_nonzero]
        y_train = y_train[mask_nonzero]
        
    reg_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('elasticnet', ElasticNet(max_iter=1000))
    ])
    param_grid = {
        'elasticnet__alpha': [0.1, 1.0, 10.0],
        'elasticnet__l1_ratio': [0.0, 0.1, 0.5, 1.0]
    }


    grid_search = GridSearchCV(reg_pipeline, param_grid, cv=5, n_jobs=-1)

    grid_search.fit(X_train, y_train)
    y_test_reg_pred = grid_search.predict(X_test)
    
    # ----- Combine binary + regression output
    if two_part:
        y_pred_combined = np.where(y_test_binary_pred == 1, y_test_reg_pred, 0.0)
    else:
        y_pred_combined = y_test_reg_pred.copy()

    # ----- Optional: save feature importances
    best_model = grid_search.best_estimator_.named_steps['elasticnet']
    feature_importance = pd.Series(index=X_train.columns, data=best_model.coef_, name='importance')
    if save_feature_importance:
        if two_part:
            path = os.path.join(results_dir, f'{prefix}_twopart_feature_importance.csv')
        else:
            path = os.path.join(results_dir, f'{prefix}_onepart_feature_importance.csv')
        pd.DataFrame(feature_importance).to_csv(path)

    return pd.DataFrame({'pred': y_pred_combined, 'true': y_test})

def predictions_indepedent_test_set_classification(X_train, X_test, features, boundary=0.1):
    
    y_train = X_train['y'].astype(float)
    y_test = X_test['y'].astype(float)
    X_train = X_train.drop(columns=X_train.columns.intersection(['tissue', 'y', 'cell_line', 'condition_cell_line', 'condition', 'binned']))
    X_test = X_test.drop(columns=X_test.columns.intersection(['tissue', 'y', 'cell_line', 'condition_cell_line', 'condition', 'binned']))
    
    features = list(set(features).intersection(set(X_test.columns)))
    X_train = X_train[features]
    X_test = X_test[features]
    
    y_train_binary = (y_train > boundary).astype(int)
    clf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    clf_pipeline.fit(X_train, y_train_binary)
    y_test_binary_pred = clf_pipeline.predict(X_test)
    y_test_binary_obs = (y_test > boundary).astype(int)

    return pd.DataFrame({'pred': y_test_binary_pred, 'true': y_test_binary_obs})

def _mcfarland_predefined_fold_cv(
    df_with_y: pd.DataFrame,
    selected_features: list[str],
    dedupe_train: bool = False,
    test_profiles_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    ElasticNet regression with train/test splits defined by the ``fold`` column.

    For each test fold, rows with ``fold == test_fold`` are held out; all other folds
    are used for training. When ``dedupe_train`` is True, training rows are deduplicated
    on (cell_line, condition) so identical observed profiles are not overweighted.

    If ``test_profiles_df`` is set, train on ``df_with_y`` and evaluate on
    ``test_profiles_df`` for the held-out fold (optional; default is train and test on
    the same profile table).
    """
    if 'fold' not in df_with_y.columns:
        raise ValueError("df_with_y must contain a 'fold' column for predefined-fold CV")
    if test_profiles_df is not None and 'fold' not in test_profiles_df.columns:
        raise ValueError("test_profiles_df must contain a 'fold' column")

    _meta_cols = [c for c in ('y', 'tissue', 'cell_line', 'condition', 'fold') if c in df_with_y.columns]
    df_with_y = df_with_y.reset_index(drop=True).copy()
    if _meta_cols:
        df_with_y = df_with_y.dropna(axis=0, subset=_meta_cols)

    if test_profiles_df is not None:
        test_profiles_df = test_profiles_df.reset_index(drop=True).copy()
        test_meta = [c for c in _meta_cols if c in test_profiles_df.columns]
        if test_meta:
            test_profiles_df = test_profiles_df.dropna(axis=0, subset=test_meta)

    selected_features = list(set(selected_features).intersection(set(df_with_y.columns)))
    if test_profiles_df is not None:
        selected_features = list(set(selected_features).intersection(set(test_profiles_df.columns)))
    if len(selected_features) == 0:
        logger.error("_mcfarland_predefined_fold_cv: no selected features intersect dataframe columns")
        return (
            pd.DataFrame(columns=['accuracy', 'split']),
            pd.DataFrame(columns=['tissue', 'cell_line', 'condition', 'pred', 'true', 'split']),
            {},
        )

    test_folds = sorted(df_with_y['fold'].dropna().unique())
    param_grid = {
        'elasticnet__alpha': [0.1, 1.0, 10.0],
        'elasticnet__l1_ratio': [0.0, 0.1, 0.5, 1.0],
    }

    outer_scores = []
    outcomes = []
    feature_importance_dict = {}

    fold_key_cols = ['cell_line', 'condition', 'fold']

    for test_fold in test_folds:
        train_df = df_with_y.loc[df_with_y['fold'] != test_fold].copy()
        if test_profiles_df is None:
            test_df = df_with_y.loc[df_with_y['fold'] == test_fold].copy()
        else:
            key_cols = [c for c in fold_key_cols if c in df_with_y.columns and c in test_profiles_df.columns]
            observed_test_keys = df_with_y.loc[df_with_y['fold'] == test_fold, key_cols].drop_duplicates()
            test_df = test_profiles_df.loc[test_profiles_df['fold'] == test_fold].copy()
            if key_cols:
                test_df = test_df.merge(observed_test_keys, on=key_cols, how='inner')

        if dedupe_train:
            train_df = train_df.drop_duplicates(subset=['cell_line', 'condition'], keep='first')

        if len(train_df) < 2 or len(test_df) < 1:
            logger.info(
                "_mcfarland_predefined_fold_cv skip fold=%s: n_train=%d n_test=%d",
                test_fold,
                len(train_df),
                len(test_df),
            )
            continue

        y_train = train_df['y'].astype(float)
        y_test = test_df['y'].astype(float).reset_index(drop=True)
        X_train = train_df.drop(columns=['y'])
        X_test = test_df.drop(columns=['y'])

        identifiers = X_test[['tissue', 'cell_line', 'condition']].reset_index(drop=True)
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        X_train = X_train.drop(columns=['tissue', 'cell_line', 'condition', 'fold'], errors='ignore').astype(float)
        X_test = X_test.drop(columns=['tissue', 'cell_line', 'condition', 'fold'], errors='ignore').astype(float)
        X_train = X_train.fillna(0.0)
        X_test = X_test.fillna(0.0)

        cv_folds = min(5, len(X_train))
        if cv_folds < 2:
            logger.info("_mcfarland_predefined_fold_cv skip fold=%s: inner cv_folds=%d", test_fold, cv_folds)
            continue

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('elasticnet', ElasticNet(max_iter=1000)),
        ])
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv_folds)
        try:
            grid_search.fit(X_train, y_train)
        except Exception as exc:
            logger.exception(
                "_mcfarland_predefined_fold_cv: GridSearchCV failed test_fold=%s: %s",
                test_fold,
                exc,
            )
            continue

        y_pred = grid_search.predict(X_test)
        best_model = grid_search.best_estimator_.named_steps['elasticnet']
        outer_acc = np.sqrt(mean_squared_error(y_test, y_pred))

        fold_name = f'fold={test_fold}'
        feature_importance_dict[fold_name] = pd.Series(
            index=X_train.columns, data=best_model.coef_, name=fold_name
        )

        split_outcomes = pd.concat(
            [identifiers, pd.DataFrame({'pred': y_pred, 'true': y_test})],
            axis=1,
        )
        outcomes.append(split_outcomes.assign(split=test_fold))
        outer_scores.append(pd.DataFrame({'accuracy': [outer_acc], 'split': [test_fold]}))
        logger.info(
            "_mcfarland_predefined_fold_cv ok: test_fold=%s RMSE_sqrt=%.4f n_train=%d n_test=%d",
            test_fold,
            outer_acc,
            len(X_train),
            len(X_test),
        )

    if outer_scores:
        outer_scores = pd.concat(outer_scores, ignore_index=True)
    else:
        outer_scores = pd.DataFrame(columns=['accuracy', 'split'])

    if outcomes:
        outcomes = pd.concat(outcomes, ignore_index=True)
    else:
        outcomes = pd.DataFrame(columns=['tissue', 'cell_line', 'condition', 'pred', 'true', 'split'])

    return outer_scores, outcomes, feature_importance_dict


def predict_McFarland_with_CPA_profiles_CV(
    n_features: int = 1000,
    results_prefix: str = 'mcfarland_CPA_fold_cv',
) -> None:
    """
    Predict McFarland drug sensitivity using CPA fold splits.

    Uses ``fold`` from predicted profiles (CPA / GEARS / scFoundation ``mcfarland_mean_*_all.csv``)
    for outer CV. Observed and baseline profiles receive CPA fold labels. For each profile
    source, the model is trained and tested on that source only.

    Saves regression scores, per-sample predictions, and feature-importance pickles for
    observed, CPA-, GEARS-, and scFoundation-predicted, no-effect, and average-effect profiles.
    """
    mean_observed_pre, mean_observed_post, mean_observed_lfc = get_McFarland_mean_data()
    mean_cpa_post, mean_cpa_lfc = get_McFarland_CPA_predictions()
    mean_cpa_post = mean_cpa_post.copy()
    mean_cpa_lfc = mean_cpa_lfc.copy()
    mean_gears_post, mean_gears_lfc = get_McFarland_GEARS_predictions()
    mean_gears_post = mean_gears_post.copy()
    mean_gears_lfc = mean_gears_lfc.copy()
    mean_scfoundation_post, mean_scfoundation_lfc = get_McFarland_scFoundation_predictions()
    mean_scfoundation_post = mean_scfoundation_post.copy()
    mean_scfoundation_lfc = mean_scfoundation_lfc.copy()
    mean_no_effect_post = get_McFarland_no_effect_predictions()
    mean_avg_post, mean_avg_lfc = get_McFarland_average_effect_predictions()
    cellline_sensitivity_info = get_McFarland_sensitivityinfo_for_profile_merge()

    def _merge_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
        return pd.merge(
            df,
            cellline_sensitivity_info,
            left_on=['cell_line', 'condition'],
            right_on=['cell_line', 'target'],
            how='left',
        )

    mean_observed_pre = _merge_sensitivity(mean_observed_pre)
    mean_observed_post = _merge_sensitivity(mean_observed_post)
    mean_observed_lfc = _merge_sensitivity(mean_observed_lfc)
    mean_cpa_post = _merge_sensitivity(mean_cpa_post)
    mean_cpa_lfc = _merge_sensitivity(mean_cpa_lfc)
    mean_gears_post = _merge_sensitivity(mean_gears_post)
    mean_gears_lfc = _merge_sensitivity(mean_gears_lfc)
    mean_scfoundation_post = _merge_sensitivity(mean_scfoundation_post)
    mean_scfoundation_lfc = _merge_sensitivity(mean_scfoundation_lfc)
    mean_no_effect_post = _merge_sensitivity(mean_no_effect_post)
    mean_avg_post = _merge_sensitivity(mean_avg_post)
    mean_avg_lfc = _merge_sensitivity(mean_avg_lfc)

    mean_observed_pre = filter_on_coefficient_of_variation(mean_observed_pre, groupby=['condition'])
    mean_observed_post = filter_on_coefficient_of_variation(mean_observed_post, groupby=['condition'])
    mean_observed_lfc = filter_on_coefficient_of_variation(mean_observed_lfc, groupby=['condition'])
    mean_cpa_post = filter_on_coefficient_of_variation(mean_cpa_post, groupby=['condition'])
    mean_cpa_lfc = filter_on_coefficient_of_variation(mean_cpa_lfc, groupby=['condition'])
    mean_gears_post = filter_on_coefficient_of_variation(mean_gears_post, groupby=['condition'])
    mean_gears_lfc = filter_on_coefficient_of_variation(mean_gears_lfc, groupby=['condition'])
    mean_scfoundation_post = filter_on_coefficient_of_variation(
        mean_scfoundation_post, groupby=['condition']
    )
    mean_scfoundation_lfc = filter_on_coefficient_of_variation(
        mean_scfoundation_lfc, groupby=['condition']
    )
    mean_no_effect_post = filter_on_coefficient_of_variation(mean_no_effect_post, groupby=['condition'])
    mean_avg_post = filter_on_coefficient_of_variation(mean_avg_post, groupby=['condition'])
    mean_avg_lfc = filter_on_coefficient_of_variation(mean_avg_lfc, groupby=['condition'])

    cpa_fold_keys = mean_cpa_post[['cell_line', 'condition', 'fold']].drop_duplicates()
    mean_observed_pre = expand_mcfarland_profiles_with_folds(mean_observed_pre, mean_cpa_post)
    mean_observed_post = expand_mcfarland_profiles_with_folds(mean_observed_post, mean_cpa_post)
    mean_observed_lfc = expand_mcfarland_profiles_with_folds(mean_observed_lfc, mean_cpa_post)
    mean_no_effect_post = expand_mcfarland_profiles_with_folds(mean_no_effect_post, mean_cpa_post)
    mean_avg_post = expand_mcfarland_profiles_with_folds(mean_avg_post, mean_cpa_post)
    mean_avg_lfc = expand_mcfarland_profiles_with_folds(mean_avg_lfc, mean_cpa_post)
    mean_cpa_post = align_mcfarland_profiles_to_fold_keys(mean_cpa_post, cpa_fold_keys)
    mean_cpa_lfc = align_mcfarland_profiles_to_fold_keys(mean_cpa_lfc, cpa_fold_keys)

    gears_fold_keys = mean_gears_post[['cell_line', 'condition', 'fold']].drop_duplicates()
    mean_gears_post = align_mcfarland_profiles_to_fold_keys(mean_gears_post, gears_fold_keys)
    mean_gears_lfc = align_mcfarland_profiles_to_fold_keys(mean_gears_lfc, gears_fold_keys)

    scfoundation_fold_keys = mean_scfoundation_post[['cell_line', 'condition', 'fold']].drop_duplicates()
    mean_scfoundation_post = align_mcfarland_profiles_to_fold_keys(
        mean_scfoundation_post, scfoundation_fold_keys
    )
    mean_scfoundation_lfc = align_mcfarland_profiles_to_fold_keys(
        mean_scfoundation_lfc, scfoundation_fold_keys
    )

    log_mcfarland_profile_expression_difference(
        mean_observed_post, mean_cpa_post, 'CPA post-treatment'
    )
    log_mcfarland_profile_expression_difference(
        mean_observed_lfc, mean_cpa_lfc, 'CPA LFC'
    )
    log_mcfarland_profile_expression_difference(
        mean_observed_post, mean_gears_post, 'GEARS post-treatment'
    )
    log_mcfarland_profile_expression_difference(
        mean_observed_lfc, mean_gears_lfc, 'GEARS LFC'
    )
    log_mcfarland_profile_expression_difference(
        mean_observed_post, mean_scfoundation_post, 'scFoundation post-treatment'
    )
    log_mcfarland_profile_expression_difference(
        mean_observed_lfc, mean_scfoundation_lfc, 'scFoundation LFC'
    )

    _meta_cols = {
        'cell_line', 'condition', 'tissue', 'fold', 'sens', 'target', 'sens_label',
        'n_cells', 'cell_type', 'y',
    }
    mean_no_effect_lfc = mean_no_effect_post.copy()
    gene_cols = [c for c in mean_no_effect_lfc.columns if c not in _meta_cols]
    mean_no_effect_lfc[gene_cols] = 0.0

    feature_to_keep = ['condition', 'tissue', 'cell_line', 'fold']
    mean_observed_pre_with_y = add_y_and_normalize(
        mean_observed_pre, 'sens', normalize=False, keep=feature_to_keep
    )
    mean_observed_post_with_y = add_y_and_normalize(
        mean_observed_post, 'sens', normalize=False, keep=feature_to_keep
    )
    mean_observed_lfc_with_y = add_y_and_normalize(
        mean_observed_lfc, 'sens', normalize=False, keep=feature_to_keep
    )
    mean_cpa_post_with_y = add_y_and_normalize(
        mean_cpa_post, 'sens', normalize=False, keep=feature_to_keep
    )
    mean_cpa_lfc_with_y = add_y_and_normalize(
        mean_cpa_lfc, 'sens', normalize=False, keep=feature_to_keep
    )
    mean_gears_post_with_y = add_y_and_normalize(
        mean_gears_post, 'sens', normalize=False, keep=feature_to_keep
    )
    mean_gears_lfc_with_y = add_y_and_normalize(
        mean_gears_lfc, 'sens', normalize=False, keep=feature_to_keep
    )
    mean_scfoundation_post_with_y = add_y_and_normalize(
        mean_scfoundation_post, 'sens', normalize=False, keep=feature_to_keep
    )
    mean_scfoundation_lfc_with_y = add_y_and_normalize(
        mean_scfoundation_lfc, 'sens', normalize=False, keep=feature_to_keep
    )
    mean_no_effect_post_with_y = add_y_and_normalize(
        mean_no_effect_post, 'sens', normalize=False, keep=feature_to_keep
    )
    mean_no_effect_lfc_with_y = add_y_and_normalize(
        mean_no_effect_lfc, 'sens', normalize=False, keep=feature_to_keep
    )
    mean_avg_post_with_y = add_y_and_normalize(
        mean_avg_post, 'sens', normalize=False, keep=feature_to_keep
    )
    mean_avg_lfc_with_y = add_y_and_normalize(
        mean_avg_lfc, 'sens', normalize=False, keep=feature_to_keep
    )

    features = feature_selection(
        mean_observed_pre_with_y.drop(columns=['y', 'fold'], errors='ignore'),
        n_features=n_features,
    )
    logger.info("predict_McFarland_with_CPA_profiles_CV: using %d features", len(features))

    # (profile_source, model_label, df_with_y, dedupe_train)
    profile_configs = [
        ('observed', 'pre_treatment', mean_observed_pre_with_y, True),
        ('observed', 'post_treatment', mean_observed_post_with_y, True),
        ('observed', 'LFC', mean_observed_lfc_with_y, True),
        ('CPA_predicted', 'post_treatment', mean_cpa_post_with_y, False),
        ('CPA_predicted', 'LFC', mean_cpa_lfc_with_y, False),
        ('GEARS_predicted', 'post_treatment', mean_gears_post_with_y, False),
        ('GEARS_predicted', 'LFC', mean_gears_lfc_with_y, False),
        ('scFoundation_predicted', 'post_treatment', mean_scfoundation_post_with_y, False),
        ('scFoundation_predicted', 'LFC', mean_scfoundation_lfc_with_y, False),
        ('no_effect', 'post_treatment', mean_no_effect_post_with_y, True),
        ('no_effect', 'LFC', mean_no_effect_lfc_with_y, True),
        ('average_effect', 'post_treatment', mean_avg_post_with_y, True),
        ('average_effect', 'LFC', mean_avg_lfc_with_y, True),
    ]

    all_scores = []
    all_predictions = []
    feature_importance: dict[str, dict] = {}

    for profile_source, model_label, df_with_y, dedupe_train in profile_configs:
        logger.info(
            "predict_McFarland_with_CPA_profiles_CV: %s / %s (rows=%d)",
            profile_source,
            model_label,
            len(df_with_y),
        )
        scores, predictions, fi = _mcfarland_predefined_fold_cv(
            df_with_y,
            features,
            dedupe_train=dedupe_train,
        )
        scores = (
            scores.assign(profile_source=profile_source)
            .assign(model=model_label)
            .assign(feature_selection=f'top{n_features} HVG')
        )
        predictions = (
            predictions.assign(profile_source=profile_source)
            .assign(model=model_label)
            .assign(feature_selection=f'top{n_features} HVG')
        )
        all_scores.append(scores)
        all_predictions.append(predictions)
        feature_importance[f'{profile_source}_{model_label}'] = fi

    results = pd.concat(all_scores, ignore_index=True)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    results.to_csv(os.path.join(results_dir, f'{results_prefix}_results.csv'))
    predictions_df.to_csv(os.path.join(results_dir, f'{results_prefix}_predictions.csv'))
    with open(os.path.join(results_dir, f'{results_prefix}_feature_importance.pkl'), 'wb') as f:
        pickle.dump(feature_importance, f)

    logger.info(
        "predict_McFarland_with_CPA_profiles_CV: wrote %s (%d score rows, %d prediction rows)",
        results_prefix,
        len(results),
        len(predictions_df),
    )


def resample_by_bins(df, n_bins=5, max_samples_per_bin=None, random_state=42):
    # Bin the response variable
    df = df.copy()
    df['bin'] = pd.qcut(df['y'], q=n_bins, duplicates='drop')

    # Optional: define max samples per bin
    max_count = max_samples_per_bin or df['bin'].value_counts().max()

    # Resample each bin to the same size
    resampled_dfs = []
    for _, group in df.groupby('bin', observed=False):
        resampled = resample(group, replace=True, n_samples=max_count, random_state=random_state)
        resampled_dfs.append(resampled)

    df_resampled = pd.concat(resampled_dfs).drop(columns='bin')

    return df_resampled.sample(frac=1, random_state=random_state)  # Shuffle

def predict_with_predicted_sciplex_profiles_CV(eval_function=predictions_indepedent_test_set_twopart,train_on_predicted=False,two_part_bool=True, results_file='sciplex_regression_predictions_CV_resampling'):
    sensitivities = get_sciplex_AUCs()
    #Observations
    mean_pre_observed, mean_post_observed, mean_lfc_observed = get_sciplex_mean_data()
    mean_pre_observed = pd.merge(mean_pre_observed, sensitivities, on=['condition', 'cell_line'], how='left').dropna()
    mean_post_observed = pd.merge(mean_post_observed, sensitivities, on=['condition', 'cell_line'], how='left').dropna()
    mean_lfc_observed = pd.merge(mean_lfc_observed, sensitivities, on=['condition', 'cell_line'], how='left').dropna()

    mean_pre_observed.index = mean_pre_observed['condition'] + '_' + mean_pre_observed['cell_line']
    mean_pre_observed = mean_pre_observed.sort_index()
    mean_post_observed.index = mean_post_observed['condition'] + '_' + mean_post_observed['cell_line']
    mean_post_observed = mean_post_observed.sort_index()
    mean_lfc_observed.index = mean_lfc_observed['condition'] + '_' + mean_lfc_observed['cell_line']
    mean_lfc_observed = mean_lfc_observed.sort_index()

    #Feature selection
    pre_data = get_sciplex_pre_treatment_data()
    features = feature_selection(pre_data, n_features=1000)

    regression_predictions = []
    classification_predictions = []
    mean_post_observed['binned'] = pd.qcut(mean_post_observed['y'], q=10, duplicates='drop', labels=False)

    # kf = KFold(n_splits=10, shuffle=True, random_state=1)
    skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

    print("Running 5-fold cross-validation...")
    for fold, (train_idx, test_idx) in enumerate(skf.split(mean_post_observed, mean_post_observed['binned'])):
        print(f"Fold {fold+1}")

        train_idx_labels = mean_post_observed.iloc[train_idx].index
        test_idx_labels = mean_post_observed.iloc[test_idx].index
        X_train_pre_observed = mean_pre_observed.loc[train_idx_labels]
        X_train_pre_observed_resampled = resample_by_bins(X_train_pre_observed, n_bins=10)
        X_test_pre_observed = mean_pre_observed.loc[test_idx_labels]

        X_train_post_observed = mean_post_observed.loc[train_idx_labels]
        X_train_post_observed_resampled = resample_by_bins(X_train_post_observed, n_bins=10)
        X_test_post_observed = mean_post_observed.loc[test_idx_labels]
        
        X_train_lfc_observed = mean_lfc_observed.loc[train_idx_labels]
        X_train_lfc_observed_resampled = resample_by_bins(X_train_lfc_observed, n_bins=10)
        X_test_lfc_observed = mean_lfc_observed.loc[test_idx_labels]    

        #Observation predictions
        pre_observation_predictions = eval_function(X_train_pre_observed_resampled,X_test_pre_observed,features, two_part=two_part_bool, save_feature_importance=True, prefix=f'pre_{results_file}_{fold}').assign(split=fold).assign(train_set='observed',test_set='observed').assign(model='pre_treatment')
        regression_predictions.append(pre_observation_predictions)
        post_observation_predictions = eval_function(X_train_post_observed_resampled,X_test_post_observed,features, two_part=two_part_bool, save_feature_importance=True, prefix=f'post_{results_file}_{fold}').assign(split=fold).assign(train_set='observed',test_set='observed').assign(model='post_treatment')
        regression_predictions.append(post_observation_predictions)
        lfc_observation_predictions = eval_function(X_train_lfc_observed_resampled,X_test_lfc_observed,features, two_part=two_part_bool, save_feature_importance=True,prefix=f'lfc_{results_file}_{fold}').assign(split=fold).assign(train_set='observed',test_set='observed').assign(model='LFC')
        regression_predictions.append(lfc_observation_predictions)

        pre_observation_predictions_classification = predictions_indepedent_test_set_classification(X_train_pre_observed_resampled, X_test_pre_observed, features).assign(split=fold).assign(test_set='observed',train_set='observed').assign(model='pre_treatment')
        classification_predictions.append(pre_observation_predictions_classification)
        post_observation_predictions_classification = predictions_indepedent_test_set_classification(X_train_post_observed_resampled, X_test_post_observed, features).assign(split=fold).assign(test_set='observed',train_set='observed').assign(model='post_treatment')
        classification_predictions.append(post_observation_predictions_classification)
        lfc_observation_predictions_classification = predictions_indepedent_test_set_classification(X_train_lfc_observed_resampled, X_test_lfc_observed, features).assign(split=fold).assign(test_set='observed',train_set='observed').assign(model='LFC')
        classification_predictions.append(lfc_observation_predictions_classification)
        
        #No effect predictions
        no_effect_predictions = get_no_effect_predictions()
        no_effect_predictions = pd.merge(no_effect_predictions, sensitivities, on=['condition', 'cell_line'], how='left').dropna()
        no_effect_predictions.index = no_effect_predictions['condition'] + '_' + no_effect_predictions['cell_line']
        no_effect_predictions = no_effect_predictions.sort_index()
        X_train_no_effect_post = no_effect_predictions.loc[train_idx_labels]
        X_train_no_effect_post_resampled = resample_by_bins(X_train_no_effect_post, n_bins=10)
        X_test_no_effect_post = no_effect_predictions.loc[test_idx_labels]

        # LFC of no effect is always 0, add for completion
        X_train_no_effect_lfc_resampled = pd.DataFrame(0, index=X_train_no_effect_post_resampled.index, columns=X_train_no_effect_post_resampled.columns)
        X_train_no_effect_lfc_resampled['y'] = X_train_no_effect_post_resampled['y']
        X_test_no_effect_lfc = pd.DataFrame(0, index=X_test_no_effect_post.index, columns=X_test_no_effect_post.columns)
        X_test_no_effect_lfc['y'] = X_test_no_effect_post['y']

        if train_on_predicted:
            post_train_df = X_train_no_effect_post_resampled.copy()
            lfc_train_df = X_train_no_effect_lfc_resampled.copy()
            save_feature_importance_bool = True
            train_set = 'No effect'
        else:
            post_train_df = X_train_post_observed_resampled.copy()
            lfc_train_df = X_train_lfc_observed_resampled.copy()
            save_feature_importance_bool = False
            train_set = 'observed'

        post_no_effect_predictions = eval_function(post_train_df,X_test_no_effect_post,features, two_part=two_part_bool, save_feature_importance=save_feature_importance_bool, prefix=f'post_{results_file}_no_effect_{fold}').assign(split=fold).assign(test_set='No effect',train_set=train_set).assign(model='post_treatment')
        regression_predictions.append(post_no_effect_predictions)
        if train_on_predicted:
            # Also evaluate on observed test set
            post_no_effect_predictions_on_observed = eval_function(post_train_df,X_test_post_observed,features, two_part=two_part_bool, save_feature_importance=False, prefix=f'post_{results_file}_no_effect_{fold}').assign(split=fold).assign(test_set='observed',train_set=train_set).assign(model='post_treatment')
            regression_predictions.append(post_no_effect_predictions_on_observed)
        lfc_no_effect_predictions = eval_function(lfc_train_df,X_test_no_effect_lfc,features, two_part=two_part_bool, save_feature_importance=save_feature_importance_bool, prefix=f'lfc_{results_file}_no_effect_{fold}').assign(split=fold).assign(test_set='No effect',train_set=train_set).assign(model='LFC')
        regression_predictions.append(lfc_no_effect_predictions)
        if train_on_predicted:
            lfc_no_effect_predictions_on_observed = eval_function(lfc_train_df,X_test_lfc_observed,features, two_part=two_part_bool, save_feature_importance=False, prefix=f'lfc_{results_file}_no_effect_{fold}').assign(split=fold).assign(test_set='observed',train_set=train_set).assign(model='LFC')
            regression_predictions.append(lfc_no_effect_predictions_on_observed)
        
        post_no_effect_predictions_classification = predictions_indepedent_test_set_classification(post_train_df, X_test_no_effect_post, features).assign(split=fold).assign(test_set='No effect', train_set=train_set).assign(model='post_treatment')
        classification_predictions.append(post_no_effect_predictions_classification)
        if train_on_predicted:
            post_no_effect_predictions_classification_on_observed = predictions_indepedent_test_set_classification(post_train_df, X_test_post_observed, features).assign(split=fold).assign(test_set='observed', train_set=train_set).assign(model='post_treatment')
            classification_predictions.append(post_no_effect_predictions_classification_on_observed)
        lfc_no_effect_predictions_classification = predictions_indepedent_test_set_classification(lfc_train_df, X_test_no_effect_lfc, features).assign(split=fold).assign(test_set='No effect', train_set=train_set).assign(model='LFC')
        classification_predictions.append(lfc_no_effect_predictions_classification)
        if train_on_predicted:
            lfc_no_effect_predictions_classification_on_observed = predictions_indepedent_test_set_classification(lfc_train_df, X_test_lfc_observed, features).assign(split=fold).assign(test_set='observed', train_set=train_set).assign(model='LFC')
            classification_predictions.append(lfc_no_effect_predictions_classification_on_observed)

        #Predicted profiles
        predictions = {'CPA':get_CPA_predictions, 
                    'GEARS':get_GEARS_predictions,
                    'GEARS_noreg':get_GEARS_noreg_predictions,
                    'scFoundation':get_scfoundation_predictions,
                    'average_effect':get_average_effect_predictions}
        for model_name, get_data_func in predictions.items():
            print(model_name)
            try:
                post_predicted, lfc_predicted = get_data_func()
            except FileNotFoundError as e:
                logger.warning(f"Skipping {model_name}: missing prediction input file ({e})")
                continue

            post_predicted = pd.merge(post_predicted, sensitivities, on=['cell_line', 'condition'], how='left').dropna()
            lfc_predicted = pd.merge(lfc_predicted, sensitivities, on=['cell_line', 'condition'], how='left').dropna()

            post_predicted.index = post_predicted['condition'] + '_' + post_predicted['cell_line']
            post_predicted = post_predicted.sort_index()
            X_train_post_predictions = post_predicted.loc[train_idx_labels]
            X_train_post_predictions_resampled = resample_by_bins(X_train_post_predictions, n_bins=10)
            X_test_post_predictions = post_predicted.loc[test_idx_labels]
            if train_on_predicted:
                post_train_df = X_train_post_predictions_resampled.copy()
                save_feature_importance_bool = True
                train_set = model_name
            else:
                post_train_df = X_train_post_observed_resampled.copy()
                save_feature_importance_bool = False
                train_set = 'observed'

            post_predictions = eval_function(post_train_df,X_test_post_predictions,features, two_part=two_part_bool, save_feature_importance=save_feature_importance_bool, prefix=f'post_{results_file}_{model_name}_{fold}').assign(split=fold).assign(test_set=model_name, train_set=train_set).assign(model='post_treatment')
            regression_predictions.append(post_predictions)
            if train_on_predicted:
                post_predictions_on_observed = eval_function(post_train_df,X_test_post_observed,features, two_part=two_part_bool, save_feature_importance=False, prefix=f'post_{results_file}_{model_name}_{fold}').assign(split=fold).assign(test_set='observed', train_set=train_set).assign(model='post_treatment')
                regression_predictions.append(post_predictions_on_observed)

            post_predictions_classification = predictions_indepedent_test_set_classification(post_train_df, X_test_post_predictions, features).assign(split=fold).assign(test_set=model_name, train_set=train_set).assign(model='post_treatment')
            classification_predictions.append(post_predictions_classification)
            if train_on_predicted:
                post_predictions_classification_on_observed = predictions_indepedent_test_set_classification(post_train_df, X_test_post_observed, features).assign(split=fold).assign(test_set='observed', train_set=train_set).assign(model='post_treatment')
                classification_predictions.append(post_predictions_classification_on_observed)

            lfc_predicted.index = lfc_predicted['condition'] + '_' + lfc_predicted['cell_line']
            lfc_predicted = lfc_predicted.sort_index()
            X_train_lfc_predictions = lfc_predicted.loc[train_idx_labels]
            X_train_lfc_predictions_resampled = resample_by_bins(X_train_lfc_predictions, n_bins=10)
            X_test_lfc_predictions = lfc_predicted.loc[test_idx_labels]
            if train_on_predicted:
                lfc_train_df = X_train_lfc_predictions_resampled.copy()
                save_feature_importance_bool = True
                train_set = model_name
            else:
                lfc_train_df = X_train_lfc_observed_resampled.copy()
                save_feature_importance_bool = False
                train_set = 'observed'

            lfc_predictions = eval_function(lfc_train_df,X_test_lfc_predictions,features, two_part=two_part_bool, save_feature_importance=save_feature_importance_bool, prefix=f'lfc_{results_file}_{model_name}_{fold}').assign(split=fold).assign(test_set=model_name, train_set=train_set).assign(model='LFC')
            regression_predictions.append(lfc_predictions)
            if train_on_predicted:
                lfc_predictions_on_observed = eval_function(lfc_train_df,X_test_lfc_observed,features, two_part=two_part_bool, save_feature_importance=False, prefix=f'lfc_{results_file}_{model_name}_{fold}').assign(split=fold).assign(test_set='observed', train_set=train_set).assign(model='LFC')
                regression_predictions.append(lfc_predictions_on_observed)
            lfc_predictions_classification = predictions_indepedent_test_set_classification(lfc_train_df, X_test_lfc_predictions, features).assign(split=fold).assign(test_set=model_name, train_set=train_set).assign(model='LFC')
            classification_predictions.append(lfc_predictions_classification)
            if train_on_predicted:
                lfc_predictions_classification_on_observed = predictions_indepedent_test_set_classification(lfc_train_df, X_test_lfc_observed, features).assign(split=fold).assign(test_set='observed', train_set=train_set).assign(model='LFC')
                classification_predictions.append(lfc_predictions_classification_on_observed)

    regression_predictions = pd.concat(regression_predictions)
    regression_predictions.to_csv(os.path.join(results_dir,f'{results_file}.csv'))

    classification_predictions = pd.concat(classification_predictions)
    classification_predictions.to_csv(os.path.join(results_dir,f'{results_file}_classification.csv'))

if __name__ == '__main__':
    log_script_start(__file__, logger)
    
    try:
        logger.info(f"Project root: {home_dir}")
        logger.info(f"Results directory: {results_dir}")
        logger.info(f"Resources directory: {resources_dir}")

        # Run prediction analyses
        # logger.info("Starting drug sensitivity prediction pipeline...")

        # logger.info("Predicting with sciplex profiles...")
        # predict_with_predicted_sciplex_profiles_CV(eval_function=predictions_indepedent_test_set_twopart, train_on_predicted=False, two_part_bool=True, results_file='sciplex_regression_predictions_CV_twostep_AUC')
        # predict_with_predicted_sciplex_profiles_CV(eval_function=predictions_indepedent_test_set_twopart, train_on_predicted=True, two_part_bool=True, results_file='sciplex_regression_predictions_selftrained_CV_twostep_AUC')

        # logger.info("Running per-treatment models...")
        # predictions_McFarland_per_treatment()
        
        # logger.info("Running per-tissue models...")
        # predictions_McFarland_per_tissue()

        # logger.info("Running full McFarland models...")
        # predictions_McFarland()
        
        # logger.info("Running seen-seen analysis...")
        # predictions_McFarland_seen_seen()
        
        # logger.info("Running leave-drug-out analysis...")
        # predictions_McFarland_LDO()
        
        # logger.info("Running leave-tissue-out analysis...")
        # predictions_McFarland_LTO()
        
        # logger.info("Running leave-one-out analysis...")
        # predictions_McFarland_LOO()

        predict_McFarland_with_CPA_profiles_CV()

        logger.info("Drug sensitivity prediction pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in drug sensitivity prediction: {str(e)}")
        raise
    
    finally:
        log_script_end(__file__, logger)