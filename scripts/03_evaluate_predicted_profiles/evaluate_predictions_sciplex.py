import os
import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import os

from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config, setup_project
from utils import setup_logging_for_script, log_script_start, log_script_end, ensure_directories_exist
from compute_de_genes import main as compute_de_genes_main

# Setup project and logging
setup_project()
logger = setup_logging_for_script(__file__)

# Set project directories using configuration
home_dir = str(config.PROJECT_ROOT)
data_dir = str(config.DATA_DIR)
resources_dir = str(config.RESOURCES_DIR)
results_dir = str(config.RESULTS_03_DIR)
figures_dir = str(config.FIGURES_03_DIR)

# Ensure directories exist
ensure_directories_exist(home_dir,data_dir,resources_dir,results_dir, figures_dir)


def transform_to_df(adata, mean=True):
    adata_df = adata.to_df()
    adata_df.index = adata.obs['CPA_cat']
    adata_df.colmns = adata.var.index
    adata_df.columns = adata_df.columns.str.split('.').str[0]

    if mean:
        adata_df = adata_df.groupby(adata_df.index).mean()
        adata_df.reset_index(inplace=True)
        
    adata_df[['cell_type', 'tissue', 'perturbation']] = adata_df['CPA_cat'].str.split('_', expand=True)
    
    return adata_df

def compute_metrics_from_means(predictions, observations, de_genes, plot=False):
    metrics = {}
    metrics_pert = {}

    metric2fct = {
           'mse': mse,
           'pearson': pearsonr,
    }
    
    for m in metric2fct.keys():
        metrics[m] = []
        metrics[m + '_de'] = []

    num_perts = len(np.unique(predictions['condition']))
    num_cols = min(4, num_perts+1)
    num_rows = int(np.ceil(num_perts / num_cols))
    
    if plot:
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(3*num_cols, 3 * num_rows))
        if num_perts > 1:
            axes = axes.flatten()
    
    for i, pert in enumerate(np.unique(predictions['condition'])):
        print('Evaluating:', pert)

        metrics_pert[pert] = {}
        for m, fct in metric2fct.items():
            predicted_expression = predictions[predictions['condition'] == pert].drop(columns=['condition','cell_type']).iloc[0,]
            true_expression = observations[observations['condition'] == pert].drop(columns=['condition','cell_type']).iloc[0,]
            if m == 'pearson':
                val = fct(predicted_expression, true_expression)[0]
                if np.isnan(val):
                    val = 0
                if plot:
                    axes[i].scatter(true_expression, predicted_expression, label='gene')
                    pert_title = pert.removesuffix('_+ctrl')

                    axes[i].set_title(f'Perturbation: {pert_title}', fontsize=10)

                    axes[i].set_ylabel('Predicted log-normalized expression', fontsize=10)
                    axes[i].set_xlabel('True log-normalized expression', fontsize=10)

            else:
                val = fct(predicted_expression, true_expression)

            metrics_pert[pert][m] = val
            metrics[m].append(metrics_pert[pert][m])

        if pert != 'ctrl':
            for m, fct in metric2fct.items():
                de_gene_subset = de_genes[(de_genes['condition'] == pert)].drop(columns=['cell_type', 'condition'])
                de_gene_subset = de_gene_subset.iloc[0,0:20].tolist()
                predicted_expression = predictions[predictions['condition'] == pert].drop(columns=['condition','cell_type'])[de_gene_subset].iloc[0,]
                true_expression = observations[observations['condition'] == pert].drop(columns=['condition','cell_type'])[de_gene_subset].iloc[0,]
                if m == 'pearson':
                    val = fct(predicted_expression, true_expression)[0]
                    if plot:
                        axes[i].scatter(true_expression, predicted_expression, label='DE gene', s=5)
                        try:
                            axes[i].plot(np.unique(true_expression), 
                                        np.poly1d(np.polyfit(true_expression, predicted_expression, 1))
                                        (np.unique(true_expression)), color='red')
                        except:
                            continue
                        axes[i].text(0.1, 0.9, f'Pearson r: {val:.2f}', transform=axes[i].transAxes)
                        axes[i].set_xlim(0, 6)
                        axes[i].set_ylim(0, 6)
                        axes[i].legend(loc='lower right')

                    if np.isnan(val):
                        val = 0
                if m == 'mse':
                    val = fct(predicted_expression, true_expression)
                    if plot:
                        axes[i].text(0.1, 0.8, f'MSE: {val:.2f}', transform=axes[i].transAxes)

                    
                metrics_pert[pert][m + '_de'] = val
                metrics[m + '_de'].append(metrics_pert[pert][m + '_de'])

        else:
            for m, fct in metric2fct.items():
                metrics_pert[pert][m + '_de'] = 0
    
    for m in metric2fct.keys():
        metrics[m] = np.mean(metrics[m])
        metrics[m + '_de'] = np.mean(metrics[m + '_de'])
    
    if plot:
        plt.tight_layout()
        plt.show()
    
    return metrics, metrics_pert


def compute_metrics_cpa(predictions, observations, de_genes, plot=False):
    metrics = {}
    metrics_pert = {}

    metric2fct = {
           'mse': mse,
           'pearson': pearsonr,
    }
    
    for m in metric2fct.keys():
        metrics[m] = []
        metrics[m + '_de'] = []

    unique_perts = list(set(observations['drug'].unique()).intersection(predictions.index))
    num_perts = len(unique_perts)
    num_cols = min(4, num_perts+1)
    num_rows = int(np.ceil(num_perts / num_cols))
    
    if plot:
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(3*num_cols, 3 * num_rows))
        if num_perts > 1:
            axes = axes.flatten()
    
    for i, pert in enumerate(unique_perts):
        print('Evaluating:', pert)

        metrics_pert[pert] = {}
        for m, fct in metric2fct.items():
            print(m)
            predicted_expression = predictions.loc[pert]
            true_expression = observations.loc[pert]
            common_columns = list(set(predictions.columns).intersection(observations.columns))
            common_columns.remove('cell_type')
            predicted_expression = list(predicted_expression[common_columns])
            true_expression = list(true_expression[common_columns])

            if m == 'pearson':
                val = fct(predicted_expression, true_expression)[0]
                if np.isnan(val):
                    val = 0
                if plot:
                    axes[i].scatter(true_expression, predicted_expression, label='gene')
                    pert_title = pert.removesuffix('_+ctrl')

                    axes[i].set_title(f'Perturbation: {pert_title}', fontsize=10)

                    axes[i].set_ylabel('Predicted log-normalized expression', fontsize=10)
                    axes[i].set_xlabel('True log-normalized expression', fontsize=10)

            else:
                val = fct(predicted_expression, true_expression)

            metrics_pert[pert][m] = val
            metrics[m].append(metrics_pert[pert][m])

        if pert != 'DMSO':
            for m, fct in metric2fct.items():
                de_genes_pert = de_genes[de_genes['drug'] == pert].values[0]
                # de_genes_pert = list(set(de_genes_pert.tolist()).intersection(results.columns.tolist()))
                de_genes_pert = de_genes_pert[1:21]
                # de_genes_pert = list(set(de_genes_pert.tolist()).intersection(results.columns.tolist()))
                print(len(set(de_genes_pert).intersection(predictions.columns.tolist())))
                predicted_expression = list(predictions.loc[pert][de_genes_pert])
                true_expression = list(observations.loc[pert][de_genes_pert])
                if m == 'pearson':
                    val = fct(predicted_expression, true_expression)[0]
                    if plot:
                        axes[i].scatter(true_expression, predicted_expression, label='DE gene', s=5)
                        axes[i].plot(np.unique(true_expression), 
                                    np.poly1d(np.polyfit(true_expression, predicted_expression, 1))
                                    (np.unique(true_expression)), color='red')
                        axes[i].text(0.1, 0.9, f'Pearson r: {val:.2f}', transform=axes[i].transAxes)
                        axes[i].set_xlim(0, 6)
                        axes[i].set_ylim(0, 6)
                        axes[i].legend(loc='lower right')

                    if np.isnan(val):
                        val = 0
                if m == 'mse':
                    val = fct(predicted_expression, true_expression)
                    if plot:
                        axes[i].text(0.1, 0.8, f'MSE: {val:.2f}', transform=axes[i].transAxes)

                    
                metrics_pert[pert][m + '_de'] = val
                metrics[m + '_de'].append(metrics_pert[pert][m + '_de'])

        else:
            for m, fct in metric2fct.items():
                metrics_pert[pert][m + '_de'] = 0
    
    for m in metric2fct.keys():
        metrics[m] = np.mean(metrics[m])
        metrics[m + '_de'] = np.mean(metrics[m + '_de'])
    
    if plot:
        plt.tight_layout()
        plt.show()
    
    return metrics, metrics_pert

def evaluate_average_effect_predictions():
    for cell_line in ['mcf7','a549','k562']:
        average_effect_predictions =  pd.read_csv(os.path.join(data_dir, 'average_effect_predictions', f'sciplex_mean_post_{cell_line}.csv'))
        observations = pd.read_csv(os.path.join(data_dir, 'observed_pseudobulk', f'sciplex_mean_post_{cell_line}.csv'), index_col=0)
        de_genes = pd.read_csv(os.path.join(results_dir, f'sciplex{cell_line}_de_genes.csv'))
        pert_to_drug = pd.read_csv(os.path.join(resources_dir, 'sciplex_drug_to_perturbation.csv'),index_col=0)
        pert_to_drug['product_name'] = pert_to_drug['product_name'].str.replace(' ', '')

        de_genes = pd.merge(de_genes, pert_to_drug, left_on=['condition'],right_on=['product_name'], how='left').drop(columns=['product_name', 'condition']).rename(columns={'target':'condition'})
        de_genes['cell_type'] = cell_line.upper()

        metrics, pert_metrics = compute_metrics_from_means(average_effect_predictions, observations, de_genes, plot=False)
        pert_metrics = pd.DataFrame.from_dict(pert_metrics, orient='index')
        pert_metrics['condition'] = pert_metrics.index
        pert_metrics = pert_metrics.melt(id_vars='condition', var_name='metric', value_name='value')
        pert_metrics['cell_type'] = cell_line.upper()
        pert_metrics['model'] = 'Average effect'

        pert_metrics.to_csv(os.path.join(results_dir, f'sciplex{cell_line}_average_effect_outcomes.csv'), index=False)

def evaluate_no_effect_predictions():
    # We take pre-treatment profiles as no-effect predictions
    for cell_line in ['mcf7','a549','k562']:
        no_effect_predictions =  pd.read_csv(os.path.join(data_dir, 'no_effect_predictions', f'sciplex_mean_post_{cell_line}.csv'))
        observations = pd.read_csv(os.path.join(data_dir, 'observed_pseudobulk', f'sciplex_mean_post_{cell_line}.csv'), index_col=0)
        de_genes = pd.read_csv(os.path.join(results_dir, f'sciplex{cell_line}_de_genes.csv'))
        pert_to_drug = pd.read_csv(os.path.join(resources_dir, 'sciplex_drug_to_perturbation.csv'),index_col=0)
        pert_to_drug['product_name'] = pert_to_drug['product_name'].str.replace(' ', '')

        de_genes = pd.merge(de_genes, pert_to_drug, left_on=['condition'],right_on=['product_name'], how='left').drop(columns=['product_name', 'condition']).rename(columns={'target':'condition'})
        de_genes['cell_type'] = cell_line.upper()

        metrics, pert_metrics = compute_metrics_from_means(no_effect_predictions, observations, de_genes, plot=False)
        pert_metrics = pd.DataFrame.from_dict(pert_metrics, orient='index')
        pert_metrics['condition'] = pert_metrics.index
        pert_metrics = pert_metrics.melt(id_vars='condition', var_name='metric', value_name='value')
        pert_metrics['cell_type'] = cell_line.upper()
        pert_metrics['model'] = 'No effect'

        pert_metrics.to_csv(os.path.join(results_dir, f'sciplex{cell_line}_no_effect_outcomes.csv'), index=False)

def evaluate_GEARS_predictions():
    for cell_line in ['mcf7','a549','k562']:
        predictions =  pd.read_csv(os.path.join(data_dir, 'GEARS_predictions', f'sciplex_mean_post_{cell_line}.csv'), index_col=0)
        predictions = predictions.rename(columns={'perturbation':'condition'})
        predictions = predictions[predictions['condition'] != 'ctrl']
        observations = pd.read_csv(os.path.join(data_dir, 'observed_pseudobulk', f'sciplex_mean_post_{cell_line}.csv'), index_col=0)
        de_genes = pd.read_csv(os.path.join(results_dir, f'sciplex{cell_line}_de_genes.csv'))
        pert_to_drug = pd.read_csv(os.path.join(resources_dir, 'sciplex_drug_to_perturbation.csv'),index_col=0)
        pert_to_drug['product_name'] = pert_to_drug['product_name'].str.replace(' ', '')

        de_genes = pd.merge(de_genes, pert_to_drug, left_on=['condition'],right_on=['product_name'], how='left').drop(columns=['product_name', 'condition']).rename(columns={'target':'condition'})
        de_genes['cell_type'] = cell_line.upper()

        metrics, pert_metrics = compute_metrics_from_means(predictions, observations, de_genes, plot=False)
        pert_metrics = pd.DataFrame.from_dict(pert_metrics, orient='index')
        pert_metrics['condition'] = pert_metrics.index
        pert_metrics = pert_metrics.melt(id_vars='condition', var_name='metric', value_name='value')
        pert_metrics['cell_type'] = cell_line.upper()
        pert_metrics['model'] = 'GEARS'

        pert_metrics.to_csv(os.path.join(results_dir, f'sciplex{cell_line}_GEARS_outcomes.csv'), index=False)

def evaluate_GEARS_noreg_predictions():
    for cell_line in ['mcf7','a549','k562']:
        predictions =  pd.read_csv(os.path.join(data_dir, 'GEARS_noreg_predictions', f'sciplex_mean_post_{cell_line}.csv'), index_col=0)
        predictions = predictions.rename(columns={'perturbation':'condition'})
        predictions = predictions[predictions['condition'] != 'ctrl']
        observations = pd.read_csv(os.path.join(data_dir, 'observed_pseudobulk', f'sciplex_mean_post_{cell_line}.csv'), index_col=0)
        de_genes = pd.read_csv(os.path.join(results_dir, f'sciplex{cell_line}_de_genes.csv'))

        pert_to_drug = pd.read_csv(os.path.join(resources_dir, 'sciplex_drug_to_perturbation.csv'),index_col=0)
        pert_to_drug['product_name'] = pert_to_drug['product_name'].str.replace(' ', '')

        de_genes = pd.merge(de_genes, pert_to_drug, left_on=['condition'],right_on=['product_name'], how='left').drop(columns=['product_name', 'condition']).rename(columns={'target':'condition'})
        de_genes['cell_type'] = cell_line.upper()

        metrics, pert_metrics = compute_metrics_from_means(predictions, observations, de_genes, plot=False)
        pert_metrics = pd.DataFrame.from_dict(pert_metrics, orient='index')
        pert_metrics['condition'] = pert_metrics.index
        pert_metrics = pert_metrics.melt(id_vars='condition', var_name='metric', value_name='value')
        pert_metrics['cell_type'] = cell_line.upper()
        pert_metrics['model'] = 'GEARS_noreg'

        pert_metrics.to_csv(os.path.join(results_dir, f'sciplex{cell_line}_GEARS_noreg_outcomes.csv'), index=False)

def evaluate_scfoundation_predictions():
    for cell_line in ['mcf7','a549','k562']:
        predictions =  pd.read_csv(os.path.join(data_dir, 'scfoundation_predictions', f'sciplex_mean_post_{cell_line}.csv'), index_col=0)
        predictions = predictions.rename(columns={'perturbation':'condition'})
        predictions = predictions[predictions['condition'] != 'ctrl']
        observations = pd.read_csv(os.path.join(data_dir, 'observed_pseudobulk', f'sciplex_mean_post_{cell_line}.csv'), index_col=0)
        de_genes = pd.read_csv(os.path.join(results_dir, f'sciplex{cell_line}_de_genes.csv'))

        pert_to_drug = pd.read_csv(os.path.join(resources_dir, 'sciplex_drug_to_perturbation.csv'),index_col=0)
        pert_to_drug['product_name'] = pert_to_drug['product_name'].str.replace(' ', '')

        de_genes = pd.merge(de_genes, pert_to_drug, left_on=['condition'],right_on=['product_name'], how='left').drop(columns=['product_name', 'condition']).rename(columns={'target':'condition'})
        de_genes['cell_type'] = cell_line.upper()

        metrics, pert_metrics = compute_metrics_from_means(predictions, observations, de_genes, plot=False)
        pert_metrics = pd.DataFrame.from_dict(pert_metrics, orient='index')
        pert_metrics['condition'] = pert_metrics.index
        pert_metrics = pert_metrics.melt(id_vars='condition', var_name='metric', value_name='value')
        pert_metrics['cell_type'] = cell_line.upper()
        pert_metrics['model'] = 'scfoundation'

        pert_metrics.to_csv(os.path.join(results_dir, f'sciplex{cell_line}_scfoundation_outcomes.csv'), index=False)

def evaluate_CPA_predictions():
    for cell_line in ['mcf7','a549','k562']:
        cpa_predictions = pd.read_csv(os.path.join(data_dir, 'CPA_predictions', f'sciplex_mean_post_{cell_line}.csv'))
        cpa_predictions = cpa_predictions[cpa_predictions['condition'] != 'ctrl']
        true_expression = pd.read_csv(os.path.join(data_dir, 'observed_pseudobulk', f'sciplex_mean_post_{cell_line}.csv'), index_col=0)
        de_genes = pd.read_csv(os.path.join(results_dir, f'sciplex{cell_line}_de_genes.csv'))

        pert_to_drug = pd.read_csv(os.path.join(resources_dir, 'sciplex_drug_to_perturbation.csv'),index_col=0)
        pert_to_drug['product_name'] = pert_to_drug['product_name'].str.replace(' ', '')

        print(cpa_predictions.head())
        print(cpa_predictions.columns)

        de_genes = pd.merge(de_genes, pert_to_drug, left_on=['condition'],right_on=['product_name'], how='left').drop(columns=['product_name', 'condition']).rename(columns={'target':'condition'})
        de_genes['cell_type'] = cell_line.upper()       

        metrics, pert_metrics = compute_metrics_from_means(cpa_predictions, true_expression, de_genes, plot=False)
        pert_metrics = pd.DataFrame.from_dict(pert_metrics, orient='index')
        pert_metrics['condition'] = pert_metrics.index
        pert_metrics = pert_metrics.melt(id_vars='condition', var_name='metric', value_name='value')
        pert_metrics['cell_type'] = cell_line.upper()
        pert_metrics['model'] = 'scfoundation'

        pert_metrics.to_csv(os.path.join(results_dir, f'sciplex{cell_line}_CPA_outcomes.csv'), index=False)
        

if __name__ == '__main__':
    log_script_start(__file__, logger)
    try:
        # Check if DE genes have been computed
        for cell_line in ['mcf7','a549','k562']:
            de_genes = pd.read_csv(os.path.join(results_dir, f'sciplex{cell_line}_de_genes.csv'))
    except Exception as e:
        logger.info(f"DE genes have not been computed yet")
        logger.info("Computing DE genes...")
        compute_de_genes_main()
    try:
        logger.info("Evaluating average effect predictions...")
        evaluate_average_effect_predictions()
        logger.info("Evaluating no effect predictions...")
        evaluate_no_effect_predictions()
        logger.info("Evaluating GEARS predictions...")
        evaluate_GEARS_predictions()
        logger.info("Evaluating GEARS noreg predictions...")
        evaluate_GEARS_noreg_predictions()
        logger.info("Evaluating scfoundation predictions...")
        evaluate_scfoundation_predictions()
        logger.info("Evaluating CPA predictions...")
        evaluate_CPA_predictions()
        logger.info("Evaluation completed successfully")
    except Exception as e:
        logger.error(f"Error in evaluate_predictions_sciplex: {str(e)}")
        raise
    log_script_end(__file__, logger)