import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import os

from config import config

# Set project directories using configuration
home_dir = str(config.PROJECT_ROOT)
data_dir = str(config.DATA_DIR)
resources_dir = str(config.RESOURCES_DIR)
results_dir = str(config.RESULTS_04_DIR)
figures_dir = str(config.FIGURES_04_DIR)

def get_predictions():
    average_effect_predictions = []
    no_effect_predictions = []
    GEARSnoreg_predictions = []
    GEARS_predictions = []
    scfoundation_predictions = []
    CPA_predictions = []
    observations = []

    for cell_line in ['mcf7', 'a549', 'k562']:
        print(f'Processing cell line: {cell_line}')

        average_effect_subset = pd.read_csv(os.path.join(data_dir, 'average_effect_predictions', f'sciplex_mean_post_{cell_line}.csv')).rename(columns={'cell_type': 'cell_line'})
        average_effect_predictions.append(average_effect_subset)

        no_effect_subset = pd.read_csv(os.path.join(data_dir, 'no_effect_predictions', f'sciplex_mean_post_{cell_line}.csv')).rename(columns={'cell_type': 'cell_line'})
        no_effect_predictions.append(no_effect_subset)

        GEARSnoreg_predictions.append(pd.read_csv(os.path.join(data_dir, 'GEARS_noreg_predictions', f'sciplex_mean_post_{cell_line}.csv'), index_col=0).rename(columns={'perturbation': 'condition'}).assign(cell_line=cell_line.upper()))
        GEARS_predictions.append(pd.read_csv(os.path.join(data_dir, 'GEARS_predictions', f'sciplex_mean_post_{cell_line}.csv'), index_col=0).rename(columns={'perturbation': 'condition'}).assign(cell_line=cell_line.upper()).drop(columns=['cell_type']))
        scfoundation_predictions.append(pd.read_csv(os.path.join(data_dir, 'scfoundation_predictions', f'sciplex_mean_post_{cell_line}.csv'), index_col=0).rename(columns={'perturbation': 'condition'}).assign(cell_line=cell_line.upper()).drop(columns=['cell_type']))
        CPA_predictions.append(pd.read_csv(os.path.join(data_dir, 'CPA_predictions', f'sciplex_mean_post_{cell_line}.csv')).rename(columns={'cell_type': 'cell_line'}))
        observations.append(pd.read_csv(os.path.join(data_dir, 'observed_pseudobulk', f'sciplex_mean_post_{cell_line}.csv'), index_col=0).assign(cell_line=cell_line.upper()))

    average_effect_predictions = pd.concat(average_effect_predictions, ignore_index=True)
    no_effect_predictions = pd.concat(no_effect_predictions, ignore_index=True)
    GEARSnoreg_predictions = pd.concat(GEARSnoreg_predictions, ignore_index=True).drop(columns=['cell_type'])
    GEARS_predictions = pd.concat(GEARS_predictions, ignore_index=True)
    scfoundation_predictions = pd.concat(scfoundation_predictions, ignore_index=True)
    CPA_predictions = pd.concat(CPA_predictions, ignore_index=True)
    observations = pd.concat(observations, ignore_index=True)

    CPA_predictions = CPA_predictions[CPA_predictions['condition']!= 'ctrl']
    GEARSnoreg_predictions = GEARSnoreg_predictions[GEARSnoreg_predictions['condition']!= 'ctrl']
    GEARS_predictions = GEARS_predictions[GEARS_predictions['condition']!= 'ctrl']
    scfoundation_predictions = scfoundation_predictions[scfoundation_predictions['condition']!= 'ctrl']
    CPA_predictions = CPA_predictions[CPA_predictions['condition']!= 'ctrl']

    all_predictions = {'CPA': CPA_predictions,
                    'GEARS_noreg': GEARSnoreg_predictions,
                    'GEARS': GEARS_predictions,
                    'scfoundation': scfoundation_predictions,
                    'no_effect': no_effect_predictions,
                    'average_effect': average_effect_predictions,
                    'observations': observations}

    return all_predictions


def get_observations():
    sciplex_observations = []
    for cl in ['mcf7', 'a549', 'k562']:
        observations = pd.read_csv(os.path.join(data_dir, 'observed_pseudobulk', f'sciplex_mean_post_{cl}.csv'), index_col=0)
        observations['cell_line'] = cl.upper()
        sciplex_observations.append(observations)

    sciplex_observations = pd.concat(sciplex_observations, ignore_index=True)

    sciplex_sensitivity = pd.read_csv(os.path.join(resources_dir, 'sciplex_sensitivity_info.csv'), index_col=0)
    sciplex_observations = pd.merge(sciplex_observations, sciplex_sensitivity[['cell_line', 'target', 'y']], left_on = ['condition', 'cell_line'], right_on = ['target', 'cell_line'], how='left')

    return sciplex_observations


def compute_correlation_with_y(df, subset = None):
    sciplex_sensitivity = pd.read_csv(os.path.join(resources_dir, 'sciplex_sensitivity_info.csv'), index_col=0)

    if subset is not None:
        subset = list(set(subset).intersection(set(df.columns.tolist())))
        df = df[subset]
    sens_values = sciplex_sensitivity[['cell_line', 'target', 'y']].rename(columns={'cell_line': 'cell_line', 'target': 'condition', 'y': 'sensitivity'})
    df_with_sensitivity = pd.merge(df, sens_values.reset_index(drop=True), on=['cell_line', 'condition'], how='left')
    df_with_sensitivity = df_with_sensitivity.drop(columns=['cell_line', 'condition'])

    return df_with_sensitivity.corrwith(df_with_sensitivity['sensitivity'], method='spearman').drop('sensitivity').fillna(0)


def compute_mse_per_gene(df, subset = None):
    sciplex_observations = get_observations()
    df = df.drop(columns=['cell_line', 'condition'])
    observations = sciplex_observations.drop(columns=['cell_line', 'condition'])

    if subset is not None:
        subset = list(set(subset).intersection(set(df.columns.tolist())))
        df = df[subset]
        observation_subset = observations[subset]

    else:
        subset = list(set(observations.columns.tolist()).intersection(set(df.columns.tolist())))
        df = df[subset]
        observation_subset = observations[subset]
        
    diff_df = df.subtract(observation_subset.values, axis=0)
    diff_df = diff_df**2

    return diff_df.mean()


def compute_mse_per_individual(df, subset = None):
    sciplex_observations = get_observations()
    df = df.set_index(['cell_line', 'condition'])
    observations = sciplex_observations.set_index(['cell_line', 'condition'])

    if subset is not None:
        subset = list(set(subset).intersection(set(df.columns.tolist())))
        df = df[subset]
        observation_subset = observations[subset]

    else:
        subset = list(set(observations.columns.tolist()).intersection(set(df.columns.tolist())))
        df = df[subset]
        observation_subset = observations[subset]

    df = df.T
    observation_subset = observation_subset.T

    diff_df = df.subtract(observation_subset.values, axis=0)

    diff_df = diff_df**2
    diff_df = diff_df.mean()
    # diff_df.index = df.index

    return diff_df


def compute_correlation_per_individual(df, subset=None):
    sciplex_observations = get_observations()
    df = df.set_index(['cell_line', 'condition'])
    observations = sciplex_observations.set_index(['cell_line', 'condition'])

    if subset is not None:
        subset = list(set(subset).intersection(set(df.columns.tolist())))
        df = df[subset]
        observation_subset = observations[subset]
    else:
        subset = list(set(observations.columns.tolist()).intersection(set(df.columns.tolist())))
        df = df[subset]
        observation_subset = observations[subset]

    correlations = df.corrwith(observation_subset, axis=1, method='pearson')
    return correlations.fillna(0)


def compute_gene_gene_correlation(df, subset = None, per_cell_line = True):
    observations = get_observations()
    # subset.remove('cell_line')
    subset.remove('condition')
    subset.remove('y')
    if subset is not None:
        subset = list(set(subset).intersection(set(df.columns.tolist())))
        df = df[subset]
        observation_subset = observations[subset]
    else:
        subset = list(set(observations.columns.tolist()).intersection(set(df.columns.tolist())))
        df = df[subset]
        observation_subset = observations[subset]

    if per_cell_line:
        results = []
        for cell_line in df['cell_line'].unique():
            df_cl = df[df['cell_line'] == cell_line].drop(columns=['cell_line'])
            df_cl_corr = df_cl.corr()
            df_cl_corr.index = pd.MultiIndex.from_product([[cell_line], df_cl_corr.index])
            results.append(df_cl_corr)

        results_df = pd.concat(results, axis=0)
        results_df = results_df.fillna(0)
    else:
        results_df = df.drop(columns=['cell_line']).corr()
        
    return results_df
