import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
import numpy as np
import pickle as pkl

import gseapy as gp
from gseapy import barplot, dotplot

from config import config

# Set project directories using configuration
home_dir = str(config.PROJECT_ROOT)
data_dir = str(config.DATA_DIR)
resources_dir = str(config.RESOURCES_DIR)
results_dir = str(config.RESULTS_04_DIR)
figures_dir = str(config.FIGURES_04_DIR)


def read_pickle(file_path, reset_index=False):
    with open(file_path, 'rb') as file:
        df_dict = pkl.load(file)

    df = []
    for k, v in df_dict.items():
        v = pd.DataFrame(v).assign(condition=k)
        if reset_index:
            v = v.reset_index()
            v.columns = [['feature', 'importance', 'condition']]
        df.append(v)

    df = pd.concat(df)

    return df

def collect_mcfarland_feature_importance():
    mcfarland_drug_to_pert = pd.read_csv(os.path.join(resources_dir, 'mcfarland_drug_to_perturbation.csv'))
    feature_importance_pancancer = pd.read_csv(os.path.join(results_dir, 'mcfarland_regression_1000_features_feature_importance.csv'), index_col=0)
    feature_importance_pancancer = feature_importance_pancancer.assign(condition='Full McFarland dataset')
    feature_importance_pancancer = feature_importance_pancancer.pivot(index=['model','condition','split'], columns='feature', values='importance')

    treatmentmodels_pre = read_pickle(os.path.join(results_dir, 'treatmentmodels_regression_feature_importance_on_pre.pkl')).assign(model='Pre')
    treatmentmodels_post = read_pickle(os.path.join(results_dir, 'treatmentmodels_regression_feature_importance_on_post.pkl')).assign(model='Post')
    treatmentmodels_lfc = read_pickle(os.path.join(results_dir, 'treatmentmodels_regression_feature_importance_on_lfc.pkl')).assign(model='Log(fold change)')

    feature_importance_treatmentmodels = pd.concat([treatmentmodels_pre, treatmentmodels_post, treatmentmodels_lfc], ignore_index=True)
    feature_importance_treatmentmodels = pd.merge(feature_importance_treatmentmodels, mcfarland_drug_to_pert, left_on='condition', right_on='target', how='left')
    feature_importance_treatmentmodels = feature_importance_treatmentmodels.drop(columns=['condition']).rename(columns={'drug':'condition'})
    feature_importance_treatmentmodels = feature_importance_treatmentmodels.pivot(index=['model','condition','split'], columns='feature', values='importance')

    tissuemodels_pre = read_pickle(os.path.join(results_dir, 'tissuemodels_regression_feature_importance_on_pre.pkl')).assign(model='Pre')
    tissuemodels_post = read_pickle(os.path.join(results_dir, 'tissuemodels_regression_feature_importance_on_post.pkl')).assign(model='Post')
    tissuemodels_lfc = read_pickle(os.path.join(results_dir, 'tissuemodels_regression_feature_importance_on_lfc.pkl')).assign(model='Log(fold change)')

    feature_importance_tissuemodels = pd.concat([tissuemodels_pre, tissuemodels_post, tissuemodels_lfc], ignore_index=True)
    feature_importance_tissuemodels = feature_importance_tissuemodels.pivot(index=['model','condition','split'], columns='feature', values='importance')

    feature_importance = pd.concat([feature_importance_pancancer, feature_importance_tissuemodels, feature_importance_treatmentmodels]).fillna(0)

    feature_importance_means = feature_importance.reset_index().groupby(['model', 'condition']).mean().reset_index().drop(columns=['split'])

    return feature_importance_pancancer, feature_importance_tissuemodels, feature_importance_treatmentmodels, feature_importance_means


def get_mcfarland_feature_importance():
    tissue_models = ['BREAST', 'CENTRAL_NERVOUS_SYSTEM', 'KIDNEY', 'LARGE_INTESTINE', 'LUNG', 'OESOPHAGUS', 'PANCREAS', 'SKIN', 'THYROID', 'URINARY_TRACT']
    treatment_models = ['Afatinib', 'Dabrafenib', 'Idasanutlin', 'Taselisib', 'Trametinib']

    feature_importance_pancancer, feature_importance_tissuemodels, feature_importance_treatmentmodels, feature_importance_means = collect_mcfarland_feature_importance()

    top_features = feature_importance_pancancer.columns.tolist()
    top_features = list(set(top_features).intersection(set(feature_importance_tissuemodels.columns.tolist() + feature_importance_treatmentmodels.columns.tolist()))) + ['condition', 'model']
    print(len(top_features))
    feature_importance_means = feature_importance_means.loc[:, top_features]
    feature_importance_means.to_csv(os.path.join(results_dir, 'mcfarland_regression_feature_importance_means.csv'))

    feature_sum = feature_importance_means.reset_index()[['condition', 'model']]
    feature_sum['sum'] = feature_importance_means.reset_index(drop=True).sum(numeric_only=True).reset_index(drop=True)
    feature_sum = feature_sum.sort_values('sum',ascending=True, key=abs)

    zero_feature_conditions = feature_sum[feature_sum['sum']==0]['condition'].tolist()
    good_performance_models = tissue_models + treatment_models + ['Full McFarland dataset']

    feature_importance_means = feature_importance_means[~feature_importance_means['condition'].isin(zero_feature_conditions)]
    feature_importance_means = feature_importance_means[feature_importance_means['condition'].isin(good_performance_models)]
    feature_importance_means.to_csv(os.path.join(results_dir, 'mcfarland_regression_feature_importance_means_filtered.csv'))


def get_sciplex_feature_importance():
    sciplex_feature_importance = []

    for i in range(0,5):
        split_feature_importance = pd.read_csv(os.path.join(results_dir, f'post_sciplex_regression_predictions_CV_twostep_AUC_{i}_twopart_feature_importance.csv'), index_col=0).T.assign(model='Post', condition='Full Sciplex dataset').assign(split=i).reset_index(drop=True)
        split_feature_importance_lfc = pd.read_csv(os.path.join(results_dir, f'LFC_sciplex_regression_predictions_CV_twostep_AUC_{i}_twopart_feature_importance.csv'), index_col=0).T.assign(model='Log(fold change)', condition='Full sciplex dataset').assign(split=i).reset_index(drop=True)
        sciplex_feature_importance.append(split_feature_importance)
        sciplex_feature_importance.append(split_feature_importance_lfc)

    sciplex_feature_importance = pd.concat(sciplex_feature_importance, ignore_index=True)
    sciplex_feature_importance = sciplex_feature_importance.drop(columns=['split']).groupby(['condition','model']).mean().reset_index()
    sciplex_lfc_feature_importance = sciplex_feature_importance[sciplex_feature_importance['model'] == 'Log(fold change)']
    sciplex_post_feature_importance = sciplex_feature_importance[sciplex_feature_importance['model'] == 'Post']

    all_feature_importance = {'post': sciplex_post_feature_importance,
                        'lfc': sciplex_lfc_feature_importance}

    models = ['no_effect', 'average_effect', 'GEARS', 'GEARS_noreg', 'CPA', 'scFoundation']

    post_df = []
    lfc_df = []
    for m in models:
        for i in range(0,5):
            split_feature_importance = pd.read_csv(os.path.join(results_dir, f'post_sciplex_regression_predictions_selftrained_CV_twostep_AUC_{m}_{i}_twopart_feature_importance.csv'), index_col=0).T.assign(model='Post', condition='Full Sciplex dataset', model_type=m).assign(split=i).reset_index(drop=True)
            split_feature_importance_lfc = pd.read_csv(os.path.join(results_dir, f'LFC_sciplex_regression_predictions_selftrained_CV_twostep_AUC_{m}_{i}_twopart_feature_importance.csv'), index_col=0).T.assign(model='Log(fold change)', condition='Full sciplex dataset', model_type=m).assign(split=i).reset_index(drop=True)
            post_df.append(split_feature_importance)
            lfc_df.append(split_feature_importance_lfc)
            
    for i in range(0,5):
            split_feature_importance = pd.read_csv(os.path.join(results_dir, f'post_sciplex_regression_predictions_selftrained_CV_twostep_AUC_{i}_twopart_feature_importance.csv'), index_col=0).T.assign(model='Post', condition='Full Sciplex dataset', model_type='Observed').assign(split=i).reset_index(drop=True)
            split_feature_importance_lfc = pd.read_csv(os.path.join(results_dir, f'LFC_sciplex_regression_predictions_selftrained_CV_twostep_AUC_{i}_twopart_feature_importance.csv'), index_col=0).T.assign(model='Log(fold change)', condition='Full sciplex dataset', model_type='Observed').assign(split=i).reset_index(drop=True)
            post_df.append(split_feature_importance)
            lfc_df.append(split_feature_importance_lfc)

    post_df = pd.concat(post_df, ignore_index=True)
    lfc_df = pd.concat(lfc_df, ignore_index=True)

    post_df = post_df.drop(columns=['split']).groupby(['condition','model', 'model_type']).mean().reset_index()
    lfc_df = lfc_df.drop(columns=['split']).groupby(['condition','model', 'model_type']).mean().reset_index()

    post_df['model_type'] = post_df['model_type'].str.replace('GEARS_noreg', 'GEARS\nOptimized')
    post_df['model_type'] = post_df['model_type'].str.replace('average_effect', 'Average effect')
    post_df['model_type'] = post_df['model_type'].str.replace('no_effect', 'No effect')

    lfc_df['model_type'] = lfc_df['model_type'].str.replace('GEARS_noreg', 'GEARS\nOptimized')
    lfc_df['model_type'] = lfc_df['model_type'].str.replace('average_effect', 'Average effect')
    lfc_df['model_type'] = lfc_df['model_type'].str.replace('no_effect', 'No effect')

    for model_type in post_df['model_type'].unique():
        all_feature_importance[f'post_{model_type}'] = post_df[post_df['model_type']==model_type].drop(columns=['model_type'])
        all_feature_importance[f'lfc_{model_type}'] = lfc_df[lfc_df['model_type']==model_type].drop(columns=['model_type'])

    with open(os.path.join(results_dir,'sciplex_feature_importances.pkl'), 'wb') as f:
        pkl.dump(all_feature_importance, f)

    return all_feature_importance

def perform_mcfarland_GSEA(feature_importance):
    for column_name in feature_importance.iloc[:,1:].columns:
        print(column_name)
        allmodel_lfc = feature_importance[['index', column_name]].sort_values(column_name).reset_index(drop=True).rename(columns={'index': 'gene_name', column_name : 'score'})
        allmodel_lfc = allmodel_lfc[allmodel_lfc['score']!= 0]
        allmodel_lfc.to_csv(os.path.join(results_dir, f'{column_name}_lfc.rnk'), sep="\t", index=False, header=False)
        try:
            pre_res = gp.prerank(
            rnk=os.path.join(results_dir, f'{column_name}_lfc.rnk'),                      # ranked gene list
            gene_sets="MSigDB_Hallmark_2020",                   # or a .gmt file or other gene set name from Enrichr
            outdir="gsea_results",                   # output directory
            permutation_num=1000,                    # number of permutations
            min_size=15,                             # minimum size of gene sets
            max_size=500,                            # maximum size of gene sets
            seed=42,
            verbose=True
        )
            
            res = pre_res.res2d.copy()
            res['-log10(FDR)'] = -np.log10(res['FDR q-val'].astype(float))
            res['Term'] = res['Term'].str.replace(r'\s*Homo.*', '', regex=True)

            res.to_csv(os.path.join(results_dir, f'GSEA_{column_name}.csv'))

        except:
            pre_res = gp.prerank(
            rnk=os.path.join(results_dir, f'{column_name}_lfc.rnk'),                      # ranked gene list
            gene_sets="MSigDB_Hallmark_2020",                   # or a .gmt file or other gene set name from Enrichr
            outdir="gsea_results",                   # output directory
            permutation_num=1000,                    # number of permutations
            min_size=1,                             # minimum size of gene sets
            max_size=1000,                            # maximum size of gene sets
            seed=42,
            verbose=True
        )
            
            res = pre_res.res2d.copy()
            res['-log10(FDR)'] = -np.log10(res['FDR q-val'].astype(float))
            res['Term'] = res['Term'].str.replace(r'\s*Homo.*', '', regex=True)

            res.to_csv(os.path.join(results_dir, f'GSEA_{column_name}.csv'))


def perform_sciplex_GSEA(feature_importance):
    for key, df in feature_importance.items():
        print(key)

        df = df.iloc[0:1,2:].T.reset_index()
        print(df.columns[1])
        df = df.sort_values(df.columns[1]).rename(columns={'index': 'gene_name', df.columns[1] : 'score'})
        df = df[df['score']!= 0]
        print(df.shape)
        key = key.replace('\n', '_')
        df.to_csv(os.path.join(results_dir, f'sciplex_{key}.rnk'), sep="\t", index=False, header=False)

        try:
            pre_res = gp.prerank(
            rnk=os.path.join(results_dir, f'sciplex_{key}.rnk'),                      # ranked gene list
            gene_sets="MSigDB_Hallmark_2020",                   # or a .gmt file or other gene set name from Enrichr
            outdir="gsea_results",                   # output directory
            permutation_num=1000,                    # number of permutations
            min_size=1,                             # minimum size of gene sets
            max_size=1000,                            # maximum size of gene sets
            seed=42,
            verbose=True
            )
            res = pre_res.res2d.copy()
            res['-log10(FDR)'] = -np.log10(res['FDR q-val'].astype(float))
            res['Term'] = res['Term'].str.replace(r'\s*Homo.*', '', regex=True)

            res.to_csv(os.path.join(results_dir, f'GSEA_sciplex_{key}.csv'))
        
        except:
            print(f'GSEA failed for {key}')
            print(df.head())

def main():
    get_mcfarland_feature_importance()
    feature_importance_means = pd.read_csv(os.path.join(results_dir, 'mcfarland_regression_feature_importance_means_filtered.csv'))
    feature_importance_ordered = feature_importance_means.loc[:, ~feature_importance_means.columns.str.contains('condition_')]
    feature_importance_ordered = feature_importance_ordered[feature_importance_ordered['model']=='Log(fold change)'].drop(columns=['model']).set_index('condition').T.reset_index()
    perform_mcfarland_GSEA(feature_importance_ordered)
    
    sciplex_feature_importance = get_sciplex_feature_importance()
    perform_sciplex_GSEA(sciplex_feature_importance)


if __name__ == '__main__':
    main()