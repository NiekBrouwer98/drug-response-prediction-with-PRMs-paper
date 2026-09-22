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
from feature_importance_io import (
    DEFAULT_MCFARLAND_FI_PREFIX,
    DEFAULT_SCIPLEX_FI_PREFIX,
    build_cpa_family_gsea_feature_tables,
    build_sciplex_gsea_feature_tables,
    load_split_cv_feature_importance_with_observed_fallback,
)

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


def get_sciplex_feature_importance(
    results_prefix: str = DEFAULT_SCIPLEX_FI_PREFIX,
    *,
    smiles: bool = True,
    gene_features_only: bool = True,
) -> dict[str, pd.DataFrame]:
    """All-model SciPlex importances from split-CV ElasticNet (SMILES pickles by default).

    Gene features only are retained for GSEA when ``gene_features_only`` is True.
    """
    fi_pickle = load_split_cv_feature_importance_with_observed_fallback(
        results_dir, results_prefix,
    )
    all_feature_importance = build_sciplex_gsea_feature_tables(
        fi_pickle, smiles=smiles, gene_features_only=gene_features_only,
    )
    with open(os.path.join(results_dir, 'sciplex_feature_importances.pkl'), 'wb') as f:
        pkl.dump(all_feature_importance, f)
    return all_feature_importance


def get_mcfarland_split_cv_feature_importance(
    results_prefix: str = DEFAULT_MCFARLAND_FI_PREFIX,
    *,
    smiles: bool = True,
    gene_features_only: bool = True,
) -> dict[str, pd.DataFrame]:
    """All-model McFarland importances from split-CV ElasticNet (SMILES pickles by default).

    Gene features only are retained for GSEA when ``gene_features_only`` is True.
    """
    fi_pickle = load_split_cv_feature_importance_with_observed_fallback(
        results_dir, results_prefix,
    )
    all_feature_importance = build_cpa_family_gsea_feature_tables(
        fi_pickle,
        condition='Full McFarland dataset',
        smiles=smiles,
        gene_features_only=gene_features_only,
    )
    with open(os.path.join(results_dir, 'mcfarland_feature_importances.pkl'), 'wb') as f:
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


def perform_cpa_family_GSEA(feature_importance: dict[str, pd.DataFrame], dataset: str) -> None:
    for key, df in feature_importance.items():
        print(key)

        df = df.iloc[0:1, 2:].T.reset_index()
        print(df.columns[1])
        df = df.sort_values(df.columns[1]).rename(columns={'index': 'gene_name', df.columns[1]: 'score'})
        df = df[df['score'] != 0]
        print(df.shape)
        key = key.replace('\n', '_')
        rnk_path = os.path.join(results_dir, f'{dataset}_{key}.rnk')
        df.to_csv(rnk_path, sep='\t', index=False, header=False)

        try:
            pre_res = gp.prerank(
                rnk=rnk_path,
                gene_sets='MSigDB_Hallmark_2020',
                outdir='gsea_results',
                permutation_num=1000,
                min_size=1,
                max_size=1000,
                seed=42,
                verbose=True,
            )
            res = pre_res.res2d.copy()
            res['-log10(FDR)'] = -np.log10(res['FDR q-val'].astype(float))
            res['Term'] = res['Term'].str.replace(r'\s*Homo.*', '', regex=True)
            res.to_csv(os.path.join(results_dir, f'GSEA_{dataset}_{key}.csv'))
        except Exception:
            print(f'GSEA failed for {dataset} {key}')
            print(df.head())


def perform_sciplex_GSEA(feature_importance):
    perform_cpa_family_GSEA(feature_importance, 'sciplex')

def main() -> None:
    sciplex_feature_importance = get_sciplex_feature_importance()
    perform_cpa_family_GSEA(sciplex_feature_importance, 'sciplex')

    mcfarland_feature_importance = get_mcfarland_split_cv_feature_importance()
    perform_cpa_family_GSEA(mcfarland_feature_importance, 'mcfarland')


if __name__ == '__main__':
    main()