import numpy as np
import pandas as pd
import os
import scanpy as sc

from config import config

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
    # cell_line_info = cell_line_info.rename(columns={'CCLE_ID':'cell_line'})
    cell_line_info[['cell_line', 'tissue']] = cell_line_info['CCLE_ID'].str.split('_',expand=True, n=1)
    cell_line_info = cell_line_info.drop_duplicates()

    return(cell_line_info)

def get_AUCs():
    cellline_information_path = os.path.join(data_dir, 'mcfarland_raw','cell_line_features')
    all_files = [os.path.join(cellline_information_path, f) for f in os.listdir(cellline_information_path) if 'metadata.csv' not in f]
    sensitivity_info = get_sensitivity_information(all_files)

    # Cap values to mitigate outliers
    sensitivity_info['sens'] = np.where(sensitivity_info['sens'] < 0 , 0, sensitivity_info['sens'])

    return sensitivity_info

def get_sens_labels(threshold):
    sensitivity_info = get_AUCs()
    
    sensitivity_info['sens_label'] = np.where(sensitivity_info['sens'] < threshold, 0, 1)

    return(sensitivity_info)

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

    return cellline_sensitivity_info


def get_sciplex_AUCs():
    filepath = os.path.join(resources_dir, 'sciplex_sensitivity_info.csv')
    if os.path.exists(filepath):
        sensitivity_df = pd.read_csv(filepath)

    else:
        sensitivity_df = []
        for cell_line in ['mcf7', 'k562', 'a549']:
            sensitivity = pd.read_csv(os.path.join(results_dir, '01_process_observed_profiles', f'sciplex{cell_line}_msd.csv'))
            sensitivity = sensitivity[(sensitivity['dose']==10000)| (sensitivity['dose']==0)][['condition','1-viability']].drop_duplicates()
            sensitivity['cell_line'] = cell_line.upper()
            sensitivity = sensitivity.rename(columns={'1-viability': 'y'})

            pert_to_drug = pd.read_csv(os.path.join(resources_dir, 'sciplex_drug_to_perturbation.csv'),index_col=0)
            pert_to_drug['product_name'] = pert_to_drug['product_name'].str.replace(' ', '')
            sensitivity = pd.merge(sensitivity, pert_to_drug, left_on='condition',right_on='product_name', how='left')

            sensitivity_df.append(sensitivity)

        sensitivity_df = pd.concat(sensitivity_df, axis=0)
        sensitivity_df.to_csv(filepath, index=False)

    sensitivity_df = sensitivity_df.drop(columns=['condition', 'product_name']).rename(columns={'target': 'condition'})

    return sensitivity_df

def get_McFarland_mean_data():
    mean_observed_pre_treatment = pd.read_csv(os.path.join(data_dir, 'observed_pseudobulk', 'mcfarland_mean_pre_all_celllines.csv'), index_col=0)
    mean_observed_post_treatment = pd.read_csv(os.path.join(data_dir, 'observed_pseudobulk', 'mcfarland_mean_post_all_celllines.csv'), index_col=0)
    mean_observed_LFC= pd.read_csv(os.path.join(data_dir, 'observed_pseudobulk', 'mcfarland_mean_LFC_all_celllines.csv'), index_col=0)

    mean_observed_pre_treatment = mean_observed_pre_treatment.rename(columns={'cell_type':'cell_line'})
    mean_observed_post_treatment = mean_observed_post_treatment.rename(columns={'cell_type':'cell_line'})
    mean_observed_LFC = mean_observed_LFC.rename(columns={'cell_type':'cell_line'})

    mean_observed_pre_treatment['cell_line'] = mean_observed_pre_treatment['cell_line'].str.split('_').str[0]
    mean_observed_post_treatment['cell_line'] = mean_observed_post_treatment['cell_line'].str.split('_').str[0]
    mean_observed_LFC['cell_line'] = mean_observed_LFC['cell_line'].str.split('_').str[0]
    
    tissues = get_tissue_labels()
    mean_observed_pre_treatment_with_tissue = pd.merge(mean_observed_pre_treatment, tissues, on='cell_line', how='left')
    mean_observed_post_treatment_with_tissue = pd.merge(mean_observed_post_treatment, tissues, on='cell_line', how='left')
    mean_observed_LFC_with_tissue = pd.merge(mean_observed_LFC, tissues, on='cell_line', how='left')

    return mean_observed_pre_treatment_with_tissue, mean_observed_post_treatment_with_tissue, mean_observed_LFC_with_tissue

def get_sciplex_pre_treatment_data():
    result = []
    for cell_line in ['mcf7', 'k562', 'a549']:
        adata = sc.read_h5ad(os.path.join(data_dir, 'sciplex_processed', f'sciplex{cell_line}.h5ad'))
        adata_df = adata.to_df()
        adata_df['product_name'] = adata.obs['product_name']
        ctrl_data = adata_df[adata_df['product_name'] == 'Vehicle']
        result.append(ctrl_data)

    pre_data = pd.concat(result,axis=0)
    pre_data = pre_data.drop(columns=['product_name'])
    return(pre_data)

def get_sciplex_mean_data():
    mean_pre_df = []
    mean_post_df = []
    mean_LFC_df = []

    for cell_line in ['mcf7', 'k562', 'a549']:
        mean_pre = pd.read_csv(os.path.join(data_dir, 'observed_pseudobulk', f'sciplex_mean_pre_{cell_line}.csv'), index_col=0).drop(columns=['cell_type'])
        mean_post = pd.read_csv(os.path.join(data_dir, 'observed_pseudobulk', f'sciplex_mean_post_{cell_line}.csv'), index_col=0).drop(columns=['cell_type'])
        mean_LFC = pd.read_csv(os.path.join(data_dir, 'observed_pseudobulk', f'sciplex_mean_LFC_{cell_line}.csv'), index_col=0)
        mean_pre['cell_line'] = cell_line.upper()
        mean_post['cell_line'] = cell_line.upper()
        mean_LFC['cell_line'] = cell_line.upper()
        mean_pre_df.append(mean_pre)
        mean_post_df.append(mean_post)
        mean_LFC_df.append(mean_LFC)

    mean_pre_df = pd.concat(mean_pre_df, axis=0)
    mean_post_df = pd.concat(mean_post_df, axis=0)
    mean_LFC_df = pd.concat(mean_LFC_df, axis=0)

    return mean_pre_df, mean_post_df, mean_LFC_df

def get_GEARS_predictions():
    post_predictions = []
    lfc_predictions = []
    for cell_line in ['mcf7', 'k562', 'a549']:
        predicted_cell_line = pd.read_csv(os.path.join(data_dir, 'GEARS_predictions', f'sciplex_mean_post_{cell_line}.csv'), index_col=0).rename(columns={'cell_type':'cell_line', 'perturbation':'condition'})
        predicted_LFC_cell_line = pd.read_csv(os.path.join(data_dir, 'GEARS_predictions', f'sciplex_mean_LFC_{cell_line}.csv')).rename(columns={'cell_type':'cell_line', 'perturbation':'condition'})
        post_predictions.append(predicted_cell_line)
        lfc_predictions.append(predicted_LFC_cell_line)

    post_predictions = pd.concat(post_predictions, axis=0)
    lfc_predictions = pd.concat(lfc_predictions, axis=0)

    return post_predictions, lfc_predictions

def get_GEARS_noreg_predictions():
    post_predictions = []
    lfc_predictions = []
    for cell_line in ['mcf7', 'k562', 'a549']:
        predicted_cell_line = pd.read_csv(os.path.join(data_dir, 'GEARS_noreg_predictions', f'sciplex_mean_post_{cell_line}.csv'), index_col=0).rename(columns={'cell_type':'cell_line', 'perturbation':'condition'})
        predicted_LFC_cell_line = pd.read_csv(os.path.join(data_dir, 'GEARS_noreg_predictions', f'sciplex_mean_LFC_{cell_line}.csv')).rename(columns={'cell_type':'cell_line', 'perturbation':'condition'})
        post_predictions.append(predicted_cell_line)
        lfc_predictions.append(predicted_LFC_cell_line)

    post_predictions = pd.concat(post_predictions, axis=0)
    lfc_predictions = pd.concat(lfc_predictions, axis=0)

    return post_predictions, lfc_predictions

def get_scfoundation_predictions():
    post_predictions = []
    lfc_predictions = []
    for cell_line in ['mcf7', 'k562', 'a549']:
        predicted_cell_line = pd.read_csv(os.path.join(data_dir, 'scfoundation_predictions', f'sciplex_mean_post_{cell_line}.csv'), index_col=0).rename(columns={'cell_type':'cell_line', 'perturbation':'condition'})
        predicted_LFC_cell_line = pd.read_csv(os.path.join(data_dir, 'scfoundation_predictions', f'sciplex_mean_LFC_{cell_line}.csv')).rename(columns={'cell_type':'cell_line', 'perturbation':'condition'})
        post_predictions.append(predicted_cell_line)
        lfc_predictions.append(predicted_LFC_cell_line)

    post_predictions = pd.concat(post_predictions, axis=0)
    lfc_predictions = pd.concat(lfc_predictions, axis=0)

    return post_predictions, lfc_predictions

def get_CPA_predictions():
    post_predictions = pd.read_csv(os.path.join(data_dir, 'CPA_predictions', 'sciplex_mean_post.csv')).rename(columns={'cell_type':'cell_line'})
    lfc_predictions = pd.read_csv(os.path.join(data_dir, 'CPA_predictions', 'sciplex_mean_LFC.csv')).rename(columns={'cell_type':'cell_line'})

    return post_predictions, lfc_predictions

def get_average_effect_predictions():
    post_predictions_df = []
    lfc_predictions_df = []
    for cell_line in ['mcf7', 'k562', 'a549']:
        post_predictions = pd.read_csv(os.path.join(data_dir, 'average_effect_predictions', f'sciplex_mean_post_{cell_line}.csv')).rename(columns={'cell_type':'cell_line'})
        lfc_predictions = pd.read_csv(os.path.join(data_dir, 'average_effect_predictions', f'sciplex_mean_LFC_{cell_line}.csv')).rename(columns={'cell_type':'cell_line'})
        post_predictions_df.append(post_predictions)
        lfc_predictions_df.append(lfc_predictions)

    post_predictions_df = pd.concat(post_predictions_df, axis=0)
    lfc_predictions_df = pd.concat(lfc_predictions_df, axis=0)
    
    return post_predictions_df, lfc_predictions_df

def get_no_effect_predictions():
    # There are no LFCs for no-effect predictions, because these are always 0
    result = []
    for cell_line in ['mcf7', 'k562', 'a549']:
        post_data = pd.read_csv(os.path.join(data_dir, 'no_effect_predictions', f'sciplex_mean_post_{cell_line}.csv'), index_col=0).rename(columns={'cell_type':'cell_line'})
        post_data['cell_line'] = cell_line.upper()
        result.append(post_data)

    result = pd.concat(result, axis=0)

    return result 

'''Helper functions'''	
def add_y_and_normalize(df_with_sensitivity, y, normalize=True, groupby=['condition'], keep=[]):

    if normalize:
        df_with_sensitivity['y'] = df_with_sensitivity.groupby(groupby)[y].transform(lambda x: (x - x.mean()) / x.std())
    else:
        df_with_sensitivity['y'] = df_with_sensitivity[y]

    columns_to_remove = list(set(['sens', 'sens_label', 'target', 'condition', 'cell_line', 'tissue']).difference(set(keep)))
    df_with_sensitivity = df_with_sensitivity.drop(columns_to_remove, axis=1)
    df_with_sensitivity = df_with_sensitivity[df_with_sensitivity['y'].notna()]
    df_with_sensitivity = df_with_sensitivity.dropna(axis=0)

    return(df_with_sensitivity)

def get_tissue_labels():
    sensitivity_info = get_sens_labels(0.2)
    drug_to_perturbation = pd.read_csv(os.path.join(resources_dir, 'mcfarland_drug_to_perturbation.csv'))
    tissues = pd.merge(sensitivity_info, drug_to_perturbation, on='drug', how='left')[['cell_line', 'tissue']].drop_duplicates()
    tissues['tissue'] = np.where(tissues['tissue'].str.contains('SKIN'), 'SKIN', tissues['tissue'])
    tissues['tissue'] = np.where(tissues['tissue'].str.contains('TO_'), tissues['tissue'].str.split('_',n=1)[1], tissues['tissue'])
    tissues = tissues.drop_duplicates()
    tissues = tissues[tissues['tissue']!='MATCHED_NORMAL_TISSUE']

    return(tissues)

def filter_on_coefficient_of_variation(df_with_sensitivity, groupby=['tissue', 'condition'], threshold=0.5):
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
    remove_columns = ['y', 'tissue', 'cell_line', 'target', 'drug','condition', 'perturbation','cell_type']
    remove_columns = list(set(remove_columns).difference(feature_subset))
    for c in remove_columns:
        if c in X_train.columns:
            X_train = X_train.drop(columns=c)

    if len(feature_subset) == 0:
        variances = X_train.var(axis=0)
        selected_features = variances.nlargest(n_features).index.tolist()
    else:
        selected_features = feature_subset

    selected_features = list(set(selected_features).intersection(set(X_train.columns)))
   
    # Only keep condition as drug indicator
    # Keep cell line and tissue to identify test instances 
    selected_features = selected_features + ['tissue', 'condition', 'cell_line']

    return selected_features
