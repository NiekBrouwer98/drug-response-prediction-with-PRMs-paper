import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
import logging

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config, setup_project
from utils import setup_logging_for_script, log_script_start, log_script_end, ensure_directories_exist

# Import local utilities
from preprocess_utils import *

# Setup project and logging
setup_project()
logger = setup_logging_for_script(__file__)

# Set project directories using configuration
path = os.getcwd()
home_dir = str(config.PROJECT_ROOT)
data_dir = str(config.DATA_DIR)
resources_dir = str(config.RESOURCES_DIR)
results_dir = str(config.RESULTS_01_DIR)
figures_dir = str(config.FIGURES_01_DIR)

# Ensure directories exist
ensure_directories_exist(home_dir,data_dir,resources_dir,results_dir, figures_dir)

class mcfarland_dataset:
    def __init__(self, cell_line=None,drug=None, filename=None, dataset=None):
        if cell_line is not None:
            self.cell_line = cell_line
        else:
            self.cell_line = self.get_cell_lines()['CCLE Name'].unique().tolist()
        if drug is not None:
            self.drug = drug
        else:
            self.drug = self.get_drugs()
        if filename is not None:
            self.filename = filename
        if dataset is not None:
            if(type(dataset) != type(ad.AnnData())):
                self.dataset = None
                print("dataset type should be of type Anndata")
            else:
                self.dataset = dataset
        self.experiment_files = None
        self.experiment2_files = None


    def set_data(self, normalize=True):
        self.set_experiment_files()
        self.set_experiment2_files()
        logger.info('Retrieved experiment files')
        merged_df = pd.DataFrame()
        for index, row in self.get_experiment_files().iterrows():
            drug_df = mcfarland_experiment()
            drug_df.set_data(row['filename'], row['target'],self.cell_line)
            if (drug_df.get_data().shape[0] == 0):
                print("No data for", row['filename'])
                continue
            else:
                if (merged_df.shape[0]==0):
                    merged_df = drug_df.get_data()
                else:
                    merged_df = ad.concat([merged_df,drug_df.get_data()], join="outer")
                    merged_df.obs_names_make_unique()

        expt2_df = pd.DataFrame()
        for index, row in self.experiment2_files.iterrows():
            drug_df = mcfarland_experiment()
            drug_df.set_data(row['filename'], row['target'],self.cell_line)
            if (expt2_df.shape[0]==0):
                expt2_df = drug_df.get_data()
            else:
                expt2_df = ad.concat([expt2_df,drug_df.get_data()], join="outer")
                expt2_df.obs_names_make_unique()

        print('Loaded all data')

        unique_cell_types = merged_df.obs.cell_type.unique()
        if expt2_df.shape[0] != 0:
            expt2_df_subset = expt2_df[expt2_df.obs.cell_type.isin(unique_cell_types)]
            merged_df_with_expt2 = ad.concat([merged_df,expt2_df_subset], join="outer")
        else:
            merged_df_with_expt2 = merged_df

        merged_df_with_expt2.obs_names_make_unique()
        print('Set all data')

        if normalize:
            #Normalize per cell line
            normalized_df = pd.DataFrame()
            for cell_line in unique_cell_types:
                subset = merged_df_with_expt2[merged_df_with_expt2.obs.cell_type == cell_line]
                sc.pp.normalize_total(subset)
                sc.pp.log1p(subset)

                if (normalized_df.shape[0]==0):
                    normalized_df = subset
                else: 
                    normalized_df = ad.concat([normalized_df,subset], join="outer")

            print('Normalized all data')

            self.dataset = normalized_df
        else:
            self.dataset = merged_df_with_expt2

        self.dataset.var['gene_name'] = self.dataset.var.index

    def get_data(self):
        return(self.dataset)
    
    def get_drug_target_matching(self):
        drug_to_target =pd.read_csv(os.path.join(resources_dir, 'mcfarland_drug_to_perturbation.csv'))
        drug_to_target = drug_to_target[['drug', 'target']]
        drug_to_target = pd.concat([drug_to_target, pd.DataFrame({'drug': ['DMSO'], 'target': ['ctrl']})], ignore_index=True)
        return(drug_to_target)

    def set_experiment_files(self):
        datasets = pd.DataFrame({'filepath': []})
        for drug in self.drug:
            drug_files = glob.glob(os.path.join(data_dir, rf"mcfarland_raw/{drug}*expt*"))
            datasets = pd.concat([datasets, pd.DataFrame({'filepath': drug_files})], ignore_index=True)
        # Use os.path to split filepath into path and filename
        datasets['path'] = datasets['filepath'].apply(lambda x: os.path.dirname(x))
        datasets['filename'] = datasets['filepath'].apply(lambda x: os.path.basename(x))
        datasets[['drug','experiment']] = datasets['filename'].str.split("_",expand=True,n=1)
        datasets = datasets[~datasets['experiment'].str.contains('.zip')]
        datasets = datasets[~datasets['experiment'].str.contains('6hr')]
        datasets = datasets[~datasets['experiment'].str.contains('expt5')]
        datasets = datasets[~datasets['experiment'].str.contains('expt2')]

        datasets = datasets.reset_index()
        datasets = datasets.merge(self.get_drug_target_matching(), on='drug', how='left')
        datasets =datasets.dropna()

        self.experiment_files = datasets

    def get_experiment_files(self):
        return(self.experiment_files)

    def set_experiment2_files(self):
        datasets = pd.DataFrame({'filepath': []})
        for drug in self.drug:
            drug_files = glob.glob(os.path.join(data_dir, rf"mcfarland_raw/{drug}*expt*"))
            datasets = pd.concat([datasets, pd.DataFrame({'filepath': drug_files})], ignore_index=True)

        # Use os.path to split filepath into path and filename
        datasets['path'] = datasets['filepath'].apply(lambda x: os.path.dirname(x))
        datasets['filename'] = datasets['filepath'].apply(lambda x: os.path.basename(x))
        datasets[['drug','experiment']] = datasets['filename'].str.split("_",expand=True,n=1)
        datasets = datasets[~datasets['experiment'].str.contains('.zip')]
        datasets = datasets[~datasets['experiment'].str.contains('6hr')]
        datasets = datasets[~datasets['experiment'].str.contains('expt5')]

        datasets_expt2 = datasets[datasets['experiment'].str.contains('expt2')]
        datasets_expt2 = datasets_expt2.merge(self.get_drug_target_matching(), on='drug', how='left')
        datasets_expt2 =datasets_expt2.dropna()

        self.experiment2_files = datasets_expt2.head(n=1)

    def get_experimemt2_files(self):
        return(self.experiment2_files)

    def get_cell_lines(self):
        all_cell_lines = pd.read_csv(os.path.join(data_dir, 'mcfarland_raw/cell_lines.csv'),sep=';')
        all_cell_lines = all_cell_lines.dropna()
        all_cell_lines[['cell_line','tissue_type']] = all_cell_lines['CCLE Name'].str.split('_',expand=True,n=1)

        return(all_cell_lines)
    
    def get_drugs(self):
        drug_to_target = pd.read_csv(os.path.join(resources_dir, 'mcfarland_drug_to_perturbation.csv'))
        drug_to_target = drug_to_target.dropna()
        drug_list = drug_to_target['drug'].unique().tolist() + ['DMSO']
        return(drug_list)
    
    def save_dataset(self, filetype='h5ad'):
        if self.dataset is None:
            raise Exception("Dataset not loaded")
        if filetype == 'h5ad':
            self.dataset.write(f'{self.filename}.h5ad')
        elif filetype == 'csv':
            self.dataset.to_df().to_csv(f'{self.filename}.csv')
        elif filetype == 'npy':
            np.save(f'{self.filename}.npy', self.dataset.to_df().to_numpy())
        else:
            raise Exception("Invalid filetype")
        
    def get_meta_data(self):         
        total_size = self.dataset.obs.shape[0]
        unique_values = self.dataset.obs.cell_type_condition.unique()
        cell_line_result = pd.DataFrame({'size': [total_size], 'unique_perturbations': [len(unique_values)], 'perturbations': [unique_values]})
        return cell_line_result
    
    def get_dataset_information(self):
        if self.dataset is None:
            raise Exception("Dataset not loaded")

        print(f'dataset contain information for {self.dataset.obs.cell_type.unique()} cell lines')
        print(f'dataset contain information for {self.dataset.obs.condition.unique()} drugs')
        print(f'number of unique cell types: {len(self.dataset.obs.cell_type.unique())}')
        print(f'number of unique conditions: {len(self.dataset.obs.condition.unique())}')
        print(f'number of samples: {self.dataset.obs.shape[0]}')
        
    def plot_umap(self, condition):
        sc.tl.pca(self.dataset,n_comps=30)
        sc.pp.neighbors(self.dataset)
        sc.tl.umap(self.dataset)
        sc.tl.leiden(self.dataset)
        sc.pl.umap(self.dataset, color=condition )


class mcfarland_experiment:
    def __init__(self):
        self.experiment = None

    def preprocess_df(self,condition):
        if self.experiment is None:
            raise Exception("Experiment not loaded")

        self.experiment.obs.index = self.experiment.obs['barcode']
        self.experiment.obs = self.experiment.obs[['singlet_ID']]
        self.experiment.obs = self.experiment.obs.rename(columns={'singlet_ID': 'cell_type'})
        self.experiment.obs['cell_type'] = self.experiment.obs['cell_type'].str.rsplit('_', n=1).str.get(0)
        self.experiment.obs['condition'] = condition
        self.experiment.obs['cell_type_condition'] =  self.experiment.obs['cell_type'] + '-' + self.experiment.obs['condition']

    def set_data(self, filename, condition, cell_line_subset=[]):
        df = sc.read_10x_mtx(os.path.join(data_dir,f'mcfarland_raw/{filename}/'))
        # Normalize per experiment
        sc.pp.normalize_total(df)
        df_classification = pd.read_csv(os.path.join(data_dir, f'mcfarland_raw/{filename}/classifications.csv'))
        df.obs['barcode'] = df.obs.index
        df.obs = pd.merge(df.obs, df_classification, on='barcode',how='left')
        if len(cell_line_subset) != 0:
            df = df[df.obs.singlet_ID.isin(cell_line_subset)]
            
        self.experiment = df
        self.preprocess_df(condition)

    def get_data(self):
        return self.experiment
    
    def get_gene_set(self):
        gene_list_df = pd.read_csv(os.path.join(resources_dir, 'OS_scRNA_gene_index.19264.tsv'), header=0, delimiter='\t')
        gene_list = list(gene_list_df['gene_name'])
        return gene_list

def process_mcfarland_all():
    all_cell_lines = mcfarland_dataset(filename='all_cell_lines')
    all_cell_lines.set_data()
    all_cell_lines.get_dataset_information()

    processed_data = process_merged_df(all_cell_lines.get_data())
    processed_data = scfoundation_preprocessing(processed_data)
    processed_dataset = mcfarland_dataset(all_cell_lines, filename=os.path.join(data_dir, 'mcfarland_processed/all_cell_lines'),dataset=processed_data)
    processed_dataset.save_dataset('h5ad')
    

def process_mcfarland_per_tissue_type(tissue_type='SKIN'):
    all_cell_lines = mcfarland_dataset().get_cell_lines()
    print(len(all_cell_lines['tissue_type'].unique()))
    tissue_lines = all_cell_lines[all_cell_lines['tissue_type']==tissue_type]['CCLE Name'].to_list()

    tissue_df = mcfarland_dataset(tissue_lines, filename=f'{tissue_type.lower()}_all_normalized')
    tissue_df.set_data()
    tissue_df.get_dataset_information()

    processed_data = process_merged_df(tissue_df.get_data())
    processed_data = scfoundation_preprocessing(processed_data)
    processed_dataset = mcfarland_dataset(tissue_lines, filename=os.path.join(data_dir, f'mcfarland_processed/{tissue_type.lower()}_all_normalized'),dataset=processed_data)
    processed_dataset.save_dataset('h5ad')


    # Also create single cell line datasets.
    processed_data = sc.read_h5ad(os.path.join(data_dir, f'mcfarland_processed/{tissue_type.lower()}_all_normalized.h5ad'))
    for cl in processed_data.obs.cell_type.unique().tolist():
        subset = processed_data[processed_data.obs.cell_type == cl]
        print(subset.shape)
        processed_dataset = mcfarland_dataset(tissue_lines,filename= os.path.join(data_dir, f'mcfarland_processed/{cl}'),dataset=subset)
        print(f'Saving dataset for {cl}')
        processed_dataset.save_dataset('h5ad')
        

'''Retrieve metadata for the cell lines'''
def get_sensitive_information(expt_files):
    cell_line_info = pd.DataFrame(columns = ['DEPMAP_ID', 'CCLE_ID', 'sens', 'drug','time'])

    for file in expt_files:
        print(file)
        single_drug_info = pd.read_csv(file)
        single_drug_info = single_drug_info[['DEPMAP_ID', 'CCLE_ID', 'sens']]
        meta_deta = str(file).replace(os.path.join(data_dir, 'mcfarland_raw/cell_line_features/'),'')
        drug_info = meta_deta.split('_')
        single_drug_info['drug'] = drug_info[0]
        if len(drug_info) > 1:
            single_drug_info['experiment'] = drug_info[1:]

        cell_line_info = pd.concat([cell_line_info, single_drug_info])

    cell_line_info= cell_line_info.drop(columns=['time','DEPMAP_ID'])
    cell_line_info = cell_line_info.rename(columns={'CCLE_ID':'cell_line'})

    return(cell_line_info)



if __name__ == "__main__":
    log_script_start(__file__, logger)
    
    try:
        logger.info("Starting McFarland data processing...")
        process_mcfarland_all()
        logger.info("McFarland data processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in McFarland data processing: {str(e)}")
        raise
    
    finally:
        log_script_end(__file__, logger)
