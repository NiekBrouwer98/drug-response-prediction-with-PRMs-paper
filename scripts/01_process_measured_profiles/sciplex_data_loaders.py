import pandas as pd
import scanpy as sc
import numpy as np
import anndata as ad
from scipy import sparse
import os
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
home_dir = str(config.PROJECT_ROOT)
data_dir = str(config.DATA_DIR)
resources_dir = str(config.RESOURCES_DIR)
results_dir = str(config.RESULTS_01_DIR)
figures_dir = str(config.FIGURES_01_DIR)

# Ensure directories exist
ensure_directories_exist(home_dir,data_dir,resources_dir,results_dir, figures_dir)


class sciplex_dataset:
    def __init__(self,filename,dataset=None, cell_line_subset=''):
        if dataset is not None:
            if(type(dataset) != type(ad.AnnData())):
                self.dataset = None
                logger.error("dataset type should be of type Anndata")
            else:
                self.dataset = dataset
        self.cell_line_subset = cell_line_subset
        self.filename = filename


    def remove_duplicate_conditions(self):
        df = self.dataset.obs[['product_name', 'condition']].copy()
        counts = df.groupby(['condition', 'product_name']).size().reset_index(name='count')
        idx_max = counts.groupby('condition')['count'].idxmax()
        largest_groups = counts.loc[idx_max]
        selected_products = largest_groups['product_name'].unique()

        self.dataset = self.dataset[self.dataset.obs['product_name'].isin(selected_products)]


    def match_target(self):
        sciplex_drug_to_perturbation = pd.read_csv(os.path.join(resources_dir, 'sciplex_drug_to_perturbation.csv'),index_col=0)
        sciplex_drug_to_perturbation['product_name'] = sciplex_drug_to_perturbation['product_name'].str.replace(' ', '')
        self.dataset.obs = pd.merge(self.dataset.obs, sciplex_drug_to_perturbation, on='product_name', how='left')
        self.dataset.obs = self.dataset.obs.rename(columns={'target':'condition'})


    def preprocess_df(self):
        if self.dataset is None:
            raise Exception("Experiment not loaded")
        if 'dose' not in self.dataset.obs.columns:
            self.dataset.obs['dose'] = 10000
        if 'cell_type' not in self.dataset.obs.columns:
            self.dataset.obs['cell_type'] = 'Unknown'
        if 'product_name' not in self.dataset.obs.columns:
            self.dataset.obs['product_name'] = 'Unknown'
        self.dataset.obs = self.dataset.obs[['cell_type', 'dose', 'product_name']]
        self.dataset.obs['cell_type'] = self.cell_line_subset
        self.dataset.obs['product_name'] = self.dataset.obs['product_name'].str.replace(' ', '')
        
        barcode_indices = self.dataset.obs.index
        self.match_target()
        self.dataset.obs.index = barcode_indices

        self.remove_duplicate_conditions()
        self.dataset.obs.index.names = ['barcode']
        self.dataset.obs['cell_type_condition'] =  self.dataset.obs['cell_type'].astype(str) + '-' + self.dataset.obs['condition'].astype(str)
        self.dataset = self.dataset[self.dataset.obs['condition'].notna()]

        self.dataset.var['feature_name'] = self.dataset.var.index.to_series().apply(lambda x: x.split('.')[0])
        self.dataset.var = self.dataset.var.rename(columns={'feature_name': 'gene_name'})
        self.dataset.var.index = self.dataset.var['gene_name']
        self.dataset.var_names_make_unique()


    def set_data(self):
        sciplex = sc.read_h5ad(os.path.join(data_dir, 'sciplex_raw', 'Srivatsan_2019_raw.h5ad'))
        sciplex = sciplex[(sciplex.obs['dose']==10000) | (sciplex.obs['vehicle']==True)]
        sciplex = sciplex[sciplex.obs['cell_type']==self.cell_line_subset].copy()

        sc.pp.normalize_total(sciplex)
        sc.pp.log1p(sciplex)
        self.dataset = sciplex
        self.preprocess_df()


    def get_data(self):
        return self.dataset
    

    def get_gene_set(self):
        gene_list_df = pd.read_csv(os.path.join(resources_dir,'OS_scRNA_gene_index.19264.tsv'), header=0, delimiter='\t')
        gene_list = list(gene_list_df['gene_name'])
        return gene_list
    
        
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
        
        
    def get_dataset_information(self):
        if self.dataset is None:
            raise Exception("Dataset not loaded")

        logger.info(f'number of unique cell types: {len(self.dataset.obs.cell_type.unique())}')
        logger.info(f'number of unique conditions: {len(self.dataset.obs.condition.unique())}')
        logger.info(f'number of samples: {self.dataset.obs.shape[0]}')


    def plot_umap(self, condition):
        sc.tl.pca(self.dataset,n_comps=30)
        sc.pp.neighbors(self.dataset)
        sc.tl.umap(self.dataset)
        sc.tl.leiden(self.dataset)
        sc.pl.umap(self.dataset, color=condition)


def process_sciplex():
    log_script_start(__file__, logger)
    logger.info("Processing SCIPLEX dataset...")
    sciplex_mcf7 = sciplex_dataset(filename='MCF7',cell_line_subset='MCF7')
    sciplex_a549 = sciplex_dataset(filename='A549',cell_line_subset='A549')
    sciplex_k562 = sciplex_dataset(filename='K562',cell_line_subset='K562')
    sciplex_mcf7.set_data()
    sciplex_mcf7.get_dataset_information()
    sciplex_a549.set_data()
    sciplex_a549.get_dataset_information()
    sciplex_k562.set_data()
    sciplex_k562.get_dataset_information()

    path = os.path.join(data_dir, 'sciplex_processed')
    if not os.path.exists(path):
        os.makedirs(path)

    # Now process the dataset to fit the input format required by GEARS.
    datasets = {'mcf7':sciplex_mcf7, 'a549':sciplex_a549, 'k562':sciplex_k562}
    for k, df in datasets.items():
        logger.info(f"Processing SCIPLEX {k} dataset...")
        processed_data = process_merged_df(df.get_data())
        processed_data = scfoundation_preprocessing(processed_data)

        processed_dataset = sciplex_dataset(dataset=processed_data, filename=f'{data_dir}/sciplex_processed/sciplex{k}')
        processed_dataset.save_dataset('h5ad')
        logger.info(f"SCIPLEX {k} dataset saved successfully")

if __name__ == '__main__':
    log_script_start(__file__, logger)
    try:
        process_sciplex()
        logger.info("SCIPLEX datasets processed successfully")
    except Exception as e:
        logger.error(f"Error in process_sciplex: {str(e)}")
        raise
    log_script_end(__file__, logger)