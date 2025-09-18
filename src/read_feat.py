import os
import json
import pandas as pd
import numpy as np
import torch

def read_feat (all_param):
    
    attr_dir = all_param['files']['node_attr']

    # read nodes with features
    drug_pubmedbert = pd.read_csv(os.path.join(attr_dir, 'attr_drug_pubmedbert.csv'), sep= ',', index_col=0)
    drug_pubmedbert = drug_pubmedbert.sort_index()
    drug_pubmedbert = torch.tensor(drug_pubmedbert.values, dtype=torch.float32)  
    
    drug_FP = pd.read_csv(os.path.join(attr_dir, 'attr_drug_fingerprint.csv'), sep= ',')
    drug_FP = drug_FP.sort_index()
    drug_FP = torch.tensor(drug_FP.values, dtype=torch.float32)  
    
    gene_pubmedbert = pd.read_csv(os.path.join(attr_dir, 'attr_protein_pubmedbert.csv'), sep= ',', index_col=0)
    gene_pubmedbert = gene_pubmedbert.sort_index()
    gene_pubmedbert = torch.tensor(gene_pubmedbert.values, dtype=torch.float32)  
    
    gene_esm2 = torch.load(os.path.join(attr_dir, 'attr_protein_emb_esm2.pt'))
    
    disease_pubmedbert = pd.read_csv(os.path.join(attr_dir, 'attr_disease_pubmedbert.csv'), sep= ',', index_col = 0)
    disease_pubmedbert = disease_pubmedbert.sort_index()
    disease_pubmedbert = torch.tensor(disease_pubmedbert.values, dtype=torch.float32)  
    
    phenotype_pubmedbert = pd.read_csv(os.path.join(attr_dir, 'attr_phenotype_pubmedbert.csv'), sep= ',', index_col = 0)
    phenotype_pubmedbert = phenotype_pubmedbert.sort_index()
    phenotype_pubmedbert = torch.tensor(phenotype_pubmedbert.values, dtype=torch.float32)  


    # make a dict of all the features
    feat = {
        'gene/protein': {
            'pubmedbert': gene_pubmedbert,
            'esm2': gene_esm2
        },
        'drug': {
            'pubmedbert': drug_pubmedbert,
            'FP': drug_FP
        },
        'disease': {
            'pubmedbert': disease_pubmedbert,
        },
        'effect/phenotype': {
            'pubmedbert': phenotype_pubmedbert,
        }
    }
    return (feat)







