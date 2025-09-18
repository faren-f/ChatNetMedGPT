# get all node ids of the second nodev type
import os
import json
import pandas as pd
import torch


def ids_nodetype2(nodes, edge):

    all_ids = {
        'drug': torch.tensor(
            nodes.loc[nodes['node_type'] == 'drug', 'node_index'].unique(), dtype=torch.int64
        ),
        'gene': torch.tensor(
            nodes.loc[nodes['node_type'] == 'gene/protein', 'node_index'].unique(), dtype=torch.int64
        ),
        'phenotype': torch.tensor(
            nodes.loc[nodes['node_type'] == 'effect/phenotype', 'node_index'].unique(), dtype=torch.int64
        )
    }
    
    # Get mapping from node_types to z_index values
    node_type_to_z = edge.groupby('node_types')['z_index'].unique().to_dict()
    
    
    all_ids_node2 = {}
    
    for z in node_type_to_z.get('disease|drug', []):
        all_ids_node2[z] = all_ids['drug']
    
    for z in node_type_to_z.get('drug|gene/protein', []):
        all_ids_node2[z] = all_ids['gene']
    
    for z in node_type_to_z.get('drug|effect/phenotype', []):
        all_ids_node2[z] = all_ids['phenotype']
        
    return(all_ids_node2)







