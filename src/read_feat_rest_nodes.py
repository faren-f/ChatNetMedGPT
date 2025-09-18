# assign random or fix features to the nodes without prior knowledge
import os
import pandas as pd
import json
import torch

def feat_rest_nodes(nodes, edge, hidden_channels):
    
    relation_type = list(edge['relation'].unique())
    mask = ['mask']
    node_types_without_feat = ['biological_process',
                               'molecular_function',
                               'cellular_component',
                               'exposure',
                               'pathway',
                               'anatomy']
    
    without_feat = {
        'node': node_types_without_feat,
        'relation': relation_type,
        'mask': mask
    }
    
    feat_rest = {}
    for k, v in without_feat.items():
        if k == 'node':
            for i in v:
                N_nodes_i = (nodes['node_type'] == i).sum()
                feat_i = torch.randn(N_nodes_i, hidden_channels)
                feat_rest[f"{i}"] = {'random': feat_i}
    
        elif k == 'relation':
             for i in v:
                 feat_i = torch.ones(1, hidden_channels)
                 feat_rest[f"{i}"] = {'fixed': feat_i}   # I should put this also fixed instead of random, but now the result is based on random
        elif k == 'mask':
            feat_i = torch.ones(1, hidden_channels)
            feat_rest[f"{k}"] = {'fixed': feat_i}  

    return (feat_rest)






    
        