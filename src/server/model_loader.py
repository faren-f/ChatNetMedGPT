import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os

import pandas as pd
import torch

from src.model_pretrain import TransformerModel

def load_embeddings():
    feat = torch.load("data/embeddings_with_feat.pt")
    return feat

def load_model(edge, nodes , mask_token):
    ##########################
    device = torch.device('cpu')
    feat = load_embeddings()

    vocab_size = mask_token + 1

    relation_type = list(edge['relation'].unique())
    node_types = list(nodes['node_type'].unique())

    mask = ['mask']
    entity = node_types + relation_type + mask

    relation_index = edge.loc[
        edge['relation'].isin(relation_type), ['relation', 'z_index']].drop_duplicates()
    mask_row = pd.DataFrame([['mask', mask_token]], columns=['relation', 'z_index'])
    relation_mask_index = pd.concat([relation_index, mask_row], ignore_index=True)

    # load the model
    tag = f"NetMedGPT"
    model_dir = "model"
    checkpoint_path = os.path.join(model_dir, f"{tag}.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    param = checkpoint['parameters']
    state_dict = checkpoint['model_state_dict']

    # instantiate the model
    model = TransformerModel(
        vocab_size,
        param['hidden_channels'],
        param['nhead'],
        param['N_encoder_layers'],
        (param['walk_length'] * 2) - 1,  # walk_length_with_relation
        device=device,
        feat=feat,
        nodes=nodes,
        entity=entity,
        relation_mask_index=relation_mask_index,
        pos_emb='fixed',
    ).to(device)

    # load the model
    model.load_state_dict(state_dict)

    model.eval()
    return model