import json
import os

import pandas as pd

from src.model_pretrain import *


def netMedGpt(input, node_type, mask_index_question):
    ##########################
    device = torch.device('cpu')
    with open("data/parameters.json", 'r') as file:
        param = json.load(file)
    model_dir = param['files']['model_dir']
    data_dir = param['files']['data_dir']
    user_response = os.path.join(param['files']['data_dir'], 'user_response')
    feat = torch.load(os.path.join(data_dir, "embeddings_with_feat.pt"))
    nodes = pd.read_csv(os.path.join(data_dir, 'nodes.csv'), sep=',')
    edge = pd.read_csv(os.path.join(data_dir, "edges.csv"))

    mask_token = edge['z_index'].max() + 1
    vocab_size = mask_token + 1

    relation_type = list(edge['relation'].unique())
    node_types = list(nodes['node_type'].unique())
    node_types_without_feat = ['biological_process',
                               'molecular_function',
                               'cellular_component',
                               'exposure',
                               'pathway',
                               'anatomy']

    mask = ['mask']
    entity = node_types + relation_type + mask

    N_top = 5
    relation_index = edge.loc[
        edge['relation'].isin(relation_type), ['relation', 'z_index']].drop_duplicates()
    mask_row = pd.DataFrame([['mask', mask_token]], columns=['relation', 'z_index'])
    relation_mask_index = pd.concat([relation_index, mask_row], ignore_index=True)

    # load the model
    tag = f"{model_dir}/NetMedGPT"

    if len(tag) > 0:
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

    nodes_at_mask = nodes.loc[nodes['node_type'] == node_type, ['node_index', 'node_name']].values
    node_ids_at_mask = torch.tensor(nodes_at_mask[:, 0].astype(int)).to(device)

    input = [int(i) for i in input.split(',')]
    sentence = torch.tensor([input])
    # mask_pos = int(mask_index_question)  # absolute position of the MASK

    # Treat --mask_index_question as an ABSOLUTE token position
    mask_pos = int(mask_index_question)

    # Safety check
    seq_len = sentence.size(1)
    if not (0 <= mask_pos < seq_len):
        raise IndexError(
            f"--mask_index_question {mask_pos} out of bounds; sequence length={seq_len}")

    if sentence[0, mask_pos].item() != mask_token:
        mask_positions = torch.where(sentence == mask_token)[1].tolist()
        raise ValueError(
            f"Token at position {mask_pos} is not MASK "
            f"(got {sentence[0, mask_pos].item()}, expected {mask_token}). "
            f"MASK positions in this sentence: {mask_positions}"
        )

    model.eval()
    input = sentence.to(device)
    with torch.no_grad():
        output = model(input)
        logits = output[:, mask_pos, node_ids_at_mask]
        probs = F.softmax(logits, dim=1)

        top_probs, top_idx = torch.topk(probs, N_top,
                                        dim=1)  # top_idx are positions in node_ids_at_mask
        top_idx = top_idx.cpu().numpy()[0]  # shape (N_top,)

        drug_names = nodes_at_mask[top_idx, 1]
    if not os.path.exists(user_response):
        os.makedirs(user_response, exist_ok=True)
    return drug_names, user_response
