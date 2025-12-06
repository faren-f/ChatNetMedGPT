import pandas as pd
import torch

from main.sentence_preprocessing import build_faiss_ip_index


def load_data():
    nodes_snake = pd.read_csv('data/nodes_snake.csv', sep= ',')
    all_node_emb = torch.load("data/node_emb_pubmedbert_for_similarity_search.pt")
    edges = pd.read_csv('data/edges.csv', sep= ',')
    nodes = pd.read_csv('data/nodes.csv', sep=',')
    relation_index = edges[['relation', 'z_index']].drop_duplicates()
    mask_token = edges['z_index'].max() + 1
    all_node_names = nodes_snake['node_name']
    index = build_faiss_ip_index(all_node_emb)

    return all_node_names, index, edges, relation_index, mask_token, nodes