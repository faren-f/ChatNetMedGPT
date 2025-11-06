import pandas as pd

from ChatNetMedGPT.sentence_preprocessing import build_faiss_ip_index


def load_data():
    nodes = pd.read_csv('../../data/nodes_snake.csv', sep= ',')
    emb_nodes = pd.read_csv("../../data/emb_pubmedbert_all_nodes.csv", index_col=0)
    edges = pd.read_csv('../../data/edges.csv', sep= ',')
    relation_index = edges[['relation', 'z_index']].drop_duplicates()
    mask_token = edges['z_index'].max() + 1
    all_node_names = nodes['node_name']
    index = build_faiss_ip_index(emb_nodes)

    return all_node_names, index, edges, relation_index, mask_token