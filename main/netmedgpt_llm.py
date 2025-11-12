import sys
import os
import json
import subprocess
import argparse

from convertor import *
from helpers import *
from sentence_preprocessing import *


with open("../data/parameters.json", 'r') as file:
    param = json.load(file)
data_dir = param['files']['data_dir']
nodes = pd.read_csv(os.path.join(param['files']['data_dir'], "nodes_snake.csv"), sep= ',')
emb_nodes = pd.read_csv(os.path.join(param['files']['data_dir'],"emb_pubmedbert_all_nodes.csv"), index_col=0)
edges = pd.read_csv(os.path.join(param['files']['data_dir'],'edges.csv'), sep= ',')
relation_index = edges[['relation', 'z_index']].drop_duplicates()
mask_token = edges['z_index'].max() + 1


parser = argparse.ArgumentParser()
parser.add_argument("--user_text", required=True, help="User query in plain English")
args = parser.parse_args()
user_text = args.user_text

## this part is when we want to convert user text to pseudo-sentence
conv = ABConverter()
sentence, node_type = conv.a_to_b(user_text)
print("A:", user_text)
print("B:", sentence, f"(tokens={len(tokenize_b(sentence))})")
print("node_type:", node_type)
print("-" * 60)

list_nodes_sentence, node_indices, sentence_indices, mask_index_question = sentence_to_token_id(sentence, mask_token, relation_index)
print("Mask index question from sentence_to_token_id:", mask_index_question)

attr_nodes = node_embedding(list_nodes_sentence)
all_node_names = nodes['node_name']

index = build_faiss_ip_index(emb_nodes)
hits_per_query = search_topk(index, attr_nodes, all_node_names, k=1)

neighbor_indices = []
neighbors = []
for i, hits in enumerate(hits_per_query):
    for name, cos, nid in hits:
        neighbors.append(name)
        neighbor_indices.append(nid) 
        print(f"  {name}  (id={nid})  cosine={cos:.4f}")

for i, index in zip(node_indices, neighbor_indices):
    sentence_indices[i] = index

sentence_str = ",".join(map(str, sentence_indices))

cmd = [
    "python", "run_netmedgpt.py",
    "--sentence", sentence_str,
    "--node_type", node_type,
    "--mask_index_question", str(mask_index_question)
]

subprocess.run(cmd)
os._exit(0)


