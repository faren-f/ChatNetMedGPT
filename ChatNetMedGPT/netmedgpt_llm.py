import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "/home/bbc8731/NetMedGPT/")))

import subprocess
import argparse

from convertor import *
from helpers import *
from sentence_preprocessing import *

# read data from here
nodes = pd.read_csv('data/nodes_snake.csv', sep= ',')
emb_nodes = pd.read_csv("data/emb_pubmedbert_all_nodes.csv", index_col=0)
edges = pd.read_csv('data/edges.csv', sep= ',')
relation_index = edges[['relation', 'z_index']].drop_duplicates()
mask_token = edges['z_index'].max() + 1
# examples_a = ["for diabetes with egft mutation what is the best traetment and do not know the adverse drug reactions",]


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

# for a in examples_a:
#     sentence = conv.a_to_b(a)
#     print("A:", a)
#     print("B:", sentence, f"(tokens={len(tokenize_b(sentence))})")
#     print("-" * 60)

# this part is when we want to back from pseudo-sentence to user text
# examples_b = [
#     "prioritized drugs are: metformin, nilotinib",
#     "TP53 disease_protein MASK indication MASK",
#     "breast cancer indication metformin drug_effect MASK"
# ]
# print("\n=== B -> A ===")
# for b in examples_b:
#     a = conv.b_to_a(b)
#     print("B:", b)
#     print("A:", a)
#     print("-" * 60)

list_nodes_sentence, node_indices, sentence_indices, mask_index_question = sentence_to_token_id(sentence, mask_token, relation_index)
print("Mask index question from sentence_to_token_id:", mask_index_question)

attr_nodes = node_embedding(list_nodes_sentence)

# ---- Example usage ----
# emb_nodes: [N, D] tensor of your 129k nodes
# node_names: list[str] of length N
# attr_nodes: [Q, D] tensor (your pseudo_sentence embeddings)

all_node_names = nodes['node_name']

index = build_faiss_ip_index(emb_nodes)
hits_per_query = search_topk(index, attr_nodes, all_node_names, k=1)

neighbor_indices = []
neighbors = []
for i, hits in enumerate(hits_per_query):
    # print(f"Query {i}")
    for name, cos, nid in hits:
        neighbors.append(name)
        neighbor_indices.append(nid) 
        print(f"  {name}  (id={nid})  cosine={cos:.4f}")

for i, index in zip(node_indices, neighbor_indices):
    sentence_indices[i] = index
    
sentence_indices

# Convert indices list to comma-separated string
sentence_str = ",".join(map(str, sentence_indices))

# Build the command to call the second script
cmd = [
    "python", "/home/bbc8731/NetMedGPT/ChatNetMedGPT/run_netmedgpt.py",
    "--sentence", sentence_str,
    "--node_type", node_type,
    "--mask_index_question", str(mask_index_question)
]

print("Running:", " ".join(cmd))
subprocess.run(cmd)



## python netmedgpt_llm.py --user_text "question"

# python netmedgpt_llm.py --user_text "for diabetes with egfr mutation what is the best treatment and what are the adverse drug reactions"

