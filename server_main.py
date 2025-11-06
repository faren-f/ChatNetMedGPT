from typing import Union

from fastapi import FastAPI

from ChatNetMedGPT.convertor import ABConverter
from ChatNetMedGPT.helpers import tokenize_b
from ChatNetMedGPT.netmedgpt import netMedGpt
from ChatNetMedGPT.sentence_preprocessing import sentence_to_token_id, node_embedding, \
    search_topk
from src.server.loader import load_data

app = FastAPI()
conv = ABConverter()
all_node_names, node_index, edges, relation_index, mask_token = load_data()


@app.get("/chat/")
def chat(user_text: Union[str, None] = None):
    sentence, node_type = conv.a_to_b(user_text)
    print("A:", user_text)
    print("B:", sentence, f"(tokens={len(tokenize_b(sentence))})")
    print("node_type:", node_type)
    print("-" * 60)

    list_nodes_sentence, node_indices, sentence_indices, mask_index_question = sentence_to_token_id(
        sentence, mask_token, relation_index)
    print("Mask index question from sentence_to_token_id:", mask_index_question)

    attr_nodes = node_embedding(list_nodes_sentence)

    hits_per_query = search_topk(node_index, attr_nodes, all_node_names, k=1)

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

    # Convert indices list to comma-separated string
    sentence_str = ",".join(map(str, sentence_indices))
    drug_names, user_response = netMedGpt(sentence_str, node_type, str(mask_index_question))
    return {"response": user_response, "predicted_drugs": drug_names}
