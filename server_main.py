import logging
from contextlib import asynccontextmanager
from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

from main.convertor import ABConverter
from main.helpers import tokenize_b
from main.sentence_preprocessing import sentence_to_token_id, node_embedding, search_topk
from src.server.loader import load_data
from src.server.model_inference import inferenceNetMedGpt
from src.server.model_loader import load_model

conv = ABConverter()
LOG = logging.getLogger(__name__)


state = {}

class DrugResponse(BaseModel):
    """Response model returning a list of recommended drug names."""
    drugs: list[str]
    node_type: str
    sentence: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    LOG.info("LIFESPACE START")
    all_node_names, node_index, edges, relation_index, mask_token, nodes = load_data()
    state['edges'] = edges
    state['relation_index'] = relation_index
    state['mask_token'] = mask_token
    state['node_index'] = node_index
    state['all_node_names'] = all_node_names
    state['nodes'] = nodes

    state['model'] = load_model(edges, nodes, mask_token)
    LOG.info("LIFESPACE Loaded")
    yield
    # --- shutdown ---
    state.clear()
    LOG.info("LIFESPACE END")

app = FastAPI(
    title="ChatNetMedGPT API",
    description=(
        "API for graph-based retrieval and recommendation of drug candidates "
        "for clinical questions using the NetMedGPT model."
    ),
    version="0.1.0",
    contact={
        "name": "Farzaneh firoozbakht, Simon Süwer",
        "email": "farzaneh.firoozbakht@uni-hamburg.de",
    },
    lifespan=lifespan,
)

@app.get(
    "/chat/",
    response_model=DrugResponse,
    summary="Recommend drugs for a clinical question",
    description=(
        "Takes a free-text clinical question, maps it to the internal graph "
        "representation, and returns a list of recommended drug names inferred "
        "by the NetMedGPT model."
    ),
    tags=["drug-recommendation"],
)
def chat(user_text: Union[str, None] = "for diabetes with egfr mutation what is the best treatment and what are the adverse drug reactions") -> DrugResponse:
    LOG.info(user_text)
    sentence, node_type = conv.a_to_b(user_text)
    LOG.info(f"A:  {user_text} B: {sentence} (tokens={len(tokenize_b(sentence))}), node_type: {node_type}")

    all_node_names = state['all_node_names']
    node_index = state['node_index']
    relation_index = state['relation_index']
    mask_token = state['mask_token']
    model = state['model']
    nodes = state['nodes']
    edges = state['edges']

    list_nodes_sentence, node_indices, sentence_indices, mask_index_question = sentence_to_token_id(
        sentence, mask_token, relation_index)
    LOG.info("Mask index question from sentence_to_token_id:", mask_index_question)

    attr_nodes = node_embedding(list_nodes_sentence)

    hits_per_query = search_topk(node_index, attr_nodes, all_node_names, k=1)
    LOG.info("Nearest neighbors found:" + str(hits_per_query))
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
    drug_names = inferenceNetMedGpt(sentence_str, node_type, str(mask_index_question), nodes, edges, model)
    return DrugResponse(drugs=drug_names.tolist(), node_type=node_type, sentence=sentence)
