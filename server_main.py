import json
import logging
from contextlib import asynccontextmanager
from typing import Union

from aiocache import Cache
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from main.convertor import ABConverter
from main.helpers import tokenize_b
from main.sentence_preprocessing import sentence_to_token_id, node_embedding, search_topk
from src.server.loader import load_data
from src.server.model_inference import inferenceNetMedGpt
from src.server.model_loader import load_model

conv = ABConverter()
cache = Cache(Cache.MEMORY, namespace="chat")
LOG = logging.getLogger(__name__)

state = {}

# New ======
def enumerate_masks(sentence):
    return sentence.count("MASK")

class Response(BaseModel):
    """Response model returning a list of recommended drug names."""
    message: str
    predictions: list[str]
    prediction_type: str
    # neighbors: list[str]
    # list_nodes_sentence: list[str]
    # sentence: str
# ==========

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

    await cache.clear()
    LOG.info("LIFESPACE Loaded")
    yield
    # --- shutdown ---
    state.clear()
    await cache.clear()
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
    root_path_in_servers=False,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get(
    "/chat",
    response_model=DrugResponse,
    summary="Recommend drugs for a clinical question",
    description=(
            "Takes a free-text clinical question, maps it to the internal graph "
            "representation, and returns a list of recommended drug names inferred "
            "by the NetMedGPT model."
    ),
    tags=["drug-recommendation"],
)
async def chat(user_text: Union[
    str, None] = "for diabetes with egfr mutation what is the best treatment and what are the adverse drug reactions") -> DrugResponse:

    # New ======
    message = []
    # ==========
    
    LOG.info(user_text)
    cached_value = await cache.get(user_text)
    if cached_value is not None:
        return cached_value
    sentence, node_type = conv.a_to_b(user_text)
    LOG.info(
        f"A:  {user_text} B: {sentence} (tokens={len(tokenize_b(sentence))}), node_type: {node_type}")
    
    # New ======
    n_mask = enumerate_masks(sentence)
    if n_mask > 1:
        message.append(
            f"More than a single question is found in the given query. For now, we only proceed with {node_type}. The user can then refine the query to get response to other questions."
        )
    else:
        message.append(
            f"The queried node type is {node_type}."
        )    
    # =======
    
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
    ambiguous_tokens = []     # new ======
    for i, hits in enumerate(hits_per_query):
        # print(f"Query {i}")
        for name, cos, nid in hits:
            neighbors.append(name)
            neighbor_indices.append(nid)
            print(f"  {name}  (id={nid})  cosine={cos:.4f}")
            # New ========
            if cos < .8:
                ambiguous_tokens.append(name)
            # =============

    for i, index in zip(node_indices, neighbor_indices):
        sentence_indices[i] = index

    sentence_str = ",".join(map(str, sentence_indices))
    # new =========
    if ambiguous_tokens:
    message.append(
        f"There are ambiguous words in user's query, which we deemed them as following: {','.join(ambiguous_tokens)}")
    # ==============
    cached_value = await cache.get(sentence_str)
    if cached_value is not None:
        return json.loads(sentence_str)

    # Convert indices list to comma-separated string
    predictions = inferenceNetMedGpt(sentence_str, node_type, str(mask_index_question), nodes, edges, model)
    # New ========
    b_text = conv.b_to_a(sentence)
    message.append(f"The queried question as the agent understood: {b_text}")
   
    response = Response(message='\n'.join(message),
                        predictions=predictions.tolist(),
                        prediction_type=node_type
                       )
    # =============
    await cache.set(user_text, response, ttl=3600)
    await cache.set(sentence_str, response, ttl=3600)
    return response
