import json
import logging
from contextlib import asynccontextmanager
from typing import Union

from aiocache import Cache
from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from requests import Response
from sse_starlette import EventSourceResponse

from main.convertor import ABConverter
from main.helpers import tokenize_b
from main.sentence_preprocessing import sentence_to_token_id, node_embedding, search_topk
from server.helper import sse_format, make_final, make_error, make_log, enumerate_masks, get_uuid
from server.models import ChatRequest, DrugResponseDTO, ChatMessage
from src.server.loader import load_data
from src.server.model_inference import inferenceNetMedGpt
from src.server.model_loader import load_model

conv = ABConverter()
cache = Cache(Cache.MEMORY, namespace="chat")
LOG = logging.getLogger(__name__)

state = {}


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
    response_model=ChatMessage,
    summary="Recommend drugs for a clinical question",
    description=(
            "Takes a free-text clinical question, maps it to the internal graph "
            "representation, and returns a list of recommended drug names inferred "
            "by the NetMedGPT model."
    ),
    tags=["drug-recommendation"],
)
async def chat(user_text: Union[
    str, None] = "for diabetes with egfr mutation what is the best treatment and what are the adverse drug reactions") -> ChatMessage:
    req = ChatRequest(message=user_text, history=[])

    final_sse_event = None
    async for event in event_gen(req):
        final_sse_event = event

    if final_sse_event is None:
        raise ValueError("No events generated")

    return json.loads(final_sse_event['data'])


@app.post("/chat/stream", tags=["drug-recommendation"],
    summary="Recommend drugs for a clinical question",
    description=(
            "Takes a free-text clinical question, maps it to the internal graph "
            "representation, and returns a list of recommended drug names inferred "
            "by the NetMedGPT model."
    ))
async def chat_stream(req: ChatRequest, request: Request):
    return EventSourceResponse(event_gen(req))


async def event_gen(req: ChatRequest):
    try:
        uid = get_uuid()
        user_text = req.message
        LOG.info("USER TEXT: %s", user_text)

        # Stream initial progress
        yield sse_format(make_log(uid, "Processing your question..."))

        try:
            sentence, node_type = conv.a_to_b(user_text)
        except ValueError as e:
            yield sse_format(make_error(uid, str(e)))
            return
        yield sse_format(make_log(uid, f"Mapped to internal type: {node_type}"))

        # Cache check
        cached_value = await cache.get(user_text)
        if cached_value is not None:
            yield sse_format(make_log(uid, "Cache hit"))
            dr = DrugResponseDTO(**cached_value)
            yield sse_format(make_final(uid, dr, dr.message))
            return

        # Tokenization
        list_nodes_sentence, node_indices, sentence_indices, mask_idx = \
            sentence_to_token_id(sentence, state["mask_token"], state["relation_index"])

        yield sse_format(make_log(uid, "Tokenized input"))

        # Embedding + top-k search
        attr_nodes = node_embedding(list_nodes_sentence)
        hits = search_topk(state["node_index"], attr_nodes, state["all_node_names"], k=1)

        yield sse_format(make_log(uid, "Nearest graph neighbors computed"))

        ambiguous = []
        neighbor_indices = []
        for query_hits in hits:
            for name, cos, nid in query_hits:
                neighbor_indices.append(nid)
                if cos < 0.8:
                    ambiguous.append(name)

        if ambiguous:
            yield sse_format(make_log(uid, f"Ambiguous terms: {', '.join(ambiguous)}"))

        # Replace tokens with node IDs
        for i, idx in zip(node_indices, neighbor_indices):
            sentence_indices[i] = idx

        encoded_sentence = ",".join(map(str, sentence_indices))

        cached_value = await cache.get(encoded_sentence)
        if cached_value is not None:
            yield sse_format(make_log(uid, "Cache hit for encoded sentence"))
            dr = DrugResponseDTO(**cached_value)
            yield sse_format(make_final(uid, dr, dr.message))
            return

        yield sse_format(make_log(uid, "Running NetMedGPT inference..."))

        predictions = inferenceNetMedGpt(
            encoded_sentence, node_type, str(mask_idx),
            state["nodes"], state["edges"], state["model"]
        )

        b_text = conv.b_to_a(sentence)
        final_msg_text = f"The agent understood your question as: {b_text}"

        yield sse_format(make_log(uid, final_msg_text))

        dr = DrugResponseDTO(
            predictions=predictions.tolist(),
            prediction_type=node_type,
            message=final_msg_text,
        )

        # Cache result
        await cache.set(user_text, dr.model_dump(), ttl=3600)
        await cache.set(encoded_sentence, dr.model_dump(), ttl=3600)

        # Final SSE message
        yield sse_format(make_final(uid, dr, final_msg_text))

    except Exception as e:
        LOG.exception("Fatal error in SSE endpoint")
        yield sse_format(make_error(f"Internal error: {str(e)}"))
