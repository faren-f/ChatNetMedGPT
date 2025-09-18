import pandas as pd
import numpy as np
import torch
import re
from transformers import AutoTokenizer, AutoModel
import faiss
import torch.nn.functional as F


####
# This code has different parts
# 1) snake_case the sentence
# 2) turn sentence_to_token_id
# 3) embed the nodes into the embedding space 
# 4) find the closest nodes to the KG 
################################################################################################################
# 1) snake_case the sentence
# convert pseudo_sentence to snake_case: in this code we first make a snake_case form of the tokens in the sentence to be comparable with the node_names in the KG 
def to_snake_token(tok: str) -> str:
    s = tok.strip().lower()
    s = re.sub(r"[^\w]+", "_", s)      # non-alnum -> underscore
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def sentence_to_snake(b: str,
               preserve_masks: bool = False,
               preserve_gene_like: bool = False) -> str:
    """
    Convert each whitespace-separated token in B to snake_case.
    Optional:
      - preserve_masks=True keeps MASK0/MASK1 uppercase.
      - preserve_gene_like=True keeps tokens like TP53 uppercase.
    """
    tokens = b.strip().split()
    out = []
    for t in tokens:
        if preserve_masks and t.upper() in {"MASK0", "MASK1"}:
            out.append(t.upper()); continue
        if preserve_gene_like and t.isupper() and any(c.isdigit() for c in t):
            out.append(t); continue
        out.append(to_snake_token(t))
    return " ".join(out)

################################################################################################################
# 2) turn sentence_to_token_id
def sentence_to_token_id(sentence, mask_token, relation_index):
    # convert sentence to snake_case form
    sentence = sentence_to_snake(sentence)
    sentence = sentence.split()
    
    # sentence_indices = [mask_token]*len(sentence) 
    # all_indeces = list(np.arange(0,len(sentence))) # first mask token for all of them

    sentence_indices = [mask_token]*9 
    all_indeces = list(np.arange(0,9)) # first mask token for all of them
    
    # identify the token of all the relations
    rel2z = dict(zip(relation_index["relation"], relation_index["z_index"]))
    
    # find the index of the of the relations in the pseudo-sentence
    relation_token_index = [
        {"index": i, "relation_name": tok, "token": rel2z[tok]}
        for i, tok in enumerate(sentence)
        if tok in rel2z
    ]
    
    relation_token_index_df = pd.DataFrame(relation_token_index, columns=["index", "relation_name", "token"])
    # print(f'relation_token_index_df: {relation_token_index_df}')
    
    # substitute index of relations in the sentence
    for i, pos in zip(relation_token_index_df['index'], relation_token_index_df['token']):
        sentence_indices[i] = pos
    
    # print(f'sentence_indices: {sentence_indices}')
    
    # remove relation indices from the sentence to find the tokens of nodes
    remaining_indices = list(set(all_indeces) - set(relation_token_index_df['index']))
    # print(f'remaining_indices: {remaining_indices}')
    
    ##### remove the indices of masks from the sentence
    mask_indices = []
    
    for tok in ['mask1', 'mask0']:
        try:
            mask_indices.append([i for i, x in enumerate(sentence) if x == tok][0])
        except:
            pass
    
    # print(f'mask_indices: {mask_indices}')
    mask_index_question = mask_indices[0] # The model only prioritize nodes for mask1, so we need to know its index 
    
    # remove mask tokensfrom the sentence to only remain node tokens
    remaining_indices = list(set(remaining_indices) - set(mask_indices))
    # print(f'remaining_indices:{remaining_indices}')
    
    # remove masks from the sentence
    set_sentence = set(sentence)- {'mask0','mask1'}
    
    # remove relations from the remaning sentence to keep only node_names
    set_sentence = set_sentence-set(relation_token_index_df['relation_name'])
    
    list_nodes_sentence = list(set_sentence)
    list_nodes_sentence
    
    node_indices = []
    for tok in list_nodes_sentence:
        node_indices.append([i for i, x in enumerate(sentence) if x == tok][0])
    
    # print(f'node_indices: {node_indices}')
    return(list_nodes_sentence, node_indices, sentence_indices, mask_index_question)

################################################################################################################
# 3) embed the nodes into the embedding space 
# Load model from HuggingFace Hub
# Mean Pooling - Take attention mask into account for correct averaging
def meanpooling(output, mask):
    embeddings = output[0] # First element of model_output contains all token embeddings
    mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
    return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

def node_embedding(list_nodes_sentence):
    tokenizer = AutoTokenizer.from_pretrained("neuml/pubmedbert-base-embeddings")
    model = AutoModel.from_pretrained("neuml/pubmedbert-base-embeddings")
    
    attr_nodes = torch.empty((0, 768))
    # Tokenize sentences
    for i in list_nodes_sentence:
        inputs = tokenizer(i, padding=True, truncation=True, max_length=512, return_tensors='pt')
    
        # Compute token embeddings
        with torch.no_grad():
            output = model(**inputs)
        
        # Perform pooling. In this case, mean pooling.
        emb = meanpooling(output, inputs['attention_mask'])
        attr_nodes = torch.concat((attr_nodes, emb), dim = 0)
    
    # print(f'Sentence embeddings: {attr_nodes}')
    return(attr_nodes)


################################################################################################################
# 4) find the closest nodes to the KG 
def build_faiss_ip_index(node_emb):
    """
    node_emb: torch.Tensor [N, D] or numpy.ndarray [N, D]
    Returns a FAISS index (inner product, i.e. cosine if normalized).
    """
    import numpy as np, faiss, torch.nn.functional as F, torch
    
    if isinstance(node_emb, torch.Tensor):
        with torch.no_grad():
            nodes = F.normalize(node_emb.float(), dim=1).cpu().numpy()
    else:
        nodes = node_emb.astype("float32", copy=False)
        nodes /= np.linalg.norm(nodes, axis=1, keepdims=True) + 1e-12

    d = nodes.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(nodes)
    return index

def search_topk(
    index: faiss.Index,
    query_emb: torch.Tensor,
    node_names: list,   # len N list of strings
    k: int = 5
):
    """
    query_emb: [Q, D] torch tensor
    Returns: list of length Q; each is list of (name, cosine, id)
    """
    with torch.no_grad():
        q = F.normalize(query_emb.detach().cpu().float(), dim=1).numpy().astype('float32', copy=False)
    sims, ids = index.search(q, k)  # sims: [Q, k], ids: [Q, k]
    results = []
    for qi in range(q.shape[0]):
        hits = []
        for j in range(k):
            nid = int(ids[qi, j])
            if nid == -1:
                continue
            hits.append((node_names[nid], float(sims[qi, j]), nid))
        results.append(hits)
    return results