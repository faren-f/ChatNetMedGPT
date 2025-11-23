import requests
from typing import List, Tuple
import re
from convertor import *

# ========= SERVER CONFIG =========
protocol = "https"
hostname = "dev.chat.cosy.bio"
host = f"{protocol}://{hostname}"
api_key = "sk-45405987006a4a4d8b4deb9e7588c6bc"
api_url = f"{host}/ollama/api/chat"
MODEL_NAME = "gpt-oss:20b"
B_MAX_TOKENS = 9

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

ALLOWED_RELATIONS = {
    'anatomy_anatomy', 'anatomy_protein_absent', 'anatomy_protein_present', 'bioprocess_bioprocess', 
    'bioprocess_protein', 'cellcomp_cellcomp', 'cellcomp_protein', 'contraindication', 'disease_disease',
    'disease_phenotype_negative', 'disease_phenotype_positive', 'disease_protein', 'drug_drug', 'drug_effect',
    'drug_protein', 'exposure_bioprocess', 'exposure_cellcomp', 'exposure_disease', 'exposure_exposure', 
    'exposure_molfunc', 'exposure_protein', 'indication', 'molfunc_molfunc', 'molfunc_protein','off_label_use', 
    'pathway_pathway', 'pathway_protein', 'phenotype_phenotype', 'phenotype_protein', 'protein_protein'
}

# ========= CHAT =========
def send_chat(system: str, user: str, model: str = MODEL_NAME, stream: bool = False) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}      
            # assistant should be added later
        ],
        "stream": stream
    }
    resp = requests.post(api_url, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"].strip()

# ========= HELPERS =========
def tokenize_b(s: str) -> List[str]:
    return s.strip().split()

def is_valid_relation(tok: str) -> bool:
    return tok in ALLOWED_RELATIONS

def looks_like_b_chain(tokens: List[str]) -> bool:
    if len(tokens) < 2:
        return False
    for i, tok in enumerate(tokens):
        if i % 2 == 1:
            if not is_valid_relation(tok):
                return False
        else:
            if tok == "":
                return False
    return True

def enforce_max_tokens(text: str, max_tokens: int = B_MAX_TOKENS) -> Tuple[bool, str]:
    toks = tokenize_b(text)
    return (len(toks) <= max_tokens, text.strip())

def revise_b(previous_b: str, a_text: str, mask_error: str = "") -> str:
    system = f"""Revise the relation chain (format B) to meet ALL constraints:
- Use ONLY relations from: {', '.join(sorted(ALLOWED_RELATIONS))}
- TOTAL TOKENS ≤ {B_MAX_TOKENS}
- Alternate ENTITY (even idx) and RELATION (odd idx) starting with ENTITY.
- Entities are one token; replace spaces with underscores.
- MASK RULES: exactly one MASK1 (the primary unknown asked by the user), any other unknowns MASK0; masks only in ENTITY slots; allowed masks: MASK1,MASK0; never use [mask].
- Output ONLY the corrected chain, no commentary."""
    error_note = f"\nValidator error: {mask_error}" if mask_error else ""
    user = f"""A: {a_text}
Previous B: {previous_b}{error_note}
Revised B (do not add new entities/relations beyond minimal fixes):"""
    return send_chat(system, user)
