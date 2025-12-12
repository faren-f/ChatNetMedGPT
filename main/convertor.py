import json
import logging
from typing import List, Optional

from main.helpers import looks_like_b_chain, tokenize_b, enforce_max_tokens, revise_b, \
    is_valid_relation, send_chat
from server.models import ChatMessage


class ModelResponseError(Exception):
    """Base exception for invalid or unusable model responses."""
    pass


# A: text
# B: pseudo-sentence
# ========= CONVERSION SETTINGS =========
B_MAX_TOKENS = 9
ALLOWED_RELATIONS = {
    'anatomy_anatomy', 'anatomy_protein_absent', 'anatomy_protein_present', 'bioprocess_bioprocess',
    'bioprocess_protein', 'cellcomp_cellcomp', 'cellcomp_protein', 'contraindication',
    'disease_disease',
    'disease_phenotype_negative', 'disease_phenotype_positive', 'disease_protein', 'drug_drug',
    'drug_effect',
    'drug_protein', 'exposure_bioprocess', 'exposure_cellcomp', 'exposure_disease',
    'exposure_exposure',
    'exposure_molfunc', 'exposure_protein', 'indication', 'molfunc_molfunc', 'molfunc_protein',
    'off_label_use',
    'pathway_pathway', 'pathway_protein', 'phenotype_phenotype', 'phenotype_protein',
    'protein_protein'
}
ALLOWED_NODE_TYPES = {"gene/protein", "drug", "effect/phenotype", "disease", "biological_process",
                      "molecular_function", "cellular_component", "exposure", "pathway", "anatomy"}

# ==== relation vocabulary & synonyms ====
RELATION_SYNONYMS = {
    "associated": "associate",
    "is associated": "disease_protein",
    "association": "disease_protein",
    "associates": "disease_protein",
    "indicate": "indication",
    "indicates": "indication",
    "has": "drug_effect",
    "adverse_effect": "drug_effect",
    "side_effect": "drug_effect",
}

MASK_PRIMARY = "MASK1"
MASK_SECONDARY = "MASK0"
ALLOWED_MASKS = {MASK_PRIMARY, MASK_SECONDARY}

# ========= PROMPTS =========
with open('main/system_A2B.txt', 'r') as f:
    SYSTEM_A_TO_B = f.read()

with open('main/system_B2A.txt', 'r') as f:
    SYSTEM_B_TO_A = f.read()

USER_A_TO_B_TEMPLATE = """Convert this A to B:

A: {text}
B:"""

USER_B_TO_A_TEMPLATE = """Convert this B to A:

B: {text}
A:"""


# ========= CONVERTER =========
class ABConverter:
    def __init__(self, max_tokens: int = B_MAX_TOKENS):
        self.max_tokens = max_tokens

    def a_to_b(self, text_a: str, attempts: int = 3,
               history: Optional[List[ChatMessage]] = None) -> str:
        system = SYSTEM_A_TO_B.format(
            allowed_relations=", ".join(sorted(ALLOWED_RELATIONS)),
            max_tokens=self.max_tokens,
            allowed_nodes=", ".join(sorted(ALLOWED_NODE_TYPES))
        )

        user = USER_A_TO_B_TEMPLATE.format(text=text_a)

        # === First model call ===
        # b = send_chat(system, user)
        raw = send_chat(system, user, history)
        logging.info("Received B from model:\nQuestion: %s\n%s", user, raw)
        # print(raw)
        # print("-*" * 60)
        b, node_type = self.parse_a_to_b_response(raw)

        def count_masks(tokens):
            n1 = sum(t == MASK_PRIMARY for t in tokens)
            n0 = sum(t == MASK_SECONDARY for t in tokens)
            legacy = sum(t.lower() in {"[mask]", "mask", "<mask>"} for t in tokens)
            return n1, n0, legacy

        def sanitize_legacy_masks(tokens):
            # Convert any [mask]/mask/<mask> to MASK0 (secondary) for now
            return [MASK_SECONDARY if t.lower() in {"[mask]", "mask", "<mask>"} else t for t in
                    tokens]

        for _ in range(attempts):
            ok_len, b = enforce_max_tokens(b, self.max_tokens)
            tokens = tokenize_b(b)
            if ok_len and looks_like_b_chain(tokens):
                return b, node_type
            # ask model to revise
            b = revise_b(b, text_a)

        # === last-resort repair ===
        tokens = tokenize_b(b)
        tokens = sanitize_legacy_masks(tokens)

        # Ensure odd positions are valid relations
        for i in range(len(tokens)):
            if i % 2 == 1 and not is_valid_relation(tokens[i]):
                tokens[i] = "indication"

        # Ensure exactly one MASK1
        n1, n0, _ = count_masks(tokens)
        if n1 == 0:
            for i in range(0, len(tokens), 2):
                if tokens[i] in ALLOWED_MASKS:
                    tokens[i] = MASK_PRIMARY
                    break
            for i in range(0, len(tokens), 2):
                if tokens[i] in ALLOWED_MASKS and tokens[i] != MASK_PRIMARY:
                    tokens[i] = MASK_SECONDARY
        elif n1 > 1:
            seen = False
            for i in range(0, len(tokens), 2):
                if tokens[i] == MASK_PRIMARY:
                    if not seen:
                        seen = True
                    else:
                        tokens[i] = MASK_SECONDARY

        tokens = tokens[:B_MAX_TOKENS]
        b = " ".join(tokens)
        return b.strip(), raw

    def b_to_a(self, text_b: str) -> str:
        system = SYSTEM_B_TO_A
        user = USER_B_TO_A_TEMPLATE.format(text=text_b)
        return send_chat(system, user)

    def parse_a_to_b_response(self, raw: str) -> (str, str):
        try:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"Model did not return valid JSON. Got: {raw}") from e

            if "error" in data:
                raise ModelResponseError(data.get("error"))

            b = data.get("chain")
            node_type = data.get("nodeType")

            if not b:
                raise ValueError(f'Model JSON missing "chain". Got: {raw}')

            if not node_type:
                raise ValueError(f'Model JSON missing "nodeType". Got: {raw}')

            return b, node_type
        except ValueError as e:
            b_line, node_line = None, None
            for line in raw.splitlines():
                if line.strip().startswith("B:"):
                    b_line = line.replace("B:", "").strip()
                if line.strip().startswith("NodeType:"):
                    node_line = line.replace("NodeType:", "").strip()

            if not b_line:
                raise ValueError(f"Model did not return B: line. Got: {raw}")
            if not node_line:
                node_line = "unknown"

            b = b_line
            node_type = node_line
            return b, node_type
