from ChatNetMedGPT.helpers import send_chat, enforce_max_tokens, tokenize_b, looks_like_b_chain, revise_b
import re
import os

path = os.system('pwd')
print(path)

# A: Human text
# B: pseudo-sentence
# ========= CONVERSION SETTINGS =========
B_MAX_TOKENS = 9
ALLOWED_RELATIONS = {
    'anatomy_anatomy', 'anatomy_protein_absent', 'anatomy_protein_present', 'bioprocess_bioprocess', 
    'bioprocess_protein', 'cellcomp_cellcomp', 'cellcomp_protein', 'contraindication', 'disease_disease',
    'disease_phenotype_negative', 'disease_phenotype_positive', 'disease_protein', 'drug_drug', 'drug_effect',
    'drug_protein', 'exposure_bioprocess', 'exposure_cellcomp', 'exposure_disease', 'exposure_exposure', 
    'exposure_molfunc', 'exposure_protein', 'indication', 'molfunc_molfunc', 'molfunc_protein','off-label use', 
    'pathway_pathway', 'pathway_protein', 'phenotype_phenotype', 'phenotype_protein', 'protein_protein'
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

# ========= PROMPTS =========
with open('ChatNetMedGPT/system_A2B.txt', 'r') as f:
    SYSTEM_A_TO_B = f.read()
    
with open('ChatNetMedGPT/system_B2A.txt', 'r') as f:
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

    def a_to_b(self, text_a: str, attempts: int = 3) -> str:
        
        system = SYSTEM_A_TO_B.format(
        allowed_relations=", ".join(sorted(ALLOWED_RELATIONS)),
        max_tokens=self.max_tokens,
        allowed_nodes=", ".join(sorted(ALLOWED_NODE_TYPES))
        )

        user = USER_A_TO_B_TEMPLATE.format(text=text_a)
    
        # === First model call ===
        # b = send_chat(system, user)
        raw = send_chat(system, user)

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
        

        for _ in range(attempts):
            ok_len, b = enforce_max_tokens(b, self.max_tokens)
            tokens = tokenize_b(b)
            if ok_len and looks_like_b_chain(tokens):
                return b, node_type
            # ask model to revise
            b = revise_b(b, text_a)
    
        # === last-resort repair ===
        tokens = tokenize_b(b)
        b = " ".join(tokens)
        return b.strip(), node_type

    def b_to_a(self, text_b: str) -> str:
        system = SYSTEM_B_TO_A
        user = USER_B_TO_A_TEMPLATE.format(text=text_b)
        return send_chat(system, user)




