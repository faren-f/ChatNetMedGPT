from helpers import send_chat, MODEL_NAME
import re
from helpers import *
import os

path = '/home/bbc8731/NetMedGPT/ChatNetMedGPT/'

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
with open(os.path.join(path, 'system_A2B.txt'), 'r') as f:
    SYSTEM_A_TO_B = f.read()
    
with open(os.path.join(path,'system_B2A.txt'), 'r') as f:
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
        
        #########Pre-normalize A with a tiny dictionary check
        GENE_SYMBOLS = None
        GENE_SET = None
        GENE_PATTERN = re.compile(r"\b[A-Za-z0-9]{3,8}\b")  # simple gene-like tokens
        
        CONFUSABLE_MAP = str.maketrans({
            "o":"0","O":"0","l":"1","I":"1","S":"5","s":"5","B":"8","Z":"2","z":"2","e":"3","E":"3"
        })
        
        def load_hgnc_symbols(path="hgnc_symbols.txt"):
            global GENE_SYMBOLS, GENE_SET
            with open(path) as f:
                GENE_SYMBOLS = [line.strip().upper() for line in f if line.strip()]
            GENE_SET = set(GENE_SYMBOLS)
        
        def normalize_gene_token(tok: str, cutoff=92):
            u = tok.upper()
            if u in GENE_SET:
                return u
            # try confusable normalization first
            conf = u.translate(CONFUSABLE_MAP)
            if conf in GENE_SET:
                return conf
            # fuzzy within small candidate space
            cand = process.extractOne(u, GENE_SYMBOLS, scorer=fuzz.WRatio, score_cutoff=cutoff)
            return cand[0] if cand else tok  # if no good match, leave as-is
        
        def normalize_A_text(a: str):
            if GENE_SYMBOLS is None:
                load_hgnc_symbols()
            def repl(m):
                tok = m.group(0)
                # Heuristic: only normalize tokens that are mostly uppercase/digits
                if sum(c.isupper() or c.isdigit() for c in tok) / len(tok) >= 0.75:
                    return normalize_gene_token(tok)
                return tok
            # quick fix for the example disease misspelling
            a = re.sub(r"\bsystemic\s+lapus\s+erythematosus\b", "systemic lupus erythematosus", a, flags=re.I)
            return GENE_PATTERN.sub(repl, a)

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
        return b.strip(), node_type

    def b_to_a(self, text_b: str) -> str:
        system = SYSTEM_B_TO_A
        user = USER_B_TO_A_TEMPLATE.format(text=text_b)
        return send_chat(system, user)




