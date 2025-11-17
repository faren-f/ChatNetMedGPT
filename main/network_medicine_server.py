import os
import requests
from urllib.parse import quote_plus

from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from sider_utils import sider_get_side_effects

load_dotenv("../.env")

mcp = FastMCP(
    name="DrugSMILES",
    host="0.0.0.0",
    port=8050,
    stateless_http=True,
)

# ---------------------------------------------------------
# 1. PubChem – SMILES, properties, synonyms
# ---------------------------------------------------------

@mcp.tool()
def pubchem_smiles(drug_name: str):
    """Get Canonical SMILES from PubChem by drug name."""
    name = quote_plus(drug_name)
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{name}/property/CanonicalSMILES/JSON"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


@mcp.tool()
def pubchem_synonyms(drug_name: str):
    """Get synonyms from PubChem by drug name."""
    name = quote_plus(drug_name)
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{name}/synonyms/JSON"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------
# 2. ChEMBL – drug targets, bioactivity, properties
# ---------------------------------------------------------

@mcp.tool()
def chembl_search(query: str):
    """Search ChEMBL for a molecule."""
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/search?q={quote_plus(query)}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


@mcp.tool()
def chembl_bioactivity(chembl_id: str):
    """Get bioactivities for a ChEMBL molecule (Ki, IC50, EC50...)."""
    url = f"https://www.ebi.ac.uk/chembl/api/data/activity?molecule_chembl_id={chembl_id}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()


@mcp.tool()
def chembl_targets(chembl_id: str):
    """Get targets associated with a ChEMBL molecule."""
    url = f"https://www.ebi.ac.uk/chembl/api/data/target?molecule_chembl_id={chembl_id}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------
# 3. DrugCentral – indications, targets, ATC
# ---------------------------------------------------------

@mcp.tool()
def drugcentral_search(query: str):
    """Search DrugCentral."""
    url = f"https://drugcentral.org/api/v1/drug?search={quote_plus(query)}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------
# 4. DGIdb – drug–gene interactions
# ---------------------------------------------------------

@mcp.tool()
def dgidb_interactions(gene: str):
    """Get drug–gene interactions for a gene from DGIdb."""
    url = f"https://dgidb.org/api/v2/interactions.json?genes={quote_plus(gene)}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


@mcp.tool()
def dgidb_drug_targets(drug_name: str):
    """Get gene targets for a drug from DGIdb."""
    url = f"https://dgidb.org/api/v2/interactions.json?drugs={quote_plus(drug_name)}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------
# 5. SIDER – drug–side effect data (download only)
# ---------------------------------------------------------

@mcp.tool()
def sider_download_url():
    """Return SIDER drug–side effect download URL."""
    return "http://sideeffects.embl.de/media/download/DRUG-SIDE_EFFECTS.tsv.gz"


@mcp.tool()
def sider_side_effects(drug_name: str):
    """Return SIDER clinical side effects for a drug."""
    return sider_get_side_effects(drug_name)


# ---------------------------------------------------------
# 6. FAERS – adverse events
# ---------------------------------------------------------

@mcp.tool()
def faers_search(drug_name: str):
    """Query FAERS OpenFDA for adverse events of a drug."""
    url = (
        "https://api.fda.gov/drug/event.json"
        f"?search=patient.drug.medicinalproduct:{quote_plus(drug_name)}"
        "&limit=100"
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------
# 7. DisGeNET – disease–gene associations
# (requires DISGENET_API_KEY in your .env)
# ---------------------------------------------------------

@mcp.tool()
def disgenet_disease_genes(disease_id: str):
    """Get genes associated with a disease from DisGeNET."""
    api_key = os.getenv("DISGENET_API_KEY")
    if not api_key:
        raise RuntimeError("DISGENET_API_KEY not set in .env")
    url = f"https://www.disgenet.org/api/gda/disease/{disease_id}"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {api_key}"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    return r.json()


@mcp.tool()
def disgenet_gene_diseases(gene_id: str):
    """Get diseases associated with a gene from DisGeNET."""
    api_key = os.getenv("DISGENET_API_KEY")
    if not api_key:
        raise RuntimeError("DISGENET_API_KEY not set in .env")
    url = f"https://www.disgenet.org/api/gda/gene/{gene_id}"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {api_key}"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------
# 8. HPO – disease phenotype data
# ---------------------------------------------------------

@mcp.tool()
def hpo_gene_to_terms(gene_symbol: str):
    """Get HPO terms associated with a gene."""
    url = f"https://hpo.jax.org/api/hpo/gene/{quote_plus(gene_symbol)}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


@mcp.tool()
def hpo_disease_to_terms(omim_id: str):
    """Get HPO terms associated with an OMIM disease ID."""
    url = f"https://hpo.jax.org/api/hpo/disease/OMIM:{omim_id}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------
# 9. STRING – protein–protein interactions
# ---------------------------------------------------------

@mcp.tool()
def string_interactions(gene: str, species: int = 9606, limit: int = 50):
    """Get STRING PPI network for a gene."""
    url = (
        f"https://string-db.org/api/json/network?"
        f"identifiers={quote_plus(gene)}&species={species}&limit={limit}"
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------
# 10. BioGRID – downloadable interactions
# ---------------------------------------------------------

@mcp.tool()
def biogrid_download_url():
    """Return BioGRID release archive URL."""
    return "https://downloads.thebiogrid.org/BioGRID/Release-Archive/"


# ---------------------------------------------------------
# 11. Reactome – pathways
# ---------------------------------------------------------

@mcp.tool()
def reactome_pathways(gene_symbol: str):
    """Query Reactome for pathways related to a gene symbol."""
    url = f"https://reactome.org/ContentService/data/query/{quote_plus(gene_symbol)}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


@mcp.tool()
def reactome_pathway_participants(pathway_id: str):
    """Get participants in a Reactome pathway."""
    url = (
        f"https://reactome.org/ContentService/data/pathway/{pathway_id}"
        "/participants"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    mcp.run(transport="stdio")
