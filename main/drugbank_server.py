import os
import requests
from urllib.parse import quote_plus

from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv("../.env")

mcp = FastMCP(
    name="DrugSMILES",
    host="0.0.0.0",
    port=8050,
    stateless_http=True,
)


@mcp.tool()
def drug_smiles(drug_query: str):
    """
    Look up SMILES for a drug using DrugBank (or PubChem as a fallback).
    `drug_query` can be a name (e.g. 'aspirin') or a DrugBank ID (e.g. 'DB00945').
    """    

    # ---------- Fallback: PubChem (no license needed, uses drug name) ----------
    encoded = quote_plus(drug_query)
    # Get canonical SMILES by name:
    # https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/<name>/property/CanonicalSMILES/JSON
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{encoded}/property/CanonicalSMILES/JSON"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    props = data.get("PropertyTable", {}).get("Properties", [])
    results = []
    for p in props:
        results.append(
            {
                "pubchem_cid": p.get("CID"),
                "name": drug_query,
                "smiles": p.get("CanonicalSMILES"),
            }
        )

    return {
        "source": "pubchem",
        "query": drug_query,
        "results": results,
    }


if __name__ == "__main__":
    transport = "stdio"
    mcp.run(transport="stdio")



