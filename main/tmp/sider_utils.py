import os
import pandas as pd

# ---------------------------------------------------------
# Local file paths (YOU must place these files here)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIDER_FOLDER = os.path.join(BASE_DIR, "data", "sider")

SIDER_DRUG_NAMES = "../data/sider/drug_names.tsv"
SIDER_SIDE_EFFECTS = "../data/sider/meddra_all_label_se.tsv"


# ---------------------------------------------------------
# Main function: lookup ADRs by drug name
# ---------------------------------------------------------
def sider_get_side_effects(drug_name: str):
    # 1. Ensure drug_names file exists
    if not os.path.exists(SIDER_DRUG_NAMES):
        raise FileNotFoundError(
            f"Missing required file: {SIDER_DRUG_NAMES}\n"
            f"Place SIDER's 'drug_names.tsv.gz' in this folder manually."
        )

    # 2. Ensure side-effects file exists
    if not os.path.exists(SIDER_SIDE_EFFECTS):
        raise FileNotFoundError(
            f"Missing required file: {SIDER_SIDE_EFFECTS}\n"
            f"Place SIDER's 'meddra_all_label_se.tsv.gz' in this folder manually."
        )

    # 3. Load local SIDER drug name table
    df_drugs = pd.read_csv(SIDER_DRUG_NAMES, sep="\t")
    print(df_drugs.columns.tolist())


    # 4. Case-insensitive match
    matches = df_drugs[df_drugs["name"].str.lower() == drug_name.lower()]
    if matches.empty:
        return {"error": f"Drug '{drug_name}' not found in drug_names.tsv.gz"}

    stitch_ids = matches["stitch_id_flat"].unique()

    # 5. Load SIDER side effects table
    df_se = pd.read_csv(SIDER_SIDE_EFFECTS, sep="\t")

    # 6. Filter by the STITCH ID
    df_filtered = df_se[df_se["stitch_id_flat"].isin(stitch_ids)]

    if df_filtered.empty:
        return {"drug": drug_name, "side_effects": []}

    # 7. Columns available in SIDER 4.1
    cols = [
        col for col in ["side_effect_name", "side_effect_frequency", "meddra_id"]
        if col in df_filtered.columns
    ]

    results = df_filtered[cols].drop_duplicates()

    return {
        "drug": drug_name,
        "side_effects": results.to_dict(orient="records"),
    }
