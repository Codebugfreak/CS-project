import requests

# Function to fetch ingredient details from PubChem
def fetch_ingredient_details(name):
    try:
        # Use PubChem's API to fetch details of the ingredient
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/MolecularFormula,MolecularWeight,CanonicalSMILES/JSON"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            properties = data["PropertyTable"]["Properties"][0]
            return {
                "name": name,
                "molecular_formula": properties.get("MolecularFormula"),
                "molecular_weight": properties.get("MolecularWeight"),
                "canonical_smiles": properties.get("CanonicalSMILES"),
            }
        else:
            return None
    except Exception as e:
        print(f"Error fetching ingredient details: {e}")
        return None

# Function to suggest optimal formulations
def suggest_formulation(product_type, ingredients):
    # Placeholder logic for optimal formulations
    suggestions = {
        "natural cream": ["Shea Butter", "Coconut Oil", "Vitamin E"],
        "perfume": ["Vanilla Extract", "Sandalwood Oil", "Rose Absolute"],
        "perfume oil": ["Jojoba Oil", "Amber Oil", "Lavender Essential Oil"],
        "hair growth cream": ["Castor Oil", "Peppermint Oil", "Biotin"],
    }
    return suggestions.get(product_type, []) + ingredients

# Function to calculate ingredient compatibility based on molecular interactions
def calculate_compatibility(ingredients):
    # Placeholder compatibility logic
    compatibility_score = len(ingredients) * 10  # Dummy score
    return compatibility_score
