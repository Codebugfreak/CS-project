import requests
import pandas as pd
import tensorflow as tf
import random
from tensorflow import keras
from keras import layers
from flask import current_app
from models import db, Plant, PhytochemicalCache
from phyto import parse_smiles, CONFIG
from train_initial_model import EXCLUDED_PHENOLS, load_trained_model
import re
from collections import Counter
from sklearn.preprocessing import StandardScaler

# Load trained model
def get_model():
    """Lazy-load the trained model only when needed."""
    global model
    if model is None:
        model = load_trained_model()
    return model



# Global dataset to store phytochemical data
dataset = pd.DataFrame(columns=["name", "molecular_formula", "molecular_weight", "canonical_smiles"])

def get_mean_molecular_weight():
    """Calculate the mean molecular weight, ensuring a fallback value if dataset is empty."""
    global dataset
    if dataset.empty:
        print("⚠️ No dataset found. Using default mean molecular weight.")
        return 250.0  # ✅ Default fallback
    return dataset["molecular_weight"].mean()



def clean_numeric(value):
    """Extract valid numeric values from strings."""
    if isinstance(value, (int, float)):
        return value
    match = re.findall(r"\d+\.\d+", str(value))
    return float(match[0]) if match else 0.0 #0.0 default



def fetch_ingredient_details(name):
    """Fetch ingredient details from PubChem, using cache if available and update dataset."""
    global dataset

    # Check if already in dataset
    existing = dataset[dataset["name"] == name]
    if not existing.empty:
        return existing.iloc[0].to_dict()

    # Check cache first
    cached = PhytochemicalCache.query.filter_by(name=name).first()
    if cached:
        row = {
            "name": name,
            "molecular_formula": cached.molecular_formula,
            "molecular_weight": cached.molecular_weight,
            "canonical_smiles": cached.canonical_smiles,
        }
        dataset = pd.concat([dataset, pd.DataFrame([row])], ignore_index=True) # Update dataset
        return row

    # Fetch from PubChem if not cached
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/JSON"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            compounds = data.get("PC_Compounds", [])
            if not compounds:
                print(f"⚠️ No compounds found for {name}")
                return None

            compound = compounds[0]
            properties = {
                prop["urn"]["label"].lower().replace(" ", "_"): prop["value"].get("sval") or prop["value"].get("fval")
                for prop in compound.get("props", [])
            }

            details = {
                "name": name,
                "molecular_formula": properties.get("molecular_formula"),
                "molecular_weight": clean_numeric(properties.get("molecular_weight")),
                "canonical_smiles": properties.get("smiles"),
            }

            # Cache results
            with current_app.app_context():
                new_entry = PhytochemicalCache(
                    name=name,
                    molecular_formula=details["molecular_formula"],
                    molecular_weight=details["molecular_weight"],
                    canonical_smiles=details["canonical_smiles"],
                )
                db.session.add(new_entry)
                db.session.commit()

            # Update dataset
            dataset = pd.concat([dataset, pd.DataFrame([details])], ignore_index=True)

            # ✅ Convert molecular weight column to numeric
            dataset["molecular_weight"] = pd.to_numeric(dataset["molecular_weight"], errors="coerce")

            return details
    except Exception as e:
        print(f"⚠️ Error fetching {name}: {str(e)}")
    
    return None


def get_phytochemicals(plant_name):
    """Retrieve phytochemicals for a plant while excluding general classes like 'Flavonoids'."""
    normalized_name = plant_name.strip().lower()
    with current_app.app_context():
        plant = Plant.query.filter(Plant.plant_name.ilike(normalized_name)).first()
        if plant and plant.phytochemicals:
            phytochemicals = set(plant.phytochemicals.split(", "))
            valid_phytochemicals = phytochemicals - EXCLUDED_PHENOLS
            return list(valid_phytochemicals)
        return None


def fetch_top_bioactive_compounds(plant_name, top_n=5):
    """Select top N bioactive compounds from a plant's phytochemical list."""
    phytochemicals = get_phytochemicals(plant_name)
    if not phytochemicals:
        return []

    return random.sample(phytochemicals, min(top_n, len(phytochemicals)))  # Randomly select bioactives


def add_to_dataset(compound_list):
    """Fetch ingredient details and update dataset."""
    global dataset
    new_data = [fetch_ingredient_details(compound) for compound in compound_list]
    new_data = [row for row in new_data if row is not None]  # ✅ Filter out None values
    if new_data:
        dataset = pd.concat([dataset, pd.DataFrame(new_data)], ignore_index=True)  # ✅ Efficient merging


def select_phytochemicals(plants, strategy="random"):
    """Select phytochemicals from a list of plants based on strategy (random or parallel)."""
    plant_phyto_dict = {}
    with current_app.app_context():
        for plant in plants:
            record = Plant.query.filter(Plant.plant_name.ilike(plant)).first()
            if record:
                phytos = [p.strip() for p in record.phytochemicals.split(",") if p.strip() and p.strip() not in EXCLUDED_PHENOLS]
                if phytos:
                    if strategy == "random":
                        plant_phyto_dict[plant] = [random.choice(phytos)]
                    else:
                        plant_phyto_dict[plant] = phytos

    # Parallel mode padding
    if strategy == "parallel" and plant_phyto_dict:
        max_len = max(len(v) for v in plant_phyto_dict.values())
        for k in plant_phyto_dict:
            plant_phyto_dict[k] += [None] * (max_len - len(plant_phyto_dict[k]))

    return plant_phyto_dict


def iterative_cascade(model, dataset, threshold=0.9, max_iterations=10):
    """Iterate over the dataset and predict the best formulation based on a threshold."""
    for _ in range(max_iterations):
        predictions = get_model().predict(dataset)
        if max(predictions) >= threshold:
            return predictions, True  # Successful prediction
    return predictions, False  # Failed after N iterations


def suggest_alternatives(ingredients):
    """Suggest alternative compounds if the formulation is not optimal."""
    return ["Alternative Compound 1", "Alternative Compound 2", "Alternative Compound 3"]


def suggest_formulation(product_type, ingredients):
    """Suggest formulation ingredients based on the product type."""
    suggestions = {
        "natural cream": ["Shea Butter", "Coconut Oil", "Vitamin E"],
        "perfume": ["Vanilla Extract", "Sandalwood Oil", "Rose Absolute"],
        "perfume oil": ["Jojoba Oil", "Amber Oil", "Lavender Essential Oil"],
        "hair growth cream": ["Castor Oil", "Peppermint Oil", "Biotin"],
    }
    return suggestions.get(product_type, []) + ingredients



def calculate_compatibility(ingredients):
    """Calculate the compatibility of selected ingredients using the trained model."""
    global dataset

    temp_data = []
    for ingredient in ingredients:
        parsed_data = parse_smiles(ingredient)
        if "compound_score" in parsed_data and "molecular_weight" in parsed_data:
            temp_data.append([parsed_data["molecular_weight"], parsed_data["compound_score"]])

    if not temp_data:
        return 0  # No valid compounds

    # Convert to DataFrame
    new_data = pd.DataFrame(temp_data, columns=["molecular_weight", "compound_score"])

    # ✅ Apply cleaning function first
    new_data["molecular_weight"] = new_data["molecular_weight"].apply(clean_numeric)

    # ✅ Convert molecular weight column to numeric after cleaning
    new_data["molecular_weight"] = pd.to_numeric(new_data["molecular_weight"], errors="coerce")

    # ✅ Ensure a valid mean molecular weight for missing values
    mean_weight = get_mean_molecular_weight() if not dataset.empty else 250.0
    new_data["molecular_weight"].fillna(mean_weight, inplace=True)

    # ✅ Append new data to global dataset instead of replacing it
    dataset = pd.concat([dataset, new_data], ignore_index=True)

    # ✅ Normalize data using StandardScaler
    scaler = StandardScaler()
    new_data[["molecular_weight", "compound_score"]] = scaler.fit_transform(dataset[["molecular_weight", "compound_score"]])

    predictions = get_model().predict(new_data)
    return float(predictions.mean())  # Avoids returning an extreme 1.0 or 0.0



def fallback_atom_count_score(smiles):
    """Fallback method: Estimate compound complexity based on atom counts in SMILES."""
    if not smiles or not isinstance(smiles, str):
        return 0  # Default score if SMILES is invalid
    
    # Extract atom symbols (e.g., C, H, O, N, S, etc.) from the SMILES string
    atoms = re.findall(r"[A-Z][a-z]?", smiles)  # Matches elements like C, O, N, Cl, etc.
    
    if not atoms:
        return 0  # No recognizable atoms found, return default score
    
    # Count occurrences of each element
    atom_counts = Counter(atoms)

    # Define atomic weightings (arbitrary but biologically relevant)
    element_weights = {
        "C": 1.0,  # Carbon is essential in organic compounds
        "H": 0.1,  # Hydrogen is common but less informative
        "O": 0.8,  # Oxygen can indicate functional groups
        "N": 0.9,  # Nitrogen often relates to bioactivity
        "S": 0.7,  # Sulfur appears in important bioactive molecules
        "P": 0.6,  # Phosphorus, often in ATP-related molecules
        "Cl": 0.5, "Br": 0.5, "I": 0.5  # Halogens, often in drugs
    }

    # Compute a simple molecular "score" based on atomic composition
    score = sum(atom_counts[element] * element_weights.get(element, 0.2) for element in atom_counts)

    return round(score, 2)  # Return a rounded score
