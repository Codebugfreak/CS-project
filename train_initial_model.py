import pandas as pd
import requests
import json
import tensorflow as tf
from tensorflow import keras
from keras import layers
from phyto import CONFIG, parse_smiles
from sklearn.impute import SimpleImputer
import os

# Define constants
EXCEL_PATH = "data/cosmetic_plants_complete.xlsx"
MODEL_SAVE_PATH = "data/pretrained_model.keras"  # Use modern Keras format

EXCLUDED_PHENOLS = {
    "Alkaloids", "Anthocyanins", "Beta-glucan", "Beta-glucans", "Betalains",
    "Catechins", "Carotenoids", "Coumarins", "Flavonoids", "Flavanols",
    "Glucosinolates", "Lignans", "Mucilage", "Phenolic acids", "Phytosterols",
    "Polyphenols", "Proanthocyanidins", "Saponins", "Tannins", "Triterpenes",
    "Triterpenoid saponins", "Tocopherols", "Tocotrienols", "Vitamin C",
    "Vitamin E", "Xanthophylls"
}

# Global variable to hold phytochemical dataset
pubchem_df = None  

def fetch_pubchem_data(phytochemical):
    """Fetch molecular data from PubChem using PC_Compounds."""
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{phytochemical}/JSON"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            print(f"⚠️ Failed to fetch data for {phytochemical}: HTTP {response.status_code}")
            return None

        data = response.json()

        # Extract compound details from PC_Compounds
        compounds = data.get("PC_Compounds", [])
        if not compounds:
            print(f"⚠️ No compounds found for {phytochemical}")
            return None
        
        compound = compounds[0]  # Use first compound found
        
        # Extract properties from 'props' list
        properties = {}
        for prop in compound.get("props", []):
            urn = prop.get("urn", {})
            label = urn.get("label", "").lower().replace(" ", "_")  # Normalize label names
            value = prop.get("value", {}).get("sval") or prop.get("value", {}).get("fval")
            if label and value:
                properties[label] = value

        # Extract necessary values safely
        smiles = properties.get("smiles")
        molecular_weight = float(properties.get("molecular_weight", 0)) if "molecular_weight" in properties else None

        return {
            'name': phytochemical,
            'molecular_weight': molecular_weight if molecular_weight else 0,
            'canonical_smiles': smiles if smiles else None,
            'compound_score': parse_smiles(smiles)['compound_score'] if smiles else 0
        }

    except requests.exceptions.RequestException as e:
        print(f"⚠️ Network error fetching {phytochemical}: {e}")
    except (KeyError, TypeError, ValueError) as e:
        print(f"⚠️ Error parsing data for {phytochemical}: {e}")
    except Exception as e:
        print(f"⚠️ Unexpected error processing {phytochemical}: {e}")

    return None

def load_phytochemical_data():
    """Fetch and process phytochemical data, returning a processed DataFrame."""
    global pubchem_df  # Ensure the dataset remains accessible globally

    plants_df = pd.read_excel(EXCEL_PATH)
    all_phytos = set()
    for phyto_list in plants_df['Phytochemicals']:
        all_phytos.update(p for p in phyto_list.split(", ") if p not in EXCLUDED_PHENOLS)

    # Fetch data from PubChem
    phytochemical_data = [fetch_pubchem_data(p) for p in all_phytos]
    pubchem_df = pd.DataFrame([p for p in phytochemical_data if p])

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    pubchem_df[['molecular_weight', 'compound_score']] = imputer.fit_transform(
        pubchem_df[['molecular_weight', 'compound_score']]
    )

    print("✅ Phytochemical data loaded successfully.")
    return pubchem_df

def build_and_train_model():
    """Train and save a new model using the phytochemical dataset."""
    global pubchem_df

    if pubchem_df is None:
        print("⚠️ No valid dataset available. Please load the phytochemical data first.")
        return
    
    # Build model
    def build_model():
        inputs = tf.keras.Input(shape=(2,))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Train model
    model = build_model()
    X_train = pubchem_df[['molecular_weight', 'compound_score']]
    y_train = pd.Series(1, index=X_train.index)  # Dummy labels

    model.fit(X_train, y_train, epochs=15, validation_split=0.2)

    # Save model
    model.save(MODEL_SAVE_PATH)
    print(f"✅ Model trained and saved as '{MODEL_SAVE_PATH}'.")

def main():
    """Main function to load phytochemical data and train the model."""
    load_phytochemical_data()
    build_and_train_model()

def load_trained_model():
    """Load the trained model."""
    return tf.keras.models.load_model(MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()
