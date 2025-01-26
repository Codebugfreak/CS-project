import requests
from rdkit import Chem

# Fetch phytochemical data from PubChem
def fetch_phytochemical_info(query):
    try:
        # Use PubChem's PUG REST API to fetch data
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{query}/property/CanonicalSMILES,MolecularFormula,IsomericSMILES,IUPACName/JSON"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            properties = data["PropertyTable"]["Properties"][0]
            return {
                "iupac_name": properties.get("IUPACName"),
                "molecular_formula": properties.get("MolecularFormula"),
                "canonical_smiles": properties.get("CanonicalSMILES"),
                "isomeric_smiles": properties.get("IsomericSMILES"),
            }
        else:
            return None
    except Exception as e:
        print(f"Error fetching data from PubChem: {e}")
        return None

# Classify compound based on features
def classify_phytochemical(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None

        # Example classification logic
        if Chem.rdMolDescriptors.CalcNumAromaticRings(mol) > 0:
            return "Flavonoid"
        elif Chem.rdMolDescriptors.CalcNumAliphaticRings(mol) > 0:
            return "Terpenoid"
        else:
            return "Other Phytochemical"

    except Exception as e:
        print(f"Error classifying compound: {e}")
        return None

# Parse SMILES string into numerical representations
def parse_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None

        atom_map = {"C": 0, "O": 1, "H": 4, "N": 3, "F": 2, "Na": 5}
        bond_map = {Chem.rdchem.BondType.SINGLE: 1, Chem.rdchem.BondType.DOUBLE: 2}

        atom_counts = {k: 0 for k in atom_map.values()}
        bond_counts = {k: 0 for k in bond_map.values()}

        for atom in mol.GetAtoms():
            atom_counts[atom_map.get(atom.GetSymbol(), -1)] += 1
        for bond in mol.GetBonds():
            bond_counts[bond_map.get(bond.GetBondType(), -1)] += 1

        return {
            "atom_counts": atom_counts,
            "bond_counts": bond_counts,
        }

    except Exception as e:
        print(f"Error parsing SMILES: {e}")
        return None
