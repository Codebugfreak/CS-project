from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import requests
from statistics import mean, stdev



# Benchmark compounds for phytochemical classes
BENCHMARKS = {
    "Phenolic": "C1=CC=C(C=C1)C2=CC(=O)C3=C(O2)C=CC(=C3O)O",  # Quercetin
    "Flavonoid": "C1=CC=C2C(=C1)C(=O)C3=C(C2=O)C=CC(=C3)O",  # Luteolin
    "Terpenoid": "CC1=CCC(CC1)C(C)=C",  # Limonene
    "Terpenoid (Aldehyde/Ketone)": "CC(=CCCC(=CC=O)C)C",  # Citral
    "Alkaloid": "CN1CCCC1C2=CC=CC=C2",  # Nicotine
    "Glycoside": "C1=CC=C(C=C1)O[C@H]2[C@H]([C@@H]([C@H](O2)CO)O)O",  # Salicin
    "Phenolic (Carboxylic Acid)": "C1=CC=C(C=C1)C(=O)O",  # Benzoic Acid
    "Other": "CC(C)CC1=CC=C(C=C1)C(C)C(C)C",  # Menthol-like structure
}

def get_benchmark_smiles(classification):
    """
    Retrieve the benchmark SMILES string for a given phytochemical class.

    Parameters:
        classification (str): The phytochemical class.

    Returns:
        str: The benchmark SMILES string.
    """
    return BENCHMARKS.get(classification, BENCHMARKS["Other"]) # dictionary comprehension


# Configuration object for weights
CONFIG = {
    "atom_weights": {"C": 1.0, "O": 3.0, "H": 0.5, "N": 2.5, "S": 2.0},
    "bond_weights": {1: 1.0, 2: 2.0, 3: 3.0, 4: 2.5},
    "bias_weights": {"atom_types": 1.5, "rings": 2.0, "functional_groups": 3.0, "base_bias": 5.0},
}

# Fetch phytochemical data from PubChem
def fetch_phytochemical_info(query, fetch_bioactivities=True):
    """
    Fetch phytochemical data, including basic properties and bioactivities, from PubChem.

    Parameters:
        query (str): Name or identifier of the compound.
        fetch_bioactivities (bool): Whether to fetch bioactivities from PubChem.

    Returns:
        dict: Compound information, including IUPAC name, molecular formula, SMILES, and bioactivities.
    """
    try:
        base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        properties_url = f"{base_url}/compound/name/{query}/property/CanonicalSMILES,MolecularFormula,IsomericSMILES,IUPACName/JSON"
        response = requests.get(properties_url)
        response.raise_for_status()

        data = response.json()
        properties = data["PropertyTable"]["Properties"][0]
        compound_cid = properties.get("CID")

        bioactivities = []
        if fetch_bioactivities and compound_cid:
            bioactivities_url = f"{base_url}/compound/cid/{compound_cid}/assaysummary/JSON"
            bio_response = requests.get(bioactivities_url)
            if bio_response.status_code == 200:
                assay_data = bio_response.json()
                if "AssaySummaries" in assay_data:
                    for assay in assay_data["AssaySummaries"]:
                        bioactivities.append({
                            "assay_type": assay.get("Category"),
                            "target": assay.get("TargetName"),
                            "target_type": assay.get("TargetType"),
                            "activity_outcome": assay.get("ActivityOutcome"),
                        })

        return {
            "iupac_name": properties.get("IUPACName"),
            "molecular_formula": properties.get("MolecularFormula"),
            "canonical_smiles": properties.get("CanonicalSMILES"),
            "isomeric_smiles": properties.get("IsomericSMILES"),
            "bioactivities": bioactivities,
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from PubChem for '{query}': {e}")
        return None

# Count functional groups
def count_functional_groups(mol):
    """
    Count the number of key functional groups in a molecule.

    Parameters:
        mol (RDKit Mol): RDKit molecule object.

    Returns:
        tuple: A dictionary of group counts and the total count.
    """
    try:
        patterns = {
            "hydroxyl": Chem.MolFromSmarts("[OH]"),
            "carbonyl": Chem.MolFromSmarts("[CX3]=[OX1]"),
            "nitro": Chem.MolFromSmarts("[NX3](=O)=O"),
            "amine": Chem.MolFromSmarts("[NX3;H2,H1,H0]"),
            "carboxyl": Chem.MolFromSmarts("C(=O)[OH]"),
            "ether": Chem.MolFromSmarts("C-O-C"),
            "sulfhydryl": Chem.MolFromSmarts("[SH]"),
            "phosphate": Chem.MolFromSmarts("P(=O)(O)(O)O"),
            "aromatic_ring": Chem.MolFromSmarts("c1ccccc1"),
        }

        # Initialize group counts
        group_counts = {name: len(mol.GetSubstructMatches(pattern)) for name, pattern in patterns.items()}
        total_count = sum(group_counts.values())
        return group_counts, total_count

    except Exception as e:
        print(f"Error counting functional groups: {e}")
        return {}, 0  # Return default values to prevent `NoneType` errors




# Classify compound
def classify_phytochemical(smiles):
    """
    Classify a compound based on its SMILES string into specific phytochemical classes.

    Parameters:
        smiles (str): SMILES representation of the compound.

    Returns:
        str: Classification of the compound (e.g., "Flavonoid", "Terpenoid", "Alkaloid", "Phenolic", "Glycoside", "Other").
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return "Unknown"

        # Check for aromatic rings (common in flavonoids and phenolics)
        if rdMolDescriptors.CalcNumAromaticRings(mol) > 0:
            # Check for hydroxyl groups (common in phenolics)
            if len(mol.GetSubstructMatches(Chem.MolFromSmarts("[OH]"))) > 0:
                return "Phenolic"
            else:
                return "Flavonoid"

        # Check for aliphatic rings (common in terpenoids)
        if rdMolDescriptors.CalcNumAliphaticRings(mol) > 0:
            # Check for carbonyl groups (common in terpenoids like citral)
            if len(mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3]=[OX1]"))) > 0:
                return "Terpenoid (Aldehyde/Ketone)"
            else:
                return "Terpenoid"

        # Check for nitrogen-containing functional groups (common in alkaloids)
        if len(mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3]"))) > 0:
            return "Alkaloid"

        # Check for glycosidic bonds (common in glycosides)
        if len(mol.GetSubstructMatches(Chem.MolFromSmarts("[C][O][C]"))) > 0:
            return "Glycoside"

        # Check for carboxylic acids (common in some phenolics and other classes)
        if len(mol.GetSubstructMatches(Chem.MolFromSmarts("C(=O)[OH]"))) > 0:
            return "Phenolic (Carboxylic Acid)"

        # Default classification for other compounds
        return "Unknown Class"

    except Exception as e:
        print(f"Error classifying compound: {e}")
        return "Error"

# Parse SMILES
def parse_smiles(smiles, benchmark_score=None, weights=None):
    try:
        smiles = str(smiles).strip()
        print(f"SMILES to process: {smiles}")

        # Parse SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError(f"Failed to parse SMILES: {smiles}")
        print(f"RDKit successfully parsed SMILES: {smiles}")

        # Count functional groups first
        group_counts, functional_groups = count_functional_groups(mol)
        print(f"Functional groups counted: {group_counts}, Total: {functional_groups}")

        # Initialize atom_counts
        atom_counts = {}
        for atom in mol.GetAtoms():
            atom_symbol = atom.GetSymbol()
            if atom_symbol not in atom_counts:
                atom_counts[atom_symbol] = 0
            atom_counts[atom_symbol] += 1
        print(f"Atom counts after processing: {atom_counts}")

        # Initialize bond_map and bond_counts
        bond_map = {
            Chem.BondType.SINGLE: 1,
            Chem.BondType.DOUBLE: 2,
            Chem.BondType.TRIPLE: 3,
            Chem.BondType.AROMATIC: 4,
        }
        bond_counts = {key: 0 for key in bond_map.values()}

        for bond in mol.GetBonds():
            bond_type = bond.GetBondType()
            if bond_type not in bond_map:
                print(f"Warning: Bond type '{bond_type}' not in bond_map, skipping.")
                continue  # Skip unknown bond types
            bond_counts[bond_map[bond_type]] += 1
        print(f"Bond counts after processing: {bond_counts}")

        # Calculate dynamic bias
        bias = (weights["bias_weights"]["atom_types"] * len(atom_counts)
                + weights["bias_weights"]["rings"] * rdMolDescriptors.CalcNumRings(mol)
                + weights["bias_weights"]["functional_groups"] * functional_groups
                + weights["bias_weights"]["base_bias"])
        print(f"Calculated dynamic bias: {bias}")

        # Calculate compound score
        compound_score = calculate_score(atom_counts, bond_counts, weights, bias)
        print(f"Calculated compound score: {compound_score}")

        # Determine oxidation/reduction tendencies using z-score
        prone_to_oxidation = False
        prone_to_reduction = False
        z_score = None

        if benchmark_score:
            scores = [benchmark_score - 10, benchmark_score, benchmark_score + 10]  # Dynamic range
            mean_score = mean(scores)
            std_dev = stdev(scores)

            if std_dev > 0:
                z_score = (compound_score - mean_score) / std_dev
                prone_to_oxidation = z_score > 1
                prone_to_reduction = z_score < -1
        print(f"Z-score: {z_score}, Oxidation: {prone_to_oxidation}, Reduction: {prone_to_reduction}")

        return {
            "atom_counts": atom_counts,
            "bond_counts": bond_counts,
            "functional_groups": group_counts,
            "compound_score": compound_score,
            "z_score": z_score,
            "prone_to_oxidation": prone_to_oxidation,
            "prone_to_reduction": prone_to_reduction,
            "dynamic_bias": bias,
        }

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return {"error": str(ve), "smiles": smiles}

    except Exception as e:
        print(f"Unexpected error while parsing SMILES '{smiles}': {e}")
        return {"error": str(e), "smiles": smiles}

# Calculate the score of a compound
def calculate_score(atom_counts, bond_counts, weights, bias=1.0):
    """
    Calculate the score of a compound based on atom and bond counts, with weights and bias.
    """
    try:
        score = bias  # Start with the bias term
        for atom, count in atom_counts.items():
            weight = weights["atom_weights"].get(atom, 0)  # Use "atom_weights" instead of "atom"
            if weight is None:
                raise ValueError(f"Missing weight for atom {atom}")
            score += weight * count  # Weighted contribution of atoms

        for bond, count in bond_counts.items():
            weight = weights["bond_weights"].get(bond, 0)  # Use "bond_weights" instead of "bond"
            if weight is None:
                raise ValueError(f"Missing weight for bond {bond}")
            score += weight * count  # Weighted contribution of bonds

        return score
    except Exception as e:
        print(f"Error in calculate_score: {e}")
        return None  # Fallback in case of failure

# Calculate the Relative Approximation Error (RAE)
def calculate_rae(compound_score, benchmark_score):
    """
    Calculate the Relative Approximation Error (RAE) between a compound and the benchmark.
    """
    return compound_score / benchmark_score
