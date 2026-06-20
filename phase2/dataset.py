import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

# Load dataset
df = pd.read_csv("data/sample.csv")

# Extract columns
smiles_list = df["SMILES"]
labels = df["label"]

# Store fingerprints
X = []

for smiles in smiles_list:

    mol = Chem.MolFromSmiles(smiles)

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=2048
    )

    X.append(np.array(fp))

# Convert to NumPy array
X = np.array(X)

# Labels
y = np.array(labels)

print("X shape:", X.shape)
print("y shape:", y.shape)