from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training:", X_train.shape)
print("Testing:", X_test.shape)

model = RandomForestClassifier()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Predictions:", predictions)
print("Actual:", y_test)
