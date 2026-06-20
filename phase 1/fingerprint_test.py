from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

smiles = "CCO"

mol = Chem.MolFromSmiles(smiles)

fp = AllChem.GetMorganFingerprintAsBitVect(
    mol,
    radius=2,
    nBits=2048
)

arr = np.array(fp)

print("Shape:", arr.shape)
print("Number of 1s:", arr.sum())
print("Positions of first few 1s:")
print(np.where(arr == 1)[0][:20])