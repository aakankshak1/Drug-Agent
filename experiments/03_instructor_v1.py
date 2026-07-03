import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def load_dataset():

    print("Loading dataset...")

    df = pd.read_csv("data/sample.csv")

    return df


def generate_fingerprints(df):

    print("Generating fingerprints...")

    smiles_list = df["SMILES"]
    labels = df["label"]

    X = []

    for smiles in smiles_list:

        mol = Chem.MolFromSmiles(smiles)

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=2,
            nBits=2048
        )

        X.append(np.array(fp))

    X = np.array(X)
    y = np.array(labels)

    return X, y


def train_random_forest(X, y):

    print("Training Random Forest...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return predictions, y_test


class Instructor:

    def run(self):

        df = load_dataset()

        X, y = generate_fingerprints(df)

        predictions, actual = train_random_forest(X, y)

        print("\n===== RESULTS =====")
        print("Predictions:", predictions)
        print("Actual:", actual)
        print("===================")


agent = Instructor()

agent.run()