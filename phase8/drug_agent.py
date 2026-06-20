import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_dataset():

    print("Loading dataset...")

    df = pd.read_csv("data/HIV.csv").head(1000)

    return df


def generate_fingerprints(df):

    print("Generating fingerprints...")

    smiles_list = df["smiles"]
    labels = df["HIV_active"]

    X = []
    y = []

    for smiles, label in zip(smiles_list, labels):

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=2,
            nBits=2048
        )

        X.append(np.array(fp))
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    print("Valid molecules:", len(X))

    return X, y


def train_model(model_name, X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    if model_name == "Random Forest":
        model = RandomForestClassifier()

    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)

    else:
        raise ValueError("Unknown model")

    print(f"Training {model_name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    return accuracy


class PlannerAgent:

    def choose_model(self, dataset_name):

        if dataset_name == "HIV":
            return "Random Forest"

        elif dataset_name == "PAMPA":
            return "Logistic Regression"

        elif dataset_name == "DAVIS":
            return "Random Forest"

        else:
            return "Random Forest"


class Instructor:

    def run(self, model_name):

        df = load_dataset()

        X, y = generate_fingerprints(df)

        accuracy = train_model(
            model_name,
            X,
            y
        )

        print("\n===== RESULTS =====")
        print("Model:", model_name)
        print("Accuracy:", accuracy)
        print("===================")


planner = PlannerAgent()

dataset = input("Enter dataset (HIV/PAMPA/DAVIS): ")

model = planner.choose_model(dataset)

print("\nPlanner selected:", model)

agent = Instructor()
agent.run(model)