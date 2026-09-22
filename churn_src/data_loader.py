# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

import os
from pathlib import Path

import pandas as pd

# preprocessing
from sklearn.model_selection import train_test_split


def load_data(test_size=0.2):
    """
    Loads the dataset and splits into train/test.
    """
    root = (
        Path(__file__).resolve().parent
    )  # --> if churn_data is to be placed in another folder use another parent
    new_path = os.path.join(root, "churn_data", "train_churn.csv")
    df = pd.read_csv(new_path)

    # We set the index to "security_no" to ensure that the data
    # is properly aligned and can be easily accessed by this unique identifier.
    # This is important for maintaining the integrity of the dataset,
    # especially when performing operations that require a unique key for each row.

    df = df.set_index("security_no")

    X = df.drop("churn_risk_score", axis=1)
    y = df["churn_risk_score"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    processed_path = Path('train_test')
    processed_path.parent.mkdir(exist_ok=True , parents=True)

    x_train.to_csv(os.path.join(processed_path, "x_train.csv"))
    x_test.to_csv(os.path.join(processed_path, "x_test.csv"))
    y_train.to_csv(os.path.join(processed_path, "y_train.csv"))
    y_test.to_csv(os.path.join(processed_path, "y_train.csv"))

    return x_train, y_train, x_test, y_test
