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
    y = df["churn_ris_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    return X_train, X_test, y_train, y_test, df
