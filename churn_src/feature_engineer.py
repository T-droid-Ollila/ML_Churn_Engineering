import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import OneHotEncoder
from churn_src.data_loader import load_data


class ChurnPreprocessor:
    """Learns transformations from training data, then applies them to unseen data.

    fit()           -> learns bin edges, medians/modes, selected features, encoder
    transform()     -> applies what was learned; never re-learns anything
    fit_transform() -> convenience for the training pass
    """

    def __init__(self):
        # Everything learned during fit lives here as instance state
        self.freq_bins = None
        self.time_bins = None
        self.age_bins = None
        self.medians = {}
        self.modes = {}
        self.most_correlated = []
        self.selected_categorical = []
        self.categorical_cols = []
        self.encoder = None
        self.is_fitted = False

    # ---------------- shared static cleaning (no learning involved) ----------------

    def _clean(self, x_df: pd.DataFrame) -> pd.DataFrame:
        df = x_df.copy()

        positive = ["Reasonable Price", "Products always in Stock",
                    "User Friendly Website", "Quality Customer Care"]
        negative = ["Too many ads", "No reason specified", "Poor Product Quality",
                    "Poor Customer Service", "Poor Website"]
        neutral = ["No reason specified"]
        mapping = {cat: "positive" for cat in positive}
        mapping.update({cat: "negative" for cat in negative})
        mapping.update({cat: "neutral" for cat in neutral})
        df.feedback = df.feedback.replace(mapping)

        df["has_referral"] = df["referral_id"].str.startswith("CID", na=False).astype(int)
        df["joined_through_referral"] = df.joined_through_referral.where(
            df.joined_through_referral.isin(["Yes", "No"]), other=np.nan
        ).astype("category")
        df["last_visit_time"] = pd.to_datetime(
            df["last_visit_time"], format="%H:%M:%S"
        ).dt.strftime("%H:%M:%S")
        df["gender"] = df["gender"].str.strip().replace("Unknown", "M")
        df["avg_time_spent"] = df["avg_time_spent"].where(df["avg_time_spent"] >= 0, np.nan)
        df["days_since_last_login"] = df["days_since_last_login"].where(
            df["days_since_last_login"] >= 0, np.nan
        )
        df["medium_of_operation"] = df["medium_of_operation"].replace("?", np.nan).fillna("Smartphone")
        df["avg_frequency_login_days"] = pd.to_numeric(df["avg_frequency_login_days"], errors="coerce")
        df["avg_frequency_login_days"] = df["avg_frequency_login_days"].where(
            df["avg_frequency_login_days"] >= 0, np.nan
        )
        df["joining_date"] = pd.to_datetime(df["joining_date"], dayfirst=True)
        df["joining_month"] = df["joining_date"].dt.strftime("%m").astype(int)
        df["joining_year"] = df["joining_date"].dt.strftime("%Y").astype(int)

        return df.drop(["joining_date", "referral_id"], axis=1)

    # ---------------- fit: learn everything from training data ----------------

    def fit(self, x_df: pd.DataFrame, y_df: pd.DataFrame) -> "ChurnPreprocessor":
        if y_df is None:
            raise ValueError("y_df is required to fit the preprocessor.")

        df = self._clean(x_df)

        # bin edges
        _, self.freq_bins = pd.qcut(df["avg_frequency_login_days"], q=5, labels=False,
                                    retbins=True, duplicates="drop")
        _, self.time_bins = pd.qcut(df["avg_time_spent"], q=5, labels=False,
                                    retbins=True, duplicates="drop")
        _, self.age_bins = pd.cut(df["age"], bins=6, labels=False, retbins=True)

        df = self._apply_bins(df)
        df = self._to_category(df)

        # medians / modes
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                self.medians[column] = df[column].median()
            else:
                self.modes[column] = df[column].mode()[0]

        df = self._fill_missing(df)

        # feature selection — the only part that genuinely needs the target
        target_col = y_df.columns[0]
        corr = pd.concat([df, y_df], axis=1).corr(numeric_only=True)
        self.most_correlated = [
            c for c in corr.columns
            if abs(corr.loc[c, target_col]) >= 0.2 and c != target_col
        ]

        y1 = y_df.copy().squeeze()
        cat_features = df.select_dtypes(include=["object", "category"])
        mi_scores = {f: mutual_info_score(df[f], y1) for f in cat_features.columns}
        mi_scores = {f: s for f, s in mi_scores.items() if s >= 0.2}
        self.selected_categorical = list(
            pd.Series(mi_scores).sort_values(ascending=False).index
        )

        selected = self._select_and_log(df)

        # encoder
        self.categorical_cols = list(
            selected.select_dtypes(include=["object", "category"]).columns
        )
        self.encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        self.encoder.fit(selected[self.categorical_cols])

        self.is_fitted = True
        return self

    # ---------------- transform: apply only; never learns ----------------

    def transform(self, x_df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted (or loaded) before transform().")

        df = self._clean(x_df)
        df = self._apply_bins(df)
        df = self._to_category(df)
        df = self._fill_missing(df)
        selected = self._select_and_log(df)

        encoded = pd.DataFrame(
            self.encoder.transform(selected[self.categorical_cols]),
            columns=self.encoder.get_feature_names_out(self.categorical_cols),
            index=selected.index,
        )
        return pd.concat([selected.drop(columns=self.categorical_cols), encoded], axis=1)

    def fit_transform(self, x_df: pd.DataFrame, y_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(x_df, y_df).transform(x_df)

    # ---------------- shared helpers used by both paths ----------------

    def _apply_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        df["avg_frequency_login_days_binned"] = pd.cut(
            df["avg_frequency_login_days"], bins=self.freq_bins, labels=False, include_lowest=True
        )
        df["avg_time_spent_binned"] = pd.cut(
            df["avg_time_spent"], bins=self.time_bins, labels=False, include_lowest=True
        )
        df["age_binned"] = pd.cut(df["age"], bins=self.age_bins, labels=False, include_lowest=True)
        return df

    def _to_category(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in df.select_dtypes(include="object").columns:
            df[column] = df[column].astype("category")
        return df

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in df.columns:
            if column in self.modes:
                df[column] = df[column].fillna(self.modes[column])
            elif column in self.medians:
                df[column] = df[column].fillna(self.medians[column])
        return df

    def _select_and_log(self, df: pd.DataFrame) -> pd.DataFrame:
        numerical = df.select_dtypes(include=[np.number])[self.most_correlated]
        selected = pd.concat([numerical, df[self.selected_categorical]], axis=1)
        for column in ["avg_transaction_value", "points_in_wallet"]:
            if column in selected.columns:
                selected[column] = np.sign(selected[column]) * np.log1p(np.abs(selected[column]))
        return selected.drop(columns=["last_visit_time"], errors="ignore")

    # ---------------- persistence — replaces your artifacts.pkl handling ----------------

    def save(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "ChurnPreprocessor":
        with open(path, "rb") as f:
            return pickle.load(f)

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_data()

    root = Path(__file__).parent.parent()
    path = os.path.join(root,'processed_data')

    preprocessor = ChurnPreprocessor()

    X_engineered_train = preprocessor.fit_transform(x_train, y_train)
    X_engineered_train.to_csv(os.path.join(path,'x_train'))
    y_train.to_csv(os.path.join(path,'y_train'))

    preprocessor.save("processed/preprocessor.pkl")

    X_engineered_test = preprocessor.transform(x_test)
    X_engineered_test.to_csv(os.path.join(path,'x_test'))
    y_test.to_csv(os.path.join(path,'y_test'))