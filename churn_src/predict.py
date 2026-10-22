import pickle
import sys

import pandas as pd
import ray
from config import root_logger
from evaluate import save_results
from feature_engineer import ChurnPreprocessor

root_logger.setLevel("WARNING")

preprocessor = ChurnPreprocessor.load(str(sys.argv[1]))
model = sys.argv[3]  # path to the model file


class ChurnPredictor:
    def __init__(self, preprocessor):
        with open(model, "rb") as f:
            self.model = pickle.load(f)
        self.preprocessor = preprocessor

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        if "churn_risk_score" in batch.columns:
            batch = batch.drop(columns=["churn_risk_score"])
        id_col = "security_no"
        ids = batch[id_col].reset_index(drop=True)

        features = batch.drop(columns=[id_col])
        transformed = self.preprocessor.transform(features)  # no y, no fit flag, no ambiguity
        batch["prediction"] = self.model.predict(transformed)
        batch["probabilities"] = self.model.predict_proba(transformed)[:, 1]
        return pd.DataFrame(
            {
                id_col: ids,
                "prediction": batch["prediction"],
                "probabilities": batch["probabilities"],
            }
        )


def predict(dataset_loc) -> pd.DataFrame:
    """Predict on the incoming dataset.

    Args:
        dataset_loc (str): dataset (with labels) to evaluate on.
        results_fp (str, optional): location to save evaluation results to. Defaults to None.

    """
    ds = ray.data.read_csv(dataset_loc)

    ds_p = ds.map_batches(
        ChurnPredictor,
        fn_constructor_kwargs={
            "preprocessor": preprocessor,
        },
        batch_format="pandas",
        concurrency=3,
    )
    # Materialize once; Ray datasets are lazy, so calling take() after saving
    # would re-run the whole pipeline.
    return ds_p.to_pandas()


if __name__ == "__main__":
    if not ray.is_initialized():
        ray.init()
    data = sys.argv[2]
    ds_p = predict(data)
    results_path = save_results(ds_p)
    print(f"Predictions saved to {results_path}")
    print(ds_p.head())
