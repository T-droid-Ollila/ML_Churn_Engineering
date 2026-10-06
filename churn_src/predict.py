import argparse
import pickle
from pathlib import Path

import pandas as pd
import ray
from config import root_logger
from evaluate import save_results
from feature_engineer import ChurnPreprocessor

root_logger.setLevel("WARNING")

args = argparse.ArgumentParser()

args.add_argument("--model_path", required=True, help="Pickled model")
args.add_argument(
    "preprocess_path", required=True, help="Path for the artifacts for preprocessing"
)
args.add_argument("--input_path", required=True, help="Path for dataset")


class ChurnPredictor:
    def __init__(self, model_path: str, preprocessor: ChurnPreprocessor):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        self.preprocessor = preprocessor

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        if "churn_risk_score" in batch.columns:
            batch = batch.drop(columns=["churn_risk_score"])
        transformed = self.preprocessor.transform(batch)  # no y, no fit flag, no ambiguity
        batch["prediction"] = self.model.predict(transformed)
        batch["probabilities"] = self.model.predict_proba(transformed)[:, 1]
        return batch


def predict(dataset_loc) -> pd.DataFrame:
    """Predict on the incoming dataset.

    Args:
        dataset_loc (str): dataset (with labels) to evaluate on.
        results_fp (str, optional): location to save evaluation results to. Defaults to None.

    """
    data = Path(dataset_loc)
    ds = ray.data.read_csv(data)

    ds_p = ds.map_batches(
        ChurnPredictor,
        fn_constructor_kwargs={
            "model_path": args.model_path,
            "preprocessor": ChurnPreprocessor.load(str(args.preprocess_path)),
        },
        batch_format="pandas",
    )

    return ds_p


if __name__ == "__main__":
    if not ray.is_initialized():
        ray.init()

    root = Path(__file__).resolve().parent.parent
    ds_p = predict(args.input_path)
    results_path = save_results(ds_p, root)
    print(f"Predictions saved to {results_path}")
    print(ds_p.take(5))
