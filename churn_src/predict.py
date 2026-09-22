import mlflow
import ray
import os
from pathlib import Path
import argparse
from churn_src.feature_engineer import ChurnPreprocessor
from churn_src.config import root_logger
import pandas as pd
import datetime

root_logger.setLevel("WARNING")

args = argparse.ArgumentParser()

args.add_argument('--model_path', required=True ,help='Pickled model')
args.add_argument('preprocess_path',required=True, help = "Path for the artifacts for preprocessing")
args.add_argument('--tracking_uri',default='https://local_host//5000',help='Tracking_url for mlflow')
args.add_argument('--input_path' , required = True , help = 'Path for dataset')
args.add_argument('--output_path',required=True)

class ChurnPredictor:
    def __init__(self, model_uri: str, preprocessor_path: str, tracking_uri: str):
        mlflow.set_tracking_uri(tracking_uri)
        self.model = mlflow.sklearn.load_model(model_uri)
        self.preprocessor = ChurnPreprocessor.load(preprocessor_path)

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        if "churn_risk_score" in batch.columns:
            batch = batch.drop(columns=["churn_risk_score"])
        transformed = self.preprocessor.transform(batch)   # no y, no fit flag, no ambiguity
        batch["prediction"] = self.model.predict(transformed)
        batch["probabilities"] = self.model.predict_proba(transformed)[:, 1]
        return batch

def predict(dataset_loc, results_fp)->None:

    """Predict on the incoming dataset.

    Args:
        run_id (str): id of the specific run to load from. Defaults to None.
        dataset_loc (str): dataset (with labels) to evaluate on.
        results_fp (str, optional): location to save evaluation results to. Defaults to None.

    """
    data = Path(dataset_loc)
    ds = ray.data.read_csv(data)

    ds_p = ds.map_batches(ChurnPredictor,fn_constructor_kwargs={
            "model_uri": args.model_path,
            "preprocess_path": args.preprocess_path,
            "tracking_uri": args.tracking_uri
        },batch_format='pandas')
    
    out_dir = Path(results_fp)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_p.write_csv(out_dir/f"batch_predictions{datetime.now().strftime('%Y%b_%a%p')}")


if __name__ == "__main__":
    if not ray.is_initialized():
        ray.init()
    predict(args.input_path,args.output_path)