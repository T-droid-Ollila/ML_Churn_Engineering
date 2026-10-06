import datetime
import os
import pickle
import sys
from pathlib import Path

import mlflow
import pandas as pd
from config import root_logger
from feature_engineer import ChurnPreprocessor
from mlflow.client import MlflowClient
from mlflow.entities import ViewType
from tqdm.notebook import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def get_best_run_id(
    client: MlflowClient,
    filter_string: str,
    metric_filter: str,
    order_by: list[str],
    override_run_id: str | None = None,
) -> str:

    # This part of the code was for analysis to come up with the last bit
    # of results for a particular experiment

    experiments = client.search_experiments(filter_string=filter_string)
    all_exp_ids = [exp.experiment_id for exp in tqdm(experiments)]

    """Find the best run by metric, unless an explicit override is given."""

    runs = client.search_runs(
        experiment_ids=all_exp_ids,
        filter_string=metric_filter,
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=order_by,
    )

    if not runs:
        raise ValueError("No best run found. Please check the experiment filters.")

    best_run_id = runs[0].info.run_id

    if override_run_id is not None and best_run_id == override_run_id:
        return override_run_id
    else:
        # Use the best_run_id from the search if not provided in command line and does not match the condition given
        return best_run_id


def promote_model_to_production(
    client: MlflowClient, run_id: str, model_path: str, model_name: str, alias: str
):
    """Promote a model to production by registering it and setting an alias."""

    model_uri = f"runs:/{run_id}/{model_path}"

    with mlflow.start_run(run_id=run_id):  # start_run, not the nonexistent mlflow.run
        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

        # model_info.registered_model_version gives the REAL version, not a hardcoded "1"
        client.set_registered_model_alias(
            name=model_name, alias=alias, version=model_version.version
        )

        mlflow.end_run()


def score_batch(
    model, data: pd.DataFrame, processor: ChurnPreprocessor, target_col: str
) -> pd.DataFrame:
    """Transform, predict, and return a results DataFrame for a batch of data."""
    target = data[target_col]
    x_val = data.drop(columns=[target_col])

    x_processed = processor.transform(x_val)
    model_predictions = model.predict(x_processed)
    model_probabilities = model.predict_proba(x_processed)[:, 1]

    results_df = pd.DataFrame(
        {
            "id": x_val.index,
            "actual": target,
            "predicted": model_predictions,
            "probabilities": model_probabilities,
        }
    )
    return results_df


def save_results(results_df: pd.DataFrame, root: Path) -> Path:
    predictions_output_dir = root / "predictions_output"
    predictions_output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%b-%d-%a_%H-%M-%S")
    jobs_output = predictions_output_dir / f"batch_predictions_{timestamp}.csv"
    results_df.to_csv(jobs_output, index=False)
    return jobs_output


if __name__ == "__main__":
    model_path = "decision_tree_model_3"
    filter_string = "name LIKE '%tree%'"
    order_by = ["metrics.best_cv_score ASC"]
    metric_filter = "metrics.best_cv_score > 0.92"

    root_logger.setLevel("ERROR")
    root = Path(__file__).resolve().parent.parent
    client = MlflowClient(
        tracking_uri=sys.argv[3]
    )  # Pass the MLflow server URI as a command line argument
    mlflow.set_tracking_uri(client.tracking_uri)

    best_run_id = get_best_run_id(client, filter_string, metric_filter, order_by, sys.argv[2])
    promote_model_to_production(
        client, best_run_id, model_path, "production_tree_model", "production"
    )
    model = mlflow.sklearn.load_model("models:/production_tree_model@production")
    f_data = pd.read_csv(root / sys.argv[1], index_col="security_no")

    pickle_path = root / "processed" / "preprocessor.pkl"
    processor = ChurnPreprocessor().load(str(pickle_path))

    results_df = score_batch(model, f_data, processor, "churn_risk_score")
    results_path = save_results(results_df, root)
    # Save the model as pickle
    prod_model_dir = root / "prod_model"
    prod_model_dir.mkdir(parents=True, exist_ok=True)
    with open(prod_model_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Predictions saved to: {results_path}")
    print(results_df.head())
