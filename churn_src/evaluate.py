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


root_logger.setLevel("ERROR")
root = Path(__file__).resolve().parent.parent
client = MlflowClient()

filter_string = "name LIKE '%tree%'"

experiments = client.search_experiments(filter_string=filter_string)
all_exp_ids = [exp.experiment_id for exp in tqdm(experiments)]

runs = client.search_runs(
    experiment_ids=8,
    filter_string="metrics.train_precision_score > 0.92",
    run_view_type=ViewType.ACTIVE_ONLY,
    order_by=["metrics.best_cv_score DESC"],
)

best_run_id = runs.info.run_id

if best_run_id is None:
    raise ValueError("No best run found. Please check the experiment and run IDs.")
elif sys.argv[2] == best_run_id:
    best_run_id = sys.argv[2]
else:
    pass  # Use the best_run_id from the search if not provided in command line

model_path = "decision_tree_model_3"

model_uri = f"runs:/{best_run_id}/{model_path}"

with mlflow.start_run(run_id=best_run_id):  # start_run, not the nonexistent mlflow.run
    model_version = mlflow.register_model(model_uri, "production_tree_model")

    # model_info.registered_model_version gives the REAL version, not a hardcoded "1"
    client.set_registered_model_alias(
        name="production_tree_model", alias="production", version=model_version.version
    )

    mlflow.end_run()

model = mlflow.sklearn.load_model("models:/production_tree_model@production")

data = pd.read_csv(root / sys.argv[1], index_col="security_no")
target = data["churn_risk_score"]
x_val = data.drop(columns=["churn_risk_score"])
processor = ChurnPreprocessor()
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

# Save the model as pickle
with open(f"{root}/prod_model/model.pkl", "wb") as f:
    pickle.dump(model, f)

predictions_output_dir = root / "predictions_output"
predictions_output_dir.mkdir(parents=True, exist_ok=True)
jobs_output = (
    predictions_output_dir
    / f"batch_predictions_{datetime.now().strftime('%Y%b%a%d_%H%M%S')}.csv"
)
results_df.to_csv(jobs_output, index=False)

if __name__ == "__main__":
    print(results_df.head())
