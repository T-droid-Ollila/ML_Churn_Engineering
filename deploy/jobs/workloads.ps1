# Store results
$env:RESULTS_DIR= "results/decision_tree/exp_3.json"
# Get and save run ID
$env:RUN_ID=$(python -c "import os; from churn_src import results_tracker; d = results_tracker.load_best_params(os.getenv('RESULTS_DIR')); print(d['mlflow_run_id'])")
# File for storing incoming results
$env:RESULTS_FILE= "results.json"
# Test data
$env:DATASET = "Customer_transactions.csv"
$env:DATASET_LOC = "churn_data/test_churn.csv"

python -m pytest --dataset-loc=$DATASET tests/data --verbose --disable-warnings > $DATASET

python .\churn_src\evaluate.py `
  --input-path $env:DATASET_LOC `
  --run-id $env:RUN_ID `
  --output-path $env:RESULTS_FILE

# python .\churn_src\predict.py  `
#   --model_path "prod_model/model.pkl" `
#   --input-path $env:DATASET `
#   --output-path $env:RESULTS_FILE


if ($LASTEXITCODE -ne 0) {
    Write-Error "Batch prediction failed"
    exit 1
}
