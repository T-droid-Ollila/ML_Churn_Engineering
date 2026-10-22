# Stored results
$env:RESULTS_DIR= "results/decision_tree/exp_3.json"
# Get and save run ID
$env:RUN_ID=$(python -c "import os; from churn_src import results_tracker; e = results_tracker.load_best_params('decision_tree', 3); print(e)")
# File for storing incoming results
$env:RESULTS_FILE= "results.json"
# Test data
$env:DATASET = "c:\Users\Simon\OneDrive\Desktop\customer_membership_transactions_20000_with_feedback.csv"

$env:DATASET_LOC = "churn_data/test_churn.csv"
$env:MLFLOW_SERVER = 'http://127.0.0.1:5000/'
$env:MODEL = 'artifacts\prod_model\model.pkl'
$env:PROCESSOR = 'artifacts/processed/preprocessor.pkl'

# python -m pytest --dataset-loc="$env:DATASET_LOC" tests/data --verbose

# python churn_src\evaluate.py $env:DATASET_LOC $env:RUN_ID $env:MLFLOW_SERVER

python -m pytest --dataset-loc="$env:DATASET" tests/data --verbose --disable-warnings

python churn_src\predict.py $env:PROCESSOR $env:DATASET

if ($LASTEXITCODE -ne 0) {
    Write-Error "Batch prediction failed"
    exit 1
}
