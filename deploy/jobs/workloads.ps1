# Store results
$env:RESULTS_DIR= "results/decision_tree/exp_1.json"
# Get and save run ID
$env:RUN_ID=$(python -c "import os; from churn_src import results_tracker; d = results_tracker.load_best_params(os.getenv('RESULTS_DIR')); print(d['run_id'])")
# File for storing incoming results
$env:RESULTS_FILE= "results.json"
# Test data
$env:DATASET = "Customer_transactions.csv"

python .\churn_src\predict.py  `
  --model-path "models:/churn_model/Production" `
  --run-id $env:RUN_ID `
  --input-path $env:DATASET `
  --output-path $env:RESULTS_FILE

python .\churn_src\evaluate.py `
  --model-path "models:/churn_model/Production" `
  --run-id $env:RUN_ID `
  --input-path $env:DATASET `
  --output-path $env:RESULTS_FILE

if ($LASTEXITCODE -ne 0) {
    Write-Error "Batch prediction failed"
    exit 1
}