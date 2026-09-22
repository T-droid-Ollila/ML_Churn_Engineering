#!/usr/bin/env bash
set -e  # exit immediately if any step fails — important for job reliability
# Tracking server
export MLFLOW_SERVER ='https://local_host//5000'
# Store results
export RESULTS_DIR= "results/decision_tree/exp_1.json"
# Get and save run ID
export RUN_ID=$(python -c "import os; from churn_src import results_tracker; d = results_tracker.load_best_params(os.getenv('RESULTS_DIR')); print(d['run_id'])")
# File for storing metrics
export RESULTS_FILE= "results/"
# Test data
export DATASET = "Customer_transactions.csv"

python -m pytest --dataset-loc=$DATASET tests/data --verbose --disable-warnings > $DATASET

python churn_src/evaluate.py \
    --run_id "$RUN_ID" \
    --input_path "$DATASET" \
    --output_path "$RESULTS_FILE" 

python churn_src/predict.py\
    --model_path "models/model.pkl" \
    --tracking_uri "$MLFLOW_SERVER" \
    --input_path "$DATASET" \
    --output_path "$RESULTS_FILE"