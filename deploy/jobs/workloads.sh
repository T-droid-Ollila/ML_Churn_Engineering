#!/usr/bin/env bash
set -e  # exit immediately if any step fails — important for job reliability
# Tracking server
export MLFLOW_SERVER ='http://127.0.0.1:5000/'
# Store results
export RESULTS_DIR= "results/decision_tree/exp_3.json"
# Get and save run ID
export RUN_ID=$(python -c "import os; from churn_src import results_tracker; d = results_tracker.load_best_params(os.getenv('RESULTS_DIR')); print(d['mlflow_run_id'])")
# File for storing metrics
export RESULTS_FILE= "results/evaluation.json"
# Test data
export DATASET = "Customer_transactions.csv"
export DATASET_LOC = "churn_data/test_churn.csv"

python -m pytest --dataset-loc=$DATASET tests/data --verbose --disable-warnings > $DATASET

python churn_src/evaluate.py \
    --validation_path "$DATASET_LOC" \
    --run_id "$RUN_ID" \
    --output_path "$RESULTS_FILE"

# python churn_src/predict.py\
#     --model_path "prod_model/model.pkl" \
#     --input_path "$DATASET" \
#     --output_path "$RESULTS_FILE"
