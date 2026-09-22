# 3. os: Operating system interface
# - Used to read environment variables

# 5. argparse: Command-line argument parser
# - Allows you to pass arguments like --model random_forest
import argparse
import os
import sys

# 6. pathlib: Object-oriented file paths
# - Better than using strings for paths (works on Windows/Linux/Mac)
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd

# 2. yaml: Reads the configuration file
# - Converts your .yaml file into a Python dictionary
import yaml
from dvclive import Live

# metrics
from sklearn import metrics

# models
from sklearn.model_selection import RandomizedSearchCV

from churn_src.config import root_logger

# Customizable libraries
from churn_src.data_loader import load_your_data
from churn_src.feature_engineer import ChurnPreprocessor
from churn_src.models import extract_models
from churn_src.results_tracker import load_best_params, save_best_params
from churn_src.visualization import (
    build_Confusion_matrix,
    dvc_visualizations,
    learning_curves)

# 1. MLflow: The experiment tracking library
#    - mlflow: Main module for logging parameters, metrics, and models
#    - mlflow.sklearn: Special module for logging scikit-learn models
#    - mlflow.xgboost: Special module for logging XGBoost models


# In essence I don't need this coz my helper functions are not
# in folder coz the sys.append check folders
sys.path.append(str(Path(__file__).parent))


# 15. Get the tracking URI from environment, or use default
# - os.getenv("KEY", "default"): If KEY doesn't exist, use "default"
# This allows you to override with a remote server (e.g., http://mlflow-server:5000)
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# 16. Set the tracking URI for MLflow
#     - All future MLflow calls will use this URI
mlflow.set_tracking_uri(tracking_uri)

# 17. Create the argument parser
parser = argparse.ArgumentParser(description="Train an ML model")

# 18. Add --config argument (optional)
#     - type=str: The value should be a string
#     - default="configs/master_config.yaml": If not provided, use this
#     - help: Description shown when user runs --help
parser.add_argument(
    "--config", type=str, default="/params.yaml", help="Path to the master config file"
)

# 19. Add --model argument (REQUIRED)
#     - required=True: The user MUST provide this
#     - This tells the script which model to train (e.g., random_forest)
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Which model to train (e.g., random_forest, xgboost)",
)

# 20. Add --mode argument (optional, defaults to "train")
#     - choices=["train", "search"]: Only allows these two values
#     - default="search": If not provided, use "search"

parser.add_argument(
    "--mode",
    type=str,
    choices=["train", "search"],
    default="search",
    help="'train' for single model, 'search' for hyperparameter search",
)

# 21. Parse the arguments
#     - This reads what the user typed on the command line
#     - args.config: The config file path
#     - args.model: The model name
#     - args.mode: "train" or "search"

args = parser.parse_args()


# 4. Load your data (tracked by DVC)
x_train, y_train, x_test, y_test = load_your_data()

preprocessor = ChurnPreprocessor()

X_engineered_train = pd.read_csv(Path('processed_data/x_train'),index_col='security_no')

y_train = pd.read_csv(Path('processed_data/y_train'),index_col='security_no')

preprocessor.save("processed/preprocessor.pkl")

X_engineered_test = pd.read_csv(Path('processed_data/x_test'),index_col='security_no')

y_test = pd.read_csv(Path('processed_data/y_test'),index_col='security_no')


def load_param(args):
    #  - Check if the model exists in the config
    #  - master_config["models"] is a dictionary of all models
    #  - If the user asks for "random_forest" but it's not in the config, ERROR
    try:
        with open(args.config) as y:
            # This produces a dictionary
            master_config = yaml.safe_load(y)
            return master_config
    except FileNotFoundError:
        sys.exception(f"Config file not found: {args.config}")
        # Exit the script with an error code
        sys.exit(1)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")
        sys.exit(1)
    if args.model not in master_config["models"]:
        raise ValueError(
            f"Model '{args.model}' not found. Available: {list(master_config['models'].keys())}"
        )


def main():
    # Load parameters
    config = load_param(args)
    model_config = config["models"][args.model]
    # Based on the design pattern of .yaml "script" active_model should not be here

    # model_params = config[model_name] # Back to this

    # - Get the specific model's configuration
    # - model_config now contains all the parameters for this model

    # 25. Check if the model is enabled
    #  If enabled: false, skip it
    if not model_config.get("enabled", True):
        print(f"⚠️ Model '{args.model}' is disabled. Set 'enabled: true' to use it.")
        sys.exit(0)  # Exit the script with success code (0)

    # 26. Print what we're doing (user feedback)
    print(f"🚀 Training model: {args.model}")
    print(f"📂 Using config: {args.config}")

    # Get shared configurations across all models

    Global_config = config["search"]
    # ------------------------------------------------------------
    # SECTION 6: MODE 1 - SINGLE MODEL TRAINING
    # ------------------------------------------------------------
    num = 1  # This is the experiment number, you can change it or make it dynamic

    if args.mode == "search":

        print(f"🔬 Running hyperparameter search for {args.model}")

        # 2. Setup MLflow
        mlflow.create_experiment(f"{model_config['experiment_name']}_{num}")

        mlflow.set_experiment(f"{model_config['experiment_name']}_{num}")

        with mlflow.start_run() as run:
            # - Log parameters dynamically to MLflow
            # - Log the model type as a parameter
            # - This helps you filter runs in the MLflow UI
            # This has to change due to structure of our param.yaml file , especially model name

            param_grid = model_config["params"]

            # 5. Initialize and train the selected model
            # No parameters were specified , the empty dict enables us do this objective
            model = extract_models(args.model, {})

            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=Global_config["n_iter"],
                cv=Global_config["cv"],
                scoring=Global_config["scoring"],
                n_jobs=Global_config["n_jobs"],
            )

            model.fit(X_engineered_train, y_train)

            # - Get the best results
            best_params = random_search.best_params_
            best_score = random_search.best_score_
            # 54. Evaluate the best model on the test set
            best_model = random_search.best_estimator_


            # 51. Print the best results (user feedback)
            print(f"\n🏆 Best CV Score: {best_score:.4f}")
            print("📝 Best Parameters:")
            for key, value in best_params.items():
                print(f"    {key}: {value}")

            # 52. Log the best parameters to MLflow
            mlflow.log_params(
                {
                    "n_iter": Global_config["n_iter"],
                    "cv": Global_config["cv"],
                    "scoring": Global_config["scoring"],
                    "n_jobs": Global_config["n_jobs"],
                }
            )
            
            for key, value in best_params.items():
                mlflow.log_param(f"best_{key}", value)

            # 53. Log the best CV score
            mlflow.log_metric("best_cv_score", best_score)

            # 6. Evaluate and track metrics

            mlflow.log_metric("f1_score", best_score)
            mlflow.set_tag("stage", "experimentation")
            mlflow.sklearn.log_model(best_model, f"{args.model}_model_{num}")
            
            # ------------------------------------------------------------
            # 🔥 CRITICAL: Save the best parameters to a file
            # ------------------------------------------------------------
            save_best_params(
                model_name=args.model,
                best_params=best_params,
                best_score=best_score,
                run_id=run.info.run_id,
                num = num
            )

            # 56. Print success message
            print("\n" + "=" * 50)
            print(f"✅ SEARCH COMPLETE! Run ID: {run.info.run_id}")
            print("=" * 50)
            # End dvc run
            run.end()
        mlflow.end_run()  # End the MLflow run

    # ------------------------------------------------------------
    # 7. MODE 2: TRAIN (Uses the saved best parameters)
    # ------------------------------------------------------------
    elif args.mode == "train":

        # ------------------------------------------------------------
        # 🔥 CRITICAL: Load the best parameters from the search
        # ------------------------------------------------------------
        saved_params = load_best_params(args.model , num)

        try:
            # Use the best parameters from the search!
            train_params = saved_params
            print("✅ Loaded best parameters from previous search:")
            for key, value in train_params.items():
                print(f"{key}: {value}")

        except Exception:
            root_logger.exception(
                f"Had a problem loading the best parameters for {args.model}.Please run the search mode first.Run search first: python src/train.py --model {args.model} --mode search "
            )

        # ------------------------------------------------------------
        # Now train the model with the best parameters
        # ------------------------------------------------------------
        mlflow.set_experiment(f"{model_config['experiment_name']}_{num}")

        with mlflow.start_run() as run:

            # ---------- SET UP DVCLive ----------
            # This creates a 'dvclive/' folder and logs everything DVC can read
            # The 'save_dvc_exp=True' flag enables tracking with 'dvc exp show'

            live = Live("evaluation", save_dvc_exp=True)

            with live:
                # Create and train the model
                model = extract_models(args.model, train_params)
                model.fit(X_engineered_train, y_train)

                train_predictions = model.predict(X_engineered_train)
                y_pred_test = model.predict(X_engineered_test)

                f1 = metrics.f1_score(y_train, train_predictions, average="weighted")
                precision = metrics.precision_score(y_train, train_predictions, average="weighted")
                recall = metrics.recall_score(y_train, train_predictions, average="weighted")

                cmap = build_Confusion_matrix(y_train, train_predictions)
                live.log_image(val=cmap, name="confusion_matrix_train.png")

                lc = learning_curves(model, X_engineered_train, y_train)
                live.log_image(val=lc, name="learning_curve_train.png")

                if hasattr(model, "predict_proba"):
                    y_train_proba= model.predict_proba(X_engineered_train)[:, 1]
                    test_predictions = model.predict_proba(X_engineered_test)[:, 1]
                    y_train_proba = y_train_proba.astype(float) if hasattr(y_train_proba, "astype") else y_train_proba
                    test_predictions = test_predictions.astype(float) if hasattr(test_predictions, "astype") else test_predictions
                elif hasattr(model, "decision_function"):
                    y_train_proba = model.decision_function(X_engineered_train)
                    test_predictions = model.decision_function(X_engineered_test)
                    y_train_proba = y_train_proba.astype(float) if hasattr(y_train_proba, "astype") else y_train_proba
                    test_predictions = test_predictions.astype(float) if hasattr(test_predictions, "astype") else test_predictions
                else:
                    raise ValueError("Model does not have predict_proba or decision_function method.")

                live.log_sklearn_plot("roc", y_train, y_train_proba, name="roc_curve_train.png")
                live.log_sklearn_plot("precision_recall", y_train, y_train_proba, name="precision_recall_curve_train.png")

                dvc_img = dvc_visualizations(y_train, y_train_proba)
                live.log_image(val=dvc_img, name="dvc_visualizations_train.png")

                train_roc_auc = metrics.roc_auc_score(y_train, y_train_proba)

                train_metrics = {
                    "train_f1_score": f1,
                    "train_precision_score": precision,
                    "train_recall_score": recall,
                    "train_roc_auc": train_roc_auc,
                }

                mlflow.log_metrics(train_metrics)
                for key, value in train_metrics.items():
                    live.log_metric(key, value)

                test_roc_auc = metrics.roc_auc_score(y_test, test_predictions)
                test_f1 = metrics.f1_score(y_test, y_pred_test, average="weighted")
                test_precision = metrics.precision_score(y_test, y_pred_test, average="weighted")
                test_recall = metrics.recall_score(y_test, y_pred_test, average="weighted")

                test_metrics = {
                    "test_roc_auc": test_roc_auc,
                    "test_f1_score": test_f1,
                    "test_precision_score": test_precision,
                    "test_recall_score": test_recall,
                }

                mlflow.log_metrics(test_metrics)
                for key, value in test_metrics.items():
                    live.log_metric(key, value)
                # 55. Log the model to MLflow

                live.end()
            mlflow.end_run()


if __name__ == "__main__":
    main()
