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

# 2. yaml: Reads the configuration file
# - Converts your .yaml file into a Python dictionary
import yaml
from dvclive import Live

# metrics
from sklearn.metrics import f1_score

# models
from sklearn.model_selection import RandomizedSearchCV

from churn_src.config import root_logger

# Customizable libraries
from churn_src.data_loader import load_your_data
from churn_src.feature_engineer import feature_engineer_full_workflow
from churn_src.models import extract_models
from churn_src.results_tracker import load_best_params, save_best_params
from churn_src.visualization import (
    build_Confusion_matrix,
    dvc_visualizations,
    learning_curves,
)

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
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")

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
X_train, y_train, X_test, y_test = load_your_data()

X_engineered_train, y_engineered_train = feature_engineer_full_workflow(X_train, y_train)

X_engineered_test, y_engineered_test = feature_engineer_full_workflow(X_test, y_test)


def load_param(args):
    #  - Check if the model exists in the config
    #  - master_config["models"] is a dictionary of all models
    #  - If the user asks for "random_forest" but it's not in the config, ERROR
    try:
        with open(args.config) as y:
            # This produces a dictionary
            master_config = yaml.safe_load(y)
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
    if args.mode == "search":

        print(f"🔬 Running hyperparameter search for {args.model}")

        # 2. Setup MLflow
        mlflow.create_experiment(f"Churn model {model_config['experiment_name']}")

        mlflow.set_experiment(f"Churn model {model_config["experiment_name"]}")

        with mlflow.start_run(run_name=f"{args.model}_search") as run:
            # - Log parameters dynamically to MLflow
            # - Log the model type as a parameter
            # - This helps you filter runs in the MLflow UI

            mlflow.log_param(
                "model_type", args.model
            )  # This has to change due to structure of our param.yaml file , especially model name
            mlflow.log_params(
                {
                    "n_iter": Global_config["n_iter"],
                    "cv": Global_config["cv"],
                    "scoring": Global_config["scoring"],
                    "n_jobs": Global_config["n_jobs"],
                }
            )
            param_grid = model_config["param"]

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

            model.fit(X_engineered_train, y_engineered_train)

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
            for key, value in best_params.items():
                mlflow.log_param(f"best_{key}", value)

            # 53. Log the best CV score
            mlflow.log_metric("best_cv_score", best_score)

            # 6. Evaluate and track metrics

            mlflow.log_metric("f1_score", f1_score)
            mlflow.set_tag("stage", "experimentation")
            # 55. Log the best model to MLflow
            mlflow.sklearn.autolog.log_model(best_model, f"{args.model}_best_model")

            # Log the best model
            mlflow.sklearn.autolog(best_model, f"{args.model}_best_model")

            # ------------------------------------------------------------
            # 🔥 CRITICAL: Save the best parameters to a file
            # ------------------------------------------------------------
            save_best_params(
                model_name=args.model,
                best_params=best_params,
                best_score=best_score,
                run_id=run.info.run_id,
            )

            # 56. Print success message
            print("\n" + "=" * 50)
            print(f"✅ SEARCH COMPLETE! Run ID: {run.info.run_id}")
            print("=" * 50)

    # ------------------------------------------------------------
    # 7. MODE 2: TRAIN (Uses the saved best parameters)
    # ------------------------------------------------------------
    elif args.mode == "train":

        # ------------------------------------------------------------
        # 🔥 CRITICAL: Load the best parameters from the search
        # ------------------------------------------------------------
        saved_params = load_best_params(args.model)

        try:
            # Use the best parameters from the search!
            train_params = saved_params
            print("✅ Loaded best parameters from previous search:")
            for key, value in train_params.items():
                print(f" {key}: {value}")

        except Exception:
            root_logger.exception(
                f"Had a problem loading the best parameters for {args.model}.Please run the search mode first.Run search first: python src/train.py --model {args.model} --mode search "
            )

        # ------------------------------------------------------------
        # Now train the model with the best parameters
        # ------------------------------------------------------------
        mlflow.set_experiment(model_config["experiment_name"])

        with mlflow.start_run() as run:

            # ---------- SET UP DVCLive ----------
            # This creates a 'dvclive/' folder and logs everything DVC can read
            # The 'save_dvc_exp=True' flag enables tracking with 'dvc exp show'

            live = Live("evaluation", save_dvc_exp=True)

            with live:

                # Log all parameters
                mlflow.log_param("model_type", args.model)
                for key, value in train_params.items():
                    mlflow.log_param(key, value)

                # Create and train the model
                model = extract_models(args.model, train_params)
                model.fit(X_engineered_train, y_engineered_train)

                # Generate predictions from our model
                train_predictions = model.predict(X_engineered_train)
                # Build the confusion matrix for training data
                cmap = build_Confusion_matrix(y_engineered_train, train_predictions)
                live.log_image(cmap, name="confusion_matrix_train.png")
                lr = learning_curves(model, X_engineered_train, y_engineered_train)
                live.log_image(lr, name="learning_curve_train.png")
                # Build precision-recall and ROC curves for training data
                # First check whether the model has predict_proba method or decision_function
                if hasattr(model, "predict_proba"):
                    train_pred_proba = model.predict_proba(X_engineered_train)[:, 1]
                elif hasattr(model, "decision_function"):
                    # For this case we need not specify the class index
                    # since decision_function returns a single score for binary classification
                    train_pred_proba = model.decision_function(X_engineered_train)
                else:
                    raise ValueError(
                        "Model does not have predict_proba or decision_function method."
                    )
                live.log_plot(
                    train_pred_proba,
                    y_engineered_train,
                    name="roc_curve_train.png",
                    kind="roc",
                )
                live.log_plot(
                    train_pred_proba,
                    y_engineered_train,
                    name="precision_recall_curve_train.png",
                    kind="prc",
                )

                dvc_img = dvc_visualizations(y_engineered_train, train_pred_proba)
                live.log_image(dvc_img, name="dvc_visualizations_train.png")

                # Evaluate
                test_score = model.f1_score(X_engineered_test, y_engineered_test)
                mlflow.log_metrics({"test_f1_score": test_score})

                print(f"📈 Test F1 Score: {test_score:.4f}")

                # Log the model
                mlflow.sklearn.autolog(model, f"{args.model}_production_model")

                # Save the run ID for reference
                with open(f"results/{args.model}_production_run.txt", "w") as f:
                    f.write(f"Run ID: {run.info.run_id}\n")
                    f.write(f"Parameters: {train_params}\n")
                    f.write(f"Test F1 Score: {test_score:.4f}\n")


if __name__ == "__main__":
    main()
