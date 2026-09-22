import json
from pathlib import Path


def save_best_params(model_name: str, best_params: dict, best_score: float, run_id: str ,num: int):
    """
    Saves the best parameters from a search to a JSON file.

    Parameters:
    -----------
    model_name : str
        The name of the model (e.g., "random_forest")
    best_params : dict
        The best hyperparameters found
    best_score : float
        The best cross-validation score
    run_id : str
        The MLflow run ID
    """
    # Create the results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Create the best_params file
    filepath = results_dir / model_name / f"exp_{num}.json"

    # Save the data
    data = {
        "model_name": model_name,
        "best_params": best_params,
        "best_score": best_score,
        "mlflow_run_id": run_id,
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

    print(f"💾 Best parameters saved to: {filepath}")
    return filepath


def load_best_params(model_name: str, num: int):
    """
    Loads the best parameters for a model from the JSON file.

    Parameters:
    -----------
    model_name : str
        The name of the model (e.g., "random_forest")
    num : int
        The experiment number

    Returns:
    --------
    dict: The best parameters, or None if not found
    """
    filepath = Path("results") / model_name / f"exp_{num}.json"

    if not filepath.exists():
        return None

    with open(filepath) as f:
        data = json.load(f)

    return data