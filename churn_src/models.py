# Models

# - Import all the models we might use
# - Each import brings in a different model class

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def extract_models(model: str, parameters: dict):
    """
    Returns a model with the given parameters.

    Parameters:
    -----------
    model_type : str
        The type of model to create (e.g., "random_forest")
    model_params : dict
        The parameters to pass to the model constructor

    Returns:
    --------
    model : sklearn estimator
        The created model object

    Example:
    --------
    >>> model = extract_models("random_forest", {"n_estimators": 100})
    >>> model.fit(X, y)

    """
    if model.lower() == "random_forest":
        return RandomForestClassifier(**parameters)
    elif model.lower() == "xgboost":
        return XGBClassifier(**parameters)
    elif model.lower() == "logistic_regression":
        return LogisticRegression(**parameters)
    elif model.lower() == "decision_tree_classifier":
        return DecisionTreeClassifier(**parameters)
    # 4. If the model_type is unknown, raise an error
    else:
        raise ValueError(f"Unknown model type: {model}")
