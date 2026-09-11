import matplotlib as plt
import numpy as np
import seaborn as sns
from mlflow.entities import Dataset
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import learning_curve

# We shall define a function for visualizations of our results of experiments


def learning_curves(model: str, X_train: Dataset, y_train):
    """
    This function is used to plot the learning curves of the model during training and validation.

    It takes the training and validation loss and accuracy as input and plots them.

    We shall use the LearningCurveDisplay from sklearn to plot the learning curves.

    """

    train_sizes = [0.1, 0.33, 0.55, 0.78, 1.0]

    train_sizes, train_scores, cross_val_scores = learning_curve(
        # The same model architecture
        model,
        # The training data
        X_train,
        y_train,
        # <-- CV USED HERE FOR DIAGNOSIS
        cv=5,
        train_sizes=train_sizes,
        scoring="neg_log_loss",
        n_jobs=-1,
    )
    pos_train_scores = -np.mean(train_scores, axis=1)
    pos_cross_val_scores = -np.mean(cross_val_scores, axis=1)
    # Plotting the learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, pos_train_scores, label="Training Log Loss")
    plt.plot(train_sizes, pos_cross_val_scores, label="Cross Validation Log Loss")
    plt.title("Number of examples vs. train and CV Log Losses")
    plt.xlabel("Total number of training and cv examples")
    plt.ylabel("Log Loss")
    plt.legend(loc="lower right")
    plt.show()
    plt.grid(True)

    return plt.figure()


# We first define our confusion matrix function
def build_Confusion_matrix(target, pred):
    """
    This function takes the target and predicted values and builds a confusion matrix.

    """
    confmat = confusion_matrix(y_true=target, y_pred=pred)
    # define plots
    fig, ax = plt.subplots(figsize=(6, 6))
    # plt.matshow(confmat , alpha = 0.6 , cmap = plt.cm.Blues)

    # I decided to use SEABORN to see the results of my experiments but I can
    # switch between plt or seaborn
    sns.heatmap(confmat, annot=True, fmt="d", ax=ax)
    # Define how the matrix is built
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=i, y=j, s=confmat[i, j], va="center", ha="center")
    ax.axis.set_ticks_bottom()
    ax.title("Confusion Matrix")
    ax.get_xlabel("True label")
    ax.get_ylabel("Pred label")
    ax.legend(["0", "1"])

    return fig


def dvc_visualizations(target, pred_proba):
    """
    This function takes predictions from probabilities of the

    predicted class and the target classes to make a roc_auc curve

    precision_recall curve.

    ________parameters________

    target : series

    pred : series

    _______Returns_________

    visualisations of the metrics produced.

    """
    # Get probabilities for the positive class

    precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(
        target, pred_proba
    )
    fpr_forest, tpr_forest, thresholds_roc_forest = roc_curve(target, pred_proba)

    # The default threshold is 0.5, so let's find the threshold for 99% recall
    #  95% precision, 90% precision and 92% precision
    Idx_99_recall, idx_for_95_precision, idx_for_92_precision, idx_for_90_precision = (
        np.argmax(recalls_forest >= 0.99),
        np.argmax(precisions_forest >= 0.95),
        np.argmax(precisions_forest >= 0.92),
        np.argmax(precisions_forest >= 0.90),
    )
    (
        threshold_99_recall,
        threshold_95_precision,
        threshold_92_precision,
        threshold_90_precision,
    ) = (
        thresholds_forest[Idx_99_recall],
        thresholds_forest[idx_for_95_precision],
        thresholds_forest[idx_for_92_precision],
        thresholds_forest[idx_for_90_precision],
    )
    # To make predictions based on the custom thresholds, we can use the following code:

    # y_pred_99_recall = (pred_proba >= threshold_99_recall).astype(int)
    # y_pred_95_precision = (pred_proba >= threshold_95_precision).astype(int)
    # y_pred_92_precision = (pred_proba >= threshold_92_precision).astype(int)
    # y_pred_90_precision = (pred_proba >= threshold_90_precision).astype(int)

    # Let's check precision and recall for these custom thresholds and print the results

    # precision_99_recall = precision_score(target, y_pred_99_recall)

    # recall_99_recall = recall_score(target, y_pred_99_recall)

    # precision_95_precision = precision_score(target, y_pred_95_precision)

    # recall_95_precision = recall_score(target, y_pred_95_precision)

    # precision_92_precision = precision_score(target, y_pred_92_precision)

    # recall_92_precision = recall_score(target, y_pred_92_precision)

    # precision_90_precision = precision_score(target, y_pred_90_precision)

    # recall_90_precision = recall_score(target, y_pred_90_precision)

    idx_threshold_roc_99_recall = (thresholds_roc_forest <= threshold_99_recall).argmax()
    fpr_99_recall, tpr_99_recall = (
        fpr_forest[idx_threshold_roc_99_recall],
        tpr_forest[idx_threshold_roc_99_recall],
    )
    idx_threshold_roc_95_precision = (
        thresholds_roc_forest <= threshold_95_precision
    ).argmax()
    fpr_95_precision, tpr_95_precision = (
        fpr_forest[idx_threshold_roc_95_precision],
        tpr_forest[idx_threshold_roc_95_precision],
    )
    idx_threshold_roc_92_precision = (
        thresholds_roc_forest <= threshold_92_precision
    ).argmax()
    fpr_92_precision, tpr_92_precision = (
        fpr_forest[idx_threshold_roc_92_precision],
        tpr_forest[idx_threshold_roc_92_precision],
    )
    idx_threshold_roc_90_precision = (
        thresholds_roc_forest <= threshold_90_precision
    ).argmax()
    fpr_90_precision, tpr_90_precision = (
        fpr_forest[idx_threshold_roc_90_precision],
        tpr_forest[idx_threshold_roc_90_precision],
    )

    # Let's plot the Precision-Recall curve and ROC curve for the Random Forest model
    fig, ax = plt.subplots(3, 1, figsize=(8, 20))

    # Precision-Recall curve
    ax[0].plot(recalls_forest, precisions_forest, label="XGB Classifier")
    # Plot the points corresponding to the custom thresholds on the Precision-Recall curve
    ax[0].scatter(
        recalls_forest[Idx_99_recall],
        precisions_forest[Idx_99_recall],
        color="green",
        label=f"Point at {threshold_99_recall:.4f}",
        zorder=5,
    )
    ax[0].scatter(
        recalls_forest[idx_for_95_precision],
        precisions_forest[idx_for_95_precision],
        color="orange",
        label=f"Point at {threshold_95_precision:.4f}",
        zorder=5,
    )
    ax[0].scatter(
        recalls_forest[idx_for_92_precision],
        precisions_forest[idx_for_92_precision],
        color="red",
        label=f"Point at {threshold_92_precision:.4f}",
        zorder=5,
    )
    ax[0].scatter(
        recalls_forest[idx_for_90_precision],
        precisions_forest[idx_for_90_precision],
        color="purple",
        label=f"Point at {threshold_90_precision:.4f}",
        zorder=5,
    )
    ax[0].set_xlabel("Recall")
    ax[0].set_ylabel("Precision")
    ax[0].set_title("Precision-Recall Curve")
    ax[0].legend(loc="best")

    # Plot precision and recall as functions of the threshold values
    ax[1].plot(
        thresholds_forest,
        precisions_forest[:-1],
        label="Precision",
        c="blue",
        linewidth=2,
    )
    ax[1].plot(thresholds_forest, recalls_forest[:-1], label="Recall", c="red", linewidth=2)
    ax[1].axvline(
        x=threshold_99_recall, c="green", linestyle="--", label="Threshold for 99% Recall"
    )
    ax[1].axvline(
        x=threshold_95_precision,
        c="orange",
        linestyle="--",
        label="Threshold for 95% Precision",
    )
    ax[1].axvline(
        x=threshold_92_precision,
        c="red",
        linestyle="--",
        label="Threshold for 92% Precision",
    )
    ax[1].axvline(
        x=threshold_90_precision,
        c="purple",
        linestyle="--",
        label="Threshold for 90% Precision",
    )
    ax[1].set_xlabel("Threshold")
    ax[1].set_ylabel("Precision / Recall")
    ax[1].set_title("Precision and Recall vs Threshold")
    ax[1].set_ylim(0.75, 1)
    ax[1].legend(loc="best")

    # Plot ROC curve
    ax[2].plot(fpr_forest, tpr_forest, linewidth=2, label="XGB Classifier")
    ax[2].scatter(
        fpr_99_recall,
        tpr_99_recall,
        color="green",
        label=f"Point at {threshold_99_recall:.4f}",
        zorder=5,
    )
    ax[2].scatter(
        fpr_95_precision,
        tpr_95_precision,
        color="orange",
        label=f"Point at {threshold_95_precision:.4f}",
        zorder=5,
    )
    ax[2].scatter(
        fpr_92_precision,
        tpr_92_precision,
        color="red",
        label=f"Point at {threshold_92_precision:.4f}",
        zorder=5,
    )
    ax[2].scatter(
        fpr_90_precision,
        tpr_90_precision,
        color="purple",
        label=f"Point at {threshold_90_precision:.4f}",
        zorder=5,
    )
    ax[2].plot(
        [0, 1], [0, 1], "k--", label="Random Classifier"
    )  # Diagonal line for random classifier
    ax[2].set_xlabel("False Positive Rate")
    ax[2].set_ylabel("True Positive Rate")
    ax[2].set_title("ROC Curve")
    ax[2].legend(loc="best")

    # return three plots from the function
    return fig
