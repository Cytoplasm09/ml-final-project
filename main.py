from src.data.load_data import load_dataset
from src.data.preprocess import clean_dataset
from src.visualization.eda import plot_eda
from src.data.split_data import split_dataset
from src.features.build_features import build_features
from src.models.train_model import plot_roc_curves
from src.models.knn_model import train_knn_model
from src.models.dt_model import train_dt_model
from src.visualization.performance import (
    plot_confusion_matrices,
    plot_performance_comparison,
)
from src.utils.helper_functions import print_header, print_info, start_step, end_step
from sklearn.metrics import confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

TRAIN_PATH = "data/raw/churn_train.csv"
TEST_PATH = "data/raw/churn_test.csv"
SUBMISSION_PATH = "output/submission.csv"

# Colors for the models in the combined ROC plot. Each model keeps its color on every curve.
DT_COLOR = "#2a6fdb"
KNN_COLOR = "#e07b1a"


def main() -> None:
    print_header("Customer Churn Prediction")

    t = start_step("Loading data")
    raw_df = load_dataset(TRAIN_PATH)
    end_step(t)
    print_info("Raw dataset shape", raw_df.shape)

    t = start_step("Cleaning data")
    clean_df, imputer = clean_dataset(raw_df, None)
    end_step(t)
    print_info("Cleaned dataset shape", clean_df.shape)

    t = start_step("Building features")
    clean_df = build_features(clean_df)
    end_step(t)

    # Plotting steps are not timed: plt.show() waits until the window is closed.
    start_step("Creating EDA visuals")
    plot_eda(clean_df)

    t = start_step("Splitting data")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(clean_df)
    end_step(t)
    print_info("Training rows", len(X_train))
    print_info("Validation rows", len(X_val))
    print_info("Test rows", len(X_test))

    t = start_step("Training k-NN")
    knn_model, k = train_knn_model(X_train, y_train)
    end_step(t)

    t = start_step("Training decision tree")
    dt_model, depth = train_dt_model(X_train, y_train)
    end_step(t)

    # Labels that show the hyperparameters chosen by the grid search
    knn_label = f"k-NN (k={k})"
    dt_label = f"DT (depth={depth})"
    print_info("Chosen k", k)
    print_info("Chosen max depth", depth)

    t = start_step("Evaluating k-NN on validation set")
    y_val_pred_knn = knn_model.predict(X_val)
    val_prob_knn = knn_model.predict_proba(X_val)[:, 1]
    end_step(t)

    t = start_step("Evaluating decision tree on validation set")
    y_val_pred_dt = dt_model.predict(X_val)
    val_prob_dt = dt_model.predict_proba(X_val)[:, 1]
    end_step(t)

    start_step("Plotting validation results")
    plot_confusion_matrices(y_val, y_val_pred_dt, y_val_pred_knn, dt_label, knn_label)
    plot_performance_comparison(y_val, y_val_pred_dt, y_val_pred_knn, dt_label, knn_label)

    # Pick the model with the higher validation AUC
    auc_knn = roc_auc_score(y_val, val_prob_knn)
    auc_dt = roc_auc_score(y_val, val_prob_dt)
    print_info(f"{knn_label} validation AUC", f"{auc_knn:.3f}")
    print_info(f"{dt_label} validation AUC", f"{auc_dt:.3f}")

    if auc_knn >= auc_dt:
        best_model, best_label, best_color = knn_model, knn_label, KNN_COLOR
    else:
        best_model, best_label, best_color = dt_model, dt_label, DT_COLOR
    print_info("Best model", best_label)

    t = start_step(f"Testing best model ({best_label})")
    y_test_pred = best_model.predict(X_test)
    test_prob = best_model.predict_proba(X_test)[:, 1]
    print_info(f"{best_label} test AUC", f"{roc_auc_score(y_test, test_prob):.3f}")
    end_step(t)

    start_step("Plotting ROC curves and test confusion matrix")
    auc_scores = plot_roc_curves(
        [
            {"label": f"{dt_label} (validation)", "y_true": y_val, "y_score": val_prob_dt, "color": DT_COLOR},
            {"label": f"{knn_label} (validation)", "y_true": y_val, "y_score": val_prob_knn, "color": KNN_COLOR},
        ],
        title="ROC Curves",
    )

    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Best Model ({best_label}) Confusion Matrix - Test Set")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    t = start_step(f"Predicting churn for test file ({best_label})")
    raw_test_df = load_dataset(TEST_PATH)
    clean_test_df, imputer = clean_dataset(raw_test_df, imputer)
    clean_test_df = build_features(clean_test_df)
    real_test_prob = best_model.predict_proba(clean_test_df)[:, 1]

    output = raw_test_df[["CustomerID"]].copy()
    output["Churn"] = real_test_prob
    os.makedirs("output", exist_ok=True)
    output.to_csv(SUBMISSION_PATH, index=False)
    end_step(t)
    print_info("Predictions saved to", SUBMISSION_PATH)

    print()
    print_header("Done")


if __name__ == "__main__":
    main()
