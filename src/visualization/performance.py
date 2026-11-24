"""Plots comparing the decision tree and KNN models."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def plot_confusion_matrices(y_test, y_pred_dt, y_pred_knn) -> None:
    """Plot confusion matrices for both models."""
    conf_dt = confusion_matrix(y_test, y_pred_dt)
    conf_knn = confusion_matrix(y_test, y_pred_knn)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(conf_dt, annot=True, fmt='d', cmap='Reds', ax=axes[0])
    axes[0].set_title('Decision Tree')
    sns.heatmap(conf_knn, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title('25-NN')
    plt.tight_layout()
    plt.show()


def plot_performance_comparison(y_test, y_pred_dt, y_pred_knn) -> None:
    """Create a bar chart comparing model metrics."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    dt_scores = [
        accuracy_score(y_test, y_pred_dt),
        precision_score(y_test, y_pred_dt, zero_division=0),
        recall_score(y_test, y_pred_dt),
        f1_score(y_test, y_pred_dt)
    ]
    knn_scores = [
        accuracy_score(y_test, y_pred_knn),
        precision_score(y_test, y_pred_knn),
        recall_score(y_test, y_pred_knn),
        f1_score(y_test, y_pred_knn)
    ]
    df = pd.DataFrame({'Metric': metrics, 'k-NN': knn_scores, 'Decision Tree': dt_scores})
    df.plot(x='Metric', kind='bar', figsize=(8, 5))
    plt.ylim(0, 1)
    plt.title('Model Performance Comparison')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from src.data.load_data import load_dataset
    from src.data.preprocess import clean_dataset
    from src.features.build_features import build_features
    from src.data.split_data import split_dataset
    from src.models.dt_model import train_dt_model
    from src.models.knn_model import train_knn_model

    raw = load_dataset("data/raw/churn_train.csv")
    clean, _ = clean_dataset(raw, None)
    features = build_features(clean)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(features)
    y_pred_dt = train_dt_model(X_train, y_train).predict(X_val)
    y_pred_knn = train_knn_model(X_train, y_train).predict(X_val)
    plot_confusion_matrices(y_val, y_pred_dt, y_pred_knn)
    plot_performance_comparison(y_val, y_pred_dt, y_pred_knn)
