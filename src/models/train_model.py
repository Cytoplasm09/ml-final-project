"""Utility helpers for model training and evaluation."""

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


def plot_roc_curves(curves: list[dict], title: str = "ROC Curves") -> list[float]:
    """Plot several ROC curves on one graph.

    Each entry in `curves` is a dict with:
        label   - name shown in the legend, e.g. "DT (validation)"
        y_true  - the true 0/1 labels
        y_score - the predicted probability of class 1
        color   - optional; give two curves the same color to show they are the same model
        style   - optional line style, e.g. "--" for a test-set curve

    The AUC of each curve is added to its legend entry.
    """
    
    plt.figure(figsize=(7, 6))
    for curve in curves:
        fpr, tpr, _ = roc_curve(curve["y_true"], curve["y_score"])
        auc = roc_auc_score(curve["y_true"], curve["y_score"])
        plt.plot(
            fpr,
            tpr,
            linestyle=curve.get("style", "-"),
            color=curve.get("color"),
            linewidth=2,
            label=f"{curve['label']}  (AUC = {auc:.3f})",
        )
    plt.plot([0, 1], [0, 1], ":", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from src.data.load_data import load_dataset
    from src.data.preprocess import clean_dataset
    from src.features.build_features import build_features
    from src.data.split_data import split_dataset
    from src.models.dt_model import train_dt_model

    raw = load_dataset("data/raw/churn_train.csv")
    clean, _ = clean_dataset(raw, None)
    features = build_features(clean)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(features)
    dt_model = train_dt_model(X_train, y_train)
    plot_roc_curves([
        {"label": "DT (validation)", "y_true": y_val, "y_score": dt_model.predict_proba(X_val)[:, 1]},
    ])
