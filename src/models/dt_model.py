import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def train_dt_model(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    """Train decision tree."""
    model = DecisionTreeClassifier(max_depth=8)
    model.fit(X_train, y_train)
    return model