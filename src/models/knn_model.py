import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_knn_model(X_train: pd.DataFrame, y_train: pd.Series) -> KNeighborsClassifier:
    """Train and return a 3-NN classifier."""
    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=25)
    )
    model.fit(X_train, y_train)
    return model
