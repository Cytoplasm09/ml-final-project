import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV


def train_knn_model(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[GridSearchCV, int]:
    """Train and return a k-NN classifier."""
    model = Pipeline([
        ("imputer", SimpleImputer(strategy='median')),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier())
    ])
    parameter = {"knn__n_neighbors": [5, 15, 25, 51, 75, 101]}
    
    cv = GridSearchCV(model, param_grid=parameter, scoring='roc_auc', cv=3, n_jobs=-1)
    cv.fit(X_train, y_train)
    k = cv.best_params_['knn__n_neighbors']
    return cv, k
