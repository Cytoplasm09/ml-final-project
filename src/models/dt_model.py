import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


def train_dt_model(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[GridSearchCV, int]:
    """Train decision tree."""
    
    parameters = {"max_depth": np.arange(2, 20), "min_samples_split": np.arange(2, 10)}
    cv = GridSearchCV(DecisionTreeClassifier(random_state=123), param_grid=parameters, scoring='roc_auc', cv=3, n_jobs=-1)
    cv.fit(X_train, y_train)
    depth = cv.best_params_['max_depth']
    min_samples_split = cv.best_params_['min_samples_split']
    return cv, depth