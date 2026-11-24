import os
import pandas as pd
from sklearn.model_selection import train_test_split
from .load_data import load_dataset


def split_dataset(df: pd.DataFrame):
    """Split the data into 60% train, 20% validation and 20% test sets.

    The split is stratified on Churn so each set keeps the same churn rate.
    Returns X_train, X_val, X_test, y_train, y_val, y_test.
    """
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=123, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    cleaned_path = "data/processed/churn_clean.csv"  # created by src.data.preprocess
    df = load_dataset(cleaned_path)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)

    os.makedirs("data/processed", exist_ok=True)
    pd.concat([X_train, y_train], axis=1).to_csv("data/processed/churn_train.csv", index=False)
    pd.concat([X_val, y_val], axis=1).to_csv("data/processed/churn_validation.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv("data/processed/churn_known_test.csv", index=False)

    print("Train, validation, and test sets have been saved to data/processed/")
