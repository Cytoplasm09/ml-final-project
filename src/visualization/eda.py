"""Exploratory plots for the churn data."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_eda(df: pd.DataFrame) -> None:
    """Plot the churn rate for each number of days since the last interaction."""
    df.groupby("Last Interaction")["Churn"].mean().plot()

    plt.title("Churn Rate on Each Day Past Last Interaction")

    plt.xlabel("Days since last interaction")
    plt.ylabel("Churn rate")

    plt.show()


if __name__ == "__main__":
    from src.data.load_data import load_dataset
    from src.data.preprocess import clean_dataset

    raw = load_dataset("data/raw/churn_train.csv")
    clean, _ = clean_dataset(raw, None)
    plot_eda(clean)
