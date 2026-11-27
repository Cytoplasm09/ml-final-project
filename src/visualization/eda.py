"""Exploratory plots for the churn data."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_eda(df: pd.DataFrame) -> None:
    
    sns.catplot(data=df, x="Churn", kind='count')
    plt.title("Distribution of Churn")
    plt.show()
    
    df["Support Calls"].hist()
    plt.title("Distribution of Support calls")
    plt.xlabel("Number of Support Calls")
    plt.grid(False)
    plt.show()
    
    sns.histplot(data=df, x="Last Interaction", hue="Churn", bins=10, multiple="fill")
    plt.show()

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
