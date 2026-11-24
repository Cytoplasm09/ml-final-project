"""Feature engineering: turn cleaned churn data into model-ready features."""

import pandas as pd
from src.data.load_data import load_dataset
from src.data.preprocess import clean_dataset

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build model features from the cleaned data and return the DataFrame."""
    
    df['Subscription Type'] = df['Subscription Type'].replace({
        'Basic': 0,
        'Standard': 1,
        'Premium': 2
        })

    df['Contract Length'] = df['Contract Length'].replace({
        'Monthly': 0,
        'Quarterly': 1,
        'Annual': 2
        })

    return df

if __name__ == "__main__":
    raw = load_dataset("data/raw/churn_train.csv")
    clean, _ = clean_dataset(raw, None)
    features = build_features(clean)
    print(features.head())
    
