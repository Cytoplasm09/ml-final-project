import os
import pandas as pd
from .load_data import load_dataset


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset by dropping rows with missing values."""
    clean = df.copy()
    
    one_coding(clean)
    
    clean['Tenure'] = clean['Tenure'].fillna(clean['Tenure'].mean())

    clean['Support Calls'] = pd.to_numeric(clean['Support Calls'].replace('none', 0))
    clean['Support Calls'] = clean['Support Calls'].fillna(clean['Support Calls'].mean())
    clean['Payment Delay'] = clean['Payment Delay'].fillna(clean['Payment Delay'].mean())
    clean['Last Interaction'] = clean['Last Interaction'].fillna(clean['Last Interaction'].mean())
    
    clean = clean.drop(columns=['Gender', 'Last Due Date', 'Last Payment Date', 'Customer Status', 'CustomerID', 'Payment Delay'])
    
    if clean.isna().sum().sum() != 0: 
        raise Exception(clean.isna().sum())
    return clean


def one_coding(df: pd.DataFrame):

    df['Support Calls'] = df['Support Calls'].replace({
        'none': 0,
        })

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


if __name__ == "__main__":
    # Load the raw dataset
    raw = load_dataset("data/raw/train.csv")
    # Clean the dataset
    cleaned = clean_dataset(raw)
    # Ensure the processed directory exists
    os.makedirs("data/processed", exist_ok=True)
    # Save the cleaned data
    processed_path = "data/processed/train.csv"
    cleaned.to_csv(processed_path, index=False)
    print(f"Cleaned data saved to {processed_path}")
