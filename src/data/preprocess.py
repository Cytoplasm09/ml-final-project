import os
import pandas as pd
from .load_data import load_dataset

#imputer
from sklearn.impute import SimpleImputer


def clean_dataset(df: pd.DataFrame, imputer: SimpleImputer) -> pd.DataFrame:
    """Clean the dataset by dropping rows with missing values."""
    clean = df.copy()
    
    one_coding(clean)
    
    # Convert date columns to datetime format
    clean['Last Due Date'] = pd.to_datetime(clean['Last Due Date'], format="%m-%d")
    clean['Last Payment Date'] = pd.to_datetime(clean['Last Payment Date'], format="%m-%d")
    # Fill in missing Payment Delay values by subtracting Last Payment Date with Last Due Date
    clean['Payment Delay'] = (clean['Last Payment Date'] - clean['Last Due Date']).dt.days
    
    clean['Support Calls'] = pd.to_numeric(clean['Support Calls'].replace('none', 0))
    
    # impute missing values only on columns with missing values
    imputer = impute_missing(clean, ['Tenure', 'Support Calls', 'Last Interaction'], imputer)

    # drop columns that are not useful for ML or leak data
    # TODO: TRY KEEPING GENDER
    clean = clean.drop(columns=['Gender', 'Last Due Date', 'Last Payment Date', 'Customer Status', 'CustomerID'])

    return clean, imputer


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


def impute_missing(df: pd.DataFrame, cols: list, imputer: SimpleImputer=None) -> SimpleImputer:
    if imputer is None: 
        imputer = SimpleImputer(strategy='median')
        imputer.fit(df[cols])
    
    df[cols] = imputer.transform(df[cols])

    return imputer

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
