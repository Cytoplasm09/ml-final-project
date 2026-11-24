import os
import pandas as pd
from .load_data import load_dataset

from sklearn.impute import SimpleImputer


def clean_dataset(df: pd.DataFrame, imputer: SimpleImputer) -> tuple[pd.DataFrame, SimpleImputer]:
    """Clean the raw churn data.

    Fills in missing values and drops columns that are identifiers or leak the target.
    Pass imputer=None for training data, so a new imputer is fitted; pass the
    returned imputer when cleaning test data, so the training medians are reused.

    Returns the cleaned DataFrame and the fitted imputer.
    """
    df = df.copy()
    
    # replace "none" with 0 for consistency
    df['Support Calls'] = pd.to_numeric(df['Support Calls'].replace('none', 0))
    
    # Convert date columns to datetime format
    df['Last Due Date'] = pd.to_datetime(df['Last Due Date'], format="%m-%d")
    df['Last Payment Date'] = pd.to_datetime(df['Last Payment Date'], format="%m-%d")
    # Fill in missing Payment Delay values by subtracting Last Payment Date with Last Due Date
    df['Payment Delay'] = (df['Last Payment Date'] - df['Last Due Date']).dt.days
    
    # impute missing values only on columns with missing values
    imputer = impute_missing(df, ['Tenure', 'Support Calls', 'Last Interaction'], imputer)

    # drop columns that are not useful for ML or leak data
    # TODO: TRY KEEPING GENDER
    df = df.drop(columns=['Gender', 'Last Due Date', 'Last Payment Date', 'Customer Status', 'CustomerID'])

    return df, imputer

def impute_missing(df: pd.DataFrame, cols: list, imputer: SimpleImputer=None) -> SimpleImputer:
    """Fill missing values in `cols` with the median, in place.

    Fits a new median imputer when `imputer` is None, otherwise reuses the given one.
    Returns the imputer so it can be applied to other data later.
    """
    if imputer is None: 
        imputer = SimpleImputer(strategy='median')
        imputer.fit(df[cols])
    
    df[cols] = imputer.transform(df[cols])

    return imputer

if __name__ == "__main__":
    # Load the raw dataset
    raw = load_dataset("data/raw/churn_train.csv")
    # Clean the dataset
    cleaned, _ = clean_dataset(raw, None)
    # Ensure the processed directory exists
    os.makedirs("data/processed", exist_ok=True)
    # Save the cleaned data
    processed_path = "data/processed/churn_clean.csv"
    cleaned.to_csv(processed_path, index=False)
    print(f"Cleaned data saved to {processed_path}")
