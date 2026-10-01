"""
Data Preprocessing Module for Predictive Maintenance AI.

Handles data loading, schema validation, integrity checks, and stratified train-test splitting.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = [
    'UDI', 'Product ID', 'Type', 'Air temperature [K]',
    'Process temperature [K]', 'Rotational speed [rpm]',
    'Torque [Nm]', 'Tool wear [min]', 'Machine failure'
]

NUMERICAL_RANGES = {
    'Air temperature [K]': (290.0, 315.0),
    'Process temperature [K]': (300.0, 325.0),
    'Rotational speed [rpm]': (1000, 3000),
    'Torque [Nm]': (0.0, 100.0),
    'Tool wear [min]': (0, 300),
}


def load_data(filepath: str = 'data/ai4i2020.csv') -> pd.DataFrame:
    """
    Load dataset from CSV file.
    
    Args:
        filepath: Path to the AI4I 2020 CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
        
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. Please download the official "
            "AI4I 2020 dataset from UCI and place it in 'data/ai4i2020.csv'."
        )
    
    df = pd.read_csv(filepath)
    return df


def validate_data(df: pd.DataFrame) -> dict:
    """
    Validate structure, missing values, and columns of the dataset.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        dict: Validation report containing summary flags and metrics.
        
    Raises:
        ValueError: If required columns are missing or data is empty.
    """
    if df.empty:
        raise ValueError("The provided DataFrame is empty.")
    
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")
    
    null_counts = df[REQUIRED_COLUMNS].isnull().sum().to_dict()
    total_nulls = sum(null_counts.values())
    
    class_counts = df['Machine failure'].value_counts().to_dict()
    total_records = len(df)
    failure_rate = (class_counts.get(1, 0) / total_records) * 100.0 if total_records > 0 else 0
    
    validation_report = {
        'total_rows': total_records,
        'total_columns': df.shape[1],
        'total_missing_values': total_nulls,
        'has_missing_values': total_nulls > 0,
        'class_distribution': class_counts,
        'failure_rate_percent': round(failure_rate, 2),
        'is_valid': total_nulls == 0 and len(missing_cols) == 0
    }
    
    return validation_report


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Perform stratified train/test split to preserve the minority failure class proportion.
    
    Args:
        df: Input DataFrame containing features and 'Machine failure'.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Random seed for reproducibility.
        
    Returns:
        tuple: (train_df, test_df)
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['Machine failure']
    )
    return train_df.copy(), test_df.copy()


if __name__ == '__main__':
    data = load_data()
    report = validate_data(data)
    print("Data Validation Report:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    
    train_data, test_data = split_data(data)
    print(f"\nTrain set size: {len(train_data)}, Test set size: {len(test_data)}")
