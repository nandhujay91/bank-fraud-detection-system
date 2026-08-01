"""
data_loader.py
Handles loading and merging raw transaction/identity data,
and performing a strict time-based train/test split.

CRITICAL: Never use a random split for fraud data. Fraud patterns evolve
over time, so a random split leaks future fraud tactics into training,
producing falsely optimistic offline metrics that won't hold in production.
"""

import pandas as pd
from pathlib import Path

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast numeric columns to the smallest dtype that fits the data,
    cutting memory usage significantly (float64 -> float32, int64 -> int32).
    Keeps object/categorical columns untouched.
    """
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min >= -2**31 and c_max <= 2**31 - 1:
                    df[col] = df[col].astype("int32")
            else:
                df[col] = df[col].astype("float32")
    return df


def load_raw_data():
    """Load and merge train_transaction + train_identity on TransactionID."""
    transaction = pd.read_csv(RAW_DATA_DIR / "train_transaction.csv")
    identity = pd.read_csv(RAW_DATA_DIR / "train_identity.csv")

    transaction = reduce_memory(transaction)
    identity = reduce_memory(identity)

    merged = transaction.merge(identity, on="TransactionID", how="left")
    return merged


def time_based_split(df: pd.DataFrame, time_col: str = "TransactionDT", test_size: float = 0.2):
    """
    Split data by time, not randomly.
    The earliest (1 - test_size) fraction of transactions -> train.
    The most recent test_size fraction -> test (simulates real deployment,
    where you only ever have past data to predict the future).
    """
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    split_index = int(len(df_sorted) * (1 - test_size))

    train_df = df_sorted.iloc[:split_index].copy()
    test_df = df_sorted.iloc[split_index:].copy()

    print(f"Train range: {train_df[time_col].min()} to {train_df[time_col].max()}")
    print(f"Test range:  {test_df[time_col].min()} to {test_df[time_col].max()}")
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    print(f"Train fraud rate: {train_df['isFraud'].mean()*100:.2f}%")
    print(f"Test fraud rate:  {test_df['isFraud'].mean()*100:.2f}%")

    return train_df, test_df


if __name__ == "__main__":
    data = load_raw_data()
    train_df, test_df = time_based_split(data)