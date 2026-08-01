"""
features.py
Feature engineering for fraud detection.

CRITICAL RULE: every feature must only use information available
BEFORE the transaction being scored (no future leakage). We enforce
this by sorting by time and using expanding/rolling windows.
"""

import pandas as pd


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-card transaction velocity features:
    - count of transactions in the past N days
    - sum of transaction amounts in the past N days
    """
    # Sort by card1 then time — this order is preserved by groupby+rolling below
    df = df.sort_values(["card1", "TransactionDT"]).reset_index(drop=True)

    df["_dt"] = pd.to_timedelta(df["TransactionDT"], unit="s")
    df_indexed = df.set_index("_dt")

    for window, label in [(1, "1d"), (7, "7d")]:
        window_str = f"{window}D"

        count_result = (
            df_indexed.groupby("card1")["TransactionID"]
            .rolling(window_str, closed="left")
            .count()
        )
        amt_result = (
            df_indexed.groupby("card1")["TransactionAmt"]
            .rolling(window_str, closed="left")
            .sum()
        )

        # Assign by position (.values), not by index — avoids the duplicate-label error
        df[f"card1_txn_count_{label}"] = count_result.values
        df[f"card1_txn_amt_sum_{label}"] = amt_result.values

    df = df.drop(columns=["_dt"])

    # Restore chronological order for downstream use
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    velocity_cols = [c for c in df.columns if "card1_txn" in c]
    df[velocity_cols] = df[velocity_cols].fillna(0)

    return df


def add_amount_deviation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    How much does this transaction's amount deviate from the card's
    historical average? Large deviations are a classic fraud signal.
    Uses expanding (all history up to now) mean/std to avoid leakage.
    """
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    grouped = df.groupby("card1")["TransactionAmt"]

    df["card1_amt_expanding_mean"] = grouped.transform(
        lambda x: x.expanding().mean().shift(1)
    )
    df["card1_amt_expanding_std"] = grouped.transform(
        lambda x: x.expanding().std().shift(1)
    )

    df["amt_deviation_from_avg"] = (
        df["TransactionAmt"] - df["card1_amt_expanding_mean"]
    ) / (df["card1_amt_expanding_std"] + 1e-5)

    # First transaction per card has no history -> fill with 0 (neutral)
    df["card1_amt_expanding_mean"] = df["card1_amt_expanding_mean"].fillna(df["TransactionAmt"])
    df["card1_amt_expanding_std"] = df["card1_amt_expanding_std"].fillna(0)
    df["amt_deviation_from_avg"] = df["amt_deviation_from_avg"].fillna(0)

    return df