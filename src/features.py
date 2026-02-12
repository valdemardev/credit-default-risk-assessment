"""Feature engineering utilities for credit risk modeling."""

from __future__ import annotations

import re
import numpy as np
import pandas as pd


def add_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features aligned with the problem statement.

    Business rule implemented (month-aligned):
        OUTSTANDING_m = max(BILL_AMTm - PAY_AMT(m+1), 0)  for m=1..5

    Notes:
        OUTSTANDING_6 cannot be computed because PAY_AMT7 is not present in the dataset.

    Parameters
    ----------
    df_in:
        Raw dataframe with UCI credit card columns.

    Returns
    -------
    pd.DataFrame
        Copy of input with new engineered columns.
    """
    df = df_in.copy()

    # Outstanding balance proxy: bill of month m minus payment from previous month (m+1)
    for m in range(1, 6):
        bill_col = f"BILL_AMT{m}"
        pay_prev_col = f"PAY_AMT{m+1}"
        out_col = f"OUTSTANDING_{m}"
        if bill_col in df.columns and pay_prev_col in df.columns:
            df[out_col] = (df[bill_col] - df[pay_prev_col]).clip(lower=0)

    # Aggregations over outstanding months available (1..5)
    out_cols = [c for c in df.columns if c.startswith("OUTSTANDING_")]
    if out_cols:
        df["OUTSTANDING_SUM"] = df[out_cols].sum(axis=1)
        df["OUTSTANDING_MAX"] = df[out_cols].max(axis=1)
        df["OUTSTANDING_MEAN"] = df[out_cols].mean(axis=1)
        df["OUTSTANDING_POS_MONTHS"] = (df[out_cols] > 0).sum(axis=1)

    # Payment ratios (stabilized to avoid division by zero / negatives)
    for m in range(1, 7):
        bill_col = f"BILL_AMT{m}"
        pay_col = f"PAY_AMT{m}"
        ratio_col = f"PAY_RATIO_{m}"
        if bill_col in df.columns and pay_col in df.columns:
            df[ratio_col] = df[pay_col] / (df[bill_col].abs() + 1.0)

    # Trends: first vs last month snapshots
    if "BILL_AMT1" in df.columns and "BILL_AMT6" in df.columns:
        df["BILL_TREND_1_6"] = df["BILL_AMT1"] - df["BILL_AMT6"]
    if "PAY_AMT1" in df.columns and "PAY_AMT6" in df.columns:
        df["PAY_TREND_1_6"] = df["PAY_AMT1"] - df["PAY_AMT6"]

    # Delinquency dynamics from PAY_1..PAY_6
    pay_status_cols = [c for c in df.columns if re.fullmatch(r"PAY_[1-6]", c)]
    if pay_status_cols:
        df["PAY_MAX"] = df[pay_status_cols].max(axis=1)
        df["PAY_MIN"] = df[pay_status_cols].min(axis=1)
        df["PAY_LATE_MONTHS"] = (df[pay_status_cols] > 0).sum(axis=1)

    return df


def drop_high_corr_features(
    df: pd.DataFrame, threshold: float = 0.98, exclude: list[str] | None = None
) -> list[str]:
    """Return highly correlated numeric columns to consider dropping."""
    exclude = exclude or []
    num_df = df.select_dtypes(include=[np.number]).drop(columns=exclude, errors="ignore")
    if num_df.shape[1] < 2:
        return []
    corr = num_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return [col for col in upper.columns if (upper[col] > threshold).any()]
