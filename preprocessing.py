import pandas as pd
import numpy as np

def add_engineered_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def safe_div(n, d):
        n = n.astype(float)
        d = d.astype(float).replace(0, np.nan)
        return (n / d).fillna(0)

    if "Clicks" in df and "Impressions" in df:
        df["CTR"] = safe_div(df["Clicks"], df["Impressions"])

    if "Spent" in df and "Clicks" in df:
        df["CPC"] = safe_div(df["Spent"], df["Clicks"])

    if "Total_Conversion" in df and "Clicks" in df:
        df["CR"] = safe_div(df["Total_Conversion"], df["Clicks"])

    if "Approved_Conversion" in df and "Total_Conversion" in df:
        df["Approval_Rate"] = safe_div(df["Approved_Conversion"], df["Total_Conversion"])

    return df

def clean_user_data(df: pd.DataFrame, expected_features: list) -> pd.DataFrame:
    # Keep only required columns; fill missing with 0
    df = df.copy()
    for col in expected_features:
        if col not in df:
            df[col] = 0
    df = df[expected_features]
    return df.fillna(0)
