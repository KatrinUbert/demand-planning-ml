import numpy as np
import pandas as pd

FEATURES = [
    "sku_index", "week_sin", "week_cos", "trend",
    "lag_1", "lag_2", "lag_4", "lag_52",
    "rolling_mean_4", "rolling_mean_8", "rolling_std_8",
]


def build_features(panel: pd.DataFrame, sku_mapping: dict[str, int]) -> pd.DataFrame:
    frame = panel.sort_values(["sku", "week"]).copy()
    frame["week"] = pd.to_datetime(frame["week"])
    frame["sku_index"] = frame["sku"].map(sku_mapping)
    if frame["sku_index"].isna().any():
        raise ValueError("Unknown SKU; retrain with its history before inference")
    week_number = frame["week"].dt.isocalendar().week.astype(int)
    frame["week_sin"] = np.sin(2 * np.pi * week_number / 52)
    frame["week_cos"] = np.cos(2 * np.pi * week_number / 52)
    frame["trend"] = (frame["week"] - pd.Timestamp("2022-01-03")).dt.days / 7
    grouped = frame.groupby("sku")["demand"]
    for lag in (1, 2, 4, 52):
        frame[f"lag_{lag}"] = grouped.shift(lag)
    for window in (4, 8):
        frame[f"rolling_mean_{window}"] = grouped.transform(
            lambda series: series.shift(1).rolling(window).mean()
        )
    frame["rolling_std_8"] = grouped.transform(
        lambda series: series.shift(1).rolling(8).std()
    )
    frame["as_of"] = frame["week"] - pd.Timedelta(days=7)
    return frame


def next_week_features(panel: pd.DataFrame, sku_mapping: dict[str, int]) -> pd.DataFrame:
    future = panel.groupby("sku", as_index=False)["week"].max()
    future["week"] += pd.Timedelta(days=7)
    future["demand"] = np.nan
    combined = pd.concat([panel, future], ignore_index=True)
    featured = build_features(combined, sku_mapping)
    return featured[featured["demand"].isna()].reset_index(drop=True)
