import numpy as np
import pandas as pd


def make_panel(weeks: int = 160, skus: int = 24, seed: int = 42) -> pd.DataFrame:
    if weeks < 100 or skus < 2:
        raise ValueError("At least 100 weeks and 2 SKUs are required")
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=weeks, freq="W-MON")
    records = []
    for index in range(skus):
        level = rng.uniform(35, 140)
        amplitude = rng.uniform(0.12, 0.35)
        phase = rng.uniform(-0.5, 0.5)
        slope = rng.uniform(-0.05, 0.15)
        previous = 0.0
        for offset, week in enumerate(dates):
            seasonal = amplitude * level * np.sin(2 * np.pi * offset / 52 + phase)
            previous = 0.55 * previous + rng.normal(0, level * 0.06)
            demand = max(0, round(level + seasonal + slope * offset + previous))
            records.append({"sku": f"SKU-{index:03d}", "week": week, "demand": demand})
    return pd.DataFrame(records).sort_values(["sku", "week"]).reset_index(drop=True)


def validate_history(panel: pd.DataFrame, minimum_weeks: int = 52) -> pd.DataFrame:
    required = {"sku", "week", "demand"}
    if not required.issubset(panel.columns):
        raise ValueError(f"Required columns: {sorted(required)}")
    result = panel.loc[:, ["sku", "week", "demand"]].copy()
    result["week"] = pd.to_datetime(result["week"], errors="raise")
    result["demand"] = pd.to_numeric(result["demand"], errors="raise")
    if result.empty or result.isna().any().any():
        raise ValueError("History must contain complete observations")
    if not np.isfinite(result["demand"]).all() or (result["demand"] < 0).any():
        raise ValueError("Demand must be finite and nonnegative")
    if result.duplicated(["sku", "week"]).any():
        raise ValueError("Duplicate SKU-week observations")
    if (result["week"].dt.dayofweek != 0).any():
        raise ValueError("Week must be a Monday")
    result = result.sort_values(["sku", "week"]).reset_index(drop=True)
    for sku, history in result.groupby("sku"):
        if len(history) < minimum_weeks:
            raise ValueError(f"{sku}: at least {minimum_weeks} weeks are required")
        if not history["week"].diff().dropna().eq(pd.Timedelta(days=7)).all():
            raise ValueError(f"{sku}: history must contain consecutive weeks")
    return result
