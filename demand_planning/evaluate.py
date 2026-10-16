import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_pinball_loss


def metrics(actual: np.ndarray, q50: np.ndarray, q85: np.ndarray) -> dict:
    actual = np.asarray(actual)
    denominator = np.abs(actual).sum()
    return {
        "mae": round(float(mean_absolute_error(actual, q50)), 4),
        "wape": round(float(np.abs(actual - q50).sum() / denominator), 4) if denominator else None,
        "pinball_q50": round(float(mean_pinball_loss(actual, q50, alpha=0.5)), 4),
        "pinball_q85": round(float(mean_pinball_loss(actual, q85, alpha=0.85)), 4),
        "coverage_q85": round(float((actual <= q85).mean()), 4),
    }


def inventory_simulation(frame: pd.DataFrame, target_column: str) -> dict:
    holding_cost = 0.6
    unit_margin = 4.0
    total_demand = fulfilled = inventory_units = lost_units = 0.0
    for _, history in frame.groupby("sku"):
        inventory = 0.0
        for row in history.sort_values("week").to_dict("records"):
            order = max(0.0, float(np.ceil(row[target_column])) - inventory)
            inventory += order
            sold = min(inventory, float(row["demand"]))
            lost = float(row["demand"]) - sold
            inventory -= sold
            total_demand += float(row["demand"])
            fulfilled += sold
            lost_units += lost
            inventory_units += inventory
    return {
        "holding_cost": round(inventory_units * holding_cost, 2),
        "lost_margin": round(lost_units * unit_margin, 2),
        "total_inventory_cost": round(inventory_units * holding_cost + lost_units * unit_margin, 2),
        "margin_after_holding": round(fulfilled * unit_margin - inventory_units * holding_cost, 2),
        "unit_fill_rate": round(fulfilled / total_demand, 4) if total_demand else None,
        "assumptions": {
            "lead_time_weeks": 0,
            "forecast_horizon_weeks": 1,
            "initial_inventory": 0,
            "holding_cost_per_unit_week": holding_cost,
            "margin_per_sold_unit": unit_margin,
            "lost_sales_no_backorders": True,
            "unlimited_supply": True,
            "currency": "synthetic units",
        },
    }
