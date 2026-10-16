import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMRegressor

from .data import make_panel
from .evaluate import inventory_simulation, metrics
from .features import FEATURES, build_features


def train(output: Path, seed: int = 42, weeks: int = 160, skus: int = 24) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    panel = make_panel(weeks=weeks, skus=skus, seed=seed)
    sku_mapping = {sku: index for index, sku in enumerate(sorted(panel["sku"].unique()))}
    frame = build_features(panel, sku_mapping).dropna(subset=FEATURES + ["demand"])
    dates = sorted(frame["week"].unique())
    train_stop = int(len(dates) * 0.6)
    validation_stop = int(len(dates) * 0.8)
    splits = {
        "train": frame[frame["week"] < dates[train_stop]].copy(),
        "validation": frame[(frame["week"] >= dates[train_stop]) & (frame["week"] < dates[validation_stop])].copy(),
        "test": frame[frame["week"] >= dates[validation_stop]].copy(),
    }
    training = splits["train"]
    models = {}
    for alpha, name in [(0.5, "q50"), (0.85, "q85")]:
        model = LGBMRegressor(
            objective="quantile", alpha=alpha, n_estimators=220,
            learning_rate=0.04, num_leaves=15, min_child_samples=25,
            reg_lambda=1.0, random_state=seed, n_jobs=1, verbosity=-1,
            deterministic=True, force_col_wise=True,
        )
        model.fit(training[FEATURES], training["demand"], categorical_feature=["sku_index"])
        models[name] = model
    residual = training["demand"] - training["rolling_mean_4"]
    residual_q50 = float(residual.quantile(0.5))
    residual_q85 = float(residual.quantile(0.85))
    report = {"data_source": "deterministic synthetic demand", "seed": seed, "splits": {}}
    scored = []
    for name, subset in splits.items():
        subset["q50"] = np.maximum(0, models["q50"].predict(subset[FEATURES]))
        subset["q85"] = np.maximum(subset["q50"], models["q85"].predict(subset[FEATURES]))
        subset["baseline_q50"] = np.maximum(0, subset["rolling_mean_4"] + residual_q50)
        subset["baseline_q85"] = np.maximum(subset["baseline_q50"], subset["rolling_mean_4"] + residual_q85)
        report["splits"][name] = {
            "start": str(subset["week"].min().date()),
            "end": str(subset["week"].max().date()),
            "rows": len(subset),
            "lightgbm": metrics(subset["demand"].to_numpy(), subset["q50"].to_numpy(), subset["q85"].to_numpy()),
            "rolling_baseline": metrics(subset["demand"].to_numpy(), subset["baseline_q50"].to_numpy(), subset["baseline_q85"].to_numpy()),
        }
        subset["split"] = name
        scored.append(subset)
    test = scored[-1]
    report["test_by_sku"] = {
        sku: metrics(group["demand"].to_numpy(), group["q50"].to_numpy(), group["q85"].to_numpy())
        for sku, group in test.groupby("sku")
    }
    report["inventory_simulation"] = {
        "split": "test", "lightgbm": inventory_simulation(test, "q85"),
        "rolling_baseline": inventory_simulation(test, "baseline_q85"),
    }
    bundle = {"models": models, "sku_mapping": sku_mapping, "features": FEATURES, "seed": seed}
    joblib.dump(bundle, output / "model.joblib")
    panel.to_csv(output / "synthetic_history.csv", index=False)
    pd.concat(scored).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame({
        "feature": FEATURES,
        "q50_gain": models["q50"].booster_.feature_importance(importance_type="gain"),
        "q85_gain": models["q85"].booster_.feature_importance(importance_type="gain"),
    }).sort_values("q50_gain", ascending=False).to_csv(output / "feature_importance.csv", index=False)
    manifest = {
        "artifact_version": 1, "model_file": "model.joblib", "features": FEATURES,
        "data_sha256": hashlib.sha256(panel.to_csv(index=False).encode()).hexdigest(),
        "seed": seed, "weeks": weeks, "skus": skus,
        "horizon_weeks": 1, "quantiles": [0.5, 0.85],
        "training_cutoff": report["splits"]["train"]["end"],
        "versions": {"python": platform.python_version(), "lightgbm": lightgbm.__version__, "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weeks", type=int, default=160)
    parser.add_argument("--skus", type=int, default=24)
    args = parser.parse_args()
    print(json.dumps(train(args.output, args.seed, args.weeks, args.skus), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
