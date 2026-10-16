import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    history = pd.read_csv(args.artifacts / "synthetic_history.csv")
    selected = history[history["sku"] == history["sku"].iloc[0]].tail(52)
    target = args.artifacts / "request.json"
    target.write_text(json.dumps({"history": selected.to_dict("records")}, indent=2), encoding="utf-8")
    print(f"Request saved to {target}")


if __name__ == "__main__":
    main()
