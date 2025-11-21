# src/4_evaluate.py
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def main(
    model_path,
    x_path,
    y_path,
    pred_path,
    metric_path
        ):
    # define paths
    pred_path = Path(pred_path)
    pred_path.mkdir(
                parents=True, 
                exist_ok=True
                    )
    metric_path = Path(metric_path)
    metric_path.mkdir(
                parents=True, 
                exist_ok=True
                    )
    pred_file = pred_path / "rf_predict.csv"
    metrics_file = metric_path / "rf_scores.json"

    # prediction
    model = joblib.load(model_path)
    X_test = pd.read_csv(x_path)
    y_test = pd.read_csv(y_path).squeeze()
    y_pred = model.predict(X_test)

    preds_df = pd.DataFrame(
                    {"y_true": y_test.values, 
                    "y_pred": y_pred},
                        )
    preds_df.to_csv(pred_file, index=False)

    # metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {"mse": mse, "rmse": rmse, "r2": r2}
    
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="models/rf_regress.joblib",
    )
    parser.add_argument(
        "--x",
        type=str,
        default="data/processed/X_test_scaled.csv",
    )
    parser.add_argument(
        "--y",
        type=str,
        default="data/processed/y_test.csv",
    )
    parser.add_argument(
        "--pred",
        type=str,
        default="data/processed",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="metrics",
    )
    args = parser.parse_args()

    main(
        args.model,
        args.x,
        args.y,
        args.pred,
        args.metric
    )
