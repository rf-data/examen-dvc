# src/models/3_train.py
import argparse
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor


def main(
    x_path,
    y_path,
    params_path,
    out_path
        ):

    X_train = pd.read_csv(x_path)
    y_train = pd.read_csv(y_path).squeeze()

    best_params = joblib.load(params_path)

    model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
        **best_params,
            )

    model.fit(X_train, y_train)
    joblib.dump(model, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--x",
        type=str,
        default="data/processed/X_train_scaled.csv",
    )
    parser.add_argument(
        "--y",
        type=str,
        default="data/processed/y_train.csv",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="models/best_params.pkl",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="models/rf_regress.joblib",
    )
    args = parser.parse_args()

    main(
        args.x,
        args.y,
        args.params,
        args.out
    )
