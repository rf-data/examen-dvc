# src/models/2_grid_search.py
import argparse
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def main(
    x_path,
    y_path,
    out_path
        ):

    X_train = pd.read_csv(x_path)
    y_train = pd.read_csv(y_path).squeeze()

    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)

    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 4],
                }

    grid = GridSearchCV(
        base_model,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
            )
    grid.fit(X_train, y_train)

    joblib.dump(grid.best_params_, out_path)


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
        "--out",
        type=str,
        default="models/best_params.pkl",
    )
    args = parser.parse_args()

    main(
        args.x,
        args.y,
        args.out
        )
