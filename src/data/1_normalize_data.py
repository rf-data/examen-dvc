# src/data/1_normalize_data.py
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def main(
    path_train,
    path_test,
    output_dir
        ):

    X_train = pd.read_csv(path_train)
    X_test = pd.read_csv(path_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        X_train_scaled, 
        columns=X_train.columns)\
        .to_csv(
            output_dir / "X_train_scaled.csv", 
            index=False
            )

    pd.DataFrame(
        X_test_scaled, 
        columns=X_test.columns
        )\
        .to_csv(
            output_dir / "X_test_scaled.csv", 
            index=False
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", 
        type=str, 
        default="data/processed/X_train.csv"
        )
    parser.add_argument(
        "--test", 
        type=str, 
        default="data/processed/X_test.csv"
        )
    parser.add_argument(
        "--out", 
        type=str, 
        default="data/processed"
        )
    args = parser.parse_args()

    main(
        args.train,
        args.test,
        args.out
        )
