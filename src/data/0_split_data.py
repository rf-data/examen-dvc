# src/data/0_split_data.py
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def main(
    raw_path, 
    output_dir, 
    test_size = 0.2, 
    random_state = 42):
    
    df = pd.read_csv(raw_path)

    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])

    X = df.drop(columns=["silica_concentrate"]).copy()
    y = df["silica_concentrate"]

    X_train, X_test, y_train, y_test = train_test_split(
                                                    X, y, 
                                                    test_size=test_size, 
                                                    random_state=random_state
                                                        )

    output_dir = output_dir.rstrip("/")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    X_train.to_csv(
        f"{output_dir}/X_train.csv", 
        index=False
        )
    X_test.to_csv(
        f"{output_dir}/X_test.csv", 
        index=False
        )
    y_train.to_csv(
        f"{output_dir}/y_train.csv", 
        index=False
        )
    y_test.to_csv(
        f"{output_dir}/y_test.csv", 
        index=False
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw", 
        type=str, 
        default="data/raw/raw.csv"
        )
    parser.add_argument(
        "--out", 
        type=str, 
        default="data/processed"
        )
    args = parser.parse_args()

    main(
        args.raw, 
        args.out
        )
