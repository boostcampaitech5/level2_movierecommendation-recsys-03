import os
import pandas as pd


def load_data(config) -> pd.DataFrame:
    train_path = os.path.join(config.base_path, config.train_path)
    return pd.read_csv(train_path)


def preprocess(df: pd.DataFrame, column_name: str, value: float):
    df[column_name] = value
    df = df.drop(["time"], axis=1)
    return df
