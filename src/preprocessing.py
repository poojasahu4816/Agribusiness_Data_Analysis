"""Preprocessing for modelling: encoding, train/test split and scaling.

Important: the scaler learns its mean and standard deviation from the TRAINING
set only, and is then applied to the test set. This avoids data leakage.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config


def preprocess(clean_df):
    df = clean_df.copy()
    keep = (config.SCALE_COLUMNS + config.CATEGORICAL_COLUMNS + [config.TARGET])
    df = df[keep]
    df = pd.get_dummies(df, columns=config.CATEGORICAL_COLUMNS, dtype=int)
    train, test = train_test_split(df, test_size=config.TEST_SIZE, random_state=config.SEED)
    train, test = train.copy(), test.copy()
    scaler = StandardScaler().fit(train[config.SCALE_COLUMNS])
    train[config.SCALE_COLUMNS] = scaler.transform(train[config.SCALE_COLUMNS])
    test[config.SCALE_COLUMNS] = scaler.transform(test[config.SCALE_COLUMNS])
    info = {
        "train_rows": len(train), "test_rows": len(test), "columns_after_encoding": train.shape[1],
        "scaler_mean": dict(zip(config.SCALE_COLUMNS, scaler.mean_.round(3))),
        "scaler_std": dict(zip(config.SCALE_COLUMNS, scaler.scale_.round(3))),
    }
    return train, test, info
