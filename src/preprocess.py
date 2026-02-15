import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df):
    # Feature Engineering
    df["Car_Age"] = 2024 - df["Year"]
    df.drop("Year", axis=1, inplace=True)

    # Drop noisy feature if exists
    if "City" in df.columns:
        df.drop("City", axis=1, inplace=True)

    # Encode categorical features
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Selling_Price_Lakh", axis=1)
    y = df["Selling_Price_Lakh"]

    return X, y


def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)
