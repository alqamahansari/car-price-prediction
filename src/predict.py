import joblib
import pandas as pd


def load_model(path):
    return joblib.load(path)


def predict_price(model, input_dict):
    df = pd.DataFrame([input_dict])
    prediction = model.predict(df)
    return prediction[0]
