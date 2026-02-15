from src.preprocess import load_data, preprocess_data, split_data
from src.train import train_model, evaluate_model, save_model
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import joblib

df = load_data("data/car.csv")

X, y = preprocess_data(df)

joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

X_train, X_test, y_train, y_test = split_data(X, y)

rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="r2")
print("Cross Validation R2 (Mean):", cv_scores.mean())

model = train_model(X_train, y_train)

evaluate_model(model, X_test, y_test)

save_model(model, "models/rf_model.pkl")