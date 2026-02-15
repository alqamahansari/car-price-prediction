import streamlit as st
import pandas as pd
import joblib

# Load model and features
model = joblib.load("models/rf_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

st.title("🚗 Car Price Predictor")

brand = st.selectbox("Brand", [
    "Maruti", "Hyundai", "Tata", "Mahindra", "Honda",
    "Toyota", "Kia", "Renault", "Skoda", "Volkswagen"
])

year = st.slider("Year", 2008, 2024, 2018)
kms = st.number_input("Driven KMs", min_value=5000, max_value=300000, step=1000)
fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
seller = st.selectbox("Seller Type", ["Dealer", "Individual"])
owner = st.selectbox("Owner", [0,1,2])

if st.button("Predict Price"):

    car_age = 2024 - year

    input_dict = {
        "Driven_kms": kms,
        "Owner": owner,
        "Car_Age": car_age
    }

    input_df = pd.DataFrame([input_dict])

    # Add missing columns
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_columns]

    prediction = model.predict(input_df)

    st.success(f"Estimated Price: ₹ {prediction[0]:.2f} Lakh")
