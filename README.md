# 🚗 Car Price Prediction (India)

### Random Forest Regression \| Structured ML Pipeline \| Dockerized Deployment


## 📌 Project Overview

This project builds a complete end-to-end Machine Learning pipeline to
predict used car selling prices in India using **Random Forest
Regression**.

It includes:

-   Modular architecture (`src/`)
-   Reproducible training pipeline
-   Cross-validation
-   Model evaluation
-   Streamlit deployment
-   Docker containerization
-   GitHub CI workflow


## 🧠 Problem Statement

Accurate pricing of used cars depends on:

-   Vehicle age
-   Brand value
-   Mileage
-   Fuel type
-   Ownership history
-   Transmission type

This project builds a supervised regression model that predicts car
price (in Lakhs INR) using structured feature engineering and ensemble
learning.


## 📊 Dataset

Structured Indian dataset (10,000 rows).

### Features:

-   Brand
-   Year
-   Driven_kms
-   Fuel_Type
-   Transmission
-   Seller_Type
-   Owner
-   City
-   Selling_Price_Lakh (Target)

Feature engineering:

Car_Age = 2024 - Year


## 🏗 Project Structure

    car-price-prediction/
    │
    ├── data/
    │   └── car.csv
    │
    ├── src/
    │   ├── preprocess.py
    │   ├── train.py
    │   └── predict.py
    │
    ├── models/               (ignored in git)
    │
    ├── train_pipeline.py
    ├── app.py
    ├── Dockerfile
    ├── requirements.txt
    └── .github/workflows/ci.yml


## 🔬 Machine Learning Approach

Algorithm Used: - Random Forest Regressor (500 estimators)

Why Random Forest?

-   Handles nonlinear relationships
-   Reduces overfitting via ensemble learning
-   Works well on tabular data
-   Robust to feature scaling

## 📈 Model Performance

  |Metric                |Value   |
  |--------------------- |--------|
  |Cross Validation R²   |\~0.92  |
  |Test R²               |\~0.93  |
  |RMSE                  |\~0.63  |
  |MAE                   |\~0.47  |

The model explains approximately 93% of variance in car price.


## ⚙️ Training Pipeline

Run from project root:

    python train_pipeline.py

Pipeline steps:

1.  Data loading
2.  Feature engineering
3.  One-hot encoding
4.  Train-test split
5.  Cross-validation
6.  Model training
7.  Model evaluation
8.  Model artifact saving


## 🌐 Streamlit Web Application

Run locally:

    streamlit run app.py

The app allows users to input car specifications and get price
predictions instantly.


## 🐳 Docker Deployment

Build:

    docker build -t car-price-app .

Run:

    docker run -p 8502:8501 car-price-app

The Dockerfile automatically trains the model during build to ensure
reproducibility.


## 🔁 CI Pipeline (GitHub Actions)

On every push to main:

-   Python environment is created
-   Dependencies installed
-   Training pipeline executed


## 🧩 Engineering Highlights

-   Modular ML architecture
-   Feature schema preservation
-   Model artifact management
-   Containerized deployment
-   CI automation


## 🚀 Future Improvements

-   Hyperparameter tuning (GridSearchCV)
-   MLflow experiment tracking
-   Data versioning (DVC)
-   Model registry integration
-   REST API deployment
-   Model drift monitoring


## 👨‍💻 Author

Mohammad Alquamah Ansari\
B.Sc. Artificial Intelligence\
Aspiring ML / GenAI Engineer

## 🏁 Conclusion

This project demonstrates how to build structured ML systems beyond
notebook experimentation, incorporating deployment and automation
practices aligned with real-world engineering workflows.
