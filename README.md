# Car Price Prediction (India)

An end-to-end Machine Learning system for predicting used car selling prices in India using Random Forest Regression, structured ML pipelines, Streamlit deployment, Docker containerization, and CI automation workflows.

The project demonstrates production-oriented ML engineering practices including modular architecture, reproducible training pipelines, deployment workflows, and automated CI integration.


## Project Overview

This project builds a supervised Machine Learning regression pipeline designed to estimate used car selling prices using structured vehicle and ownership features.

The system integrates:

- Modular ML architecture
- Feature engineering workflows
- Random Forest Regression
- Cross-validation evaluation
- Streamlit web deployment
- Docker containerization
- GitHub Actions CI pipeline
- Reproducible training workflow


## Problem Statement

Accurate pricing of used vehicles depends on multiple factors such as:

- Vehicle age
- Brand value
- Mileage
- Fuel type
- Ownership history
- Transmission type
- Seller category
- City/location

This project predicts used car prices (in Lakhs INR) using ensemble-based supervised regression techniques.


## Dataset

Structured Indian used-car dataset containing approximately 10,000 rows.

### Features

- Brand
- Year
- Driven_kms
- Fuel_Type
- Transmission
- Seller_Type
- Owner
- City
- Selling_Price_Lakh (Target Variable)


## Feature Engineering

The project includes structured preprocessing and engineered features such as:

```text
Car_Age = 2024 - Year
```

Additional preprocessing workflows include:

- One-hot encoding
- Feature schema preservation
- Structured train-test splitting


## Project Structure

```text
car-price-prediction/
│
├── .github/
│   └── workflows/
│
├── data/
│   └── car.csv
│
├── notebooks/
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── .gitignore
├── Dockerfile
├── app.py
├── requirements.txt
├── train_pipeline.py
└── README.md
```


## Machine Learning Approach

### Algorithm Used

- Random Forest Regressor
- 500 estimators


## Why Random Forest?

Random Forest was selected because it:

- Handles nonlinear relationships effectively
- Reduces overfitting through ensemble learning
- Performs strongly on structured tabular datasets
- Provides stable regression performance
- Requires minimal feature scaling


## Model Performance

| Metric | Value |
|--------|--------|
| Cross Validation R² | ~0.92 |
| Test R² | ~0.93 |
| RMSE | ~0.63 |
| MAE | ~0.47 |

The model explains approximately 93% of variance in used car prices.


## Training Pipeline

Run the complete ML workflow from the project root:

```bash
python train_pipeline.py
```

### Pipeline Steps

1. Data loading  
2. Feature engineering  
3. One-hot encoding  
4. Train-test split  
5. Cross-validation  
6. Model training  
7. Model evaluation  
8. Model artifact generation  


## Streamlit Web Application

Launch the application locally:

```bash
streamlit run app.py
```

The web interface allows users to:

- Input vehicle specifications
- Generate real-time price predictions
- Interact with the trained ML model


## Docker Deployment

### Build Docker Image

```bash
docker build -t car-price-app .
```

### Run Docker Container

```bash
docker run -p 8502:8501 car-price-app
```

The Docker workflow automatically trains the model during build to ensure reproducibility and deployment consistency.


## CI Pipeline (GitHub Actions)

The repository integrates GitHub Actions for Continuous Integration workflows.

### CI Workflow Includes

- Python environment setup
- Dependency installation
- Training pipeline execution
- Build validation

The workflow executes automatically on pushes to the main branch.


## Engineering Highlights

- Modular ML architecture
- Structured preprocessing workflows
- Feature schema preservation
- Reproducible training pipeline
- Streamlit deployment
- Docker containerization
- CI automation
- Deployment-ready project structure


## Technologies Used

### Machine Learning & Data Science
- Python
- Scikit-learn
- Pandas
- NumPy

### Deployment & MLOps
- Docker
- GitHub Actions
- Streamlit

### Tools
- Git & GitHub


## Learning Outcomes

Through this project, I explored:

- Ensemble learning workflows
- Regression system development
- Feature engineering pipelines
- Streamlit deployment
- Docker containerization
- CI/CD automation
- Structured ML architecture
- Reproducible ML workflows


## Future Improvements

- Hyperparameter tuning using GridSearchCV
- MLflow experiment tracking
- Data versioning using DVC
- REST API deployment
- Model registry integration
- Model drift monitoring
- Cloud deployment workflows
- Explainable AI visualizations


## Note

Model artifacts and generated files are excluded from version control to maintain repository cleanliness and lightweight deployment workflows.


## Research & Application Areas

This project relates to:

- Machine Learning Engineering
- Regression Systems
- Predictive Analytics
- Auto Pricing Intelligence
- MLOps Foundations
- AI Deployment Systems


## Contributing

Suggestions and improvements are welcome.

Areas of interest include:

- MLOps
- Regression Optimization
- Explainable AI
- Deployment Engineering
- CI/CD Automation


## Author

**Mohammad Alquamah Ansari**  
B.Sc. Artificial Intelligence

GitHub: https://github.com/alqamahansari  
Portfolio: https://alqamahansari.github.io/


## Conclusion

This project demonstrates how structured Machine Learning systems can move beyond notebook experimentation by integrating modular pipelines, deployment workflows, automation, and reproducibility practices aligned with real-world ML engineering environments.
