# 🩺 Diabetes Risk Prediction System

An AI-powered Diabetes Risk Prediction web application built using **Machine Learning**, **Random Forest Classifier**, and **Streamlit** with a modern futuristic healthcare UI.

---

# 🚀 Live Demo
https://vikas-diabetes-model.streamlit.app/

---

# 📌 Project Overview

This project predicts whether a person is at risk of diabetes based on several medical attributes such as:

- Pregnancies
- Glucose Level
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

The application uses a trained **Random Forest Classifier** model to make predictions and provide confidence scores.

---

# 🧠 Machine Learning Workflow

The complete ML pipeline followed in this project:

1. Problem Definition
2. Data Collection
3. Data Understanding & EDA
4. Data Preprocessing
5. Feature Scaling
6. Model Training
7. Model Evaluation
8. Hyperparameter Tuning
9. Model Selection
10. Deployment

---

# 📊 Exploratory Data Analysis (EDA)

Performed:
- Histograms
- Correlation Heatmap
- Boxplots
- Outlier Analysis
- Missing Value Handling
- Duplicate Checking

---

# 🤖 Models Used

| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | 0.77 | 0.66 |
| Random Forest | 0.77 | 0.69 |
| XGBoost | 0.75 | 0.67 |

✅ Final Selected Model: **Random Forest Classifier**

Reason:
- Best balance between Accuracy and F1 Score
- Better handling of nonlinear relationships
- Strong overall prediction performance

---

# ⚙️ Hyperparameter Tuning

Performed using:
- GridSearchCV

Tuned Models:
- Logistic Regression
- Random Forest
- XGBoost

---

# 🛠️ Tech Stack

## Machine Learning
- Python
- Scikit-learn
- Pandas
- NumPy
- XGBoost

## Visualization
- Matplotlib
- Seaborn

## Deployment
- Streamlit

## Model Serialization
- Joblib

---

# 🎨 UI Features

- Futuristic AI healthcare landing page
- Dark premium theme
- Responsive design
- Real-time prediction system
- Confidence score display

---

# 📂 Project Structure

```bash
Diabetes-Prediction-System/
│
├── app.py
├── diabetes_model.pkl
├── columns.pkl
├── requirements.txt
├── README.md
│
├── notebooks/
│   └── Diabetesnotebook.ipynb
│
└── dataset/
    └── diabetes.csv