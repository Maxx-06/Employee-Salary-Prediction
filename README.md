# 🧠 Employee Salary Prediction using Machine Learning

This project uses machine learning to predict employee salaries based on features such as education level, years of experience, location, and job role. It is designed to assist HR departments and recruiters in making more data-driven decisions.

---

## 📌 Problem Statement

Manual salary decisions are often inconsistent and biased. Our goal is to build a predictive model that can estimate employee salaries based on historical data, improving fairness and accuracy in compensation.

---

## 🚀 Project Overview

- **Project Type**: Regression Problem
- **Input Features**: Age, Education, Experience, Industry, Job Role, Location, etc.
- **Target Variable**: Salary
- **Tech Stack**:
  - Python
  - Pandas, NumPy
  - Matplotlib, Seaborn
  - Scikit-learn
  - Streamlit (for web app)

---

## 🛠️ System Development Approach

| Component        | Technology        |
|------------------|-------------------|
| Programming Lang | Python             |
| Libraries        | Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn |
| Web App          | Streamlit          |
| IDE              | Jupyter Notebook / Google Colab |
| Deployment       | Localhost (Streamlit) |

---

## 📊 Dataset

- **Source**: [Add your dataset link here – e.g., Kaggle or UCI]
- **Format**: CSV
- **Size**: Approx. N rows × M columns
- **Features Example**:
  - `age`, `education`, `experience`, `industry`, `job_title`, `location`, `salary`

---

## 🔍 Steps Followed (Algorithm & Deployment)

1. **Data Collection**
2. **Data Preprocessing**
   - Handling missing values
   - Label encoding categorical columns
3. **Exploratory Data Analysis (EDA)**
   - Correlation matrix
   - Outlier detection
4. **Feature Selection**
5. **Model Building**
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
6. **Model Evaluation**
   - R² Score, MAE, RMSE
7. **Web App Deployment using Streamlit**

---

## ✅ Results

- **Best Model**: Random Forest Regressor
- **R² Score**: ~0.86 (example)
- **Insights**:
  - Experience and Education are key predictors of salary.

---

## 📱 Streamlit Web App

- The model is deployed using **Streamlit** to create a simple UI.
- Users can input employee details and get real-time salary predictions.

> Run locally using:
```bash
streamlit run app.py
