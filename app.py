import streamlit as st
import pandas as pd
import joblib


model = joblib.load("gb_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Income Prediction App")
st.write("Fill out the form below to predict whether income is >50K or <=50K.")

# 🧾 User Input Form
with st.form("input_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)

    workclass = st.selectbox("Workclass", [
        "Private", "Self-emp-not-inc", "Local-gov", "Not-listed",
        "State-gov", "Self-emp-inc", "Federal-gov"
    ])

    education = st.selectbox("Education", [
        "HS-grad", "Some-college", "Bachelors", "Masters",
        "Assoc-voc", "11th", "Assoc-acdm", "10th",
        "Prof-school", "7th-8th", "9th", "12th", "Doctorate"
    ])

    marital_status = st.selectbox("Marital Status", [
        "Married-civ-spouse", "Never-married", "Divorced",
        "Separated", "Widowed", "Married-spouse-absent"
    ])

    occupation = st.selectbox("Occupation", [
        "Prof-specialty", "Craft-repair", "Exec-managerial", "Adm-clerical",
        "Sales", "Other-service", "Machine-op-inspct", "Transport-moving",
        "Not-listed", "Handlers-cleaners", "Tech-support", "Farming-fishing",
        "Protective-serv", "Priv-house-serv", "Armed-Forces"
    ])

    relationship = st.selectbox("Relationship", [
        "Husband", "Not-in-family", "Own-child", "Unmarried", "Wife", "Other-relative"
    ])

    race = st.selectbox("Race", [
        "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"
    ])

    gender = st.radio("Gender", ["Male", "Female"])

    capital_gain = st.number_input("Capital Gain", min_value=0, step=1)
    capital_loss = st.number_input("Capital Loss", min_value=0, step=1)
    hours_per_week = st.slider("Hours per Week", 1, 99, 40)

    native_country = st.selectbox("Native Country", [
        "United-States", "Mexico", "Philippines", "Germany",
        "Canada", "India", "England", "Other"
    ])

    submitted = st.form_submit_button("Predict")


if submitted:
    input_dict = {
        "age": age,
        "workclass": workclass,
        "education": education,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "gender": gender,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "native-country": native_country
    }

    input_df = pd.DataFrame([input_dict])

    
    input_df['workclass'] = input_df['workclass'].replace(['Without-pay', 'Never-worked'], 'Not-listed')
    input_df['occupation'] = input_df['occupation'].replace(['Priv-house-serv', 'Protective-serv', 'Armed-Forces'], 'Not-listed')
    input_df['native-country'] = input_df['native-country'].replace([
        'Mexico', 'Philippines', 'Germany', 'Canada', 'India', 'England'], 'Other')

    # 🧠 One-hot encode
    input_encoded = pd.get_dummies(input_df)

    
    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    
    input_encoded = input_encoded[model_columns]

    # 📈 Predict using regression model
    prediction = model.predict(input_encoded)[0]
    result = ">50K" if prediction >= 0.5 else "<=50K"

    # ✅ Displaying  Result from model :
    st.success(f"Predicted Income: {result}")
    st.write(f"Prediction Score (0–1): {round(prediction, 4)}")


# from sklearn.linear_model import LinearRegression

# model = LinearRegression()
# model.fit(X_train, y_train)




# y_pred = model.predict(X_test)




# from sklearn.metrics import mean_squared_error, r2_score

# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print("Mean Squared Error:", mse)
# print("R² Score:", r2)