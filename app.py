import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


np.random.seed(42)

ages = np.random.randint(18, 70, 200)
bmis = np.random.uniform(16, 40, 200)

X = np.column_stack((ages, bmis))

y = []
for age, bmi in X:
    if bmi < 23:
        y.append(0)
    elif bmi < 30:
        y.append(1)
    else:
        y.append(2)

y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=300))
])

model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


def predict_risk(age, bmi):
    prediction = model.predict([[age, bmi]])[0]

    if prediction == 0:
        return "Low Risk"
    elif prediction == 1:
        return "Medium Risk"
    else:
        return "High Risk"


def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 24.9:
        return "Normal Weight"
    elif bmi < 29.9:
        return "Overweight"
    else:
        return "Obese"

st.title("BMI ML Health Risk Predictor")

st.write("Model Accuracy:", round(accuracy * 100, 2), "%")

height = st.number_input("Enter Height (cm)", 100.0, 250.0)
weight = st.number_input("Enter Weight (kg)", 30.0, 200.0)
age = st.number_input("Enter Age", 1)

if st.button("Predict"):

    bmi = round(weight / ((height/100)**2), 2)

    risk = predict_risk(age, bmi)
    category = bmi_category(bmi)

    st.subheader(f"Your BMI: {bmi}")
    st.info(f"BMI Category: {category}")
    st.success(f"Predicted Risk Level: {risk}")
