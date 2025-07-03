import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
with open("scaler.pkl", "rb") as f:
    scaler =pickle.load(f)
with open("model.pkl", "rb") as f:
    model =pickle.load(f)

st.title("Churn Prediction Application")

st.divider()

st.write("Please enter the values and hit the predict button for getting a prediction.")

st.divider()

age = st.number_input("Enter Age", min_value =10, max_value=100, value=30)
tenure = st.number_input("Enter Tenure", min_value =0, max_value=130, value=10)
monthycharge = st.number_input("Enter Monthly Charge", min_value =30, max_value=150)
gender = st.selectbox("Enter The Gender", ["Male","Female"])

st.divider()

predictbutton = st.button("Predict")
if predictbutton:
    gender_selected = 1 if gender =="female" else 0
    X =[age, gender, tenure, monthycharge]

    X1 = np.array[X]
    X_array = scaler.transform[[X1]]
    prediction = model.predict[X_array][0]
    predicted = "Yes" if prediction ==1 else "No"
    st.write(f"Predicted: {predicted}")

else:
    st.write("Please enter the values and use predict button")



    

