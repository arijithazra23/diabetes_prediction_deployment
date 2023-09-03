"""
arijit.hazra : Diabetes Prediction Webapp
"""
import pickle
import numpy as np
import streamlit as st
from diabetes_prediction import model

# with open("diabetes_pred_model.pkl", "rb") as f:
#     model = pickle.load(f)

# user_input = (5, 166, 72, 19, 175, 25.8, 0.587, 51)
check_diabetic = {0: "Non Diabetic", 1: "Diabetic"}


def predict_results(user_input):
    input_array = [np.array(user_input)]
    results_int = model.predict(input_array)[0]
    results_output = check_diabetic[results_int]
    return results_output


def main():
    st.title("Diabetes Prediction Webapp")

    pregnancy_val = st.text_input("Enter Pregnancy value between (0 -17) ")
    glucose_val = st.text_input("Enter Glucose value between (0 -199) ")
    bloodpressure_val = st.text_input("Enter BloodPressure value between (0 -122) ")
    skinthickness_val = st.text_input("Enter SkinThickness value between (0 -99) ")
    insulin_val = st.text_input("Enter Insulin value between (0 -846) ")
    bmi_val = st.text_input("Enter BMI value between (0 -67) ")
    diabetespedigree_val = st.text_input("Enter DiabetesPedigreeFunction value between (0.07 -2.42) ")
    age_val = st.text_input("Enter Age value between (21 -81) ")

    prediction_results = ""
    if st.button("Submit to Check Diabetes Prediction"):
        user_input_vals = (pregnancy_val, glucose_val, bloodpressure_val, skinthickness_val, insulin_val, bmi_val, \
                           diabetespedigree_val, age_val)

        prediction_results = predict_results(user_input_vals)

    st.success("Patient is :  {}".format(prediction_results))


if __name__ == "__main__":
    main()




