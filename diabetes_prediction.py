import pickle
import numpy as np

with open("diabetes_pred_model.pkl", "rb") as f:
    model = pickle.load(f)

user_input = (5, 166, 72, 19, 175, 25.8, 0.587, 51)
check_diabetic = {0: "Non Diabetic", 1: "Diabetic"}

def predict_results(user_input):
    input_array = [np.array(user_input)]
    results_int = model.predict(input_array)[0]
    results_output = check_diabetic[results_int]
    return results_output


pred_res = predict_results(user_input)
print("User is : {} ".format(pred_res))
