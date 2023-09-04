import pickle
import uvicorn
from fastapi import FastAPI
import json
from pydantic import BaseModel
import requests

check_diabetic = {0: "Non Diabetic", 1: "Diabetic"}

app = FastAPI()


class ModelInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    Bmi: float
    DiabetesPedigree: float
    Age: int


# Load the Saved Model
with open("diabetes_pred_model.pkl", "rb") as f:
    model = pickle.load(f)


# Make an instance of FastAPI
@app.post('/diabetes_prediction')
def diabetes_pred(input_params: ModelInput):
    input_data = input_params.model_dump_json()
    input_dict = json.loads(input_data)

    pregnancies_val = input_dict['Pregnancies']
    glucose_val = input_dict['Glucose']
    bloodpressure_val = input_dict['BloodPressure']
    skinthickness_val = input_dict['SkinThickness']
    insulin_val = input_dict['Insulin']
    bmi_val = input_dict['Bmi']
    diabetespedigree_val = input_dict['DiabetesPedigree']
    age_val = input_dict['Age']

    model_results_as_int = model.predict(
        [pregnancies_val, glucose_val, bloodpressure_val, skinthickness_val, insulin_val, bmi_val, diabetespedigree_val, age_val])
    results_output = check_diabetic[model_results_as_int[0]]
    return results_output


if __name__ == "__main__":
    diabetes_pred(ModelInput)