import json
import requests

url = 'http://127.0.0.1:8000/diabetes_prediction'

input_data_for_model = {
    'Pregnancies': 6,
    'Glucose': 148,
    'BloodPressure': 72,
    'SkinThickness': 35,
    'Insulin': 0,
    'Bmi': 33,
    'DiabetesPedigree': .0627,
    'Age': 50
}

input_json = json.dumps(input_data_for_model)
print(input_json)

response = requests.post(url=url, data=input_json)
print(response)
