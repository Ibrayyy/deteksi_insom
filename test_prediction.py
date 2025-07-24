import pandas as pd
import joblib
from app import preprocess_input

# Load model
model = joblib.load('rforest_model.joblib')

# Test data
test_data = {
    'Gender': 'Perempuan',
    'Age': 30,
    'Occupation': 'Engineer',
    'Sleep Duration': 7.0,
    'Quality of Sleep': 7,
    'Physical Activity Level': 5,
    'Stress Level': 5,
    'BMI Category': 'Normal',
    'Blood Pressure': '120/80',
    'Heart Rate': 80,
    'Daily Steps': 8000
}

# Preprocess data
result = preprocess_input(test_data)

# Make prediction
prediction = model.predict(result)
proba = model.predict_proba(result)

print('Prediction:', prediction[0])
print('Probability:', proba[0])
print('Shape:', result.shape) 