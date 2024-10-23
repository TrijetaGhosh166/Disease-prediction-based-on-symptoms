# 1. Library imports
import uvicorn
from fastapi import FastAPI
from diseasenote import SymptomData  # Import SymptomData instead of BankNote
import numpy as np
import pickle
import pandas as pd

# 2. Create the app object
app = FastAPI()

# Load the trained model for disease prediction
pickle_in = open("disease.pkl", "rb")
classifier = pickle.load(pickle_in)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, Welcome to the Disease Prediction API!'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome to the Disease Prediction API, ': f'{name}'}

# 5. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted disease
@app.post('/predict')
def predict_disease(data: SymptomData):
    # Convert the data into dictionary format
    data = data.dict()

    # Extract symptoms from the incoming data
    symptoms = [
        data['stomach_pain'], data['acidity'], data['ulcers_on_tongue'],
        data['vomiting'], data['cough'], data['fatigue'], data['high_fever'],
        data['headache'], data['nausea'], data['loss_of_appetite'],
        data['muscle_pain'], data['dizziness']
    ]

    # Make a prediction using the classifier
    prediction = classifier.predict([symptoms])
    
    # Assuming your model returns the disease name directly
    predicted_disease = prediction[0]
    
    return {
        'predicted_disease': predicted_disease
    }

# 6. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1:8000', port=8000)
