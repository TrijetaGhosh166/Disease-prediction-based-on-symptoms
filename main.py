# 1. Library imports
import uvicorn  # ASGI server
from fastapi import FastAPI
from diseasenote import SymptomData  # Import your SymptomData class
import pickle
import numpy as np

# 2. Create the app object
app = FastAPI()

# Load the trained model for disease prediction
pickle_in = open("disease.pkl", "rb")
classifier = pickle.load(pickle_in)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, Welcome to the Disease Prediction API!'}

# 4. Route to welcome a user by name
@app.get('/welcome/{name}')
def get_name(name: str):
    return {'Welcome to the Disease Prediction API, ': f'{name}'}

# 5. Expose the prediction functionality
@app.post('/predict')
def predict_disease(data: SymptomData):
    # Convert the data into a dictionary format
    data = data.dict()

    # Extract symptoms from the incoming data
    symptoms = [
        data['stomach_pain'], data['acidity'], data['ulcers_on_tongue'],
        data['vomiting'], data['cough'], data['fatigue'],
        data['high_fever'], data['headache'], data['nausea'],
        data['loss_of_appetite'], data['muscle_pain'], data['dizziness']
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
    uvicorn.run(app, host='127.0.0.1', port=8000)
