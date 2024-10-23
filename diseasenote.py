from pydantic import BaseModel

# Class describing symptoms for disease prediction
class SymptomData(BaseModel):
    stomach_pain: float
    acidity: float
    ulcers_on_tongue: float
    vomiting: float
    cough: float
    fatigue: float
    high_fever: float
    headache: float
    nausea: float
    loss_of_appetite: float
    muscle_pain: float
    dizziness: float


