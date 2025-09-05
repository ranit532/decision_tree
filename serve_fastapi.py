from fastapi import FastAPI
from pydantic import BaseModel
from utils.io_utils import latest_joblib
import os
import joblib
import numpy as np

app = FastAPI()

class WineFeatures(BaseModel):
    features: list[float]

# Load the latest model
def load_model():
    model_path = latest_joblib(os.path.join(os.path.dirname(__file__),'models'))
    if model_path:
        return joblib.load(model_path)
    return None

model = load_model()

@app.get('/health')
def health():
    return {'status':'ok'}

@app.get('/model')
def model_info():
    m = latest_joblib(os.path.join(os.path.dirname(__file__),'models'))
    return {'model': os.path.basename(m) if m else None}

@app.post('/predict')
def predict(wine_features: WineFeatures):
    if model is None:
        return {"error": "Model not found"}
    
    features = np.array(wine_features.features).reshape(1, -1)
    prediction = model.predict(features)
    
    # Assuming the model returns a class index, you might want to map it to a class name
    # For the wine dataset, the classes are 0, 1, 2
    return {"prediction": int(prediction[0])}