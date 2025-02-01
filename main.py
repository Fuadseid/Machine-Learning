from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle
import logging

# Initialize FastAPI
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific frontend URLs in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None  # Handle missing model gracefully

# Define input schema
class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.post("/predict")
async def predict(input: DiabetesInput):
    """Predicts diabetes based on input data."""
    if model is None:
        return {"error": "Model not available. Please check the server logs."}
    
    try:
        input_data = np.array([[  # Convert input to NumPy array
            input.Pregnancies, input.Glucose, input.BloodPressure, input.SkinThickness,
            input.Insulin, input.BMI, input.DiabetesPedigreeFunction, input.Age
        ]])

        prediction = model.predict(input_data)
        result = int(prediction[0])
        logging.info(f"Prediction: {result}")
        return {"prediction": result}
    
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return {"error": "An error occurred while making the prediction."}

# To run this application, use:
# uvicorn main:app --reload
