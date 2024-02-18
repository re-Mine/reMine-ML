from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Define input data schema
class InputData(BaseModel):
    numbers_matrix: list

# Load the pre-trained model
model = joblib.load("predict.pkl")

# Define a predict endpoint
@app.post("/predict/")
async def predict(data: InputData):
    try:
        # Convert input to numpy array
        input_matrix = np.array(data.numbers_matrix)
        
        # Validate input matrix shape
        if input_matrix.shape[1] != 5:
            raise HTTPException(status_code=400, detail="Each row should contain exactly 5 numbers.")
        
        # Perform inference
        predictions = model.predict(input_matrix)
        
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
