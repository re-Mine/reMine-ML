from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

class InputData(BaseModel):
    numbers: list

model = joblib.load("line_predict.pkl")

@app.post("/predict/")
async def predict(data: InputData):
    try:
        if len(data.numbers) != 7:
            raise HTTPException(status_code=400, detail="Please provide exactly 5 numbers.")
        
        input_numbers = np.array(data.numbers).reshape(1, -1)
        
        prediction = model.predict(input_numbers)
        
        intercept = model.intercept_[0]
        slope = model.coef_[0]
        
        return {"prediction": prediction[0], "intercept": intercept, "slope": slope}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
