from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

# Define input data schema
class InputData(BaseModel):
    text: str

# Load the pre-trained model
model = tf.keras.models.load_model('/best_model.h5')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)
vocab_size = len(tokenizer.word_index) + 1

# Define a predict endpoint
@app.post("/predict/")
async def predict(data: InputData):
    try:
        # Preprocess the input data
        processed_data = preprocess_data(data.text)
        # Perform inference
        prediction = model.predict(processed_data)
        # Get class prediction
        predicted_class = get_predicted_class(prediction)
        return {"predicted_class": predicted_class}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def preprocess_data(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    return text

def get_predicted_class(prediction):
    predicted_class = tf.argmax(prediction, axis=-1).numpy()[0]
    return predicted_class.item()
