# reMine-ML

## Introduction
```
👊 A model that predicts emotions based on situational data
```  

## How to run?
### Train
```
python emotion_model.py
``` 
### Predict
```
#with best_model.h5
sentiment_model_best = tf.keras.models.load_model(BEST_MODEL_NAME, custom_objects={'TFBertForSequenceClassification': TFBertForSequenceClassification})

predicted_value = sentiment_model_best.predict(test_x)
predicted_label = np.argmax(predicted_value, axis = 1)
``` 
### Details
- Use TensorFlow
