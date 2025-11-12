import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

MODEL_PATH = Path("model/linear_model.keras")
app = FastAPI(title="TF Linear Regression API")

origins = ["*"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

if not MODEL_PATH.exists():
    import train_and_save 
    train_and_save.train_and_save()
model = tf.keras.models.load_model(MODEL_PATH)

class PredictIn(BaseModel):
    x: float 

class PredictOut(BaseModel):
    y_hat: float

@app.get("/")
def root():
    return {"status": "ok", "model": "linear_regression_tf"}

@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    x = np.array([[payload.x]], dtype="float32")
    y_hat = float(model.predict(x, verbose=0)[0, 0])
    return {"y_hat": y_hat}
