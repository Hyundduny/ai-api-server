from typing import Union
from fastapi import FastAPI

# model.py를 가져온다.
import model

# API 서버를 생성한다.
app = FastAPI()

operator_model = None

# 모델을 호출한다.
@app.post("/model/train/{type}") 
def train_model(type: str):
    global operator_model
    operator_model = model.Model(type)
    operator_model.train()
    return {"result": "OK"}

@app.post("/model/load/{type}") 
def load_model(type: str):
    global operator_model
    operator_model = model.Model(type)
    operator_model.load_model()
    return {"result": "OK"}

# 모델의 예측 기능을 호출한다.
@app.get("/model/test") 
def test_model():
    if operator_model is None:
        return {"error": "Model not loaded. Please load a model first."}
    
    output = operator_model.predict()
    return {"prediction": output.tolist()}