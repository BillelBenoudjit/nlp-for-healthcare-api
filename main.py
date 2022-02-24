from fastapi import FastAPI
from ner import predict

from pydantic import BaseModel


class Data(BaseModel):
    text: str


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello to nlp-for-helathcare api"}


@app.post("/ner")
async def predict_entities(data: Data):
    predictions = await predict(data.text)
    return predictions
