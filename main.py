from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ner import predict

from pydantic import BaseModel


class Data(BaseModel):
    text: str


app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello to the nlp-for-helathcare api"}


@app.post("/ner")
async def predict_entities(data: Data):
    predictions = await predict(data.text)
    return predictions
