from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from camembert import predict_ner_camembert
from bluebert import predict_ner_bluebert
# from ner import predict_helper

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


@app.post("/ner/camembert")
async def predict_entities(data: Data):
    predictions = await predict_ner_camembert(data.text)
    return predictions


@app.post("/ner/bluebert")
async def predict_entities(data: Data):
    predictions = await predict_ner_bluebert(data.text)
    return predictions


"""
@app.post("/ner/clinicalbert")
async def predict_entities(data: Data):
    predictions = await predict_helper(data.text)
    return predictions
"""
