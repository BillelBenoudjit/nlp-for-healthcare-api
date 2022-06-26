from fastapi import APIRouter
from pydantic import BaseModel

from server.treatments.camembert import predict_ner_camembert
from server.treatments.crf import crf
from server.preprocess import format_text, text2features


class Data(BaseModel):
    text: str


router = APIRouter()


@router.post("/camembert")
async def predict_entities(data: Data):
    predictions = await predict_ner_camembert(data.text)
    return predictions


@router.post("/crf")
async def predict_entities(data: Data):
    sentences = []
    tokens = format_text(data.text)
    sentences.append(tokens)
    features = [text2features(sentence) for sentence in sentences]
    predictions = crf(data.text, tokens, features, 'server/data/crf_model_DEFT8.sav')
    return predictions
