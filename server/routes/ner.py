from fastapi import APIRouter
from pydantic import BaseModel

from server.treatments.camembert import predict_ner_camembert


class Data(BaseModel):
    text: str


router = APIRouter()


@router.post("/camembert")
async def predict_entities(data: Data):
    predictions = await predict_ner_camembert(data.text)
    return predictions
