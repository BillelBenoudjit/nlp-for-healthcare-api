from fastapi import APIRouter
from typing import List

from sentence_similarity import predict_similar_cases

router = APIRouter()


@router.post("/")
async def predict_similarity(data: List):
    predictions = await predict_similar_cases("server/data/camembert-large-semantic-sim-2epochs", data)
    return predictions
