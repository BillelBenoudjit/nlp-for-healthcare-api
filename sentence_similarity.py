import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers import util


async def predict_similar_cases(model_id, data):
    model = SentenceTransformer(model_id)
    query_embedding = model.encode(data[0], convert_to_tensor=True, show_progress_bar=False)
    compare_embeddings = model.encode(data[1:], convert_to_tensor=True, show_progress_bar=False)

    hits = util.semantic_search(query_embedding, compare_embeddings, top_k=3)
    predictions = hits[0]  # Get the hits for the first query
    return predictions

