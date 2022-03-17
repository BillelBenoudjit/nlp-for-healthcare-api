from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from typing import List

from camembert import predict_ner_camembert
from english_ner import predict_english_ner
from sentence_similarity import predict_similar_cases

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


@app.post("/ner/ncbi/bluebert")
async def predict_entities(data: Data):
    id2label = {0: 'O', 1: 'B-disease', 2: 'I-disease'}
    model_id = "data/clinical_ner_bluebert_pubmed_mimic_uncased"
    predictions = await predict_english_ner(data.text, model_id, id2label)
    return predictions


@app.post("/ner/ncbi/clinicalbert")
async def predict_entities(data: Data):
    id2label = {0: 'O', 1: 'B-disease', 2: 'I-disease'}
    model_id = "data/clinical_ner_Bio_ClinicalBert"
    predictions = await predict_english_ner(data.text, model_id, id2label)
    return predictions


@app.post("/ner/bc5cdr/bluebert")
async def predict_entities(data: Data):
    id2label = {0: 'I-Disease', 1: 'B-Chemical', 2: 'I-Chemical', 3: 'O', 4: 'B-Disease'}
    model_id = "data/BC5CDR_bluebert_pubmed_mimic_uncased"
    predictions = await predict_english_ner(data.text, model_id, id2label)
    return predictions


@app.post("/ner/bionlp13cg/bluebert")
async def predict_entities(data: Data):
    id2label = {0: 'I-Immaterial_anatomical_entity', 1: 'B-Developing_anatomical_structure',
                2: 'B-Pathological_formation',
                3: 'O', 4: 'B-Cancer', 5: 'I-Gene_or_gene_product', 6: 'I-Anatomical_system', 7: 'B-Organism_substance',
                8: 'I-Amino_acid', 9: 'I-Developing_anatomical_structure', 10: 'I-Tissue', 11: 'I-Organism',
                12: 'B-Organism_subdivision', 13: 'I-Organism_substance', 14: 'I-Cellular_component',
                15: 'I-Organism_subdivision', 16: 'B-Simple_chemical', 17: 'B-Immaterial_anatomical_entity',
                18: 'B-Cellular_component', 19: 'B-Tissue', 20: 'B-Cell', 21: 'B-Amino_acid', 22: 'I-Cancer',
                23: 'I-Simple_chemical', 24: 'I-Multi-tissue_structure', 25: 'I-Cell', 26: 'B-Gene_or_gene_product',
                27: 'B-Organism', 28: 'I-Pathological_formation', 29: 'I-Organ', 30: 'B-Organ',
                31: 'B-Multi-tissue_structure', 32: 'B-Anatomical_system'}
    model_id = "data/BioNLP13CG_bluebert_5epochs"
    predictions = await predict_english_ner(data.text, model_id, id2label)
    return predictions


@app.post("/sts")
async def predict_similarity(data: List):
    predictions = await predict_similar_cases("data/camembert-large-semantic-sim-2epochs", data)
    return predictions
