from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.routes.patient import router as PatientRouter
from server.routes.ner import router as NamedEntityRecognitionRouter
from server.routes.sts import router as SemanticSimilarityRouter


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


app.include_router(PatientRouter, tags=["Patient"], prefix="/patient")
app.include_router(NamedEntityRecognitionRouter, tags=["Named Entity Recognition"], prefix="/ner")
app.include_router(SemanticSimilarityRouter, tags=["Semantic Similarity"], prefix="/sts")


@app.get("/")
async def root():
    return {"message": "Hello to the nlp-for-helathcare api"}
