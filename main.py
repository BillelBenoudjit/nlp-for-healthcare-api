from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.routes.patient import router as PatientRouter
from server.routes.ner import router as NamedEntityRecognitionRouter
from server.routes.sts import router as SemanticSimilarityRouter
import uvicorn

from decouple import config


PORT = config("PORT")

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(PORT))
