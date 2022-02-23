from fastapi import FastAPI
from ner import predict

from pydantic import BaseModel

class Data(BaseModel):
    text: str

app = FastAPI()

# sentence = "un homme âgé de 61 ans."
# predictions = predict(sentence)
# print(predictions)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.post("/ner")
async def predict_entities(data: Data):
    predictions = await predict(data.text)
    return {"predictions": predictions}
