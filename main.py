import logging

import spacy

from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware


logger = logging.getLogger()
app = FastAPI()

class DataModel(BaseModel):
    text: str


nlp = spacy.load('./models/reviews_1')

def get_prediction(data):
    doc= nlp(data.text)
    logger.info(doc.cats)
    # TODO: Get max prediction
    return doc


@app.post("/api/predict", response_class=ORJSONResponse)
def inference(data: DataModel):
    #doc = get_prediction(data)
    response = {'predicted rating': 1}

    return ORJSONResponse(response)