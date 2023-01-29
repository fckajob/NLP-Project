import logging

import spacy
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from fastapi.responses import ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware

logger= logging.getLogger()
app= FastAPI()
origins=["http://localhost",
"http://localhost:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

nlp = spacy.load('./models/reviews_1_balanced_full')
def get_prediction(data):
    doc= nlp(data.text)
    #TODO: Get max prediction
    logger.info(doc.cats)
    return doc

@app.post("/api/predict",response_class=ORJSONResponse)
def interface(data:DataModel):
    doc=get_prediction(data)
    response={

                'predictedrating' : int(max(doc.cats,key=lambda k:doc.cats[k])),
                'probability': max(doc.cats.values())
              }
    return ORJSONResponse(response)



