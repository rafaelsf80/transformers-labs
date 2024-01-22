from fastapi import FastAPI, Request

import json
import numpy as np
import pickle
import os

from transformers import BartTokenizer, TFBartForConditionalGeneration

app = FastAPI()


@app.get(os.environ['AIP_HEALTH_ROUTE'], status_code=200)
def health():
    return {"status": "healthy"}


@app.post(os.environ['AIP_PREDICT_ROUTE'])
async def predict(request: Request):
    body = await request.json()

    instances = body["instances"]

    from transformers import MarianMTModel, MarianTokenizer

    outputs = []
    for instance in instances:
        #model_name = 'opus-mt-tr-en-finetuned-tr-to-en/checkpoint-38000'
        model_name = '../model-output-tr-en'
        tokenizer1 = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        translated = model.generate(**tokenizer1(instance, return_tensors="pt", padding=True))

        outputs.append([tokenizer1.decode(t, skip_special_tokens=True) for t in translated])   

    return {"predictions": [outputs]}