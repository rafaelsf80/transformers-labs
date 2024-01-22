from fastapi import FastAPI, Request

import json
import numpy as np
import pickle
import os

from transformers import BartTokenizer, TFBartForConditionalGeneration

model = TFBartForConditionalGeneration.from_pretrained("facebook/bart-large")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

app = FastAPI()


#@app.get(os.environ['AIP_HEALTH_ROUTE'], status_code=200)
@app.get('ping', status_code=200)

def health():
    return {"status": "healthy"}


@app.post(os.environ['AIP_PREDICT_ROUTE'])
async def predict(request: Request):
    body = await request.json()

    instances = body["instances"]
    
    outputs = []
    for instance in instances:
        inputs = tokenizer([instance["in_text"]], max_length=1024, return_tensors="tf")
        summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=len(instance["in_text"].split(" "))//4)
        output = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        outputs.append(output[0])

    return {"predictions": [outputs]}