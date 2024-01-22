""" Translation from Turkish to English using AutoTokenizer, AutoModelForSeq2SeqLM classes
"""

import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("Using Helsinki-NLP models from the Language Technology Research Group at the University of Helsinki")
print(f"Transformers version: {transformers.__version__}")

text = """beklentimin altında bir ürün kaliteli değil"""
# Note Helsinki-NLP/opus-mt-trk-en is not the same as Helsinki-NLP/opus-mt-tr-en
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tr-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-tr-en")

input_ids = tokenizer(text, return_tensors="pt").input_ids
outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3)

print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))