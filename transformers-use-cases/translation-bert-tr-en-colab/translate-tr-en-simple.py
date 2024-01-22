""" Translation from Turkish to English using pipeline class (simple way) 
"""

import transformers
from transformers import pipeline

print("Using Helsinki-NLP models from the Language Technology Research Group at the University of Helsinki")
print(f"Transformers version: {transformers.__version__}")

text = """beklentimin altında bir ürün kaliteli değil"""
# Note Helsinki-NLP/opus-mt-trk-en is not the same as Helsinki-NLP/opus-mt-tr-en
translator = pipeline("translation_tr_to_en", model="Helsinki-NLP/opus-mt-tr-en")

outputs = translator(text, clean_up_tokenization_spaces=True) # min_length=100 forces to output 100 words minimum

print(outputs[0]['translation_text'])



