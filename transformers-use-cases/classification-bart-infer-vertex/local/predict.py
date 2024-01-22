from transformers import MarianMTModel, MarianTokenizer

src_text = ['Bu odun yanmaz.']
#model_name = 'opus-mt-tr-en-finetuned-tr-to-en/checkpoint-38000'
model_name = '../model'
tokenizer1 = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

translated = model.generate(**tokenizer1(src_text, return_tensors="pt", padding=True))

print([tokenizer1.decode(t, skip_special_tokens=True) for t in translated]) 