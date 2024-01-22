# https://tsmatz.wordpress.com/2022/11/25/huggingface-japanese-summarization/

from datasets import load_dataset
import numpy as np 

ds = load_dataset("csebuetnlp/xlsum", name="spanish")
print(ds)
print(ds["train"][0])

from transformers import AutoTokenizer
t5_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

def tokenize_sample_data(data):
  # Max token size is 14536 and 215 for inputs and labels, respectively.
  # Here I restrict these token size.
  input_feature = t5_tokenizer(data["text"], truncation=True, max_length=1024)
  label = t5_tokenizer(data["summary"], truncation=True, max_length=128)
  return {
    "input_ids": input_feature["input_ids"],
    "attention_mask": input_feature["attention_mask"],
    "labels": label["input_ids"],
  }

tokenized_ds = ds.map(
  tokenize_sample_data,
  remove_columns=["id", "url", "title", "summary", "text"],
  batched=True,
  batch_size=128)


import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





# #### METRICS

# import evaluate
# import numpy as np
# from nltk.tokenize import RegexpTokenizer

# rouge_metric = evaluate.load("rouge")

# # define function for custom tokenization
# def tokenize_sentence(arg):
#   encoded_arg = t5_tokenizer(arg)
#   return t5_tokenizer.convert_ids_to_tokens(encoded_arg.input_ids)

# # define function to get ROUGE scores with custom tokenization
# def metrics_func(eval_arg):
#   preds, labels = eval_arg
#   # Replace -100
#   labels = np.where(labels != -100, labels, t5_tokenizer.pad_token_id)
#   # Convert id tokens to text
#   text_preds = t5_tokenizer.batch_decode(preds, skip_special_tokens=True)
#   text_labels = t5_tokenizer.batch_decode(labels, skip_special_tokens=True)
#   # Insert a line break (\n) in each sentence for ROUGE scoring
#   # (Note : Please change this code, when you perform on other languages except for Japanese)
#   text_preds = [(p if p.endswith(("!", "！", "?", "？", "。")) else p + "。") for p in text_preds]
#   text_labels = [(l if l.endswith(("!", "！", "?", "？", "。")) else l + "。") for l in text_labels]
#   sent_tokenizer_jp = RegexpTokenizer(u'[^!！?？。]*[!！?？。]')
#   text_preds = ["\n".join(np.char.strip(sent_tokenizer_jp.tokenize(p))) for p in text_preds]
#   text_labels = ["\n".join(np.char.strip(sent_tokenizer_jp.tokenize(l))) for l in text_labels]
#   # compute ROUGE score with custom tokenization
#   return rouge_metric.compute(
#     predictions=text_preds,
#     references=text_labels,
#     tokenizer=tokenize_sentence
#   )

import os
from transformers import AutoModelForSeq2SeqLM

# load local model
model = (AutoModelForSeq2SeqLM
         .from_pretrained("./model_trained_for_summarization_spanish")
         .to(device))



from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(
  t5_tokenizer,
  model=model,
  return_tensors="pt")


# Predict with test data (first 5 rows)

from torch.utils.data import DataLoader

sample_dataloader = DataLoader(
  tokenized_ds["test"].with_format("torch"),
  collate_fn=data_collator,
  batch_size=5)
for batch in sample_dataloader:
  with torch.no_grad():
    preds = model.generate(
      batch["input_ids"].to(device),
      num_beams=15,
      num_return_sequences=1,
      no_repeat_ngram_size=1,
      remove_invalid_values=True,
      max_length=128,
    )
  labels = batch["labels"]
  break

# Replace -100 (see above)
labels = np.where(labels != -100, labels, t5_tokenizer.pad_token_id)

# Convert id tokens to text
text_preds = t5_tokenizer.batch_decode(preds, skip_special_tokens=True)
print(labels)
text_labels = t5_tokenizer.batch_decode(labels, skip_special_tokens=True)

# Show result
print("***** Input's Text *****")
print(ds["test"]["text"][0])
print("***** Summary Text (True Value) *****")
print(text_labels[0])
print("***** Summary Text (Generated Text) *****")
print(text_preds[0])
