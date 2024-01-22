# https://tsmatz.wordpress.com/2022/11/25/huggingface-japanese-summarization/

from datasets import load_dataset

ds = load_dataset("csebuetnlp/xlsum", name="spanish")
print(type(ds))
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

# see https://huggingface.co/docs/transformers/main_classes/configuration
mt5_config = AutoConfig.from_pretrained(
  "google/mt5-small",
  max_length=128,
  length_penalty=0.6,
  no_repeat_ngram_size=2,
  num_beams=15,
)
model = (AutoModelForSeq2SeqLM
         .from_pretrained("google/mt5-small", config=mt5_config)
         .to(device))




from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(
  t5_tokenizer,
  model=model,
  return_tensors="pt")


# ### METRICS

import evaluate
import numpy as np
from nltk.tokenize import RegexpTokenizer

rouge_metric = evaluate.load("rouge")

# define function for custom tokenization
def tokenize_sentence(arg):
  encoded_arg = t5_tokenizer(arg)
  return t5_tokenizer.convert_ids_to_tokens(encoded_arg.input_ids)

# define function to get ROUGE scores with custom tokenization
def metrics_func(eval_arg):
  preds, labels = eval_arg
  # Replace -100
  labels = np.where(labels != -100, labels, t5_tokenizer.pad_token_id)
  # Convert id tokens to text
  text_preds = t5_tokenizer.batch_decode(preds, skip_special_tokens=True)
  text_labels = t5_tokenizer.batch_decode(labels, skip_special_tokens=True)
  # Insert a line break (\n) in each sentence for ROUGE scoring
  # (Note : Please change this code, when you perform on other languages except for Japanese)
  text_preds = [(p if p.endswith(("!", "！", "?", "？", "。")) else p + "。") for p in text_preds]
  text_labels = [(l if l.endswith(("!", "！", "?", "？", "。")) else l + "。") for l in text_labels]
  sent_tokenizer_jp = RegexpTokenizer(u'[^!！?？。]*[!！?？。]')
  text_preds = ["\n".join(np.char.strip(sent_tokenizer_jp.tokenize(p))) for p in text_preds]
  text_labels = ["\n".join(np.char.strip(sent_tokenizer_jp.tokenize(l))) for l in text_labels]
  # compute ROUGE score with custom tokenization
  return rouge_metric.compute(
    predictions=text_preds,
    references=text_labels,
    tokenizer=tokenize_sentence
  )



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
print(metrics_func([preds, labels]))


# # FINETUNING

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
  output_dir = "mt5-summarize-spanish-checkpoints",
  log_level = "error",
  num_train_epochs = 10,
  learning_rate = 5e-4,
  lr_scheduler_type = "linear",
  warmup_steps = 90,
  optim = "adafactor",
  weight_decay = 0.01,
  per_device_train_batch_size = 2,
  per_device_eval_batch_size = 1,
  gradient_accumulation_steps = 16,
  evaluation_strategy = "steps",
  eval_steps = 100,
  predict_with_generate=True,
  generation_max_length = 128,
  save_steps = 500,
  logging_steps = 10,
  push_to_hub = False
)

from transformers import Seq2SeqTrainer
trainer = Seq2SeqTrainer(
  model = model,
  args = training_args,
  data_collator = data_collator,
  compute_metrics = metrics_func,
  train_dataset = tokenized_ds["train"],
  eval_dataset = tokenized_ds["validation"].select(range(20)),
  tokenizer = t5_tokenizer,
)

trainer.train()


import os
from transformers import AutoModelForSeq2SeqLM

# save fine-tuned model in local
os.makedirs("./model_trained_for_summarization_spanish", exist_ok=True)
if hasattr(trainer.model, "module"):
  trainer.model.module.save_pretrained("./model_trained_for_summarization_spanish")
else:
  trainer.model.save_pretrained("./model_trained_for_summarization_spanish")

# load local model
model = (AutoModelForSeq2SeqLM
         .from_pretrained("./model_trained_for_summarization_spanish")
         .to(device))

from torch.utils.data import DataLoader

# Predict with test data (first 5 rows)
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
text_labels = t5_tokenizer.batch_decode(labels, skip_special_tokens=True)

# Show result
print("***** Input's Text *****")
print(ds["test"]["text"][2])
print("***** Summary Text (True Value) *****")
print(text_labels[2])
print("***** Summary Text (Generated Text) *****")
print(text_preds[2])
