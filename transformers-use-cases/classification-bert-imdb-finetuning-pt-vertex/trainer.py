import base64
import json
import os
import random
import sys

import google.auth
from google.cloud import aiplatform
from google.cloud.aiplatform import gapic as aip
from google.cloud.aiplatform import hyperparameter_tuning as hpt
from google.protobuf.json_format import MessageToDict

import datasets
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import ClassLabel, Sequence, load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          EvalPrediction, Trainer, TrainingArguments,
                          default_data_collator)

print(f"Notebook runtime: {'GPU' if torch.cuda.is_available() else 'CPU'}")
print(f"PyTorch version : {torch.__version__}")
print(f"Transformers version : {datasets.__version__}")
print(f"Datasets version : {transformers.__version__}")

APP_NAME = "finetuned-bert-classifier"

os.environ["TOKENIZERS_PARALLELISM"] = "false"


dataset = load_dataset("imdb")
dataset


print(
    "Total # of rows in training dataset {} and size {:5.2f} MB".format(
        dataset["train"].shape[0], dataset["train"].size_in_bytes / (1024 * 1024)
    )
)
print(
    "Total # of rows in test dataset {} and size {:5.2f} MB".format(
        dataset["test"].shape[0], dataset["test"].size_in_bytes / (1024 * 1024)
    )
)

dataset["train"][0]


label_list = dataset["train"].unique("label")
label_list

def show_random_elements(dataset, num_examples=2):
    assert num_examples <= len(
        dataset
    ), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
        elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
            df[column] = df[column].transform(
                lambda x: [typ.feature.names[i] for i in x]
            )
    
batch_size = 16
max_seq_length = 128
model_name_or_path = "bert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    use_fast=True,
)
# 'use_fast' ensure that we use fast tokenizers (backed by Rust) from the 🤗 Tokenizers library.
     

tokenizer("Hello, this is one sentence!")

example = dataset["train"][4]
print(example)

tokenizer(
    ["Hello", ",", "this", "is", "one", "sentence", "split", "into", "words", "."],
    is_split_into_words=True,
)

# Dataset loading repeated here to make this cell idempotent
# Since we are over-writing datasets variable
dataset = load_dataset("imdb")

# Mapping labels to ids
# NOTE: We can extract this automatically but the `Unique` method of the datasets
# is not reporting the label -1 which shows up in the pre-processing.
# Hence the additional -1 term in the dictionary
label_to_id = {1: 1, 0: 0, -1: 0}


def preprocess_function(examples):
    """
    Tokenize the input example texts
    NOTE: The same preprocessing step(s) will be applied
    at the time of inference as well.
    """
    args = (examples["text"],)
    result = tokenizer(
        *args, padding="max_length", max_length=max_seq_length, truncation=True
    )

    # Map labels to IDs (not necessary for GLUE tasks)
    if label_to_id is not None and "label" in examples:
        result["label"] = [label_to_id[example] for example in examples["label"]]

    return result


# apply preprocessing function to input examples
dataset = dataset.map(preprocess_function, batched=True, load_from_cache_file=True)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path, num_labels=len(label_list)
)
   

args = TrainingArguments(
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=1,
    weight_decay=0.01,
    output_dir="/tmp/cls",
)

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
     

trainer = Trainer(
    model,
    args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=default_data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()


saved_model_local_path = "./"

trainer.save_model(saved_model_local_path)

history = trainer.evaluate()




