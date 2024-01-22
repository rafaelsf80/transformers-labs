"""https://medium.com/@tskumar1320/how-to-fine-tune-pre-trained-language-translation-model-3e8a6aace9f"""

import argparse
import logging

import glob
import os
from google.cloud import storage

import numpy as np
import transformers
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

MODEL_CHECKPOINT = "Helsinki-NLP/opus-mt-tr-en"
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

metric = load_metric("sacrebleu")


def get_args():
  '''Parses args.'''

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--epochs',
      required=False,
      default=3,
      type=int,
      help='number of epochs')
  parser.add_argument(
      '--job_dir',
      required=True,
      type=str,
      help='bucket to store saved model, without gs://')
  args = parser.parse_args()
  return args


def preprocess_function(examples):     

    prefix = ""
    max_input_length = 128
    max_target_length = 128
    source_lang = "tr"
    target_lang = "en"

    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def main():
    args = get_args()

    logging.info(f"Transformers version: {transformers.__version__}")
    raw_datasets = load_dataset("wmt16", "tr-en")
    logging.info(raw_datasets["train"][0])

    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

    logging.info("Training arguments, including checkpoints ....")
    batch_size = 16
    model_name = MODEL_CHECKPOINT.split("/")[-1]
    parameters = Seq2SeqTrainingArguments(
        f"{model_name}-finetuned-tr-to-en",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True    
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        parameters,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    logging.info("Finetuning training ....")
    trainer.train()

    # Does it work in GCS ??
    logging.info("Saving model to GCS ....")

    #trainer.save_model(f'{args.job_dir}/model_output')
    trainer.save_model(f'model_output')


    destination_bucket_name = f'{args.job_dir}'  # args.job_dir without gs://
    directory_path = "model_output" # local
    destination_blob_name = "model_output_tr_en"
    # Upload folder to GCS
    client = storage.Client()
    rel_paths = glob.glob(directory_path + '/**', recursive=True)
    bucket = client.get_bucket(destination_bucket_name)
    for local_file in rel_paths:
        remote_path = f'{destination_blob_name}/{"/".join(local_file.split(os.sep)[1:])}'
        print(remote_path)
        if os.path.isfile(local_file):
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)


if __name__ == "__main__":
    main()