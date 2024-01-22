import os
import pandas as pd

RAW_DATASET_FILE = "dataset_raw_tr_en.txt"
TRAIN_EVAL_SPLIT = 0.95 # 95%

def prepare_translation_datasets(data_path):
    with open(os.path.join(data_path, RAW_DATASET_FILE), "r", encoding="utf-8") as f:
        all_text = f.readlines()
        all_text = [text.strip("\n").strip() for text in all_text]
        turkish_text = all_text[0::4]
        english_text = all_text[2::4]  

    data = []
    for turkish, english in zip(turkish_text, english_text):
        data.append(["translate turkish to english", turkish, english])
        data.append(["translate english to turkish", english, turkish])

    train_df = pd.DataFrame(data[:int(len(data)*TRAIN_EVAL_SPLIT)], columns=["prefix", "input_text", "target_text"])
    eval_df = pd.DataFrame(data[int(len(data)*TRAIN_EVAL_SPLIT):], columns=["prefix", "input_text", "target_text"])
  
    return train_df, eval_df

train_df, eval_df = prepare_translation_datasets("data") # directory

# train and eval datasets in tsv
train_df.to_csv("data/train.tsv", sep="\t")
eval_df.to_csv("data/eval.tsv", sep="\t")