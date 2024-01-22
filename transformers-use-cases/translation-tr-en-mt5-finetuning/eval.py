import logging
import sacrebleu
import pandas as pd
from simpletransformers.t5 import T5Model, T5Args


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


model_args = T5Args()
model_args.max_length = 512
model_args.length_penalty = 1
model_args.num_beams = 10

model = T5Model("mt5", "outputs", args=model_args)

# Prepare data for evaluation

eval_df = pd.read_csv("data/eval.tsv", sep="\t").astype(str)

turkish_truth = [eval_df.loc[eval_df["prefix"] == "translate english to turkish"]["target_text"].tolist()]
to_turkish = eval_df.loc[eval_df["prefix"] == "translate english to turkish"]["input_text"].tolist()

english_truth = [eval_df.loc[eval_df["prefix"] == "translate turkish to english"]["target_text"].tolist()]
to_english = eval_df.loc[eval_df["prefix"] == "translate turkish to english"]["input_text"].tolist()

# Predict

turkish_preds = model.predict(to_turkish)

eng_trk_bleu = sacrebleu.corpus_bleu(turkish_preds, turkish_truth)
print("--------------------------")
print("English to Turkish: ", eng_trk_bleu.score)

english_preds = model.predict(to_english)

trk_eng_bleu = sacrebleu.corpus_bleu(english_preds, english_truth)
print("Turkish to English: ", trk_eng_bleu.score)