""" BERT pre-training with Hugging Face, using Wikitext
    No finetuning is performed. Typically you would generally use it for fine-tuning
    on a downstream task,.
    Model and tokenizer are uploaded to Vertex AI Model Registry
"""

import glob
import nltk
import random
import logging
import json
import os

import datasets
import transformers
import tensorflow as tf
from tensorflow import keras

from google.cloud import storage

print(f"Tensorflow version : {tf.__version__}")
print(f"Keras version : {keras.__version__}")
print(f"Transformers version : {datasets.__version__}")
print(f"Datasets version : {transformers.__version__}")


nltk.download("punkt")
# Only log error messages
tf.get_logger().setLevel(logging.ERROR)
# Set random seed
tf.keras.utils.set_random_seed(42)


### Define variables
TOKENIZER_BATCH_SIZE = 256  # Batch-size to train the tokenizer on
TOKENIZER_VOCABULARY = 25000  # Total number of unique subwords the tokenizer can have

BLOCK_SIZE = 128  # Maximum number of tokens in an input sample
NSP_PROB = 0.50  # Probability that the next sentence is the actual next sentence in NSP
SHORT_SEQ_PROB = 0.1  # Probability of generating shorter sequences to minimize the mismatch between pretraining and fine-tuning.
MAX_LENGTH = 512  # Maximum number of tokens in an input sample after padding

MLM_PROB = 0.2  # Probability with which tokens are masked in MLM

TRAIN_BATCH_SIZE = 2  # Batch-size for pretraining the model on
MAX_EPOCHS = 1  # Recommended min 50 epochs for decent performance
LEARNING_RATE = 1e-4  # Learning rate for training the model

MODEL_CHECKPOINT = "bert-base-cased"  # Name of pretrained model from 🤗 Model Hub

# Load the WikiText dataset
from datasets import load_dataset

# set ignore_verification=True
# https://github.com/huggingface/datasets/issues/2969
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", ignore_verifications=True)

print(dataset)

## Training a new Tokenizer
all_texts = [
    doc for doc in dataset["train"]["text"] if len(doc) > 0 and not doc.startswith(" =")
]

def batch_iterator():
    for i in range(0, len(all_texts), TOKENIZER_BATCH_SIZE):
        yield all_texts[i : i + TOKENIZER_BATCH_SIZE]

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

tokenizer = tokenizer.train_new_from_iterator(
    batch_iterator(), vocab_size=TOKENIZER_VOCABULARY
)

## Data Pre-processing
dataset["train"] = dataset["train"].select([i for i in range(1000)])
dataset["validation"] = dataset["validation"].select([i for i in range(1000)])

# We define the maximum number of tokens after tokenization that each training sample
# will have
max_num_tokens = BLOCK_SIZE - tokenizer.num_special_tokens_to_add(pair=True)


def prepare_train_features(examples):
    """Function to prepare features for NSP task

    Arguments:
      examples: A dictionary with 1 key ("text")
        text: List of raw documents (str)
    Returns:
      examples:  A dictionary with 4 keys
        input_ids: List of tokenized, concatnated, and batched
          sentences from the individual raw documents (int)
        token_type_ids: List of integers (0 or 1) corresponding
          to: 0 for senetence no. 1 and padding, 1 for sentence
          no. 2
        attention_mask: List of integers (0 or 1) corresponding
          to: 1 for non-padded tokens, 0 for padded
        next_sentence_label: List of integers (0 or 1) corresponding
          to: 1 if the second sentence actually follows the first,
          0 if the senetence is sampled from somewhere else in the corpus
    """

    # Remove un-wanted samples from the training set
    examples["document"] = [
        d.strip() for d in examples["text"] if len(d) > 0 and not d.startswith(" =")
    ]
    # Split the documents from the dataset into it's individual sentences
    examples["sentences"] = [
        nltk.tokenize.sent_tokenize(document) for document in examples["document"]
    ]
    # Convert the tokens into ids using the trained tokenizer
    examples["tokenized_sentences"] = [
        [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)) for sent in doc]
        for doc in examples["sentences"]
    ]

    # Define the outputs
    examples["input_ids"] = []
    examples["token_type_ids"] = []
    examples["attention_mask"] = []
    examples["next_sentence_label"] = []

    for doc_index, document in enumerate(examples["tokenized_sentences"]):
        current_chunk = []  # a buffer stored current working segments
        current_length = 0
        i = 0

        # We *usually* want to fill up the entire sequence since we are padding
        # to `block_size` anyways, so short sequences are generally wasted
        # computation. However, we *sometimes*
        # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
        # sequences to minimize the mismatch between pretraining and fine-tuning.
        # The `target_seq_length` is just a rough target however, whereas
        # `block_size` is a hard limit.
        target_seq_length = max_num_tokens

        if random.random() < SHORT_SEQ_PROB:
            target_seq_length = random.randint(2, max_num_tokens)

        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    # `a_end` is how many segments from `current_chunk` go into the `A`
                    # (first) sentence.
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []

                    if len(current_chunk) == 1 or random.random() < NSP_PROB:
                        is_random_next = True
                        target_b_length = target_seq_length - len(tokens_a)

                        # This should rarely go for more than one iteration for large
                        # corpora. However, just to be careful, we try to make sure that
                        # the random document is not the same as the document
                        # we're processing.
                        for _ in range(10):
                            random_document_index = random.randint(
                                0, len(examples["tokenized_sentences"]) - 1
                            )
                            if random_document_index != doc_index:
                                break

                        random_document = examples["tokenized_sentences"][
                            random_document_index
                        ]
                        random_start = random.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j])
                            if len(tokens_b) >= target_b_length:
                                break
                        # We didn't actually use these segments so we "put them back" so
                        # they don't go to waste.
                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments
                    else:
                        is_random_next = False
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])

                    input_ids = tokenizer.build_inputs_with_special_tokens(
                        tokens_a, tokens_b
                    )
                    # add token type ids, 0 for sentence a, 1 for sentence b
                    token_type_ids = tokenizer.create_token_type_ids_from_sequences(
                        tokens_a, tokens_b
                    )

                    padded = tokenizer.pad(
                        {"input_ids": input_ids, "token_type_ids": token_type_ids},
                        padding="max_length",
                        max_length=MAX_LENGTH,
                    )

                    examples["input_ids"].append(padded["input_ids"])
                    examples["token_type_ids"].append(padded["token_type_ids"])
                    examples["attention_mask"].append(padded["attention_mask"])
                    examples["next_sentence_label"].append(1 if is_random_next else 0)
                    current_chunk = []
                    current_length = 0
            i += 1

    # We delete all the un-necessary columns from our dataset
    del examples["document"]
    del examples["sentences"]
    del examples["text"]
    del examples["tokenized_sentences"]

    return examples


tokenized_dataset = dataset.map(
    prepare_train_features,
    batched=True,
    remove_columns=["text"],
    num_proc=1,
)


from transformers import DataCollatorForLanguageModeling

collater = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROB, return_tensors="tf"
)

train = tokenized_dataset["train"].to_tf_dataset(
    columns=["input_ids", "token_type_ids", "attention_mask"],
    label_cols=["labels", "next_sentence_label"],
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    collate_fn=collater,
)

validation = tokenized_dataset["validation"].to_tf_dataset(
    columns=["input_ids", "token_type_ids", "attention_mask"],
    label_cols=["labels", "next_sentence_label"],
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    collate_fn=collater,
)

## Defining the model
from transformers import BertConfig
config = BertConfig.from_pretrained(MODEL_CHECKPOINT)

from transformers import TFBertForPreTraining
model = TFBertForPreTraining(config)

# Now we define our optimizer and compile the model. The loss calculation is handled
# internally and so we need not worry about that!

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

model.compile(optimizer=optimizer)

logging.info("Training ....")
model.fit(train, validation_data=validation, epochs=MAX_EPOCHS)

# Save tokenizer and model locally
logging.info("Saving model locally ....")
tokenizer.save_pretrained(f'model_tokenizer')
model.save_pretrained('model_output', saved_model=True)

output_directory = os.environ['AIP_MODEL_DIR']

logging.info("Saving model and tokenizer to GCS ....")
logging.info(f'Exporting SavedModel to: {output_directory}')

# extract GCS bucket_name from AIP_MODEL_DIR, ex: argolis-vertex-europewest4
bucket_name = output_directory.split("/")[2] # without gs://

# extract GCS object_name from AIP_MODEL_DIR, ex: aiplatform-custom-training-2023-02-22-16:31:12.167/model/
object_name = "/".join(output_directory.split("/")[3:])

directory_path = "model_output" # local
# Upload model to GCS
client = storage.Client()
rel_paths = glob.glob(directory_path + '/**', recursive=True)
bucket = client.get_bucket(bucket_name)
for local_file in rel_paths:
    remote_path = f'{object_name}{"/".join(local_file.split(os.sep)[1:])}'
    logging.info(remote_path)
    if os.path.isfile(local_file):
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(local_file)


directory_path = "model_tokenizer" # local
# Upload tokenizer to GCS
client = storage.Client()
rel_paths = glob.glob(directory_path + '/**', recursive=True)
bucket = client.get_bucket(bucket_name)
for local_file in rel_paths:
    remote_path = f'{object_name}{"/".join(local_file.split(os.sep)[1:])}'
    logging.info(remote_path)
    if os.path.isfile(local_file):
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(local_file)